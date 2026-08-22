from dataclasses import dataclass
import importlib
import inspect

import torch
from torch import nn

from .mfreinforce import MFReinforce
from .timing import synchronized_time


@dataclass(frozen=True)
class DiscreteTransportConfig:
    n_train: int | None = None
    lr: float | None = None
    n_particles: int | None = None
    n_logit_gradient: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    lambda_: float | None = None
    eta: float | None = None
    simplex_sigma: float = 1.0
    flow: str = "exact"
    n_flow_particles: int | None = None
    baseline: bool = True
    reuse_state_gradient: bool = True
    seed: int = 0


class DiscreteTransport(MFReinforce):
    def __init__(self, env, policy=None, config=DiscreteTransportConfig()):
        super().__init__(env, policy=policy, config=config)

    @property
    def lambda_(self):
        if self.config.lambda_ is not None:
            return self.config.lambda_
        return getattr(self.env.config, "transport_lambda", 0.2)

    @property
    def eta(self):
        if self.config.eta is not None:
            return self.config.eta
        return getattr(self.env.config, "transport_eta", self.lambda_)

    def sample_simplex(self, generator):
        logits = torch.zeros(self.env.n_states, dtype=self.env.dtype, device=self.env.device)
        logits[:-1] = self.config.simplex_sigma * torch.randn(
            self.env.n_states - 1, dtype=self.env.dtype, device=self.env.device, generator=generator
        )
        return torch.softmax(logits, dim=0)

    def sample_simplex_batch(self, n_particles, generator):
        logits = torch.zeros(n_particles, self.env.n_states, dtype=self.env.dtype, device=self.env.device)
        logits[:, :-1] = self.config.simplex_sigma * torch.randn(
            n_particles,
            self.env.n_states - 1,
            dtype=self.env.dtype,
            device=self.env.device,
            generator=generator,
        )
        return torch.softmax(logits, dim=-1)

    def perturb_law(self, law, q, scale):
        return (1.0 - scale) * law + scale * q

    def simplex_score_h(self, q):
        q = q.clamp_min(1e-12)
        z = torch.log(q[..., :-1] / q[..., -1:])
        a = -z / self.config.simplex_sigma**2
        return a / q[..., :-1] + a.sum(dim=-1, keepdim=True) / q[..., -1:] - 1.0 / q[..., :-1] + 1.0 / q[..., -1:]

    def estimate_state_sensitivities(self, laws, seed, initial_distribution=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        sensitivities = [
            torch.zeros(self.env.n_states, self.n_parameters, dtype=self.env.dtype, device=self.env.device)
            for _ in range(self.horizon + 1)
        ]
        states = [self.initial_states(self.n_logit_gradient, generator, initial_distribution)]
        action_log_probs = []
        h_values = []

        for t in range(self.horizon):
            q = self.sample_simplex_batch(self.n_logit_gradient, generator)
            perturbed_law = self.perturb_law(laws[t], q, self.eta)
            actions, log_probs = self.sample_actions_with_log_probs(t, states[-1], perturbed_law, generator)
            action_log_probs.append(log_probs)
            h_values.append(self.simplex_score_h(q))

            with torch.no_grad():
                states.append(self.sample_next_state(t, states[-1], perturbed_law, actions, generator))

        factor = (1.0 - self.eta) / self.eta
        for target_t in range(1, self.horizon + 1):
            numerator = torch.zeros(self.env.n_states - 1, self.n_parameters, dtype=self.env.dtype, device=self.env.device)

            law_score_sum = torch.zeros(
                self.n_logit_gradient, self.n_parameters, dtype=self.env.dtype, device=self.env.device
            )
            for s in range(target_t):
                law_score_sum = law_score_sum - factor * (h_values[s] @ sensitivities[s][:-1])

            target_states = states[target_t]
            mask = target_states < self.env.n_states - 1
            numerator.index_add_(0, target_states[mask], law_score_sum[mask])
            action_gradients = self.state_indexed_log_prob_gradients(
                torch.stack(action_log_probs[:target_t]), target_states
            )
            numerator = numerator + action_gradients[:-1]

            sensitivities[target_t][:-1] = numerator / self.n_logit_gradient
            sensitivities[target_t][-1] = -sensitivities[target_t][:-1].sum(dim=0)

        return sensitivities

    def trajectory_gradient(self, laws, sensitivities, seed, initial_distribution=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        base_state = self.initial_state(generator, initial_distribution)
        state = self.initial_state(generator, initial_distribution)
        base_rewards = []
        rewards = []
        policy_score = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        perturbation_score = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)

        for t in range(self.horizon):
            law = laws[t]
            base_action, _ = self.sample_action(t, base_state, law, generator)
            q = self.sample_simplex(generator)
            perturbed_law = self.perturb_law(law, q, self.lambda_)
            action, _ = self.sample_action(t, state, perturbed_law, generator)
            base_rewards.append(self.env.reward(base_state, law, base_action))
            rewards.append(self.env.reward(state, perturbed_law, action))

            policy_score = policy_score + self.log_prob_gradient(t, state, perturbed_law, action)
            perturbation_score = perturbation_score + self.simplex_score_h(q) @ sensitivities[t][:-1]

            with torch.no_grad():
                base_state = self.sample_next_state(t, base_state, law, base_action, generator)
                state = self.sample_next_state(t, state, perturbed_law, action, generator)

        q_terminal = self.sample_simplex(generator)
        terminal_law = self.perturb_law(laws[-1], q_terminal, self.lambda_)
        terminal_reward = self.env.terminal_reward(state, terminal_law)
        base_terminal_reward = self.env.terminal_reward(base_state, laws[-1])
        perturbation_score = perturbation_score + self.simplex_score_h(q_terminal) @ sensitivities[-1][:-1]

        trajectory_score = policy_score - ((1.0 - self.lambda_) / self.lambda_) * perturbation_score
        trajectory_return = self.discounted_returns(rewards, terminal_reward)[0]
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        return trajectory_score, trajectory_return, base_return

    def batched_trajectory_gradient(self, laws, sensitivities, seed, initial_distribution=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        base_states = self.initial_states(self.n_particles, generator, initial_distribution)
        states = self.initial_states(self.n_particles, generator, initial_distribution)
        base_rewards = []
        rewards = []
        action_log_probs = []
        law_scores = []

        for t in range(self.horizon):
            law = laws[t]
            base_actions, _ = self.sample_actions_with_log_probs(t, base_states, law.expand(self.n_particles, -1), generator)
            q = self.sample_simplex_batch(self.n_particles, generator)
            perturbed_law = self.perturb_law(law, q, self.lambda_)
            actions, log_probs = self.sample_actions_with_log_probs(t, states, perturbed_law, generator)
            base_rewards.append(self.env.reward(base_states, law, base_actions))
            rewards.append(self.env.reward(states, perturbed_law, actions))
            action_log_probs.append(log_probs)
            law_scores.append(self.simplex_score_h(q) @ sensitivities[t][:-1])

            with torch.no_grad():
                base_states = self.sample_next_state(t, base_states, law, base_actions, generator)
                states = self.sample_next_state(t, states, perturbed_law, actions, generator)

        q_terminal = self.sample_simplex_batch(self.n_particles, generator)
        terminal_law = self.perturb_law(laws[-1], q_terminal, self.lambda_)
        terminal_reward = self.env.terminal_reward(states, terminal_law)
        base_terminal_reward = self.env.terminal_reward(base_states, laws[-1])
        law_scores.append(self.simplex_score_h(q_terminal) @ sensitivities[-1][:-1])

        returns = self.discounted_returns(rewards, terminal_reward)[0]
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        advantages = returns - returns.mean() if self.config.baseline else returns
        action_score = torch.stack(action_log_probs).sum(dim=0)
        action_gradient = self.flat_grad((action_score * advantages.detach()).sum())
        perturbation_score = torch.stack(law_scores).sum(dim=0)
        law_gradient = -((1.0 - self.lambda_) / self.lambda_) * (
            perturbation_score * advantages.detach().unsqueeze(-1)
        ).sum(dim=0)
        return (action_gradient + law_gradient) / self.n_particles, base_return.mean()

    def estimate_gradient(self, seed):
        law_generator = torch.Generator(device=self.env.device)
        law_generator.manual_seed(seed + 30_000)
        initial_distribution = self.sample_initial_distribution(law_generator)
        laws, _ = self.mean_field_law_flow(seed=seed + 20_000, initial_distribution=initial_distribution)
        shared_sensitivities = None
        if self.config.reuse_state_gradient:
            shared_sensitivities = self.estimate_state_sensitivities(
                laws,
                seed + 10_000,
                initial_distribution=initial_distribution,
            )

        if shared_sensitivities is not None:
            return self.batched_trajectory_gradient(
                laws,
                shared_sensitivities,
                seed,
                initial_distribution=initial_distribution,
            )

        gradient = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        return_sum = torch.zeros((), dtype=self.env.dtype, device=self.env.device)
        scores = []
        returns = []

        for b in range(self.n_particles):
            sensitivities = self.estimate_state_sensitivities(
                laws,
                seed + 10_000 + b,
                initial_distribution=initial_distribution,
            )

            score, trajectory_return, base_return = self.trajectory_gradient(
                laws,
                sensitivities,
                seed + b,
                initial_distribution=initial_distribution,
            )
            if self.config.baseline:
                scores.append(score)
                returns.append(trajectory_return)
                return_sum = return_sum + base_return.detach()
            else:
                gradient = gradient + score * trajectory_return.detach()
                return_sum = return_sum + base_return.detach()

        if self.config.baseline:
            scores = torch.stack(scores)
            objectives = torch.stack(returns)
            advantages = objectives - objectives.mean()
            gradient = (scores * advantages.detach().unsqueeze(-1)).sum(dim=0)

        gradient = gradient / self.n_particles
        objective = return_sum / self.n_particles
        return gradient, objective


def train_discrete_transport(env, policy=None, config=DiscreteTransportConfig()):
    return DiscreteTransport(env, policy=policy, config=config).train()


@dataclass(frozen=True)
class ContinuousTransportConfig:
    n_train: int | None = None
    lr: float | None = None
    n_particles: int | None = None
    n_law_gradient: int | None = None
    n_law_particles: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    lambda_: float | None = None
    eta: float | None = None
    rho: float | None = None
    law_chart: str = "mean"
    min_affine_scale: float | None = None
    flow: str = "exact"
    n_flow_particles: int | None = None
    baseline: bool = True
    reuse_state_gradient: bool = True
    seed: int = 0


class ContinuousTransport:
    def __init__(self, env, policy=None, config=ContinuousTransportConfig()):
        if hasattr(env, "n_states"):
            raise TypeError("ContinuousTransport is for continuous state spaces. Use DiscreteTransport instead.")
        if not hasattr(env, "sample_initial"):
            raise TypeError("ContinuousTransport requires an environment with sample_initial.")

        self.env = env
        self.config = config
        self.policy = self._make_policy() if policy is None else policy
        if config.law_chart not in {"mean", "gaussian"}:
            raise ValueError("law_chart must be either 'mean' or 'gaussian'.")
        if config.flow not in {"exact", "particle"}:
            raise ValueError("flow must be either 'exact' or 'particle'.")

    def trainable_parameters(self):
        if isinstance(self.policy, nn.Module):
            return list(self.policy.parameters())
        return [self.policy]

    def optimizer(self):
        return torch.optim.Adam(self.trainable_parameters(), lr=self.lr)

    @property
    def n_train(self):
        return self.env.config.n_train if self.config.n_train is None else self.config.n_train

    @property
    def lr(self):
        return self.env.config.lr if self.config.lr is None else self.config.lr

    @property
    def n_particles(self):
        return self.env.config.n_particles if self.config.n_particles is None else self.config.n_particles

    @property
    def n_law_gradient(self):
        if self.config.n_law_gradient is not None:
            return self.config.n_law_gradient
        if hasattr(self.env.config, "n_law_gradient"):
            return self.env.config.n_law_gradient
        return getattr(self.env.config, "n_logit_gradient", 10)

    @property
    def n_law_particles(self):
        return self.n_particles if self.config.n_law_particles is None else self.config.n_law_particles

    @property
    def n_flow_particles(self):
        return self.n_law_particles if self.config.n_flow_particles is None else self.config.n_flow_particles

    @property
    def horizon(self):
        return self.env.config.T if self.config.horizon is None else self.config.horizon

    @property
    def validation_interval(self):
        return (
            self.env.config.validation_interval
            if self.config.validation_interval is None
            else self.config.validation_interval
        )

    @property
    def lambda_(self):
        if self.config.lambda_ is not None:
            return self.config.lambda_
        if hasattr(self.env.config, "transport_lambda"):
            return self.env.config.transport_lambda
        perturbation_scale = getattr(self.env.config, "perturbation_scale", 0.0)
        return perturbation_scale if perturbation_scale > 0.0 else 0.1

    @property
    def eta(self):
        if self.config.eta is not None:
            return self.config.eta
        if hasattr(self.env.config, "transport_eta"):
            return self.env.config.transport_eta
        return self.lambda_

    @property
    def rho(self):
        if self.config.rho is not None:
            return self.config.rho
        return getattr(self.env.config, "rho", 1.0)

    @property
    def discount(self):
        if hasattr(self.env.config, "discount"):
            return self.env.config.discount
        return getattr(self.env.config, "gamma", 1.0)

    @property
    def n_parameters(self):
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    def _make_policy(self):
        if hasattr(self.env, "zero_policy"):
            return nn.Parameter(self.env.zero_policy())

        module = importlib.import_module(type(self.env).__module__)
        policy_class = getattr(module, f"{type(self.env).__name__}Policy")
        return policy_class(self.env.config)

    def policy_time(self, t):
        if isinstance(self.policy, nn.Module):
            return torch.tensor(float(t), dtype=self.env.dtype, device=self.env.device)
        return t

    def set_flat_gradient(self, gradient):
        offset = 0
        for parameter in self.trainable_parameters():
            next_offset = offset + parameter.numel()
            parameter.grad = gradient[offset:next_offset].reshape_as(parameter).clone()
            offset = next_offset

    def flatten_grads(self, grads):
        pieces = []
        for parameter, grad in zip(self.trainable_parameters(), grads):
            if grad is None:
                pieces.append(torch.zeros_like(parameter).reshape(-1))
            else:
                pieces.append(grad.reshape(-1))
        return torch.cat(pieces)

    def flat_grad(self, value, retain_graph=False):
        grads = torch.autograd.grad(value, self.trainable_parameters(), allow_unused=True, retain_graph=retain_graph)
        return self.flatten_grads(grads)

    def initial_state(self, generator):
        return self.env.sample_initial(1, generator).reshape(())

    def initial_states(self, n_particles, generator):
        return self.env.sample_initial(n_particles, generator)

    def law_argument(self, moment):
        if moment.ndim > 0 and moment.shape[-1] == 2:
            return moment[..., 0]
        return moment

    def sample_action(self, t, state, law, generator):
        with torch.no_grad():
            if hasattr(self.env, "sample_action"):
                action = self.env.sample_action(self.policy, t, state, self.law_argument(law), generator)
            else:
                action = self.env.policy(self.policy, self.policy_time(t), state, self.law_argument(law)).sample()
        return action.reshape(())

    def sample_actions_with_log_probs(self, t, states, law, generator):
        action_law = self.env.policy(self.policy, self.policy_time(t), states, self.law_argument(law))
        with torch.no_grad():
            if hasattr(self.env, "sample_action"):
                actions = self.env.sample_action(self.policy, t, states, self.law_argument(law), generator)
            else:
                actions = action_law.sample()
        return actions, action_law.log_prob(actions.detach())

    def log_prob_gradient(self, t, state, law, action):
        action_law = self.env.policy(self.policy, self.policy_time(t), state, self.law_argument(law))
        log_prob = action_law.log_prob(action)
        grads = torch.autograd.grad(log_prob, self.trainable_parameters(), allow_unused=True)
        return self.flatten_grads(grads)

    def sample_next_state(self, t, state, law, action, generator):
        signature = inspect.signature(self.env.sample)
        if "t" in signature.parameters:
            return self.env.sample(state, self.law_argument(law), action, generator, t=t).reshape_as(state)
        return self.env.sample(state, self.law_argument(law), action, generator).reshape_as(state)

    def sample_perturbation(self, generator, scale):
        if scale <= 0.0:
            raise ValueError("ContinuousTransport perturbation scales must be positive.")
        if self.config.min_affine_scale is not None:
            raise ValueError("min_affine_scale changes the perturbation law and invalidates the Gaussian score.")
        zeta = self.rho * torch.randn((), dtype=self.env.dtype, device=self.env.device, generator=generator)
        affine_scale = 1.0 + scale * zeta
        beta = self.rho * torch.randn((), dtype=self.env.dtype, device=self.env.device, generator=generator)
        return zeta, beta, affine_scale

    def sample_perturbation_batch(self, n_particles, generator, scale):
        if scale <= 0.0:
            raise ValueError("ContinuousTransport perturbation scales must be positive.")
        if self.config.min_affine_scale is not None:
            raise ValueError("min_affine_scale changes the perturbation law and invalidates the Gaussian score.")
        zeta = self.rho * torch.randn(n_particles, dtype=self.env.dtype, device=self.env.device, generator=generator)
        affine_scale = 1.0 + scale * zeta
        beta = self.rho * torch.randn(n_particles, dtype=self.env.dtype, device=self.env.device, generator=generator)
        return zeta, beta, affine_scale

    def perturb_mean(self, mean, zeta, beta, scale):
        return (1.0 + scale * zeta) * mean + scale * beta

    def perturb_moment(self, moment, zeta, beta, scale):
        affine_scale = 1.0 + scale * zeta
        mean = affine_scale * moment[0] + scale * beta
        variance = affine_scale.square() * moment[1]
        return torch.stack([mean, variance], dim=-1)

    def mean_score_h(self, mean, perturbed_mean, scale):
        variance = scale**2 * self.rho**2 * (mean.square() + 1.0)
        centered = perturbed_mean - mean
        variance = variance.clamp_min(1e-12)
        return centered / variance + mean * scale**2 * self.rho**2 * (centered.square() / variance.square() - 1.0 / variance)

    def gaussian_score(self, moment, zeta, beta, affine_scale, scale, sensitivity):
        mean_coefficient = affine_scale * beta / (scale * self.rho**2)
        log_std_coefficient = affine_scale * (zeta - beta * moment[0]) / (scale * self.rho**2) - 1.0
        return mean_coefficient.unsqueeze(-1) * sensitivity[0] + log_std_coefficient.unsqueeze(-1) * sensitivity[1]

    def transport_score(self, moment, perturbed_moment, zeta, beta, affine_scale, scale, sensitivity):
        if self.config.law_chart == "mean":
            return self.mean_score_h(moment[0], perturbed_moment[..., 0], scale).unsqueeze(-1) * sensitivity[0]
        if self.config.law_chart == "gaussian":
            return self.gaussian_score(moment, zeta, beta, affine_scale, scale, sensitivity)
        raise ValueError(f"Unknown law chart: {self.config.law_chart}")

    def mean_field_moment_flow(self, horizon=None, seed=None):
        horizon = self.horizon if horizon is None else horizon

        if self.config.flow == "exact" and hasattr(self.env, "moment_flow"):
            with torch.no_grad():
                means, variances = self.env.moment_flow(self.policy, lambda_=0.0)
            moments = torch.stack([means[: horizon + 1], variances[: horizon + 1].clamp_min(1e-12)], dim=-1)
            return list(moments.detach())

        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(self.config.seed if seed is None else seed)
        states = self.env.sample_initial(self.n_flow_particles, generator)
        moments = [torch.stack([states.mean(), states.var(unbiased=False).clamp_min(1e-12)]).detach()]

        for t in range(horizon):
            moment = moments[-1]
            actions = self.sample_actions_for_population(t, states, moment, generator)
            with torch.no_grad():
                states = self.sample_next_states_for_population(t, states, moment, actions, generator)
            moments.append(torch.stack([states.mean(), states.var(unbiased=False).clamp_min(1e-12)]).detach())

        return moments

    def mean_field_mean_flow(self, horizon=None, seed=None):
        return [moment[0] for moment in self.mean_field_moment_flow(horizon=horizon, seed=seed)]

    def sample_actions_for_population(self, t, states, law, generator):
        with torch.no_grad():
            if hasattr(self.env, "sample_action"):
                return self.env.sample_action(self.policy, t, states, self.law_argument(law), generator)
            return self.env.policy(self.policy, self.policy_time(t), states, self.law_argument(law)).sample()

    def sample_next_states_for_population(self, t, states, law, actions, generator):
        signature = inspect.signature(self.env.sample)
        if "t" in signature.parameters:
            return self.env.sample(states, self.law_argument(law), actions, generator, t=t)
        return self.env.sample(states, self.law_argument(law), actions, generator)

    def discounted_returns(self, rewards, terminal_reward):
        values = [None] * (len(rewards) + 1)
        values[-1] = terminal_reward
        for t in range(len(rewards) - 1, -1, -1):
            values[t] = rewards[t] + self.discount * values[t + 1]
        return values

    def estimate_moment_sensitivities(self, moments, seed):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        sensitivities = [
            torch.zeros(2, self.n_parameters, dtype=self.env.dtype, device=self.env.device)
            for _ in range(self.horizon + 1)
        ]
        states = [self.initial_states(self.n_law_gradient, generator)]
        action_log_probs = []
        perturbations = []

        for t in range(self.horizon):
            zeta, beta, affine_scale = self.sample_perturbation_batch(self.n_law_gradient, generator, self.eta)
            perturbed_moment = self.perturb_moment(moments[t], zeta, beta, self.eta)
            action, log_prob = self.sample_actions_with_log_probs(t, states[-1], perturbed_moment, generator)
            action_log_probs.append(log_prob)
            perturbations.append((zeta, beta, affine_scale, perturbed_moment))

            with torch.no_grad():
                states.append(self.sample_next_state(t, states[-1], perturbed_moment, action, generator))

        for target_t in range(1, self.horizon + 1):
            mean_estimate = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
            second_moment_estimate = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)

            score = torch.zeros(self.n_law_gradient, self.n_parameters, dtype=self.env.dtype, device=self.env.device)
            for s in range(target_t):
                zeta, beta, affine_scale, perturbed_moment = perturbations[s]
                score = score + self.transport_score(
                    moments[s], perturbed_moment, zeta, beta, affine_scale, self.eta, sensitivities[s]
                )

            state = states[target_t]
            log_probs = torch.stack(action_log_probs[:target_t])
            mean_estimate = mean_estimate + (state.unsqueeze(-1) * score).sum(dim=0)
            mean_estimate = mean_estimate + self.flat_grad(
                (log_probs * state.detach().unsqueeze(0)).sum(), retain_graph=True
            )
            second_moment_estimate = second_moment_estimate + (state.square().unsqueeze(-1) * score).sum(dim=0)
            second_moment_estimate = second_moment_estimate + self.flat_grad(
                (log_probs * state.square().detach().unsqueeze(0)).sum(), retain_graph=target_t < self.horizon
            )

            mean_gradient = mean_estimate / self.n_law_gradient
            second_moment_gradient = second_moment_estimate / self.n_law_gradient
            variance_gradient = second_moment_gradient - 2.0 * moments[target_t][0] * mean_gradient

            sensitivities[target_t][0] = mean_gradient
            sensitivities[target_t][1] = variance_gradient / (2.0 * moments[target_t][1].clamp_min(1e-12))

        return sensitivities

    def estimate_mean_sensitivities(self, moments, seed):
        return [sensitivity[0] for sensitivity in self.estimate_moment_sensitivities(moments, seed)]

    def trajectory_gradient(self, moments, sensitivities, seed):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        base_state = self.initial_state(generator)
        state = self.initial_state(generator)
        base_rewards = []
        rewards = []
        score_terms = []

        for t in range(self.horizon):
            moment = moments[t]
            base_action = self.sample_action(t, base_state, moment, generator)
            zeta, beta, affine_scale = self.sample_perturbation(generator, self.lambda_)
            perturbed_moment = self.perturb_moment(moment, zeta, beta, self.lambda_)
            action = self.sample_action(t, state, perturbed_moment, generator)

            law_score = self.transport_score(
                moment, perturbed_moment, zeta, beta, affine_scale, self.lambda_, sensitivities[t]
            )
            action_score = self.log_prob_gradient(t, state, perturbed_moment, action)
            score_terms.append(law_score + action_score)
            base_rewards.append(self.env.reward(base_state, self.law_argument(moment), base_action))
            rewards.append(self.env.reward(state, perturbed_moment[0], action))

            with torch.no_grad():
                base_state = self.sample_next_state(t, base_state, moment, base_action, generator)
                state = self.sample_next_state(t, state, perturbed_moment, action, generator)

        zeta, beta, affine_scale = self.sample_perturbation(generator, self.lambda_)
        terminal_moment = self.perturb_moment(moments[-1], zeta, beta, self.lambda_)
        terminal_reward = self.env.terminal_reward(state, terminal_moment[0])
        base_terminal_reward = self.env.terminal_reward(base_state, self.law_argument(moments[-1]))
        score_terms.append(
            self.transport_score(moments[-1], terminal_moment, zeta, beta, affine_scale, self.lambda_, sensitivities[-1])
        )

        returns = self.discounted_returns(rewards, terminal_reward)
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        return torch.stack(score_terms), torch.stack(returns), base_return

    def batched_trajectory_gradient(self, moments, sensitivities, seed):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        base_states = self.initial_states(self.n_particles, generator)
        states = self.initial_states(self.n_particles, generator)
        base_rewards = []
        rewards = []
        action_log_probs = []
        law_scores = []

        for t in range(self.horizon):
            moment = moments[t]
            base_action, _ = self.sample_actions_with_log_probs(t, base_states, moment, generator)
            zeta, beta, affine_scale = self.sample_perturbation_batch(self.n_particles, generator, self.lambda_)
            perturbed_moment = self.perturb_moment(moment, zeta, beta, self.lambda_)
            action, log_prob = self.sample_actions_with_log_probs(t, states, perturbed_moment, generator)

            law_scores.append(
                self.transport_score(moment, perturbed_moment, zeta, beta, affine_scale, self.lambda_, sensitivities[t])
            )
            action_log_probs.append(log_prob)
            base_rewards.append(self.env.reward(base_states, self.law_argument(moment), base_action))
            rewards.append(self.env.reward(states, self.law_argument(perturbed_moment), action))

            with torch.no_grad():
                base_states = self.sample_next_state(t, base_states, moment, base_action, generator)
                states = self.sample_next_state(t, states, perturbed_moment, action, generator)

        zeta, beta, affine_scale = self.sample_perturbation_batch(self.n_particles, generator, self.lambda_)
        terminal_moment = self.perturb_moment(moments[-1], zeta, beta, self.lambda_)
        terminal_reward = self.env.terminal_reward(states, self.law_argument(terminal_moment))
        base_terminal_reward = self.env.terminal_reward(base_states, self.law_argument(moments[-1]))
        law_scores.append(
            self.transport_score(moments[-1], terminal_moment, zeta, beta, affine_scale, self.lambda_, sensitivities[-1])
        )

        returns = torch.stack(self.discounted_returns(rewards, terminal_reward))
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        advantages = returns - returns.mean(dim=1, keepdim=True) if self.config.baseline else returns
        action_gradient = self.flat_grad((torch.stack(action_log_probs) * advantages[:-1].detach()).sum())
        law_gradient = (torch.stack(law_scores) * advantages.detach().unsqueeze(-1)).sum(dim=(0, 1))
        return (action_gradient + law_gradient) / self.n_particles, base_return.mean()

    def estimate_gradient(self, seed):
        moments = self.mean_field_moment_flow(seed=seed + 20_000)
        shared_sensitivities = None
        if self.config.reuse_state_gradient:
            shared_sensitivities = self.estimate_moment_sensitivities(moments, seed + 10_000)

        if shared_sensitivities is not None:
            return self.batched_trajectory_gradient(moments, shared_sensitivities, seed)

        gradient = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        scores = []
        returns = []
        objectives = []

        for particle in range(self.n_particles):
            sensitivities = shared_sensitivities
            if sensitivities is None:
                sensitivities = self.estimate_moment_sensitivities(moments, seed + 10_000 + particle)

            score, trajectory_returns, objective = self.trajectory_gradient(moments, sensitivities, seed + particle)
            objectives.append(objective.detach())

            if self.config.baseline:
                scores.append(score)
                returns.append(trajectory_returns)
            else:
                gradient = gradient + (score * trajectory_returns.detach().unsqueeze(-1)).sum(dim=0)

        if self.config.baseline:
            scores = torch.stack(scores)
            returns = torch.stack(returns)
            returns = returns - returns.mean(dim=0, keepdim=True)
            gradient = (scores * returns.detach().unsqueeze(-1)).sum(dim=(0, 1))

        gradient = gradient / self.n_particles
        objective = torch.stack(objectives).mean()
        return gradient, objective

    def evaluate(self, n_particles=None, horizon=None, seed=None):
        n_particles = self.n_particles if n_particles is None else n_particles
        horizon = getattr(self.env.config, "T_val", self.horizon) if horizon is None else horizon
        moments = self.mean_field_moment_flow(horizon=horizon, seed=seed)
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(self.config.seed if seed is None else seed)

        states = self.env.sample_initial(n_particles, generator)
        rewards = []
        for t in range(horizon):
            moment = moments[t]
            actions = self.sample_actions_for_population(t, states, moment, generator)
            rewards.append(self.env.reward(states, moment[0], actions))
            with torch.no_grad():
                states = self.sample_next_states_for_population(t, states, moment, actions, generator)

        terminal = self.env.terminal_reward(states, moments[-1][0])
        return self.discounted_returns(rewards, terminal)[0].mean()

    def train(self):
        setup_started_at = synchronized_time(self.env.device)
        optimizer = self.optimizer()
        setup_seconds = synchronized_time(self.env.device) - setup_started_at
        history = {
            "objective": [],
            "validation_objective": [],
            "gradient_norm": [],
            "train_step_seconds": [],
            "validation_seconds": [],
            "setup_seconds": [setup_seconds],
        }

        for episode in range(self.n_train):
            step_started_at = synchronized_time(self.env.device)
            gradient, objective = self.estimate_gradient(self.config.seed + episode * (self.n_particles + 1))

            optimizer.zero_grad()
            self.set_flat_gradient(-gradient)
            optimizer.step()
            objective_value = float(objective.detach().cpu())
            gradient_norm_value = float(gradient.norm().detach().cpu())
            history["train_step_seconds"].append(synchronized_time(self.env.device) - step_started_at)

            history["objective"].append(objective_value)
            history["gradient_norm"].append(gradient_norm_value)

            if self.validation_interval and (episode + 1) % self.validation_interval == 0:
                validation_started_at = synchronized_time(self.env.device)
                with torch.no_grad():
                    validation = self.evaluate(seed=self.config.seed + self.n_train)
                validation_value = float(validation.detach().cpu())
                history["validation_seconds"].append(synchronized_time(self.env.device) - validation_started_at)
                history["validation_objective"].append(validation_value)

        return self.policy, history


def train_continuous_transport(env, policy=None, config=ContinuousTransportConfig()):
    return ContinuousTransport(env, policy=policy, config=config).train()
