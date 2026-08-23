from dataclasses import dataclass
import math
import importlib
import inspect

import torch
from torch import nn

from .mfreinforce import MFReinforce
from .reinforce import exact_continuous_validation_objective
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


@dataclass(frozen=True)
class AdaptiveDiscreteTransportConfig(DiscreteTransportConfig):
    adaptive_checkpoint_interval: int = 500
    adaptive_replications: int = 4
    contraction_lambda: float = 0.5
    contraction_eta: float = 0.75
    target_bias_lambda: float = 0.25
    target_bias_eta: float = 0.25
    bias_order_lambda: float = 1.0
    bias_order_eta: float = 1.0
    controller_lr_lambda: float = 0.05
    controller_lr_eta: float = 0.05
    controller_beta1_lambda: float = 0.9
    controller_beta1_eta: float = 0.9
    controller_beta2_lambda: float = 0.999
    controller_beta2_eta: float = 0.999
    controller_eps_lambda: float = 1e-8
    controller_eps_eta: float = 1e-8
    diagnostic_delta: float = 1e-12
    lambda_min: float = 0.025
    lambda_max: float = 0.95
    eta_min: float = 0.2
    eta_max: float = 0.98
    direction_cosine_min: float = 0.0
    direction_norm_min: float = 1e-12
    direction_z: float = 1.0


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

    def estimate_state_sensitivities(self, laws, seed, initial_distribution=None, eta=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        eta = self.eta if eta is None else eta

        sensitivities = [
            torch.zeros(self.env.n_states, self.n_parameters, dtype=self.env.dtype, device=self.env.device)
            for _ in range(self.horizon + 1)
        ]
        states = [self.initial_states(self.n_logit_gradient, generator, initial_distribution)]
        action_log_probs = []
        h_values = []

        for t in range(self.horizon):
            q = self.sample_simplex_batch(self.n_logit_gradient, generator)
            perturbed_law = self.perturb_law(laws[t], q, eta)
            actions, log_probs = self.sample_actions_with_log_probs(t, states[-1], perturbed_law, generator)
            action_log_probs.append(log_probs)
            h_values.append(self.simplex_score_h(q))

            with torch.no_grad():
                states.append(self.sample_next_state(t, states[-1], perturbed_law, actions, generator))

        factor = (1.0 - eta) / eta
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

    def batched_trajectory_components(self, laws, seed, initial_distribution=None, lambda_=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)
        lambda_ = self.lambda_ if lambda_ is None else lambda_

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
            perturbed_law = self.perturb_law(law, q, lambda_)
            actions, log_probs = self.sample_actions_with_log_probs(t, states, perturbed_law, generator)
            base_rewards.append(self.env.reward(base_states, law, base_actions))
            rewards.append(self.env.reward(states, perturbed_law, actions))
            action_log_probs.append(log_probs)
            law_scores.append(self.simplex_score_h(q))

            with torch.no_grad():
                base_states = self.sample_next_state(t, base_states, law, base_actions, generator)
                states = self.sample_next_state(t, states, perturbed_law, actions, generator)

        q_terminal = self.sample_simplex_batch(self.n_particles, generator)
        terminal_law = self.perturb_law(laws[-1], q_terminal, lambda_)
        terminal_reward = self.env.terminal_reward(states, terminal_law)
        base_terminal_reward = self.env.terminal_reward(base_states, laws[-1])
        law_scores.append(self.simplex_score_h(q_terminal))

        returns = self.discounted_returns(rewards, terminal_reward)[0]
        base_return = self.discounted_returns(base_rewards, base_terminal_reward)[0]
        advantages = returns - returns.mean() if self.config.baseline else returns
        action_score = torch.stack(action_log_probs).sum(dim=0)
        action_gradient = self.flat_grad((action_score * advantages.detach()).sum())
        weights = advantages.detach()
        return action_gradient, weights, law_scores, base_return.mean()

    def combine_batched_trajectory_components(self, action_gradient, weights, law_scores, sensitivities, lambda_=None):
        lambda_ = self.lambda_ if lambda_ is None else lambda_
        law_gradient = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        for index, simplex_score in enumerate(law_scores):
            law_gradient = law_gradient + (simplex_score.transpose(0, 1) @ weights) @ sensitivities[index][:-1]
        law_gradient = -((1.0 - lambda_) / lambda_) * law_gradient
        return (action_gradient + law_gradient) / self.n_particles

    def batched_trajectory_gradient(self, laws, sensitivities, seed, initial_distribution=None):
        action_gradient, weights, law_scores, base_return = self.batched_trajectory_components(
            laws,
            seed,
            initial_distribution=initial_distribution,
        )
        gradient = self.combine_batched_trajectory_components(action_gradient, weights, law_scores, sensitivities)
        return gradient, base_return

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


class AdaptiveDiscreteTransport(DiscreteTransport):
    def __init__(self, env, policy=None, config=AdaptiveDiscreteTransportConfig()):
        super().__init__(env, policy=policy, config=config)
        self._lambda = self._clamp_scale(config.lambda_ if config.lambda_ is not None else 0.2, config.lambda_min, config.lambda_max)
        self._eta = self._clamp_scale(config.eta if config.eta is not None else 0.8, config.eta_min, config.eta_max)
        self._rho_lambda = self._scale_to_logit(self._lambda, config.lambda_min, config.lambda_max)
        self._rho_eta = self._scale_to_logit(self._eta, config.eta_min, config.eta_max)
        self._moment_lambda = 0.0
        self._moment_eta = 0.0
        self._second_lambda = 0.0
        self._second_eta = 0.0
        self._controller_steps = 0

    @property
    def lambda_(self):
        return self._lambda

    @property
    def eta(self):
        return self._eta

    @staticmethod
    def _clamp_scale(value, lower, upper):
        return min(max(float(value), lower), upper)

    @staticmethod
    def _scale_to_logit(value, lower, upper):
        ratio = (value - lower) / (upper - lower)
        ratio = min(max(ratio, 1e-12), 1.0 - 1e-12)
        return math.log(ratio / (1.0 - ratio))

    @staticmethod
    def _logit_to_scale(rho, lower, upper):
        sigmoid = 1.0 / (1.0 + math.exp(-rho))
        return lower + (upper - lower) * sigmoid

    @staticmethod
    def _covariance_trace(samples):
        if samples.shape[0] < 2:
            return torch.zeros((), dtype=samples.dtype, device=samples.device)
        centered = samples - samples.mean(dim=0)
        return centered.square().sum() / (samples.shape[0] - 1)

    @staticmethod
    def _cosine(left, right):
        denominator = left.norm() * right.norm()
        if not torch.isfinite(denominator).item() or denominator.item() <= 0:
            return float("nan")
        return float((left @ right / denominator).detach().cpu())

    def _controller_update(self, name, signal):
        config = self.config
        beta1 = getattr(config, f"controller_beta1_{name}")
        beta2 = getattr(config, f"controller_beta2_{name}")
        step_size = getattr(config, f"controller_lr_{name}")
        eps = getattr(config, f"controller_eps_{name}")

        moment_name = f"_moment_{name}"
        second_name = f"_second_{name}"
        rho_name = f"_rho_{name}"

        moment = beta1 * getattr(self, moment_name) + (1.0 - beta1) * signal
        second = beta2 * getattr(self, second_name) + (1.0 - beta2) * signal**2
        setattr(self, moment_name, moment)
        setattr(self, second_name, second)

        corrected_moment = moment / (1.0 - beta1**self._controller_steps)
        corrected_second = second / (1.0 - beta2**self._controller_steps)
        rho = getattr(self, rho_name) - step_size * corrected_moment / (corrected_second**0.5 + eps)
        setattr(self, rho_name, rho)

    def _update_scales_from_logits(self):
        config = self.config
        self._lambda = self._logit_to_scale(self._rho_lambda, config.lambda_min, config.lambda_max)
        self._eta = self._logit_to_scale(self._rho_eta, config.eta_min, config.eta_max)

    def adaptive_diagnostic(self, seed):
        config = self.config
        lambda_plus = self.lambda_
        eta_plus = self.eta
        lambda_minus = max(config.lambda_min, config.contraction_lambda * lambda_plus)
        eta_minus = max(config.eta_min, config.contraction_eta * eta_plus)
        effective_c_lambda = max(lambda_minus / lambda_plus, 1e-12)
        effective_c_eta = max(eta_minus / eta_plus, 1e-12)

        g_pp = []
        g_mp = []
        g_pm = []
        g_mm = []
        for replication in range(config.adaptive_replications):
            base_seed = seed + replication * 100_000
            law_generator = torch.Generator(device=self.env.device)
            law_generator.manual_seed(base_seed + 30_000)
            initial_distribution = self.sample_initial_distribution(law_generator)
            laws, _ = self.mean_field_law_flow(seed=base_seed + 20_000, initial_distribution=initial_distribution)

            sensitivities_plus = self.estimate_state_sensitivities(
                laws,
                base_seed + 10_000,
                initial_distribution=initial_distribution,
                eta=eta_plus,
            )
            sensitivities_minus = self.estimate_state_sensitivities(
                laws,
                base_seed + 40_000,
                initial_distribution=initial_distribution,
                eta=eta_minus,
            )
            components_plus = self.batched_trajectory_components(
                laws,
                base_seed,
                initial_distribution=initial_distribution,
                lambda_=lambda_plus,
            )
            components_minus = self.batched_trajectory_components(
                laws,
                base_seed + 50_000,
                initial_distribution=initial_distribution,
                lambda_=lambda_minus,
            )

            action_gradient, weights, law_scores, _ = components_plus
            g_pp.append(
                self.combine_batched_trajectory_components(
                    action_gradient,
                    weights,
                    law_scores,
                    sensitivities_plus,
                    lambda_=lambda_plus,
                ).detach()
            )
            g_pm.append(
                self.combine_batched_trajectory_components(
                    action_gradient,
                    weights,
                    law_scores,
                    sensitivities_minus,
                    lambda_=lambda_plus,
                ).detach()
            )

            action_gradient, weights, law_scores, _ = components_minus
            g_mp.append(
                self.combine_batched_trajectory_components(
                    action_gradient,
                    weights,
                    law_scores,
                    sensitivities_plus,
                    lambda_=lambda_minus,
                ).detach()
            )
            g_mm.append(
                self.combine_batched_trajectory_components(
                    action_gradient,
                    weights,
                    law_scores,
                    sensitivities_minus,
                    lambda_=lambda_minus,
                ).detach()
            )

        g_pp = torch.stack(g_pp)
        g_mp = torch.stack(g_mp)
        g_pm = torch.stack(g_pm)
        g_mm = torch.stack(g_mm)

        delta_lambda = 0.5 * ((g_pp - g_mp) + (g_pm - g_mm))
        delta_eta = 0.5 * ((g_pp - g_pm) + (g_mp - g_mm))
        mean_delta_lambda = delta_lambda.mean(dim=0)
        mean_delta_eta = delta_eta.mean(dim=0)

        discrepancy_lambda = (
            mean_delta_lambda.norm().square() - self._covariance_trace(delta_lambda) / config.adaptive_replications
        ).clamp_min(0.0)
        discrepancy_eta = (
            mean_delta_eta.norm().square() - self._covariance_trace(delta_eta) / config.adaptive_replications
        ).clamp_min(0.0)
        bias_lambda = discrepancy_lambda / (1.0 - effective_c_lambda**config.bias_order_lambda) ** 2
        bias_eta = discrepancy_eta / (1.0 - effective_c_eta**config.bias_order_eta) ** 2
        variance = self._covariance_trace(g_pp)

        z_lambda = float(torch.log((bias_lambda + config.diagnostic_delta) / (config.target_bias_lambda * variance + config.diagnostic_delta)).detach().cpu())
        z_eta = float(torch.log((bias_eta + config.diagnostic_delta) / (config.target_bias_eta * variance + config.diagnostic_delta)).detach().cpu())

        mean_pp = g_pp.mean(dim=0)
        mean_mp = g_mp.mean(dim=0)
        mean_pm = g_pm.mean(dim=0)
        mean_mm = g_mm.mean(dim=0)
        lambda_high = 0.5 * (mean_pp + mean_pm)
        lambda_low = 0.5 * (mean_mp + mean_mm)
        eta_high = 0.5 * (mean_pp + mean_mp)
        eta_low = 0.5 * (mean_pm + mean_mm)
        lambda_cosine = self._cosine(lambda_high, lambda_low)
        eta_cosine = self._cosine(eta_high, eta_low)

        if (
            lambda_high.norm().item() > config.direction_norm_min
            and lambda_low.norm().item() > config.direction_norm_min
            and math.isfinite(lambda_cosine)
            and lambda_cosine < config.direction_cosine_min
        ):
            z_lambda = max(z_lambda, config.direction_z)
        if (
            eta_high.norm().item() > config.direction_norm_min
            and eta_low.norm().item() > config.direction_norm_min
            and math.isfinite(eta_cosine)
            and eta_cosine < config.direction_cosine_min
        ):
            z_eta = max(z_eta, config.direction_z)

        self._controller_steps += 1
        self._controller_update("lambda", z_lambda)
        self._controller_update("eta", z_eta)
        self._update_scales_from_logits()

        return {
            "adaptive_lambda_before": lambda_plus,
            "adaptive_eta_before": eta_plus,
            "adaptive_lambda_after": self.lambda_,
            "adaptive_eta_after": self.eta,
            "adaptive_lambda_contracted": lambda_minus,
            "adaptive_eta_contracted": eta_minus,
            "adaptive_z_lambda": z_lambda,
            "adaptive_z_eta": z_eta,
            "adaptive_bias_lambda": float(bias_lambda.detach().cpu()),
            "adaptive_bias_eta": float(bias_eta.detach().cpu()),
            "adaptive_variance": float(variance.detach().cpu()),
            "adaptive_lambda_cosine": lambda_cosine,
            "adaptive_eta_cosine": eta_cosine,
        }

    def train(self):
        setup_started_at = synchronized_time(self.env.device)
        optimizer = self.optimizer()
        setup_seconds = synchronized_time(self.env.device) - setup_started_at
        history = {
            "objective": [],
            "validation_objective": [],
            "gradient_norm": [],
            "lambda": [],
            "eta": [],
            "adaptive_step": [],
            "adaptive_lambda_before": [],
            "adaptive_eta_before": [],
            "adaptive_lambda_after": [],
            "adaptive_eta_after": [],
            "adaptive_lambda_contracted": [],
            "adaptive_eta_contracted": [],
            "adaptive_z_lambda": [],
            "adaptive_z_eta": [],
            "adaptive_bias_lambda": [],
            "adaptive_bias_eta": [],
            "adaptive_variance": [],
            "adaptive_lambda_cosine": [],
            "adaptive_eta_cosine": [],
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

            checkpoint_interval = self.config.adaptive_checkpoint_interval
            if checkpoint_interval and (episode + 1) % checkpoint_interval == 0:
                diagnostic = self.adaptive_diagnostic(self.config.seed + 1_000_000 + episode * 10_000)
                history["adaptive_step"].append(episode + 1)
                for key, value in diagnostic.items():
                    history[key].append(value)

            history["train_step_seconds"].append(synchronized_time(self.env.device) - step_started_at)
            history["objective"].append(objective_value)
            history["gradient_norm"].append(gradient_norm_value)
            history["lambda"].append(self.lambda_)
            history["eta"].append(self.eta)

            if self.validation_interval and (episode + 1) % self.validation_interval == 0:
                validation_started_at = synchronized_time(self.env.device)
                with torch.no_grad():
                    validation = self.evaluate(seed=self.config.seed + self.n_train)
                validation_value = float(validation.detach().cpu())
                history["validation_seconds"].append(synchronized_time(self.env.device) - validation_started_at)
                history["validation_objective"].append(validation_value)

        return self.policy, history


def train_discrete_transport(env, policy=None, config=DiscreteTransportConfig()):
    return DiscreteTransport(env, policy=policy, config=config).train()


def train_adaptive_discrete_transport(env, policy=None, config=AdaptiveDiscreteTransportConfig()):
    return AdaptiveDiscreteTransport(env, policy=policy, config=config).train()


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
        if hasattr(self.env, "objective"):
            return exact_continuous_validation_objective(self.env, self.policy)

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
