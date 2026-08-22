from dataclasses import dataclass
import importlib
import inspect

import torch
from torch import nn

from .discrete_validation import evaluate_initial_distributions, evaluate_law, mean_field_next_law
from .timing import synchronized_time


@dataclass(frozen=True)
class MFReinforceConfig:
    n_train: int | None = None
    lr: float | None = None
    n_particles: int | None = None
    n_logit_gradient: int | None = None
    horizon: int | None = None
    validation_interval: int | None = None
    perturbation_eta: float | None = None
    flow: str = "exact"
    n_flow_particles: int | None = None
    baseline: bool = True
    reuse_state_gradient: bool = True
    seed: int = 0


class MFReinforce:
    def __init__(self, env, policy=None, config=MFReinforceConfig()):
        if not hasattr(env, "n_states") or not hasattr(env, "n_actions"):
            raise TypeError("MFReinforce only supports finite discrete state and action spaces.")

        self.env = env
        self.config = config
        self.policy = self._make_policy() if policy is None else policy
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
    def n_flow_particles(self):
        return self.n_particles if self.config.n_flow_particles is None else self.config.n_flow_particles

    @property
    def n_logit_gradient(self):
        if self.config.n_logit_gradient is not None:
            return self.config.n_logit_gradient
        return getattr(self.env.config, "n_logit_gradient", 10)

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
    def perturbation_eta(self):
        if self.config.perturbation_eta is not None:
            return self.config.perturbation_eta
        return getattr(self.env.config, "perturbation_eta", 1.0)

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

    def state_indexed_log_prob_gradients(self, log_probs, target_states):
        gradients = []
        for state_index in range(self.env.n_states):
            weights = (target_states == state_index).to(log_probs.dtype)
            gradients.append(self.flat_grad((log_probs * weights.unsqueeze(0)).sum(), retain_graph=True))
        return torch.stack(gradients)

    def sample_initial_distribution(self, generator):
        if hasattr(self.env, "sample_initial_distribution"):
            return self.env.sample_initial_distribution(generator)
        return self.env.initial_distribution

    def validation_initial_distributions(self):
        if hasattr(self.env, "validation_initial_distributions"):
            return list(self.env.validation_initial_distributions())
        return [self.env.initial_distribution]

    def initial_state(self, generator, initial_distribution=None):
        probabilities = self.sample_initial_distribution(generator) if initial_distribution is None else initial_distribution
        return torch.multinomial(probabilities, 1, generator=generator).reshape(())

    def initial_states(self, n_particles, generator, initial_distribution=None):
        probabilities = self.sample_initial_distribution(generator) if initial_distribution is None else initial_distribution
        return torch.multinomial(probabilities, n_particles, replacement=True, generator=generator)

    def sample_action(self, t, state, law, generator):
        probabilities = self.env.policy(self.policy, self.policy_time(t), state, law)
        probabilities = probabilities.clamp_min(1e-12)
        probabilities = probabilities / probabilities.sum()
        action = torch.multinomial(probabilities, 1, generator=generator).reshape(())
        return action, probabilities

    def sample_actions_with_log_probs(self, t, states, laws, generator):
        probabilities = self.env.policy(self.policy, self.policy_time(t), states, laws)
        probabilities = probabilities.clamp_min(1e-12)
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
        actions = torch.multinomial(
            probabilities.detach().reshape(-1, probabilities.shape[-1]), 1, generator=generator
        ).reshape(states.shape)
        log_probs = torch.log(probabilities.gather(-1, actions.unsqueeze(-1)).squeeze(-1))
        return actions, log_probs

    def log_prob_gradient(self, t, state, law, action):
        probabilities = self.env.policy(self.policy, self.policy_time(t), state, law)
        probabilities = probabilities.clamp_min(1e-12)
        probabilities = probabilities / probabilities.sum()
        log_prob = torch.log(probabilities[action])
        grads = torch.autograd.grad(log_prob, self.trainable_parameters(), allow_unused=True)
        return self.flatten_grads(grads)

    def sample_next_state(self, t, state, law, action, generator):
        signature = inspect.signature(self.env.sample)
        if "t" in signature.parameters:
            return self.env.sample(state, law, action, generator, t=t)
        return self.env.sample(state, law, action, generator)

    def log_law(self, law):
        return law.clamp_min(1e-12).log()

    def perturbed_law(self, log_law, noise):
        return torch.softmax(log_law + self.perturbation_eta * noise, dim=-1)

    def mean_field_next_law(self, t, law):
        return mean_field_next_law(self.env, self.policy, self.policy_time, t, law)

    def mean_field_law_flow(self, horizon=None, seed=None, initial_distribution=None):
        horizon = self.horizon if horizon is None else horizon
        if self.config.flow == "particle":
            return self.particle_law_flow(horizon, seed, initial_distribution=initial_distribution)

        initial_distribution = self.env.initial_distribution if initial_distribution is None else initial_distribution
        laws = [initial_distribution.detach()]
        log_laws = [self.log_law(laws[0])]

        for t in range(horizon):
            next_law = self.mean_field_next_law(t, laws[-1]).detach()
            laws.append(next_law)
            log_laws.append(self.log_law(next_law))

        return laws, log_laws

    def particle_law_flow(self, horizon, seed=None, initial_distribution=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(self.config.seed + 20_000 if seed is None else seed)
        initial_distribution = self.sample_initial_distribution(generator) if initial_distribution is None else initial_distribution
        states = torch.multinomial(initial_distribution, self.n_flow_particles, replacement=True, generator=generator)
        laws = [self.empirical_law(states)]
        log_laws = [self.log_law(laws[0])]

        for t in range(horizon):
            law = laws[-1]
            probabilities = self.env.policy(self.policy, self.policy_time(t), states, law)
            probabilities = probabilities.clamp_min(1e-12)
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            actions = torch.multinomial(
                probabilities.reshape(-1, probabilities.shape[-1]), 1, generator=generator
            ).reshape(states.shape)

            with torch.no_grad():
                states = self.sample_next_state(t, states, law, actions, generator)

            next_law = self.empirical_law(states)
            laws.append(next_law)
            log_laws.append(self.log_law(next_law))

        return laws, log_laws

    def empirical_law(self, states):
        counts = torch.bincount(states, minlength=self.env.n_states)
        return (counts.to(self.env.dtype) / states.numel()).detach()

    def discounted_returns(self, rewards, terminal_reward):
        values = [None] * (len(rewards) + 1)
        values[-1] = terminal_reward
        for t in range(len(rewards) - 1, -1, -1):
            values[t] = rewards[t] + self.discount * values[t + 1]
        return values

    def evaluate_law(self, initial_distribution, horizon):
        return evaluate_law(
            self.env,
            self.policy,
            self.policy_time,
            self.discount,
            initial_distribution,
            horizon,
        )

    def estimate_state_logit_gradients(self, log_laws, seed, initial_distribution=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        gradients = [
            torch.zeros(self.env.n_states, self.n_parameters, dtype=self.env.dtype, device=self.env.device)
            for _ in range(self.horizon + 1)
        ]
        laws = [torch.softmax(log_law, dim=0) for log_law in log_laws]

        for target_t in range(1, self.horizon + 1):
            numerator = torch.zeros(self.env.n_states, self.n_parameters, dtype=self.env.dtype, device=self.env.device)

            base_states = self.initial_states(self.n_logit_gradient, generator, initial_distribution)
            perturbed_states = self.initial_states(self.n_logit_gradient, generator, initial_distribution)
            noises = torch.randn(
                target_t,
                self.n_logit_gradient,
                self.env.n_states,
                dtype=self.env.dtype,
                device=self.env.device,
                generator=generator,
            )
            law_scores = []
            action_log_probs = []

            for s in range(target_t):
                law = laws[s]
                perturbed_law = self.perturbed_law(log_laws[s].unsqueeze(0), noises[s])

                base_actions, _ = self.sample_actions_with_log_probs(
                    s, base_states, law.expand(self.n_logit_gradient, -1), generator
                )
                perturbed_actions, log_probs = self.sample_actions_with_log_probs(
                    s, perturbed_states, perturbed_law, generator
                )
                law_scores.append(noises[s] @ gradients[s] / self.perturbation_eta)
                action_log_probs.append(log_probs)

                with torch.no_grad():
                    base_states = self.sample_next_state(s, base_states, law, base_actions, generator)
                    perturbed_states = self.sample_next_state(
                        s, perturbed_states, perturbed_law, perturbed_actions, generator
                    )

            numerator.index_add_(0, perturbed_states, torch.stack(law_scores).sum(dim=0))
            numerator = numerator + self.state_indexed_log_prob_gradients(
                torch.stack(action_log_probs), perturbed_states
            )

            grad_mu = numerator / self.n_logit_gradient
            gradients[target_t] = grad_mu / laws[target_t].clamp_min(1e-12).unsqueeze(-1)

        return gradients

    def trajectory_gradient(self, log_laws, logit_gradients, seed, initial_distribution=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        base_state = self.initial_state(generator, initial_distribution)
        perturbed_state = self.initial_state(generator, initial_distribution)
        noises = torch.randn(
            self.horizon + 1, self.env.n_states, dtype=self.env.dtype, device=self.env.device, generator=generator
        )
        rewards = []
        perturbed_rewards = []
        score_terms = []

        for t in range(self.horizon):
            law = torch.softmax(log_laws[t], dim=0)
            perturbed_law = self.perturbed_law(log_laws[t], noises[t])

            base_action, _ = self.sample_action(t, base_state, law, generator)
            perturbed_action, _ = self.sample_action(t, perturbed_state, perturbed_law, generator)
            law_score = noises[t] @ logit_gradients[t] / self.perturbation_eta
            action_score = self.log_prob_gradient(t, perturbed_state, perturbed_law, perturbed_action)
            score_terms.append(law_score + action_score)

            rewards.append(self.env.reward(base_state, law, base_action))
            perturbed_rewards.append(self.env.reward(perturbed_state, perturbed_law, perturbed_action))

            with torch.no_grad():
                base_state = self.sample_next_state(t, base_state, law, base_action, generator)
                perturbed_state = self.sample_next_state(t, perturbed_state, perturbed_law, perturbed_action, generator)

        terminal_law = torch.softmax(log_laws[-1], dim=0)
        perturbed_terminal_law = self.perturbed_law(log_laws[-1], noises[-1])
        terminal_reward = self.env.terminal_reward(base_state, terminal_law)
        perturbed_terminal_reward = self.env.terminal_reward(perturbed_state, perturbed_terminal_law)
        score_terms.append(noises[-1] @ logit_gradients[-1] / self.perturbation_eta)

        returns = self.discounted_returns(perturbed_rewards, perturbed_terminal_reward)
        base_returns = self.discounted_returns(rewards, terminal_reward)
        return torch.stack(score_terms), torch.stack(returns), base_returns[0]

    def batched_trajectory_gradient(self, log_laws, logit_gradients, seed, initial_distribution=None):
        generator = torch.Generator(device=self.env.device)
        generator.manual_seed(seed)

        base_states = self.initial_states(self.n_particles, generator, initial_distribution)
        perturbed_states = self.initial_states(self.n_particles, generator, initial_distribution)
        noises = torch.randn(
            self.horizon + 1,
            self.n_particles,
            self.env.n_states,
            dtype=self.env.dtype,
            device=self.env.device,
            generator=generator,
        )
        rewards = []
        perturbed_rewards = []
        law_scores = []
        action_log_probs = []

        for t in range(self.horizon):
            law = torch.softmax(log_laws[t], dim=0)
            perturbed_law = self.perturbed_law(log_laws[t].unsqueeze(0), noises[t])

            base_actions, _ = self.sample_actions_with_log_probs(
                t, base_states, law.expand(self.n_particles, -1), generator
            )
            perturbed_actions, log_probs = self.sample_actions_with_log_probs(
                t, perturbed_states, perturbed_law, generator
            )
            law_scores.append(noises[t] @ logit_gradients[t] / self.perturbation_eta)
            action_log_probs.append(log_probs)

            rewards.append(self.env.reward(base_states, law, base_actions))
            perturbed_rewards.append(self.env.reward(perturbed_states, perturbed_law, perturbed_actions))

            with torch.no_grad():
                base_states = self.sample_next_state(t, base_states, law, base_actions, generator)
                perturbed_states = self.sample_next_state(t, perturbed_states, perturbed_law, perturbed_actions, generator)

        terminal_law = torch.softmax(log_laws[-1], dim=0)
        perturbed_terminal_law = self.perturbed_law(log_laws[-1].unsqueeze(0), noises[-1])
        terminal_reward = self.env.terminal_reward(base_states, terminal_law)
        perturbed_terminal_reward = self.env.terminal_reward(perturbed_states, perturbed_terminal_law)
        law_scores.append(noises[-1] @ logit_gradients[-1] / self.perturbation_eta)

        returns = torch.stack(self.discounted_returns(perturbed_rewards, perturbed_terminal_reward))
        advantages = returns - returns.mean(dim=1, keepdim=True) if self.config.baseline else returns
        action_gradient = self.flat_grad((torch.stack(action_log_probs) * advantages[:-1].detach()).sum())
        law_gradient = (torch.stack(law_scores) * advantages.detach().unsqueeze(-1)).sum(dim=(0, 1))
        base_returns = self.discounted_returns(rewards, terminal_reward)
        return (action_gradient + law_gradient) / self.n_particles, base_returns[0].mean()

    def estimate_gradient(self, seed):
        law_generator = torch.Generator(device=self.env.device)
        law_generator.manual_seed(seed + 30_000)
        initial_distribution = self.sample_initial_distribution(law_generator)
        _, log_laws = self.mean_field_law_flow(seed=seed + 20_000, initial_distribution=initial_distribution)
        shared_logit_gradients = None
        if self.config.reuse_state_gradient:
            shared_logit_gradients = self.estimate_state_logit_gradients(
                log_laws,
                seed + 10_000,
                initial_distribution=initial_distribution,
            )

        if shared_logit_gradients is not None:
            return self.batched_trajectory_gradient(
                log_laws,
                shared_logit_gradients,
                seed,
                initial_distribution=initial_distribution,
            )

        gradient = torch.zeros(self.n_parameters, dtype=self.env.dtype, device=self.env.device)
        scores = []
        returns = []
        objectives = []
        for k in range(self.n_particles):
            logit_gradients = self.estimate_state_logit_gradients(
                log_laws,
                seed + 10_000 + k,
                initial_distribution=initial_distribution,
            )
            score, trajectory_returns, particle_objective = self.trajectory_gradient(
                log_laws,
                logit_gradients,
                seed + k,
                initial_distribution=initial_distribution,
            )
            objectives.append(particle_objective)
            if self.config.baseline:
                scores.append(score)
                returns.append(trajectory_returns)
            else:
                gradient = gradient + (score * trajectory_returns.detach().unsqueeze(-1)).sum(dim=0)

        if self.config.baseline:
            scores = torch.stack(scores)
            returns = torch.stack(returns)
            advantages = returns - returns.mean(dim=0, keepdim=True)
            gradient = (scores * advantages.detach().unsqueeze(-1)).sum(dim=(0, 1))

        gradient = gradient / self.n_particles
        return gradient, torch.stack(objectives).mean()

    def evaluate(self, n_particles=None, horizon=None, seed=None):
        """Evaluate by exact deterministic population recursion.

        The n_particles and seed arguments are kept for API compatibility.
        """
        horizon = getattr(self.env.config, "T_val", self.horizon) if horizon is None else horizon
        return evaluate_initial_distributions(
            self.env,
            self.policy,
            self.policy_time,
            self.discount,
            self.validation_initial_distributions(),
            horizon,
        )

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


def train_mfreinforce(env, policy=None, config=MFReinforceConfig()):
    return MFReinforce(env, policy=policy, config=config).train()
