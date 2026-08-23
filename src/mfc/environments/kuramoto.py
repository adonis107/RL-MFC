from dataclasses import dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class KuramotoConfig:
    coupling: float = 0.8
    diffusion: float = 0.08
    dt: float = 0.1
    target_phase: float = 0.0
    sync_weight: float = 1.0
    lock_weight: float = 0.5
    terminal_sync_weight: float = 5.0
    terminal_lock_weight: float = 2.0
    action_weight: float = 0.02
    action_scale: float = 2.0
    tau: float = 0.25
    rho: float = 1.0
    hidden_width: int = 64
    T: int = 20
    T_val: int = 40
    n_train: int = 10_000
    lr: float = 1e-3
    n_particles: int = 500
    n_law_gradient: int = 20
    n_flow_particles: int = 1024
    validation_particles: int = 4096
    validation_seed: int = 9173
    validation_interval: int = 10
    dtype: torch.dtype = torch.float32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class KuramotoPolicy(nn.Module):
    def __init__(self, config=KuramotoConfig()):
        super().__init__()
        self.config = config
        self.device = config.device
        self.net = nn.Sequential(
            nn.Linear(9, config.hidden_width),
            nn.Tanh(),
            nn.Linear(config.hidden_width, config.hidden_width),
            nn.Tanh(),
            nn.Linear(config.hidden_width, 1),
        )
        self.to(self.device)

    def forward(self, t, phase, law):
        if t.ndim == 0:
            t = t.expand(phase.shape)
        t_feature = (t / max(self.config.T - 1, 1)).to(dtype=phase.dtype).unsqueeze(-1)
        cos_phase = torch.cos(phase)
        sin_phase = torch.sin(phase)

        if law.ndim == 1:
            c = law[0].expand_as(phase)
            s = law[1].expand_as(phase)
        else:
            c = law[..., 0]
            s = law[..., 1]

        interaction = s * cos_phase - c * sin_phase
        order = torch.sqrt((c.square() + s.square()).clamp_min(1e-12))
        target = torch.as_tensor(self.config.target_phase, dtype=phase.dtype, device=phase.device)
        features = torch.stack(
            [
                t_feature.squeeze(-1),
                cos_phase,
                sin_phase,
                c,
                s,
                interaction,
                order,
                torch.sin(target - phase),
                torch.cos(target - phase),
            ],
            dim=-1,
        )
        return self.config.action_scale * torch.tanh(self.net(features).squeeze(-1))


class Kuramoto:
    law_feature_dim = 2

    def __init__(self, config=KuramotoConfig()):
        self.config = config
        self.dtype = config.dtype
        self.device = config.device

    def sample_initial(self, n_particles, generator):
        return (2.0 * math.pi) * torch.rand(
            n_particles,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        ) - math.pi

    def empirical_law(self, states):
        return torch.stack([torch.cos(states).mean(), torch.sin(states).mean()]).detach()

    def state_law_features(self, states):
        return torch.stack([torch.cos(states), torch.sin(states)], dim=-1)

    def law_argument(self, law):
        return law

    def policy(self, theta, t, state, law):
        mean = theta(t, state, law)
        return torch.distributions.Normal(mean, self.config.tau)

    def sample_action(self, theta, t, states, law, generator):
        time = torch.tensor(float(t), dtype=self.dtype, device=self.device)
        mean = theta(time, states, law)
        noise = torch.randn(states.shape, dtype=states.dtype, device=states.device, generator=generator)
        return mean + self.config.tau * noise

    def interaction_field(self, states, law):
        c = law[..., 0] if law.ndim > 1 else law[0]
        s = law[..., 1] if law.ndim > 1 else law[1]
        return s * torch.cos(states) - c * torch.sin(states)

    def sample(self, states, law, actions, generator, t=None):
        drift = self.config.coupling * self.interaction_field(states, law) + actions
        noise = torch.randn(states.shape, dtype=states.dtype, device=states.device, generator=generator)
        return states + self.config.dt * drift + (2.0 * self.config.diffusion * self.config.dt) ** 0.5 * noise

    def local_disagreement(self, states, law):
        c = law[..., 0] if law.ndim > 1 else law[0]
        s = law[..., 1] if law.ndim > 1 else law[1]
        return 1.0 - torch.cos(states) * c - torch.sin(states) * s

    def phase_lock_cost(self, states):
        target = torch.as_tensor(self.config.target_phase, dtype=states.dtype, device=states.device)
        return 1.0 - torch.cos(states - target)

    def reward(self, states, law, actions):
        cost = self.config.dt * (
            self.config.sync_weight * self.local_disagreement(states, law)
            + self.config.lock_weight * self.phase_lock_cost(states)
            + self.config.action_weight * actions.square()
        )
        return -cost

    def terminal_reward(self, states, law):
        cost = (
            self.config.terminal_sync_weight * self.local_disagreement(states, law)
            + self.config.terminal_lock_weight * self.phase_lock_cost(states)
        )
        return -cost

    def sample_law_perturbation(self, generator, scale):
        if scale <= 0.0:
            raise ValueError("Kuramoto transport scale must be positive.")
        beta = self.config.rho * torch.randn(2, dtype=self.dtype, device=self.device, generator=generator)
        zeta = torch.zeros_like(beta)
        affine_scale = torch.ones_like(beta)
        return zeta, beta, affine_scale

    def sample_law_perturbation_batch(self, n_particles, generator, scale):
        if scale <= 0.0:
            raise ValueError("Kuramoto transport scale must be positive.")
        beta = self.config.rho * torch.randn(
            n_particles,
            2,
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        zeta = torch.zeros_like(beta)
        affine_scale = torch.ones_like(beta)
        return zeta, beta, affine_scale

    def perturb_law_features(self, law, zeta, beta, scale):
        perturbed = law + scale * beta
        norm = perturbed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return torch.where(norm > 0.995, 0.995 * perturbed / norm, perturbed)

    def transport_score(self, law, perturbed_law, zeta, beta, affine_scale, scale, sensitivity):
        coefficient = (perturbed_law - law) / (scale**2 * self.config.rho**2)
        return (coefficient.unsqueeze(-1) * sensitivity).sum(dim=-2)

    def objective(self, policy, lambda_=0.0):
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.config.validation_seed)
        states = self.sample_initial(self.config.validation_particles, generator)
        rewards = []
        for t in range(self.config.T_val):
            law = self.empirical_law(states)
            actions = self.sample_action(policy, t, states, law, generator)
            rewards.append(self.reward(states, law, actions))
            with torch.no_grad():
                states = self.sample(states, law, actions, generator, t=t)

        terminal = self.terminal_reward(states, self.empirical_law(states))
        return torch.stack(rewards).sum(dim=0).add(terminal).mean()
