from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class DistributionConfig:
    c_mov: float = 0.01
    hidden_width: int = 256
    T: int = 5
    T_val: int = 5
    gamma: float = 1.0
    n_train: int = 100_000
    lr: float = 1e-4
    n_particles: int = 500
    n_logit_gradient: int = 10
    validation_interval: int = 10
    target_distribution: tuple[float, ...] = (0.02, 0.04, 0.09, 0.16, 0.19, 0.19, 0.16, 0.09, 0.04, 0.02)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DistributionPolicy(nn.Module):
    def __init__(self, config=DistributionConfig()):
        super().__init__()
        self.device = config.device
        self.n_states = 10
        self.n_actions = 3
        self.net = nn.Sequential(
            nn.Linear(1 + self.n_states, config.hidden_width),
            nn.Tanh(),
            nn.Linear(config.hidden_width, config.hidden_width),
            nn.Tanh(),
            nn.Linear(config.hidden_width, self.n_states * self.n_actions),
        )
        self.to(self.device)

    def forward(self, t, mu):
        if t.ndim == 0:
            t = t.expand(mu.shape[:-1])
        t = t.unsqueeze(-1)
        logits = self.net(torch.cat([t, mu], dim=-1))
        logits = logits.reshape(*mu.shape[:-1], self.n_states, self.n_actions)
        return torch.softmax(logits, dim=-1)

class Distribution:
    def __init__(self, config=DistributionConfig()):
        self.LEFT, self.STAY, self.RIGHT = 0, 1, 2
        self.n_states, self.n_actions = 10, 3

        self.config = config
        self.dtype = torch.float32
        self.device = config.device

        self.initial_distribution = torch.full((10,), 0.1, dtype=self.dtype, device=self.device)
        self.target_distribution = torch.tensor(config.target_distribution, dtype=self.dtype, device=self.device)
        self.action_values = torch.tensor([-1, 0, 1], dtype=torch.long, device=self.device)

    def sample_initial_distribution(self, generator):
        weights = -torch.rand(self.n_states, dtype=self.dtype, device=self.device, generator=generator).clamp_min(1e-12).log()
        return weights / weights.sum()

    def transition(self, states, mu, actions):
        moves = self.action_values[actions]
        next_states = (states + moves) % self.n_states
        return torch.nn.functional.one_hot(next_states, num_classes=self.n_states).to(self.dtype)

    def sample(self, states, mu, actions, generator):
        return (states + self.action_values[actions]) % self.n_states

    def reward(self, states, mu, actions):
        movement_cost = self.config.c_mov * self.action_values[actions].abs().to(self.dtype)
        distribution_cost = (mu - self.target_distribution).square().sum(dim=-1)
        return -movement_cost - distribution_cost

    def terminal_reward(self, states, mu):
        distribution_cost = (mu - self.target_distribution).square().sum(dim=-1)
        return torch.zeros_like(states, dtype=self.dtype) - distribution_cost

    def policy(self, theta, t, state, mu):
        probabilities = theta(t, mu) if callable(theta) else theta
        if state.ndim == 0:
            return probabilities[..., state, :]
        if probabilities.ndim == 2:
            return probabilities[state]

        index = state.unsqueeze(-1).unsqueeze(-1).expand(*state.shape, 1, self.n_actions)
        return torch.gather(probabilities, dim=-2, index=index).squeeze(-2)

    def optimal_policy(self):
        raise NotImplementedError("No closed-form optimal policy is specified for the distribution environment.")

    def optimal_theta(self):
        raise NotImplementedError("No closed-form optimal policy is specified for the distribution environment.")
