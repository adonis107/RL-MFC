from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class PortfolioConfig:
    T: int = 10
    x0_mean: float = 1.0
    x0_variance: float = 0.04
    risk_free_return: float = 1.0
    excess_return_mean: float = 0.02
    excess_return_volatility: float = 0.08
    chi: float = 10.0
    mean_field_penalty: float = 2.0
    tau: float = 0.2
    rho: float = 1.0
    perturbation_scale: float = 0.0
    return_distribution: str = "normal"
    student_t_df: float = 5.0
    discount: float = 1.0
    n_train: int = 10_000
    lr: float = 1e-2
    n_particles: int = 500
    validation_interval: int = 10
    dtype: torch.dtype = torch.float64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Portfolio:
    def __init__(self, config=PortfolioConfig()):
        self.config = config
        self.dtype = config.dtype
        self.device = config.device
        self.n_params = 2

        self.s = torch.full((config.T,), config.risk_free_return, dtype=self.dtype, device=self.device)
        self.rbar = torch.full((config.T,), config.excess_return_mean, dtype=self.dtype, device=self.device)
        self.sigma_R = torch.full((config.T,), config.excess_return_volatility, dtype=self.dtype, device=self.device)
        self.h = self.rbar.square() + self.sigma_R.square()
        self.tau = torch.full((config.T,), config.tau, dtype=self.dtype, device=self.device)
        self.initial_law = torch.tensor([config.x0_mean, config.x0_variance], dtype=self.dtype, device=self.device)

    def sample_initial(self, n_particles, generator):
        noise = torch.randn(n_particles, dtype=self.dtype, device=self.device, generator=generator)
        return self.config.x0_mean + self.config.x0_variance**0.5 * noise

    def sample_excess_returns(self, t, shape, generator):
        if self.config.return_distribution == "normal":
            noise = torch.randn(shape, dtype=self.dtype, device=self.device, generator=generator)
        elif self.config.return_distribution == "student_t":
            normal = torch.randn(shape, dtype=self.dtype, device=self.device, generator=generator)
            concentration = torch.full(shape, 0.5 * self.config.student_t_df, dtype=self.dtype, device=self.device)
            chi_square = 2.0 * torch._standard_gamma(concentration, generator=generator)
            noise = normal / (chi_square / self.config.student_t_df).sqrt()
            noise = noise * ((self.config.student_t_df - 2.0) / self.config.student_t_df) ** 0.5
        else:
            raise ValueError(f"Unknown return distribution: {self.config.return_distribution}")

        return self.rbar[t] + self.sigma_R[t] * noise

    def policy_mean(self, theta, t, states, mu_mean):
        k, ell = theta[t]
        return k * (states - mu_mean) + ell

    def policy(self, theta, t, state, mu):
        return torch.distributions.Normal(self.policy_mean(theta, t, state, mu), self.tau[t])

    def sample_action(self, theta, t, states, mu_mean, generator):
        noise = torch.randn(states.shape, dtype=states.dtype, device=states.device, generator=generator)
        return self.policy_mean(theta, t, states, mu_mean) + self.tau[t] * noise

    def transition(self, states, mu, actions, t):
        mean = self.s[t] * states + self.rbar[t] * actions
        variance = self.sigma_R[t].square() * actions.square()
        return torch.stack([mean, variance], dim=-1)

    def sample(self, states, mu, actions, generator, t):
        returns = self.sample_excess_returns(t, states.shape, generator)
        return self.s[t] * states + returns * actions

    def reward(self, states, mu, actions):
        gamma = self.config.mean_field_penalty
        if gamma == 0.0:
            return torch.zeros_like(states, dtype=self.dtype)
        return torch.zeros_like(states, dtype=self.dtype) - gamma * mu.square()

    def terminal_reward(self, states, mu):
        return states - self.config.chi * (states - mu).square()

    def moment_flow(self, theta, lambda_=None):
        lambda_ = self.config.perturbation_scale if lambda_ is None else lambda_
        mu = [torch.tensor(self.config.x0_mean, dtype=self.dtype, device=self.device)]
        Sigma = [torch.tensor(self.config.x0_variance, dtype=self.dtype, device=self.device)]

        for t in range(self.config.T):
            k, ell = theta[t]
            mu.append(self.s[t] * mu[t] + self.rbar[t] * ell)
            Sigma.append(
                (self.s[t].square() + 2.0 * self.s[t] * self.rbar[t] * k + self.h[t] * k.square()) * Sigma[t]
                + self.h[t] * self.tau[t].square()
                + self.sigma_R[t].square() * ell.square()
                + self.h[t] * k.square() * lambda_**2 * self.config.rho**2
            )

        return torch.stack(mu), torch.stack(Sigma)

    def objective(self, theta, lambda_=None):
        lambda_ = self.config.perturbation_scale if lambda_ is None else lambda_
        mu, Sigma = self.moment_flow(theta, lambda_)
        value = mu[-1] - self.config.chi * (Sigma[-1] + lambda_**2 * self.config.rho**2)
        gamma = self.config.mean_field_penalty
        if gamma != 0.0:
            perturbation = lambda_**2 * self.config.rho**2
            running = mu[: self.config.T].square() + perturbation
            value = value - gamma * running.sum()
        return value

    def exact_gradient(self, theta, lambda_=None):
        lambda_ = self.config.perturbation_scale if lambda_ is None else lambda_
        mu, Sigma = self.moment_flow(theta, lambda_)
        p = torch.empty(self.config.T + 1, dtype=self.dtype, device=self.device)
        adj_var = torch.empty(self.config.T + 1, dtype=self.dtype, device=self.device)

        p[-1] = 1.0
        adj_var[-1] = -self.config.chi

        for t in range(self.config.T - 1, -1, -1):
            k = theta[t, 0]
            p[t] = self.s[t] * p[t + 1] - 2.0 * self.config.mean_field_penalty * mu[t]
            adj_var[t] = (
                self.s[t].square() + 2.0 * self.s[t] * self.rbar[t] * k + self.h[t] * k.square()
            ) * adj_var[t + 1]

        grad = torch.empty_like(theta)
        for t in range(self.config.T):
            k, ell = theta[t]
            grad[t, 0] = 2.0 * adj_var[t + 1] * (
                (self.s[t] * self.rbar[t] + self.h[t] * k) * Sigma[t]
                + self.h[t] * k * lambda_**2 * self.config.rho**2
            )
            grad[t, 1] = self.rbar[t] * p[t + 1] + 2.0 * self.sigma_R[t].square() * ell * adj_var[t + 1]

        return grad

    def zero_policy(self):
        return torch.zeros(self.config.T, self.n_params, dtype=self.dtype, device=self.device)

    def optimal_theta(self):
        theta = torch.empty(self.config.T, self.n_params, dtype=self.dtype, device=self.device)
        B = self.rbar.square() / self.h
        product = torch.ones((), dtype=self.dtype, device=self.device)

        for t in range(self.config.T - 1, -1, -1):
            theta[t, 0] = -self.s[t] * self.rbar[t] / self.h[t]
            theta[t, 1] = self.rbar[t] * product / (2.0 * self.config.chi * self.sigma_R[t].square())
            product = product / (self.s[t] * (1.0 - B[t]))

        if self.config.mean_field_penalty == 0.0:
            return theta

        # The running mean-field penalty leaves the optimal k_t unchanged, since k_t
        # enters only through the variance recursion. The objective stays quadratic and
        # concave in ell, so grad_ell J is affine and the optimum solves a linear system.
        theta[:, 1] = 0.0
        base = self.exact_gradient(theta, lambda_=0.0)[:, 1]
        hessian = torch.empty(self.config.T, self.config.T, dtype=self.dtype, device=self.device)
        for t in range(self.config.T):
            probe = theta.clone()
            probe[t, 1] = 1.0
            hessian[:, t] = self.exact_gradient(probe, lambda_=0.0)[:, 1] - base
        theta[:, 1] = torch.linalg.solve(hessian, -base)
        return theta

    def optimal_policy(self):
        return self.optimal_theta()
