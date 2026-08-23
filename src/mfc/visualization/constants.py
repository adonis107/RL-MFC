from mfc.environments import (
    Advertising,
    AdvertisingConfig,
    AdvertisingPolicy,
    Cybersecurity,
    CybersecurityConfig,
    CybersecurityPolicy,
    Distribution,
    DistributionConfig,
    DistributionPolicy,
    Kuramoto,
    KuramotoConfig,
    KuramotoPolicy,
    LQ,
    LQConfig,
    Portfolio,
    PortfolioConfig,
    TwoState,
    TwoStateConfig,
)


ENVIRONMENTS = {
    "advertising": (Advertising, AdvertisingConfig),
    "cybersecurity": (Cybersecurity, CybersecurityConfig),
    "distribution": (Distribution, DistributionConfig),
    "kuramoto": (Kuramoto, KuramotoConfig),
    "lq": (LQ, LQConfig),
    "portfolio": (Portfolio, PortfolioConfig),
    "twostate": (TwoState, TwoStateConfig),
}

POLICIES = {
    "advertising": AdvertisingPolicy,
    "cybersecurity": CybersecurityPolicy,
    "distribution": DistributionPolicy,
    "kuramoto": KuramotoPolicy,
}

STATE_LABELS = {
    "advertising": ["not customer", "customer"],
    "cybersecurity": ["DI", "DS", "UI", "US"],
    "distribution": [str(i) for i in range(10)],
    "twostate": ["0", "1"],
}
