"""Monte Carlo pricing of European options, with uncertainty estimates.

Under Black–Scholes, S_T can be sampled exactly (no time-stepping error):

    S_T = S0 exp[(r - q - sigma^2/2) T + sigma sqrt(T) Z],   Z ~ N(0, 1)

The price estimator is the sample mean of discounted payoffs

    V_hat = e^{-rT} (1/n) sum_i payoff(S_T^(i))

and by the CLT its standard error is  s / sqrt(n), where s is the sample
standard deviation of the discounted payoffs. Error therefore shrinks like
1/sqrt(n): 4x more paths halves the error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black_scholes import OptionType


@dataclass(frozen=True)
class MCResult:
    price: float
    std_error: float
    n_paths: int
    confidence: float = 0.95

    @property
    def ci(self) -> tuple[float, float]:
        """Two-sided normal-approximation confidence interval."""
        z = norm.ppf(0.5 + self.confidence / 2)
        return self.price - z * self.std_error, self.price + z * self.std_error


def payoff(S_T: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
    if option_type == "call":
        return np.maximum(S_T - K, 0.0)
    if option_type == "put":
        return np.maximum(K - S_T, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def summarize(discounted_payoffs: np.ndarray, confidence: float = 0.95) -> MCResult:
    n = discounted_payoffs.size
    return MCResult(
        price=float(discounted_payoffs.mean()),
        std_error=float(discounted_payoffs.std(ddof=1) / np.sqrt(n)),
        n_paths=n,
        confidence=confidence,
    )


def bs_mc_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    n_paths: int = 100_000,
    antithetic: bool = False,
    seed: int | None = None,
    confidence: float = 0.95,
) -> MCResult:
    """Monte Carlo price of a European option under Black–Scholes.

    With ``antithetic=True``, each normal draw Z is paired with -Z and the two
    payoffs are averaged before computing the standard error, so the reported
    error correctly reflects the variance reduction (pairs are i.i.d.; the
    individual paths are not).
    """
    rng = np.random.default_rng(seed)
    drift = (r - q - 0.5 * sigma**2) * T
    vol = sigma * np.sqrt(T)
    discount = np.exp(-r * T)

    if antithetic:
        Z = rng.standard_normal(n_paths // 2)
        pay_plus = payoff(S0 * np.exp(drift + vol * Z), K, option_type)
        pay_minus = payoff(S0 * np.exp(drift - vol * Z), K, option_type)
        samples = discount * 0.5 * (pay_plus + pay_minus)
    else:
        Z = rng.standard_normal(n_paths)
        samples = discount * payoff(S0 * np.exp(drift + vol * Z), K, option_type)

    return summarize(samples, confidence)
