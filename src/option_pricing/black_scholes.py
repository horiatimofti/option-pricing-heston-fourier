"""Closed-form Black–Scholes prices for European calls and puts.

Model (risk-neutral measure Q), constant r, q, sigma:

    dS_t = (r - q) S_t dt + sigma S_t dW_t

Price of a European call with strike K and maturity T:

    C = S0 e^{-qT} N(d1) - K e^{-rT} N(d2)
    d1 = [ln(S0/K) + (r - q + sigma^2 / 2) T] / (sigma sqrt(T))
    d2 = d1 - sigma sqrt(T)

The put follows from put–call parity:  C - P = S0 e^{-qT} - K e^{-rT}.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

OptionType = Literal["call", "put"]


def _d1_d2(S0, K, T, r, sigma, q):
    vol_sqrt_t = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def bs_price(
    S0: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: float,
    sigma: ArrayLike,
    q: float = 0.0,
    option_type: OptionType = "call",
) -> np.ndarray:
    """Black–Scholes price of a European call or put.

    All array-like arguments are broadcast, so a vector of strikes returns a
    vector of prices. T is in years; r, q, sigma are annualised and continuous.
    """
    S0, K, T, sigma = (np.asarray(x, dtype=float) for x in (S0, K, T, sigma))
    if np.any(T <= 0) or np.any(sigma <= 0):
        raise ValueError("T and sigma must be strictly positive")

    d1, d2 = _d1_d2(S0, K, T, r, sigma, q)
    disc_S = S0 * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)

    if option_type == "call":
        return disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    if option_type == "put":
        return disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def bs_vega(S0, K, T, r, sigma, q: float = 0.0) -> np.ndarray:
    """dPrice/dsigma (same for calls and puts). Used by the implied-vol solver."""
    S0, K, T, sigma = (np.asarray(x, dtype=float) for x in (S0, K, T, sigma))
    d1, _ = _d1_d2(S0, K, T, r, sigma, q)
    return S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
