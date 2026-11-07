"""Black–Scholes implied volatility: the sigma that reproduces a given price.

Used later to turn Heston prices into an implied-volatility smile.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .black_scholes import OptionType, bs_price


def implied_vol(
    price: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float = 0.0,
    option_type: OptionType = "call",
    lo: float = 1e-6,
    hi: float = 5.0,
) -> float:
    """Solve bs_price(sigma) = price with Brent's method.

    Returns NaN if the price lies outside the no-arbitrage bounds, where no
    volatility can reproduce it.
    """

    def objective(sigma: float) -> float:
        return float(bs_price(S0, K, T, r, sigma, q, option_type)) - price

    try:
        return brentq(objective, lo, hi, xtol=1e-10)
    except ValueError:
        return float(np.nan)
