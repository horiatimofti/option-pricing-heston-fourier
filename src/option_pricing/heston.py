"""Heston stochastic-volatility model — TO BE IMPLEMENTED (steps 3–4).

Risk-neutral dynamics:

    dS_t = (r - q) S_t dt + sqrt(v_t) S_t dW_t^S
    dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW_t^v
    d<W^S, W^v>_t = rho dt

Parameters: v0 (initial variance), kappa (mean-reversion speed), theta
(long-run variance), xi (vol of vol), rho (spot–vol correlation).
Feller condition 2 kappa theta > xi^2 keeps v_t strictly positive.

Planned functions:
  * heston_char_func(u, ...)  characteristic function of ln S_T
                              (use the "little Heston trap" form of
                              Albrecher et al. 2007 for numerical stability)
  * heston_fft_price(...)     Carr–Madan (1999) FFT over a strike grid
  * heston_mc_price(...)      Euler / full-truncation scheme for v_t,
                              returning an MCResult for comparison with FFT
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HestonParams:
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def feller_satisfied(self) -> bool:
        return 2 * self.kappa * self.theta > self.xi**2


def heston_char_func(*args, **kwargs):
    raise NotImplementedError("Step 3: characteristic function of ln S_T")


def heston_fft_price(*args, **kwargs):
    raise NotImplementedError("Step 3: Carr–Madan FFT pricing")


def heston_mc_price(*args, **kwargs):
    raise NotImplementedError("Step 4: Monte Carlo under Heston")
