import numpy as np
import pytest

from option_pricing import bs_mc_price, bs_price, implied_vol

S0, K, T, r, sigma, q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.0


def test_textbook_value():
    # Hull's standard ATM example: S=K=100, T=1, r=5%, sigma=20% -> C ≈ 10.4506
    assert bs_price(S0, K, T, r, sigma) == pytest.approx(10.450583572185565, abs=1e-10)


@pytest.mark.parametrize("q_", [0.0, 0.03])
def test_put_call_parity(q_):
    strikes = np.linspace(60, 140, 17)
    c = bs_price(S0, strikes, T, r, sigma, q_, "call")
    p = bs_price(S0, strikes, T, r, sigma, q_, "put")
    np.testing.assert_allclose(c - p, S0 * np.exp(-q_ * T) - strikes * np.exp(-r * T))


def test_call_bounds():
    c = bs_price(S0, np.linspace(50, 150, 11), T, r, sigma)
    assert np.all(c >= np.maximum(S0 - np.linspace(50, 150, 11) * np.exp(-r * T), 0) - 1e-12)
    assert np.all(c <= S0)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("antithetic", [False, True])
def test_mc_matches_analytical(option_type, antithetic):
    exact = bs_price(S0, K, T, r, sigma, q, option_type)
    res = bs_mc_price(S0, K, T, r, sigma, q, option_type, n_paths=400_000,
                      antithetic=antithetic, seed=42)
    # Within 4 standard errors (fails by chance with prob. ~6e-5)
    assert abs(res.price - exact) < 4 * res.std_error


def test_mc_error_scales_like_inverse_sqrt_n():
    small = bs_mc_price(S0, K, T, r, sigma, n_paths=10_000, seed=1)
    large = bs_mc_price(S0, K, T, r, sigma, n_paths=160_000, seed=1)
    assert large.std_error / small.std_error == pytest.approx(0.25, rel=0.1)


def test_implied_vol_roundtrip():
    for vol in (0.1, 0.2, 0.45):
        price = float(bs_price(S0, 110.0, T, r, vol))
        assert implied_vol(price, S0, 110.0, T, r) == pytest.approx(vol, abs=1e-8)
