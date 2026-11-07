# Option Pricing under Black–Scholes and Heston: Monte Carlo and Fourier Methods

[![CI](https://github.com/horiatimofti/option-pricing-heston-fourier/actions/workflows/ci.yml/badge.svg)](https://github.com/horiatimofti/option-pricing-heston-fourier/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Pricing European options under two models and with two numerical methods, with an
emphasis on numerical accuracy and honest error reporting.

Two separate comparisons are kept apart:

- **Model comparison:** Black–Scholes (constant volatility) vs Heston (stochastic variance).
- **Method comparison:** Monte Carlo vs Fourier inversion, run on the **same** Heston model and
  parameters so that any difference is numerical, not a modelling difference.

## Status

| Step | Component | Status |
|---|---|---|
| 1 | Black–Scholes analytical price (calls, puts) | ✅ done, tested |
| 2 | Black–Scholes Monte Carlo with standard error, CI, convergence | ✅ done, tested |
| 3 | Heston pricing via Carr–Madan FFT | ⏳ planned |
| 4 | Heston Monte Carlo, compared with FFT | ⏳ planned |
| 5 | Interactive app (strikes, implied vol, MC uncertainty, convergence) | 🟡 Black–Scholes part done |

## Setup

Requires [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 automatically if needed).

```bash
uv sync                          # create .venv and install dependencies (incl. dev tools)
uv run pytest                    # run the tests
uv run streamlit run app.py      # launch the interactive demo
```

## Layout

```
src/option_pricing/
  black_scholes.py   closed-form prices and vega
  monte_carlo.py     MC estimator, standard error, confidence interval, antithetic variates
  implied_vol.py     Black–Scholes implied volatility (Brent root-finding)
  heston.py          Heston model (to be implemented)
tests/               correctness tests (textbook values, put–call parity, MC vs analytical)
notebooks/           numbered, curated analysis notebooks
reports/figures/     figures used in this README
app.py               Streamlit demo
```

## Methods

**Black–Scholes.** Closed-form price; puts via put–call parity
`C − P = S₀e^{−qT} − Ke^{−rT}`.

**Monte Carlo.** Under Black–Scholes, `S_T` is sampled exactly, so the only error is
statistical. The estimator is the mean of discounted payoffs; its standard error is
`s/√n`, giving a CLT-based confidence interval. Antithetic variates pair `Z` with `−Z`;
the standard error is computed on pair averages so it reflects the true variance reduction.

## Results

To be filled in from actual runs once the Heston components are implemented.

## Limitations

- European payoffs only; no early exercise.
- Constant interest rate and dividend yield.
- Confidence intervals rely on the normal approximation (CLT), which is reliable only for
  sufficiently large path counts.

## References

- Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities.
- Heston, S. (1993). A closed-form solution for options with stochastic volatility.
- Carr, P. & Madan, D. (1999). Option valuation using the fast Fourier transform.
- Albrecher, H. et al. (2007). The little Heston trap.

## License

MIT — see [LICENSE](LICENSE).
