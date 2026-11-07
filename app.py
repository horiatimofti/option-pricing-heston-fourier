"""Interactive demo. Run with:  uv run streamlit run app.py

Currently covers Black–Scholes (analytical vs Monte Carlo).
Heston tabs will be added once the Fourier and MC implementations exist.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from option_pricing import bs_mc_price, bs_price

st.set_page_config(page_title="Option Pricing: BS & Heston", layout="wide")
st.title("Option pricing — Black–Scholes and Heston")

with st.sidebar:
    st.header("Market & contract")
    S0 = st.number_input("Spot S₀", value=100.0, min_value=0.01)
    K = st.number_input("Strike K", value=100.0, min_value=0.01)
    T = st.number_input("Maturity T (years)", value=1.0, min_value=0.01)
    r = st.number_input("Risk-free rate r", value=0.05, step=0.005, format="%.4f")
    q = st.number_input("Dividend yield q", value=0.0, step=0.005, format="%.4f")
    sigma = st.number_input("Volatility σ", value=0.20, min_value=0.001, step=0.01)
    option_type = st.radio("Option", ["call", "put"], horizontal=True)
    st.header("Monte Carlo")
    n_paths = st.select_slider("Paths", [1_000, 10_000, 100_000, 1_000_000], value=100_000)
    antithetic = st.checkbox("Antithetic variates")
    seed = st.number_input("Seed", value=42, step=1)

tab_bs, tab_heston = st.tabs(["Black–Scholes", "Heston (coming next)"])

with tab_bs:
    exact = float(bs_price(S0, K, T, r, sigma, q, option_type))
    mc = bs_mc_price(S0, K, T, r, sigma, q, option_type, n_paths, antithetic, int(seed))
    lo, hi = mc.ci

    c1, c2, c3 = st.columns(3)
    c1.metric("Analytical", f"{exact:.4f}")
    c2.metric("Monte Carlo", f"{mc.price:.4f}", f"{mc.price - exact:+.4f} vs analytical")
    c3.metric("95% CI", f"[{lo:.4f}, {hi:.4f}]", f"SE {mc.std_error:.4f}", delta_color="off")

    left, right = st.columns(2)

    with left:
        st.subheader("Price across strikes")
        strikes = np.linspace(0.5 * S0, 1.5 * S0, 101)
        fig = go.Figure(go.Scatter(x=strikes, y=bs_price(S0, strikes, T, r, sigma, q, option_type),
                                   name="Analytical"))
        fig.add_vline(x=S0, line_dash="dot", annotation_text="S₀")
        fig.update_layout(xaxis_title="Strike", yaxis_title="Price", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Monte Carlo convergence")
        ns = np.unique(np.logspace(2, np.log10(max(n_paths, 1000)), 20).astype(int))
        rows = [bs_mc_price(S0, K, T, r, sigma, q, option_type, int(n), antithetic, int(seed))
                for n in ns]
        df = pd.DataFrame({"n": ns, "price": [x.price for x in rows],
                           "lo": [x.ci[0] for x in rows], "hi": [x.ci[1] for x in rows]})
        fig = go.Figure([
            go.Scatter(x=df.n, y=df.hi, line_width=0, showlegend=False),
            go.Scatter(x=df.n, y=df.lo, fill="tonexty", line_width=0, name="95% CI"),
            go.Scatter(x=df.n, y=df.price, mode="lines+markers", name="MC estimate"),
        ])
        fig.add_hline(y=exact, line_dash="dash", annotation_text="Analytical")
        fig.update_layout(xaxis_type="log", xaxis_title="Number of paths",
                          yaxis_title="Price", height=400)
        st.plotly_chart(fig, use_container_width=True)

with tab_heston:
    st.info("Heston Fourier (Carr–Madan FFT) and Monte Carlo pricing are not implemented yet. "
            "See `src/option_pricing/heston.py`.")
