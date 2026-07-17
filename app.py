"""Newsvendor Monte Carlo Lab — an interactive Streamlit front end for the
single-period newsvendor problem, built on the simulation engine in newsvendor.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from newsvendor import (
    DISTRIBUTIONS,
    NewsvendorInputs,
    analytical_q_star,
    convergence_series,
    ppf,
    profit_distribution_at_q,
    run_simulation,
    sensitivity_heatmap,
)

# --------------------------------------------------------------------------------------
# Palette + chart chrome (light-mode validated palette; see project's dataviz reference)
# --------------------------------------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

BLUE = "#2a78d6"       # categorical slot 1 — primary series
AQUA = "#1baf7a"        # categorical slot 2 — simulated reference
VIOLET = "#4a3aa7"      # categorical slot 5 — analytical reference
RED = "#e34948"         # categorical slot 6 — shortage / risk
BLUE_BAND = "#b7d3f6"   # sequential step 150 — CI band fill
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]  # sequential blue 100->700

FONT = dict(family="system-ui, -apple-system, 'Segoe UI', sans-serif", color=INK)


def base_layout(height=420, **overrides):
    layout = dict(
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=FONT,
        margin=dict(l=10, r=10, t=40, b=10),
        hoverlabel=dict(bgcolor=SURFACE, font=FONT, bordercolor=BASELINE),
    )
    layout.update(overrides)
    return layout


def style_axes(fig, x_title="", y_title=""):
    fig.update_xaxes(
        title=dict(text=x_title, font=dict(color=INK_SECONDARY, size=13)),
        showgrid=False,
        showline=True,
        linecolor=BASELINE,
        tickfont=dict(color=INK_MUTED, size=11),
        zeroline=False,
    )
    fig.update_yaxes(
        title=dict(text=y_title, font=dict(color=INK_SECONDARY, size=13)),
        showgrid=True,
        gridcolor=GRIDLINE,
        showline=False,
        tickfont=dict(color=INK_MUTED, size=11),
        zeroline=False,
    )
    return fig


st.set_page_config(page_title="Newsvendor Monte Carlo Lab", layout="wide")

st.title("Newsvendor Monte Carlo Lab")
st.caption(
    "A single-period inventory decision: choose an order quantity **Q** before observing "
    "stochastic demand **D**, trading the cost of ordering too little against the cost of "
    "ordering too much."
)

# --------------------------------------------------------------------------------------
# Sidebar — model inputs
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.header("Model inputs")

    st.subheader("Economics")
    price = st.number_input("Selling price ($/unit)", min_value=0.01, value=25.0, step=1.0)
    cost = st.number_input("Unit cost ($/unit)", min_value=0.0, value=12.0, step=1.0)
    salvage = st.number_input("Salvage value ($/unit)", min_value=0.0, value=5.0, step=1.0)
    with st.expander("Advanced: goodwill / stockout penalty"):
        stockout_penalty = st.number_input(
            "Extra penalty per unit of unmet demand ($)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            help="On top of the lost margin, an additional goodwill or contractual penalty per unit short.",
        )

    st.subheader("Demand distribution")
    dist_name = st.selectbox("Distribution", list(DISTRIBUTIONS.keys()))
    dist_params = {}
    for key, label, default in DISTRIBUTIONS[dist_name]["params"]:
        dist_params[key] = st.number_input(label, value=float(default))

    st.subheader("Simulation settings")
    n_reps = st.select_slider(
        "Replications per order quantity", options=[1000, 2000, 5000, 10000, 20000, 50000, 100000], value=20000
    )
    grid_points = st.slider("Order-quantity grid resolution", 30, 250, 120)
    seed = st.number_input("Random seed", value=42, step=1)

# --------------------------------------------------------------------------------------
# Validate inputs
# --------------------------------------------------------------------------------------
errors = []
if cost <= salvage:
    errors.append("Unit cost must be greater than salvage value (otherwise there is no overage cost).")
if price <= cost:
    errors.append("Selling price must be greater than unit cost (otherwise there is no underage cost).")
if dist_name == "Uniform" and dist_params["high"] <= dist_params["low"]:
    errors.append("Uniform maximum demand must exceed minimum demand.")

if errors:
    for e in errors:
        st.error(e)
    st.stop()

inputs = NewsvendorInputs(
    price=price, cost=cost, salvage=salvage, stockout_penalty=stockout_penalty,
    dist_name=dist_name, dist_params=dist_params,
)
dist_params_items = tuple(sorted(dist_params.items()))


# --------------------------------------------------------------------------------------
# Cached simulation runs
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner="Running Monte Carlo simulation…")
def cached_run_simulation(price, cost, salvage, stockout_penalty, dist_name, dist_params_items, n_reps, grid_points, seed):
    dp = dict(dist_params_items)
    inp = NewsvendorInputs(price, cost, salvage, stockout_penalty, dist_name, dp)
    lo = max(ppf(dist_name, 0.005, dp), 0.0)
    hi = ppf(dist_name, 0.995, dp)
    q_grid = np.linspace(lo, hi, grid_points)
    return run_simulation(inp, q_grid, n_reps, seed)


@st.cache_data(show_spinner="Sampling profit distribution…")
def cached_profit_distribution(price, cost, salvage, stockout_penalty, dist_name, dist_params_items, q, n_reps, seed):
    dp = dict(dist_params_items)
    inp = NewsvendorInputs(price, cost, salvage, stockout_penalty, dist_name, dp)
    return profit_distribution_at_q(inp, q, n_reps, seed)


@st.cache_data(show_spinner="Checking convergence…")
def cached_convergence(price, cost, salvage, stockout_penalty, dist_name, dist_params_items, q, n_reps, seed):
    dp = dict(dist_params_items)
    inp = NewsvendorInputs(price, cost, salvage, stockout_penalty, dist_name, dp)
    return convergence_series(inp, q, n_reps, seed)


@st.cache_data(show_spinner="Computing sensitivity grid…")
def cached_sensitivity(price, salvage, stockout_penalty, dist_name, dist_params_items, cost_lo, cost_hi):
    dp = dict(dist_params_items)
    cost_values = np.linspace(cost_lo, cost_hi, 18)
    variability_mults = np.linspace(0.5, 1.75, 18)
    grid = sensitivity_heatmap(price, salvage, stockout_penalty, dist_name, dp, cost_values, variability_mults)
    return cost_values, variability_mults, grid


results = cached_run_simulation(
    price, cost, salvage, stockout_penalty, dist_name, dist_params_items, n_reps, grid_points, seed
)
q_star_analytical = analytical_q_star(inputs)
q_star_sim = results["q_sim_star"]
gap_pct = abs(q_star_sim - q_star_analytical) / max(q_star_analytical, 1e-9) * 100

# --------------------------------------------------------------------------------------
# KPI row
# --------------------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Critical ratio", f"{inputs.critical_ratio:.3f}")
k2.metric("Analytical Q*", f"{q_star_analytical:,.1f}")
k3.metric("Simulated Q*", f"{q_star_sim:,.1f}")
k4.metric("Gap vs. analytical", f"{gap_pct:.2f}%")
k5.metric("Expected profit at Q*", f"${results['profit_sim_star']:,.0f}")

st.divider()

tab_curve, tab_dist, tab_sensitivity, tab_method = st.tabs(
    ["Profit vs. order quantity", "Profit distribution", "Sensitivity analysis", "Methodology & validation"]
)

# --------------------------------------------------------------------------------------
# Tab 1: Profit vs order quantity
# --------------------------------------------------------------------------------------
with tab_curve:
    st.subheader("Expected profit across candidate order quantities")
    q_grid = results["q_grid"]
    mean_profit = results["mean_profit"]
    ci = results["ci95"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([q_grid, q_grid[::-1]]),
        y=np.concatenate([mean_profit + ci, (mean_profit - ci)[::-1]]),
        fill="toself", fillcolor=BLUE_BAND, opacity=0.5,
        line=dict(width=0), hoverinfo="skip", showlegend=False, name="95% CI",
    ))
    fig.add_trace(go.Scatter(
        x=q_grid, y=mean_profit, mode="lines", line=dict(color=BLUE, width=2),
        name="Expected profit",
        hovertemplate="Q = %{x:,.0f}<br>Expected profit = $%{y:,.0f}<extra></extra>",
    ))
    fig.add_vline(x=q_star_analytical, line=dict(color=VIOLET, width=2, dash="dash"))
    fig.add_annotation(x=q_star_analytical, y=1.0, yref="paper", yanchor="bottom",
                        text=f"Analytical Q* = {q_star_analytical:,.0f}", showarrow=False,
                        font=dict(color=VIOLET, size=12))
    fig.add_vline(x=q_star_sim, line=dict(color=AQUA, width=2, dash="dot"))
    fig.add_annotation(x=q_star_sim, y=0.90, yref="paper", yanchor="bottom",
                        text=f"Simulated Q* = {q_star_sim:,.0f}", showarrow=False,
                        font=dict(color=AQUA, size=12))
    fig.update_layout(**base_layout(height=460, showlegend=False))
    style_axes(fig, "Order quantity (Q)", "Expected profit ($)")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Shaded band is a 95% confidence interval on the simulated expected profit "
        f"(±1.96·SE across {n_reps:,} replications per Q). Dashed line marks the closed-form "
        "critical-ratio optimum; dotted line marks the simulation's argmax."
    )

# --------------------------------------------------------------------------------------
# Tab 2: Profit distribution at a chosen Q
# --------------------------------------------------------------------------------------
with tab_dist:
    st.subheader("Distribution of profit outcomes at a chosen order quantity")
    q_choice = st.slider(
        "Order quantity to inspect", min_value=float(q_grid[0]), max_value=float(q_grid[-1]),
        value=float(q_star_sim), step=max((q_grid[-1] - q_grid[0]) / 200, 1e-6),
    )
    outcomes = cached_profit_distribution(
        price, cost, salvage, stockout_penalty, dist_name, dist_params_items, q_choice, n_reps, seed
    )
    mean_p = outcomes.mean()
    p5, p95 = np.percentile(outcomes, [5, 95])

    d1, d2, d3 = st.columns(3)
    d1.metric("Expected profit at this Q", f"${mean_p:,.0f}")
    d2.metric("5th–95th percentile range", f"${p5:,.0f} – ${p95:,.0f}")
    d3.metric("Std. deviation of profit", f"${outcomes.std(ddof=1):,.0f}")

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=outcomes, nbinsx=60, marker=dict(color=BLUE, line=dict(color=SURFACE, width=1)),
        name="Profit outcomes", hovertemplate="Profit ≈ $%{x:,.0f}<br>Count = %{y}<extra></extra>",
    ))
    hist_fig.add_vline(x=mean_p, line=dict(color=INK, width=2, dash="dash"))
    hist_fig.add_annotation(x=mean_p, y=1.0, yref="paper", yanchor="bottom",
                             text=f"Mean = ${mean_p:,.0f}", showarrow=False, font=dict(color=INK, size=12))
    hist_fig.add_vrect(x0=p5, x1=p95, fillcolor=AQUA, opacity=0.08, line_width=0)
    hist_fig.update_layout(**base_layout(height=420, showlegend=False))
    style_axes(hist_fig, "Simulated profit ($)", "Frequency")
    st.plotly_chart(hist_fig, use_container_width=True)
    st.caption("Shaded band marks the 5th–95th percentile range of simulated profit outcomes at this Q.")

# --------------------------------------------------------------------------------------
# Tab 3: Sensitivity analysis
# --------------------------------------------------------------------------------------
with tab_sensitivity:
    st.subheader("How the optimal order quantity moves with cost and demand variability")
    cost_lo = max(salvage + 0.5, 0.5)
    cost_hi = price - 0.5
    if cost_hi <= cost_lo:
        st.info("Widen the gap between salvage value and selling price to explore cost sensitivity.")
    else:
        cost_values, variability_mults, grid = cached_sensitivity(
            price, salvage, stockout_penalty, dist_name, dist_params_items, cost_lo, cost_hi
        )
        heat = go.Figure(data=go.Heatmap(
            x=variability_mults, y=cost_values, z=grid,
            colorscale=[[i / (len(SEQ_RAMP) - 1), c] for i, c in enumerate(SEQ_RAMP)],
            colorbar=dict(title=dict(text="Q*", font=dict(color=INK_SECONDARY)), tickfont=dict(color=INK_MUTED)),
            hovertemplate="Variability × %{x:.2f}<br>Unit cost = $%{y:.1f}<br>Analytical Q* = %{z:,.0f}<extra></extra>",
        ))
        heat.add_trace(go.Scatter(
            x=[1.0], y=[cost], mode="markers",
            marker=dict(color=RED, size=12, line=dict(color=SURFACE, width=2)),
            name="Current inputs", hovertemplate="Current inputs<extra></extra>",
        ))
        heat.update_layout(**base_layout(height=460, showlegend=False))
        style_axes(heat, "Demand variability multiplier (1.0 = current spread)", "Unit cost ($)")
        st.plotly_chart(heat, use_container_width=True)
        if dist_name == "Poisson":
            st.caption(
                "Poisson demand ties variance to the mean, so the variability multiplier is not independently "
                "adjustable for this distribution — the heatmap holds it fixed. The red marker shows your current inputs."
            )
        else:
            st.caption(
                "Color is the closed-form optimal Q* as unit cost and demand dispersion vary, holding price and "
                "salvage fixed. The red marker shows your current inputs."
            )

# --------------------------------------------------------------------------------------
# Tab 4: Methodology & validation
# --------------------------------------------------------------------------------------
with tab_method:
    st.subheader("Model")
    st.latex(r"\text{profit}(d, Q) = p\cdot\min(d,Q) + s\cdot\max(Q-d,0) - c\cdot Q - g\cdot\max(d-Q,0)")
    st.markdown(
        "where $p$ = selling price, $c$ = unit cost, $s$ = salvage value, and $g$ = stockout penalty."
    )
    st.latex(r"F(Q^*) = \frac{c_u}{c_u + c_o}, \qquad c_u = p - c + g, \qquad c_o = c - s")

    st.subheader("Validation: simulation vs. closed form")
    val_df = pd.DataFrame({
        "Quantity": ["Analytical Q* (critical ratio)", "Simulated Q* (grid argmax)"],
        "Order quantity": [q_star_analytical, q_star_sim],
        "Expected profit": [
            float(np.interp(q_star_analytical, q_grid, mean_profit)),
            results["profit_sim_star"],
        ],
    })
    st.dataframe(val_df.style.format({"Order quantity": "{:,.1f}", "Expected profit": "${:,.0f}"}), hide_index=True)
    st.caption(
        f"Gap between simulated and analytical Q*: **{gap_pct:.2f}%** "
        f"(grid resolution: {grid_points} points; expect the gap to shrink as grid resolution and replications increase)."
    )

    st.subheader("Monte Carlo convergence")
    conv = cached_convergence(
        price, cost, salvage, stockout_penalty, dist_name, dist_params_items, q_star_sim, n_reps, seed
    )
    conv_fig = go.Figure()
    conv_fig.add_trace(go.Scatter(
        x=np.arange(1, len(conv) + 1), y=conv, mode="lines", line=dict(color=BLUE, width=2),
        name="Running mean profit",
        hovertemplate="Replications = %{x:,}<br>Running mean profit = $%{y:,.0f}<extra></extra>",
    ))
    conv_fig.add_hline(y=results["profit_sim_star"], line=dict(color=INK_MUTED, width=1, dash="dash"))
    conv_fig.update_layout(**base_layout(height=360, showlegend=False))
    style_axes(conv_fig, "Replications", "Running mean profit at simulated Q* ($)")
    st.plotly_chart(conv_fig, use_container_width=True)
    st.caption("The running mean should stabilize as replications accumulate — evidence the estimate has converged.")

    st.subheader("Demand distribution summary (from this run's samples)")
    m1, m2 = st.columns(2)
    m1.metric("Simulated demand mean", f"{results['demand_mean']:,.1f}")
    m2.metric("Simulated demand std. dev.", f"{results['demand_std']:,.1f}")
