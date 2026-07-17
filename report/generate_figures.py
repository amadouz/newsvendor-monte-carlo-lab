"""Generates the static figures (and LaTeX table rows) used in the final report,
by driving the same simulation engine (newsvendor.py) that powers the Streamlit app.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from newsvendor import (
    NewsvendorInputs,
    analytical_q_star,
    convergence_series,
    ppf,
    profit_distribution_at_q,
    run_simulation,
    sensitivity_heatmap,
    scale_dispersion,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette (validated, from the project's dataviz reference)
# ---------------------------------------------------------------------------
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
VIOLET = "#4a3aa7"
BLUE_BAND = "#b7d3f6"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#1c5cab", "#0d366b"]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

SEED = 42
N_REPS = 50000
GRID_POINTS = 150

# Baseline economics: cu = 13, co = 7, critical ratio = 0.65
PRICE, COST, SALVAGE, PENALTY = 25.0, 12.0, 5.0, 0.0
BASE_MEAN, BASE_STD = 200.0, 40.0

baseline = NewsvendorInputs(PRICE, COST, SALVAGE, PENALTY, "Normal", {"mean": BASE_MEAN, "std": BASE_STD})

lo = max(ppf("Normal", 0.005, baseline.dist_params), 0.0)
hi = ppf("Normal", 0.995, baseline.dist_params)
q_grid = np.linspace(lo, hi, GRID_POINTS)
res = run_simulation(baseline, q_grid, N_REPS, SEED)
q_star_a = analytical_q_star(baseline)
q_star_s = res["q_sim_star"]
gap_pct = abs(q_star_s - q_star_a) / q_star_a * 100

print(f"critical ratio = {baseline.critical_ratio:.4f}")
print(f"analytical Q*  = {q_star_a:.2f}")
print(f"simulated Q*   = {q_star_s:.2f}")
print(f"gap            = {gap_pct:.3f}%")
print(f"E[profit] at Q*= {res['profit_sim_star']:.2f}")

# ---------------------------------------------------------------------------
# Figure 1: expected profit vs order quantity
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.fill_between(q_grid, res["mean_profit"] - res["ci95"], res["mean_profit"] + res["ci95"],
                 color=BLUE_BAND, alpha=0.6, linewidth=0, label="95% CI")
ax.plot(q_grid, res["mean_profit"], color=BLUE, linewidth=2, label="Simulated expected profit")
ax.axvline(q_star_a, color=VIOLET, linestyle="--", linewidth=2, label=f"Analytical $Q^*$ = {q_star_a:.0f}")
ax.axvline(q_star_s, color=AQUA, linestyle=":", linewidth=2, label=f"Simulated $Q^*$ = {q_star_s:.0f}")
ax.set_xlabel("Order quantity $Q$")
ax.set_ylabel("Expected profit ($)")
ax.set_title("Expected profit across candidate order quantities", color=INK, fontsize=12)
ax.legend(frameon=False, fontsize=9, loc="lower center")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_profit_curve.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: profit distribution histogram at Q*
# ---------------------------------------------------------------------------
outcomes = profit_distribution_at_q(baseline, q_star_s, N_REPS, SEED)
mean_p = outcomes.mean()
p5, p95 = np.percentile(outcomes, [5, 95])

fig, ax = plt.subplots(figsize=(6.5, 4.0))
ax.hist(outcomes, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
ax.axvspan(p5, p95, color=AQUA, alpha=0.08, linewidth=0)
ax.axvline(mean_p, color=INK, linestyle="--", linewidth=2, label=f"Mean = \\${mean_p:,.0f}")
ax.set_xlabel("Simulated profit ($)")
ax.set_ylabel("Frequency")
ax.set_title(f"Profit distribution at $Q$ = {q_star_s:.0f}  (shaded: 5th–95th pct.)", color=INK, fontsize=12)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_profit_hist.png"), dpi=200)
plt.close(fig)
print(f"hist: mean={mean_p:.2f} p5={p5:.2f} p95={p95:.2f} std={outcomes.std(ddof=1):.2f}")

# ---------------------------------------------------------------------------
# Figure 3: sensitivity heatmap (unit cost x demand variability multiplier)
# ---------------------------------------------------------------------------
cost_values = np.linspace(SALVAGE + 0.5, PRICE - 0.5, 40)
variability_mults = np.linspace(0.5, 1.75, 40)
heat = sensitivity_heatmap(PRICE, SALVAGE, PENALTY, "Normal", baseline.dist_params, cost_values, variability_mults)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
cmap = mpl.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
im = ax.imshow(heat, origin="lower", aspect="auto", cmap=cmap,
               extent=[variability_mults[0], variability_mults[-1], cost_values[0], cost_values[-1]])
ax.scatter([1.0], [COST], color="#e34948", s=70, edgecolor=SURFACE, linewidth=1.5, zorder=5,
           label="Current inputs")
cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Analytical $Q^*$", color=INK_SECONDARY)
cbar.ax.yaxis.set_tick_params(color=INK_MUTED, labelcolor=INK_MUTED)
ax.set_xlabel("Demand variability multiplier (1.0 = baseline $\\sigma$)")
ax.set_ylabel("Unit cost ($)")
ax.set_title("Optimal order quantity vs. cost and demand variability", color=INK, fontsize=12)
leg = ax.legend(frameon=True, fontsize=9, loc="upper right")
leg.get_frame().set_facecolor(SURFACE)
leg.get_frame().set_edgecolor(BASELINE)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_sensitivity_heatmap.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 4: Monte Carlo convergence
# ---------------------------------------------------------------------------
conv = convergence_series(baseline, q_star_s, N_REPS, SEED)
fig, ax = plt.subplots(figsize=(6.5, 3.6))
ax.plot(np.arange(1, len(conv) + 1), conv, color=BLUE, linewidth=1.5)
ax.axhline(res["profit_sim_star"], color=INK_MUTED, linestyle="--", linewidth=1)
ax.set_xlabel("Replications")
ax.set_ylabel(r"Running mean profit at $Q^*$ (\$)")
ax.set_title("Monte Carlo convergence of expected profit", color=INK, fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "fig_convergence.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------------------
# Cross-distribution comparison table (statistical rigor across distributions)
# ---------------------------------------------------------------------------
mean_target, std_target = 200.0, 40.0
cv = std_target / mean_target
sigma_ln = np.sqrt(np.log(1 + cv ** 2))
mu_ln = np.log(mean_target) - sigma_ln ** 2 / 2
gamma_scale = std_target ** 2 / mean_target
gamma_shape = mean_target / gamma_scale
half_width = std_target * np.sqrt(12) / 2

dists = {
    "Normal": {"mean": mean_target, "std": std_target},
    "Uniform": {"low": mean_target - half_width, "high": mean_target + half_width},
    "Poisson": {"lam": mean_target},
    "Lognormal": {"mu": mu_ln, "sigma": sigma_ln},
    "Gamma": {"shape": gamma_shape, "scale": gamma_scale},
}

print("\nCross-distribution comparison (price=25, cost=12, salvage=5, critical ratio=0.65):")
rows = []
for name, params in dists.items():
    inp = NewsvendorInputs(PRICE, COST, SALVAGE, PENALTY, name, params)
    qa = analytical_q_star(inp)
    lo_d = max(ppf(name, 0.005, params), 0.0)
    hi_d = ppf(name, 0.995, params)
    grid_d = np.linspace(lo_d, hi_d, GRID_POINTS)
    res_d = run_simulation(inp, grid_d, N_REPS, SEED)
    qs = res_d["q_sim_star"]
    gap = abs(qs - qa) / qa * 100
    dstd = res_d["demand_std"]
    row = (name, qa, qs, gap, res_d["profit_sim_star"], dstd)
    rows.append(row)
    print(f"  {name:10s}  Q*_analytical={qa:7.2f}  Q*_sim={qs:7.2f}  gap={gap:5.2f}%  "
          f"E[profit]=${res_d['profit_sim_star']:8.2f}  demand_std={dstd:6.2f}")

print("\nLaTeX table rows:")
for name, qa, qs, gap, ep, dstd in rows:
    print(f"{name} & {qa:.1f} & {qs:.1f} & {gap:.2f}\\% & \\${ep:,.0f} & {dstd:.1f} \\\\")
