"""Monte Carlo engine and analytical benchmark for the single-period newsvendor problem.

Model
-----
A retailer chooses an order quantity Q before observing stochastic demand D.
For a realized demand d:

    sales      = min(d, Q)
    leftover   = max(Q - d, 0)
    shortage   = max(d - Q, 0)
    profit     = price * sales + salvage * leftover - cost * Q - stockout_penalty * shortage

The expected-profit-maximizing order quantity satisfies the critical-ratio
condition F(Q*) = cu / (cu + co), where:

    cu = price - cost + stockout_penalty   (underage cost: margin + goodwill lost per unit short)
    co = cost - salvage                    (overage cost: cost tied up per unit left over)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy import stats


def _normal_support(p):
    return max(0.0, p["mean"] - 5 * p["std"]), p["mean"] + 5 * p["std"]


DISTRIBUTIONS: dict[str, dict] = {
    "Normal": {
        "params": [("mean", "Mean demand", 200.0), ("std", "Std. deviation", 40.0)],
        "rvs": lambda rng, n, p: rng.normal(p["mean"], p["std"], n),
        "ppf": lambda q, p: stats.norm.ppf(q, p["mean"], p["std"]),
    },
    "Uniform": {
        "params": [("low", "Minimum demand", 100.0), ("high", "Maximum demand", 300.0)],
        "rvs": lambda rng, n, p: rng.uniform(p["low"], p["high"], n),
        "ppf": lambda q, p: stats.uniform.ppf(q, p["low"], p["high"] - p["low"]),
    },
    "Poisson": {
        "params": [("lam", "Mean demand (lambda)", 200.0)],
        "rvs": lambda rng, n, p: rng.poisson(p["lam"], n).astype(float),
        "ppf": lambda q, p: stats.poisson.ppf(q, p["lam"]),
    },
    "Lognormal": {
        "params": [("mu", "Log-mean (mu)", 5.2), ("sigma", "Log-std (sigma)", 0.3)],
        "rvs": lambda rng, n, p: rng.lognormal(p["mu"], p["sigma"], n),
        "ppf": lambda q, p: stats.lognorm.ppf(q, p["sigma"], scale=np.exp(p["mu"])),
    },
    "Gamma": {
        "params": [("shape", "Shape (k)", 16.0), ("scale", "Scale (theta)", 12.5)],
        "rvs": lambda rng, n, p: rng.gamma(p["shape"], p["scale"], n),
        "ppf": lambda q, p: stats.gamma.ppf(q, p["shape"], scale=p["scale"]),
    },
}


@dataclass(frozen=True)
class NewsvendorInputs:
    price: float
    cost: float
    salvage: float
    stockout_penalty: float
    dist_name: str
    dist_params: dict = field(default_factory=dict)

    @property
    def cu(self) -> float:
        return self.price - self.cost + self.stockout_penalty

    @property
    def co(self) -> float:
        return self.cost - self.salvage

    @property
    def critical_ratio(self) -> float:
        return self.cu / (self.cu + self.co)


def ppf(dist_name: str, q, dist_params: dict):
    return DISTRIBUTIONS[dist_name]["ppf"](q, dist_params)


def analytical_q_star(inputs: NewsvendorInputs) -> float:
    q_star = ppf(inputs.dist_name, inputs.critical_ratio, inputs.dist_params)
    return float(max(q_star, 0.0))


def sample_demand(dist_name: str, dist_params: dict, n_reps: int, rng: np.random.Generator) -> np.ndarray:
    d = DISTRIBUTIONS[dist_name]["rvs"](rng, n_reps, dist_params)
    return np.clip(d, 0, None)


def profit(demand: np.ndarray, q_grid: np.ndarray, inputs: NewsvendorInputs) -> np.ndarray:
    """Vectorized profit over (replications x order quantities)."""
    d = demand[:, None]
    q = q_grid[None, :]
    sales = np.minimum(d, q)
    leftover = np.maximum(q - d, 0)
    shortage = np.maximum(d - q, 0)
    return (
        inputs.price * sales
        + inputs.salvage * leftover
        - inputs.cost * q
        - inputs.stockout_penalty * shortage
    )


def run_simulation(inputs: NewsvendorInputs, q_grid: np.ndarray, n_reps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    demand = sample_demand(inputs.dist_name, inputs.dist_params, n_reps, rng)
    pi = profit(demand, q_grid, inputs)
    mean_profit = pi.mean(axis=0)
    std_profit = pi.std(axis=0, ddof=1)
    ci95 = 1.96 * std_profit / np.sqrt(n_reps)
    service_level = (demand[:, None] <= q_grid[None, :]).mean(axis=0)
    best_idx = int(np.argmax(mean_profit))
    return {
        "q_grid": q_grid,
        "mean_profit": mean_profit,
        "ci95": ci95,
        "service_level": service_level,
        "best_idx": best_idx,
        "q_sim_star": float(q_grid[best_idx]),
        "profit_sim_star": float(mean_profit[best_idx]),
        "demand_mean": float(demand.mean()),
        "demand_std": float(demand.std(ddof=1)),
    }


def profit_distribution_at_q(inputs: NewsvendorInputs, q: float, n_reps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    demand = sample_demand(inputs.dist_name, inputs.dist_params, n_reps, rng)
    return profit(demand, np.array([q]), inputs)[:, 0]


def convergence_series(inputs: NewsvendorInputs, q: float, n_reps: int, seed: int) -> np.ndarray:
    """Cumulative running-mean profit at a fixed Q, to visualize Monte Carlo convergence."""
    outcomes = profit_distribution_at_q(inputs, q, n_reps, seed)
    return np.cumsum(outcomes) / np.arange(1, n_reps + 1)


def scale_dispersion(dist_name: str, dist_params: dict, mult: float) -> dict:
    """Return a copy of dist_params with the distribution's dispersion scaled by mult."""
    p = dict(dist_params)
    if dist_name == "Normal":
        p["std"] = p["std"] * mult
    elif dist_name == "Uniform":
        mid = (p["low"] + p["high"]) / 2
        half = (p["high"] - p["low"]) / 2 * mult
        p["low"], p["high"] = mid - half, mid + half
    elif dist_name == "Lognormal":
        p["sigma"] = p["sigma"] * mult
    elif dist_name == "Gamma":
        p["scale"] = p["scale"] * mult
    # Poisson: variance is tied to the mean, dispersion is not independently adjustable.
    return p


def sensitivity_heatmap(
    price: float,
    salvage: float,
    stockout_penalty: float,
    dist_name: str,
    dist_params: dict,
    cost_values: np.ndarray,
    variability_mults: np.ndarray,
) -> np.ndarray:
    """Analytical Q* over a (cost, variability multiplier) grid. Rows=cost, cols=variability."""
    grid = np.empty((len(cost_values), len(variability_mults)))
    for i, cost in enumerate(cost_values):
        cu = price - cost + stockout_penalty
        co = cost - salvage
        ratio = cu / (cu + co)
        for j, mult in enumerate(variability_mults):
            scaled_params = scale_dispersion(dist_name, dist_params, mult)
            grid[i, j] = max(ppf(dist_name, ratio, scaled_params), 0.0)
    return grid
