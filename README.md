# Newsvendor Monte Carlo Lab

Project: *Use Monte Carlo simulation to develop a newsvendor problem application.*
ISYE 6644, Summer 2026 — Ahmadou Diallo (solo group).

## What's here

- `newsvendor.py` — the simulation engine: demand distributions, the vectorized
  profit function, the Monte Carlo grid search over order quantities, and the
  closed-form critical-ratio benchmark `F(Q*) = c_u / (c_u + c_o)`.
- `app.py` — the interactive Streamlit front end.
- `.streamlit/config.toml` — app theme.
- `requirements.txt` — Python dependencies.

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens in a browser at `http://localhost:8501`.

## What the app shows

1. **KPIs** — critical ratio, analytical Q*, simulated Q*, the gap between them,
   and expected profit at the optimum.
2. **Profit vs. order quantity** — expected profit across a grid of candidate Q,
   with a 95% confidence band and both optima marked.
3. **Profit distribution** — histogram of simulated profit outcomes at a
   user-chosen Q, with mean and a 5th–95th percentile band.
4. **Sensitivity analysis** — a heatmap of the analytical optimal Q as unit cost
   and demand variability change.
5. **Methodology & validation** — the model equations, a simulated-vs-analytical
   comparison table, and a Monte Carlo convergence plot.

All inputs (price, cost, salvage, optional stockout penalty, demand distribution
and its parameters, replication count, grid resolution, random seed) are
adjustable from the sidebar and drive every chart on the page.
