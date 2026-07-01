# TODO — Improve Sales & Demand Forecasting Project

## Step 1 — Notebook outputs
- [ ] Update the forecasting notebook(s) to perform walk-forward validation (rolling-origin).
- [ ] Train the final/best model using the chosen criterion.
- [ ] Export `forecast.json` (historical + next-horizon forecast + intervals if added) and `metrics.json`.
- [ ] Save model artifact(s) (e.g., joblib) for reproducibility.


## Step 2 — Dashboard integration
- [x] Refactor `index.html` to fetch/load `forecast.json` and `metrics.json`.

- [ ] Replace hardcoded arrays (hS, fLR, fD, etc.) with data from JSON.
- [ ] Make KPIs and narrative sections compute peak/low and totals from loaded forecast.


## Step 3 — Documentation
- [ ] Update `README.md` with exact run steps:
  - how to generate JSON artifacts
  - how to preview the dashboard locally

## Step 4 — Verification
- [ ] Run the notebook/script and ensure JSON files are produced.
- [ ] Open dashboard and confirm charts/KPIs render correctly.

