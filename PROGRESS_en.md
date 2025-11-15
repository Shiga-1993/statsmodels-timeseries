# StatsModels Investigation Progress (English)

## Overview
- Target: Monthly stock log returns (`DATA/stock_prices_monthly.csv`) with Nikkei 225 (`DATA/n225_monthly.csv`) log returns as the exogenous regressor; exploring Prophet and SARIMAX prototypes.
- `analysis.py`: Single-company Prophet + SARIMAX workflow that produces charts, CSVs, and Markdown reports.
- `experiments.py`: Batch evaluation tool across multiple tickers/hyperparameters. Controls Prophet regressor lags, SARIMAX lags, level/log target, etc. via CLI flags.

## Completed Experiments (log-return target)
1. **Prototype (`analysis.py`)**
   - Ticker: Meikyo Enterprise.
   - Result: Prophet and SARIMAX both achieve RMSE ≈ 0.178; Nikkei coefficient positive and significant on SARIMAX. Outputs stored in `outputs/prototype/`.
2. **Multi-company sweeps (`experiments.py`)**
   - Tickers: 13 names with sufficient monthly history starting in the early 2000s.
   - Grid: Prophet seasonality_mode {additive, multiplicative}, changepoint_prior_scale {0.05, 0.2}, seasonality_prior_scale {5,15}; SARIMAX orders {(1,0,1),(1,1,1),(2,0,1)}, lags {0,1,2,3,6}.
   - Output: `outputs/experiments_all_loglag0/`.
3. **Lag stress test**
   - Prophet: Add regressor lag = 1.
   - SARIMAX: Expand lag grid to {0,1,2,3,6,9,12}.
   - Output: `outputs/experiments_all_reglag1/`.
   - Aggregated summary: `outputs/experiments_all_summary.md`.

## Key Observations
- **Prophet**
  - 9 of 13 tickers prefer multiplicative seasonality, meaning the bulk of variation can be captured via seasonality/trend tuning. Swapping regressor lags (0–2) yields negligible RMSE differences (≤0.01); N225 helps only marginally, so seasonality adjustments take priority.
  - Lagging N225 by one month improves RMSE for only 3/13 tickers and often causes unstable coefficients. At present the N225 regressor has limited explanatory power in Prophet compared with seasonal effects.
  - Next steps: Standardize returns, try level targets, or add alternative covariates; emphasize scaling/spec revisions.
- **SARIMAX**
  - 10 of 13 tickers select lag = 1 (N225 log return) with significant coefficients (p < 0.05), indicating the Nikkei exerts a sizable effect on next-month returns.
  - Lags 6 or 12 only help a handful of names and coefficients remain small. SARIMAX should default to lag 1 with minor adjustments as needed.
  - ConvergenceWarnings still appear; track `mle_retvals['converged']` and BIC to filter marginal fits.

## Handoff Guide
1. **Environment**
   - Required packages: `prophet`, `statsmodels`, `yfinance`, `scikit-learn`.
   - `DATA/n225_monthly.csv` already formatted for `analysis.py` / `experiments.py`.
2. **Scripts**
   - Single-company analysis: `python analysis.py --company "Ticker" [--level-target] [--outdir outputs/...].`
   - Batch experiments: `python experiments.py --companies ... --prophet-reg-lag ... --sarimax-lags "0 1 2 3 6 9 12" --outdir outputs/...`.
   - Include low-data tickers by lowering `--min-obs`.
3. **Reference outputs**
   - `outputs/experiments_all_summary.md` captures best settings and commentary.
   - Each `outputs/experiments*/` directory includes raw CSVs, `summary.json`, and optional `experiment_logs.csv`.

### Prototype runs (single-company roundup)
- `outputs/prototype`: Original baseline with Nikkei log returns; Prophet/SARIMAX RMSE ≈ 0.178, SARIMAX CV best lag = 1 (β ≈ 1.34).
- `outputs/prototype_default`: Same ticker rerun with current `analysis.py` defaults. Prophet RMSE slightly higher (~0.186) after newer scaling; SARIMAX still RMSE ≈ 0.178 and lag=1, proving consistency.
- `outputs/prototype_default_explicit`: Uses `--exog-use-level` to feed raw Nikkei levels. RMSEs unchanged; model effectively differences internally, so log returns remain preferable for interpretability.
- `outputs/prototype_macro`: Replaces exogenous driver with `ALT_DATA/macro_index.csv`. RMSE degrades slightly; macro composite does not provide the same explanatory power as Nikkei. Conclusion: keep Nikkei log return with lag 1 as the default.

## TODO Ideas
- Re-evaluate Prophet with standardized inputs / extra covariates (`--level-target` trials).
- Extend SARIMAX to include seasonal terms `(P,D,Q,m=12)` and automate residual diagnostics (Ljung–Box, ARCH).
- Add confidence intervals / convergence flags to `experiments.py` outputs.
- Use `analysis.py` to generate plots/reports whenever deeper visualization is required.

## Standardization tests
- `--standardize`: Z-score y and N225 simultaneously. RMSE worsened across all 13 names.
- `--standardize-n225`: Standardizing only N225 makes coefficients interpretable with minimal impact on RMSE.
- Level target + standardization: Performed poorly (RMSE worse by orders of magnitude). Stick with log returns.
- Seasonal SARIMAX (`--sarimax-seasonal`): No improvement; failed to converge on many tickers.
- Prophet lag sweeps (`outputs/experiments_stdN_reglag{0..6}/`, `outputs/prophet_lag_summary.*`): Best lags scatter but improvements ≤0.01 RMSE; lag 0–2 is usually sufficient.
- SARIMAX diagnostics (`sarimax_diagnostics.py`, `outputs/diagnostics_stdN/`): Residual tests (Ljung-Box, ARCH) still flag autocorrelation/heteroskedasticity; consider richer AR/MA/GARCH structures.
