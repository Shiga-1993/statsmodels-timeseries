#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monthly stock vs. development-plan SARIMAX with lag sweep & TS-CV
- Inputs:
    * Stock: CSV (date, adj_close) or yfinance download
    * Dev plan: CSV (date, project_count, floor_area, ...)
- Outputs:
    * PNG: time series overlay, lag-RMSE curve, residual ACF
    * CSV: cross-val metrics by lag, final-insample/out-of-sample prediction
Notes:
    - Linux GUI-less safe (no plt.show())
    - Large fonts, no figure titles by default
"""

import os
import warnings
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, grangercausalitytests
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------------
# Config defaults
# ----------------------------------
DEFAULT_LAGS = [0, 3, 6, 9, 12]   # months
DEFAULT_ORDER = (1, 1, 1)         # ARIMA order
DEFAULT_SORDER = (0, 0, 0, 0)     # seasonal (P,D,Q,m)
FORECAST_H = 6                    # months ahead for final forecast

# ----------------------------------
# Utils
# ----------------------------------
def set_matplotlib_rc(fontsize=16):
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": fontsize,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "figure.autolayout": True,
    })

def monthify(df, date_col="date", how="last"):
    """
    Convert to month-start index with pandas resample('MS').
    how: 'last' (last valid obs), 'mean', 'sum', etc.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # Use 'MS' (Month Start) to avoid end-of-month pitfalls.  :contentReference[oaicite:2]{index=2}
    if how == "last":
        out = df.resample("MS").last()
    elif how == "mean":
        out = df.resample("MS").mean()
    elif how == "sum":
        out = df.resample("MS").sum()
    else:
        raise ValueError("Unsupported resample method")
    return out

def make_lagged_exog(exog_df, lag):
    """Shift exogenous features by 'lag' months (positive=use past info)."""
    if lag == 0:
        return exog_df.copy()
    return exog_df.shift(lag)

def ts_cross_val(endog, exog, order, sorder, n_splits=5):
    """
    Expanding-window CV using sklearn TimeSeriesSplit.  :contentReference[oaicite:3]{index=3}
    Returns list of dicts with fold metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    records = []
    y = endog.values.astype(float)
    X = exog.values.astype(float) if exog is not None else None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(y), 1):
        y_tr, y_te = y[train_idx], y[test_idx]
        X_tr = X[train_idx] if X is not None else None
        X_te = X[test_idx] if X is not None else None

        # Guard: SARIMAX needs > parameters length
        if len(y_tr) < (order[0]+order[2]+order[1]+2):
            continue

        model = SARIMAX(
            y_tr, order=order,
            seasonal_order=sorder,
            exog=X_tr,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit(disp=False)

        yhat = res.forecast(steps=len(y_te), exog=X_te)
        rmse = np.sqrt(mean_squared_error(y_te, yhat))
        mae  = mean_absolute_error(y_te, yhat)
        rec = {"fold": fold, "rmse": rmse, "mae": mae, "aic": res.aic}
        records.append(rec)

    return pd.DataFrame.from_records(records)

def plot_overlay(df, stock_col, exog_cols, outpath):
    set_matplotlib_rc(16)
    fig, ax1 = plt.subplots(figsize=(10,5))
    ax1.plot(df.index, df[stock_col], label=stock_col)
    ax1.set_xlabel("Date")
    ax1.set_ylabel(stock_col)

    # Normalize exog for visualization scale
    for c in exog_cols:
        series = (df[c] - df[c].mean()) / (df[c].std() + 1e-9)
        ax1.plot(df.index, series, label=f"{c} (z-score)")
    ax1.legend(loc="best")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def plot_lag_rmse(cv_metrics_by_lag, outpath):
    set_matplotlib_rc(16)
    fig, ax = plt.subplots(figsize=(8,4))
    xs = []
    ys = []
    for lag, dfm in cv_metrics_by_lag.items():
        xs.append(lag)
        ys.append(dfm["rmse"].mean())
    ax.bar(xs, ys)
    ax.set_xlabel("Lag Δ (months)")
    ax.set_ylabel("CV RMSE")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

def plot_resid_acf(residuals, outpath, nlags=24):
    set_matplotlib_rc(16)
    acfs = acf(residuals, nlags=nlags, fft=True, missing="drop")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.stem(range(len(acfs)), acfs, use_line_collection=True)
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF of residuals")
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------
# Main pipeline
# ----------------------------------
def main(args):
    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load stock
    if args.stock_csv:
        stock = pd.read_csv(args.stock_csv)  # columns: date, adj_close
    else:
        # Optional: yfinance fetch (research/education OSS).  :contentReference[oaicite:4]{index=4}
        import yfinance as yf
        data = yf.download(args.ticker, period=args.period, interval="1d", auto_adjust=True)  # :contentReference[oaicite:5]{index=5}
        data = data.reset_index()[["Date", "Close"]].rename(columns={"Date":"date","Close":"adj_close"})
        stock = data

    stock_m = monthify(stock, date_col="date", how="last")  # last daily→monthly  :contentReference[oaicite:6]{index=6}
    stock_m = stock_m.rename(columns={"adj_close": "adj_close"})

    # 2) Load dev-plan
    dev = pd.read_csv(args.dev_csv)  # columns: date, project_count, floor_area, ...
    dev_m = monthify(dev, date_col="date", how=args.dev_agg)  # 'sum' など業務定義に合わせる

    # 3) Join and make target (optionally log-return)
    df = stock_m.join(dev_m, how="inner").dropna(how="all")
    if args.use_log_return:
        df["y"] = np.log(df["adj_close"]).diff()
    else:
        df["y"] = df["adj_close"]  # レベルを直接予測
    df = df.dropna()

    # Feature columns (all dev-plan columns)
    exog_cols = [c for c in df.columns if c not in ["adj_close", "y"]]

    # 4) Lag sweep with TS-CV
    cv_metrics_by_lag = {}
    for lag in args.lags:
        X_lag = make_lagged_exog(df[exog_cols], lag)
        tmp = pd.concat([df["y"], X_lag], axis=1).dropna()
        y_endog = tmp["y"]
        X_exog = tmp.drop(columns=["y"])
        metrics = ts_cross_val(y_endog, X_exog, order=args.order, sorder=args.sorder, n_splits=args.n_splits)
        if len(metrics) == 0:
            continue
        metrics["lag"] = lag
        cv_metrics_by_lag[lag] = metrics

    # Save CV metrics
    if cv_metrics_by_lag:
        cv_all = pd.concat(cv_metrics_by_lag.values(), ignore_index=True)
        cv_all.to_csv(os.path.join(args.outdir, "cv_metrics_by_lag.csv"), index=False)

    # 5) Choose best lag by mean RMSE
    if not cv_metrics_by_lag:
        raise RuntimeError("No CV results. Check data coverage/length.")
    lag_best = min(cv_metrics_by_lag, key=lambda L: cv_metrics_by_lag[L]["rmse"].mean())

    # 6) Fit final model on full data with best lag, then forecast H months
    X_best = make_lagged_exog(df[exog_cols], lag_best)
    final = pd.concat([df["y"], X_best], axis=1).dropna()
    y_final = final["y"]
    X_final = final.drop(columns=["y"])

    model = SARIMAX(
        y_final,
        order=args.order,
        seasonal_order=args.sorder,
        exog=X_final,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = model.fit(disp=False)

    # Forecast requires future exog (naなら直近値ホールド等:業務設計で変更)
    last_idx = final.index[-1]
    future_idx = pd.date_range(last_idx + pd.offsets.MonthBegin(), periods=args.forecast_h, freq="MS")
    # naive hold-last for exog; 実務では将来想定を入力
    X_future = pd.DataFrame(
        np.tile(X_final.iloc[-1:].values, (args.forecast_h, 1)),
        index=future_idx, columns=X_final.columns
    )
    y_fcst = res.forecast(steps=args.forecast_h, exog=X_future)

    # 7) Save outputs
    # Time series overlay
    vis_df = df.copy()
    vis_df = vis_df.join(X_best, how="left", rsuffix="_lagged")
    plot_overlay(vis_df, stock_col=("y" if not args.use_log_return else "y"), 
                 exog_cols=[c for c in X_best.columns], 
                 outpath=os.path.join(args.outdir, "overlay_timeseries.png"))

    # Lag vs RMSE
    plot_lag_rmse(cv_metrics_by_lag, os.path.join(args.outdir, "lag_rmse.png"))

    # Residual ACF
    plot_resid_acf(residuals=res.resid, outpath=os.path.join(args.outdir, "resid_acf.png"), nlags=24)

    # Save predictions
    pred_insample = res.get_prediction()
    pred_df = pd.DataFrame({
        "y_true": y_final,
        "y_fitted": pred_insample.predicted_mean
    }, index=y_final.index)
    pred_df.to_csv(os.path.join(args.outdir, "insample_fit.csv"))

    fcst_df = pd.DataFrame({"y_forecast": y_fcst}, index=future_idx)
    fcst_df.to_csv(os.path.join(args.outdir, "forecast.csv"))

    # 8) Optional: Granger (diagnostic only, 注意:系列整形が必要)  :contentReference[oaicite:7]{index=7}
    if args.do_granger and len(exog_cols) >= 1:
        # Example with first exog column
        z = pd.concat([df["y"], df[exog_cols[0]]], axis=1).dropna()
        # statsmodels expects cols order: [y, x] for "x causes y"
        try:
            _ = grangercausalitytests(z[[ "y", exog_cols[0] ]], maxlag=max(DEFAULT_LAGS))
        except Exception:
            pass

    print(f"[DONE] Best lag Δ={lag_best} (months). Files saved under: {args.outdir}")

# ----------------------------------
# CLI
# ----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stock_csv", type=str, default=None,
                        help="CSV with columns: date, adj_close. If omitted, use yfinance.")
    parser.add_argument("--ticker", type=str, default="MSFT",
                        help="Ticker for yfinance when stock_csv is omitted.")
    parser.add_argument("--period", type=str, default="10y",
                        help="History period for yfinance (e.g., '10y', '5y').")
    parser.add_argument("--dev_csv", type=str, required=True,
                        help="CSV with columns: date, project_count, floor_area, ...")
    parser.add_argument("--dev_agg", type=str, default="sum", choices=["sum","mean","last"],
                        help="Monthly aggregation for dev metrics.")
    parser.add_argument("--use_log_return", action="store_true",
                        help="Use log return of adj_close as target y.")
    parser.add_argument("--lags", type=int, nargs="+", default=DEFAULT_LAGS)
    parser.add_argument("--order", type=int, nargs=3, default=DEFAULT_ORDER)
    parser.add_argument("--sorder", type=int, nargs=4, default=DEFAULT_SORDER)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--forecast_h", type=int, default=FORECAST_H)
    parser.add_argument("--do_granger", action="store_true")
    parser.add_argument("--outdir", type=str, default="out_sarimax")
    args = parser.parse_args()
    main(args)

