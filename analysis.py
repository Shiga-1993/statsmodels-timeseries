#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prototype time-series analysis for monthly stock data with N225 exogenous regressor.

Outputs
-------
- High-resolution figures for Prophet and SARIMAX fits
- CSV exports for prepared data, forecasts, coefficients, and CV metrics
- Markdown report summarizing key findings
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf


def set_plot_style(fontsize: int = 18) -> None:
    """Global matplotlib styling aimed at publication-quality figures."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": fontsize,
            "axes.labelsize": fontsize,
            "axes.titlesize": fontsize,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "lines.linewidth": 2.5,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "figure.autolayout": True,
        }
    )


def load_stock_series(csv_path: Path, company: str, use_log_return: bool) -> pd.DataFrame:
    """Return MS-frequency dataframe with target series for the selected company."""
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Company"] == company].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Log_Return"] = pd.to_numeric(df["Log_Return"], errors="coerce")
    df = df.sort_values("Date")

    if use_log_return:
        df["y"] = df["Log_Return"]
    else:
        df["y"] = df["Close"]

    df = df.set_index("Date")[["y", "Close"]].dropna(subset=["y"])
    df = df.asfreq("MS")
    return df


def load_n225_series(n225_path: Path) -> pd.Series:
    df = pd.read_csv(n225_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")
    series = pd.to_numeric(df["Log_Return"], errors="coerce")
    series = series.asfreq("MS").dropna()
    return series.rename("n225_ret")


def prepare_dataset(stock_csv: Path, n225_csv: Path, company: str, use_log_return: bool) -> pd.DataFrame:
    stock = load_stock_series(stock_csv, company, use_log_return)
    n225 = load_n225_series(n225_csv)
    df = stock.join(n225, how="inner").dropna()
    return df


def run_prophet(df: pd.DataFrame, outdir: Path, horizon: int = 6) -> Tuple[Dict[str, float], pd.DataFrame]:
    set_plot_style()
    df_prophet = (
        df.reset_index()
        .rename(columns={"Date": "ds", "y": "y", "n225_ret": "n225_ret"})
        [["ds", "y", "n225_ret"]]
    )

    model = Prophet(weekly_seasonality=False, daily_seasonality=False, yearly_seasonality=True)
    model.add_regressor("n225_ret")
    model.fit(df_prophet)

    in_sample = model.predict(df_prophet[["ds", "n225_ret"]])
    y_true = df_prophet["y"].to_numpy()
    y_hat = in_sample["yhat"].to_numpy()
    rmse = float(np.sqrt(mean_squared_error(y_true, y_hat)))
    mae = float(mean_absolute_error(y_true, y_hat))

    fig1 = model.plot(in_sample)
    fig1.savefig(outdir / "prophet_fit.png")
    plt.close(fig1)

    fig2 = model.plot_components(in_sample)
    fig2.savefig(outdir / "prophet_components.png")
    plt.close(fig2)

    coeffs = regressor_coefficients(model)
    coeffs.to_csv(outdir / "prophet_coefficients.csv", index=False)

    in_sample_out = pd.DataFrame(
        {
            "ds": df_prophet["ds"],
            "y": df_prophet["y"],
            "n225_ret": df_prophet["n225_ret"],
            "yhat": in_sample["yhat"],
            "yhat_lower": in_sample["yhat_lower"],
            "yhat_upper": in_sample["yhat_upper"],
        }
    )
    in_sample_out.to_csv(outdir / "prophet_in_sample.csv", index=False)

    last_date = df_prophet["ds"].max()
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    avg_n225 = df_prophet["n225_ret"].tail(12).mean()
    future = pd.DataFrame(
        {
            "ds": future_dates,
            "n225_ret": np.repeat(avg_n225, len(future_dates)),
        }
    )
    future_fc = model.predict(future)
    future_fc_out = future_fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    future_fc_out.to_csv(outdir / "prophet_forecast.csv", index=False)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "n225_coef": float(coeffs.loc[coeffs["regressor"] == "n225_ret", "coef"].iloc[0]),
    }
    return metrics, future_fc_out


def ts_cross_val(y: pd.Series, X: pd.DataFrame, order: Tuple[int, int, int], n_splits: int) -> pd.DataFrame:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    y_np = y.values.astype(float)
    X_np = X.values.astype(float) if X is not None else None
    for fold, (train_idx, test_idx) in enumerate(tscv.split(y_np), 1):
        y_tr, y_te = y_np[train_idx], y_np[test_idx]
        X_tr = X_np[train_idx] if X_np is not None else None
        X_te = X_np[test_idx] if X_np is not None else None
        if len(y_tr) < sum(order) + 2:
            continue
        model = SARIMAX(
            y_tr,
            order=order,
            exog=X_tr,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        y_pred = res.forecast(steps=len(y_te), exog=X_te)
        rmse = float(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae = float(mean_absolute_error(y_te, y_pred))
        rows.append({"fold": fold, "rmse": rmse, "mae": mae, "aic": float(res.aic)})
    return pd.DataFrame(rows)


def run_sarimax(
    df: pd.DataFrame,
    outdir: Path,
    lags: Iterable[int] = (0, 1, 3, 6),
    order: Tuple[int, int, int] = (1, 0, 1),
    n_splits: int = 5,
) -> Dict[str, float]:
    set_plot_style()
    metrics_by_lag: Dict[int, pd.DataFrame] = {}
    for lag in lags:
        data = pd.concat([df["y"], df["n225_ret"].shift(lag)], axis=1).dropna()
        data.columns = ["y", "n225_lag"]
        if len(data) < 40:
            continue
        cv = ts_cross_val(data["y"], data[["n225_lag"]], order, n_splits)
        if not cv.empty:
            cv["lag"] = lag
            metrics_by_lag[lag] = cv

    if not metrics_by_lag:
        raise RuntimeError("SARIMAX: no valid CV splits—check data length.")

    cv_all = pd.concat(metrics_by_lag.values(), ignore_index=True)
    cv_all.to_csv(outdir / "sarimax_cv_metrics.csv", index=False)
    lag_best = min(metrics_by_lag, key=lambda L: metrics_by_lag[L]["rmse"].mean())

    final_df = pd.concat([df["y"], df["n225_ret"].shift(lag_best)], axis=1).dropna()
    final_df.columns = ["y", "n225_lag"]
    y_final = final_df["y"]
    X_final = final_df[["n225_lag"]]

    model = SARIMAX(
        y_final,
        order=order,
        exog=X_final,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    pred = res.get_prediction()
    pred_df = pd.DataFrame({"y_true": y_final, "y_fitted": pred.predicted_mean}, index=y_final.index)
    pred_df.to_csv(outdir / "sarimax_in_sample.csv")

    rmse = float(np.sqrt(mean_squared_error(y_final, pred.predicted_mean)))
    mae = float(mean_absolute_error(y_final, pred.predicted_mean))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_final.index, y_final, label="Actual")
    ax.plot(pred_df.index, pred_df["y_fitted"], label="Fitted", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Log Return")
    ax.legend()
    fig.savefig(outdir / "sarimax_fit.png")
    plt.close(fig)

    residuals = res.resid
    acf_vals = acf(residuals, nlags=24, fft=True, missing="drop")
    fig_acf, ax_acf = plt.subplots(figsize=(10, 4))
    ax_acf.stem(range(len(acf_vals)), acf_vals)
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Residual ACF")
    fig_acf.savefig(outdir / "sarimax_residual_acf.png")
    plt.close(fig_acf)

    (outdir / "sarimax_summary.txt").write_text(res.summary().as_text())

    lag_plot = cv_all.groupby("lag")["rmse"].mean()
    fig_lag, ax_lag = plt.subplots(figsize=(8, 4))
    ax_lag.bar(lag_plot.index.astype(str), lag_plot.values)
    ax_lag.set_xlabel("Lag (months)")
    ax_lag.set_ylabel("CV RMSE")
    fig_lag.savefig(outdir / "sarimax_lag_rmse.png")
    plt.close(fig_lag)

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "best_lag": lag_best,
        "coef_n225": float(res.params.get("n225_lag", np.nan)),
    }
    return metrics


def build_report(
    outdir: Path,
    company: str,
    df: pd.DataFrame,
    prophet_metrics: Dict[str, float],
    sarimax_metrics: Dict[str, float],
    prophet_future: pd.DataFrame,
) -> None:
    report_path = outdir / "report.md"
    period_start = df.index.min().date()
    period_end = df.index.max().date()
    n_obs = len(df)
    mean_ret = df["y"].mean()
    std_ret = df["y"].std()

    future_head = prophet_future.iloc[0]
    future_tail = prophet_future.iloc[-1]

    lines = [
        f"# 時系列分析サマリー: {company}",
        "",
        "## データ整形",
        f"- 対象期間: {period_start} ～ {period_end}（{n_obs} ヶ月）",
        "- 目的変数: 月次ログリターン (株価)",
        "- 外生変数: 日経平均 (N225) 月次ログリターン",
        f"- ログリターン平均 {mean_ret:.4f}, 標準偏差 {std_ret:.4f}",
        "",
        "## Prophet",
        f"- RMSE: {prophet_metrics['rmse']:.4f}",
        f"- MAE: {prophet_metrics['mae']:.4f}",
        f"- N225 係数推定: {prophet_metrics['n225_coef']:.4f}",
        f"- 直近12ヶ月平均のN225リターンを入力した6ヶ月予測: 初月 {future_head['ds'].date()} の期待値 {future_head['yhat']:.4f}, "
        f"最終月 {future_tail['ds'].date()} の期待値 {future_tail['yhat']:.4f}",
        "- 図: `prophet_fit.png`, `prophet_components.png`",
        "",
        "## SARIMAX",
        f"- RMSE: {sarimax_metrics['rmse']:.4f}",
        f"- MAE: {sarimax_metrics['mae']:.4f}",
        f"- CVベストラグ（月）: {sarimax_metrics['best_lag']}",
        f"- N225 係数推定: {sarimax_metrics['coef_n225']:.4f}",
        "- 図: `sarimax_fit.png`, `sarimax_lag_rmse.png`, `sarimax_residual_acf.png`",
        "",
        "## 考察メモ",
        "1. Prophet と SARIMAX の双方で N225 ログリターンの係数がプラスである場合、N225 上昇が当該銘柄の翌月ログリターンを押し上げる傾向。",
        "2. 係数の大きさや統計的有意性の比較で、外部要因の説明力をレポート本文で補足してください。",
    ]
    report_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prophet & SARIMAX prototype for stock vs N225.")
    parser.add_argument("--stock-csv", type=Path, default=Path("DATA/stock_prices_monthly.csv"))
    parser.add_argument("--n225-csv", type=Path, default=Path("DATA/n225_monthly.csv"))
    parser.add_argument("--company", type=str, default="明豊エンタープライズ")
    parser.add_argument(
        "--level-target",
        dest="use_log_return",
        action="store_false",
        help="Use price level instead of log return as target.",
    )
    parser.set_defaults(use_log_return=True)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/prototype"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = prepare_dataset(args.stock_csv, args.n225_csv, args.company, args.use_log_return)
    df.to_csv(outdir / "prepared_dataset.csv")

    prophet_metrics, prophet_future = run_prophet(df, outdir)
    sarimax_metrics = run_sarimax(df, outdir)
    build_report(outdir, args.company, df, prophet_metrics, sarimax_metrics, prophet_future)

    summary = {
        "company": args.company,
        "prophet": prophet_metrics,
        "sarimax": sarimax_metrics,
        "outdir": str(outdir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
