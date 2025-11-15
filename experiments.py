#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hyper-parameter exploration across companies for Prophet & SARIMAX with N225 exogenous regressor.

Generates CSV logs with train/test metrics and coefficient diagnostics to help
identify robust parameter ranges and lag structures.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.utilities import regressor_coefficients
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis import prepare_dataset

# ----------------------------------------------------------------------
# Configurations
# ----------------------------------------------------------------------
DEFAULT_COMPANIES = [
    "明豊エンタープライズ",
    "日本エスコン",
    "フェイスネットワーク",
    "霞ヶ関キャピタル",
    "日神グループホールディングス",
]

# default grids (can be overridden via CLI)
DEFAULT_PROPHET_MODES = ["additive", "multiplicative"]
DEFAULT_PROPHET_CPS = [0.05, 0.2]
DEFAULT_PROPHET_SPS = [5.0, 15.0]
DEFAULT_SARIMAX_ORDERS = [(1, 0, 1), (1, 1, 1), (2, 0, 1)]
DEFAULT_SARIMAX_LAGS = [0, 1, 2, 3, 6]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def evaluate_prophet(
    df: pd.DataFrame,
    params: Dict[str, float],
    test_horizon: int,
    reg_lag: int,
    exog_col: str,
) -> Dict[str, float]:
    data = df.reset_index().rename(columns={"Date": "ds"})
    data["reg_input"] = data[exog_col].shift(reg_lag)
    data = data.dropna(subset=["reg_input"])

    if len(data) <= test_horizon + 24:
        raise ValueError("Not enough observations for Prophet split.")

    train = data.iloc[:-test_horizon]
    test = data.iloc[-test_horizon:]

    model = Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=params["seasonality_mode"],
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
    )
    model.add_regressor("reg_input")
    model.fit(train)

    train_pred = model.predict(train[["ds", "reg_input"]])
    test_pred = model.predict(test[["ds", "reg_input"]])
    coeffs = regressor_coefficients(model)
    coef = float(coeffs.loc[coeffs["regressor"] == "reg_input", "coef"].iloc[0])

    result = {
        "train_rmse": rmse(train["y"], train_pred["yhat"]),
        "train_mae": mae(train["y"], train_pred["yhat"]),
        "test_rmse": rmse(test["y"], test_pred["yhat"]),
        "test_mae": mae(test["y"], test_pred["yhat"]),
        "exog_coef": coef,
        "params": params,
        "reg_lag": reg_lag,
    }
    return result


def make_lagged(df: pd.DataFrame, lag: int, exog_col: str) -> pd.DataFrame:
    lagged = pd.concat([df["y"], df[exog_col].shift(lag)], axis=1).dropna()
    lagged.columns = ["y", "exog_lag"]
    return lagged


def evaluate_sarimax(
    df: pd.DataFrame,
    order: Tuple[int, int, int],
    lag: int,
    n_splits: int,
    seasonal_order: Tuple[int, int, int, int],
    exog_col: str,
) -> Dict[str, float]:
    data = make_lagged(df, lag, exog_col)
    if len(data) < max(60, (n_splits + 1) * 10):
        raise ValueError("Not enough observations for SARIMAX split.")

    splits = min(n_splits, max(2, len(data) // 30))
    tscv = TimeSeriesSplit(n_splits=splits)
    rows = []
    values = data["y"].values.astype(float)
    exog_values = data[["exog_lag"]].values.astype(float)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(values), 1):
        y_tr, y_te = values[train_idx], values[test_idx]
        x_tr, x_te = exog_values[train_idx], exog_values[test_idx]
        model = SARIMAX(
            y_tr,
            order=order,
            seasonal_order=seasonal_order,
            exog=x_tr,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        y_pred = res.forecast(steps=len(y_te), exog=x_te)
        rows.append(
            {
                "fold": fold,
                "rmse": rmse(y_te, y_pred),
                "mae": mae(y_te, y_pred),
                "aic": float(res.aic),
            }
        )

    cv_df = pd.DataFrame(rows)
    final_model = SARIMAX(
        data["y"],
        order=order,
        seasonal_order=seasonal_order,
        exog=data[["exog_lag"]],
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False)
    fitted = final_model.fittedvalues

    result = {
        "cv_rmse": cv_df["rmse"].mean(),
        "cv_mae": cv_df["mae"].mean(),
        "cv_aic": cv_df["aic"].mean(),
        "insample_rmse": rmse(data["y"], fitted),
        "insample_mae": mae(data["y"], fitted),
        "exog_coef": float(final_model.params.get("exog_lag", np.nan)),
        "exog_pvalue": float(final_model.pvalues.get("exog_lag", np.nan)),
        "lag": lag,
        "order": order,
        "seasonal_order": seasonal_order,
        "splits": splits,
        "converged": bool(final_model.mle_retvals.get("converged", True)),
    }
    return result


def describe_params(params: Dict[str, float]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


# ----------------------------------------------------------------------
# Main routine
# ----------------------------------------------------------------------
def parse_order(text: str) -> Tuple[int, int, int]:
    parts = [int(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Order must have three comma-separated integers, e.g., 1,0,1")
    return tuple(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter sweeps for Prophet & SARIMAX.")
    parser.add_argument("--stock-csv", type=Path, default=Path("DATA/stock_prices_monthly.csv"))
    parser.add_argument("--exog-csv", type=Path, default=Path("DATA/n225_monthly.csv"))
    parser.add_argument("--n225-csv", dest="exog_csv", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--exog-date-col", type=str, default="Date")
    parser.add_argument("--exog-value-col", type=str, default="Close_N225")
    parser.add_argument("--exog-use-level", action="store_true", help="Use raw values instead of log returns for exogenous series.")
    parser.add_argument("--exog-name", type=str, default="n225_ret")
    parser.add_argument("--companies", nargs="+", default=DEFAULT_COMPANIES)
    parser.add_argument("--min-obs", type=int, default=150, help="Minimum observations required per company.")
    parser.add_argument("--prophet-test-h", type=int, default=24, help="Holdout horizon for Prophet evaluation.")
    parser.add_argument("--prophet-modes", nargs="+", default=DEFAULT_PROPHET_MODES)
    parser.add_argument("--prophet-cps", nargs="+", type=float, default=DEFAULT_PROPHET_CPS)
    parser.add_argument("--prophet-sps", nargs="+", type=float, default=DEFAULT_PROPHET_SPS)
    parser.add_argument("--prophet-reg-lag", type=int, default=0, help="Lag (months) applied to N225 regressor for Prophet.")
    parser.add_argument("--sarimax-splits", type=int, default=5)
    parser.add_argument("--sarimax-orders", nargs="+", type=parse_order, default=DEFAULT_SARIMAX_ORDERS)
    parser.add_argument("--sarimax-seasonal", nargs="+", default=["0,0,0,0"], help="Seasonal order tuples P,D,Q,m (comma separated).")
    parser.add_argument("--sarimax-lags", nargs="+", type=int, default=DEFAULT_SARIMAX_LAGS)
    parser.add_argument("--level-target", action="store_true", help="Use price level as target instead of log return.")
    parser.add_argument("--standardize", action="store_true", help="Z-score y and N225 (shorthand for enabling both flags below).")
    parser.add_argument("--standardize-target", action="store_true", help="Z-score y only.")
    parser.add_argument("--standardize-n225", action="store_true", help="Z-score N225 series only.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    prophet_rows: List[Dict[str, object]] = []
    sarimax_rows: List[Dict[str, object]] = []
    logs: List[Dict[str, str]] = []

    prophet_grid = [
        {
            "seasonality_mode": mode,
            "changepoint_prior_scale": cps,
            "seasonality_prior_scale": sps,
        }
        for mode in args.prophet_modes
        for cps in args.prophet_cps
        for sps in args.prophet_sps
    ]

    seasonal_orders = [tuple(int(x) for x in val.split(",")) for val in args.sarimax_seasonal]
    exog_col = args.exog_name

    for company in args.companies:
        try:
            df = prepare_dataset(
                args.stock_csv,
                args.exog_csv,
                company,
                use_log_return=not args.level_target,
                exog_date_col=args.exog_date_col,
                exog_value_col=args.exog_value_col,
                exog_log_return=not args.exog_use_level,
                exog_alias=exog_col,
            )
            std_target = args.standardize or args.standardize_target
            std_exog = args.standardize or args.standardize_n225
            if std_target or std_exog:
                df = df.copy()
                if std_target:
                    mean = df["y"].mean()
                    std = df["y"].std()
                    if std == 0 or np.isnan(std):
                        raise ValueError("Target std=0; cannot standardize.")
                    df["y"] = (df["y"] - mean) / std
                if std_exog:
                    mean = df[exog_col].mean()
                    std = df[exog_col].std()
                    if std == 0 or np.isnan(std):
                        raise ValueError("Exogenous series std=0; cannot standardize.")
                    df[exog_col] = (df[exog_col] - mean) / std
        except Exception as exc:
            logs.append({"company": company, "stage": "load", "error": str(exc)})
            continue

        if len(df) < args.min_obs:
            logs.append(
                {
                    "company": company,
                    "stage": "skip",
                    "error": f"observations {len(df)} < min {args.min_obs}",
                }
            )
            continue

        # Prophet sweeps
        for params in prophet_grid:
            try:
                metrics = evaluate_prophet(
                    df,
                    params,
                    test_horizon=args.prophet_test_h,
                    reg_lag=args.prophet_reg_lag,
                    exog_col=exog_col,
                )
                row = {
                    "company": company,
                    "seasonality_mode": params["seasonality_mode"],
                    "changepoint_prior_scale": params["changepoint_prior_scale"],
                    "seasonality_prior_scale": params["seasonality_prior_scale"],
                    "reg_lag": metrics.pop("reg_lag"),
                    **metrics,
                }
                prophet_rows.append(row)
            except Exception as exc:
                logs.append(
                    {
                        "company": company,
                        "stage": "prophet",
                        "params": describe_params(params),
                        "error": str(exc),
                    }
                )

        # SARIMAX sweeps
        for order, lag, seas in itertools.product(args.sarimax_orders, args.sarimax_lags, seasonal_orders):
            try:
                metrics = evaluate_sarimax(
                    df,
                    order,
                    lag,
                    n_splits=args.sarimax_splits,
                    seasonal_order=seas,
                    exog_col=exog_col,
                )
                row = {
                    "company": company,
                    "order": order,
                    "lag": lag,
                    "seasonal_order": seas,
                    **metrics,
                }
                sarimax_rows.append(row)
            except Exception as exc:
                logs.append(
                    {
                        "company": company,
                        "stage": "sarimax",
                        "params": f"order={order}, lag={lag}",
                        "error": str(exc),
                    }
                )

    if prophet_rows:
        pd.DataFrame(prophet_rows).to_csv(outdir / "prophet_experiments.csv", index=False)
    if sarimax_rows:
        pd.DataFrame(sarimax_rows).to_csv(outdir / "sarimax_experiments.csv", index=False)
    if logs:
        pd.DataFrame(logs).to_csv(outdir / "experiment_logs.csv", index=False)

    summary = {
        "prophet_trials": len(prophet_rows),
        "sarimax_trials": len(sarimax_rows),
        "companies": args.companies,
        "outdir": str(outdir),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
