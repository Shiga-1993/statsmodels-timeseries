#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate consolidated visualizations for analysis.py outputs.

Usage:
    python visualize_results.py --analysis-dir outputs/prototype_default
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_csv(path: Path, **kwargs) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, **kwargs)


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def visualize(analysis_dir: Path, exog_col: str, rolling_window: int, out_file: Path, show: bool) -> None:
    analysis_dir = analysis_dir.resolve()
    prepared = _load_csv(analysis_dir / "prepared_dataset.csv")
    if prepared is None:
        raise FileNotFoundError(f"prepared_dataset.csv not found in {analysis_dir}")
    date_col = prepared.columns[0]
    prepared[date_col] = pd.to_datetime(prepared[date_col])
    prepared = prepared.rename(columns={date_col: "Date"})

    prophet_insample = _load_csv(analysis_dir / "prophet_in_sample.csv")
    sarimax_insample = _load_csv(analysis_dir / "sarimax_in_sample.csv", parse_dates=[0])
    if sarimax_insample is not None:
        date_col = sarimax_insample.columns[0]
        sarimax_insample = sarimax_insample.rename(columns={date_col: "Date"})
    prophet_forecast = _load_csv(analysis_dir / "prophet_forecast.csv")
    sarimax_cv = _load_csv(analysis_dir / "sarimax_cv_metrics.csv")

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    ax = axes.ravel()

    # Panel 1: Prophet fit + forecast
    ax0 = ax[0]
    ax0.plot(prepared["Date"], prepared["y"], label="Actual", color="black", linewidth=1.2)
    prophet_rmse = prophet_mae = np.nan
    if prophet_insample is not None:
        prophet_insample["ds"] = pd.to_datetime(prophet_insample["ds"])
        ax0.plot(prophet_insample["ds"], prophet_insample["yhat"], label="Prophet (in-sample)", color="#1f77b4")
        prophet_rmse = _rmse(prophet_insample["y"].values, prophet_insample["yhat"].values)
        prophet_mae = _mae(prophet_insample["y"].values, prophet_insample["yhat"].values)
    if prophet_forecast is not None:
        prophet_forecast["ds"] = pd.to_datetime(prophet_forecast["ds"])
        ax0.plot(prophet_forecast["ds"], prophet_forecast["yhat"], "--", label="Prophet forecast", color="#1f77b4")
        ax0.fill_between(
            prophet_forecast["ds"],
            prophet_forecast["yhat_lower"],
            prophet_forecast["yhat_upper"],
            color="#1f77b4",
            alpha=0.15,
        )
    ax0.set_title("Prophet: Actual vs Prediction")
    ax0.set_xlabel("Date")
    ax0.set_ylabel("Log return")
    ax0.legend()

    # Panel 2: SARIMAX fit
    ax1 = ax[1]
    ax1.plot(prepared["Date"], prepared["y"], label="Actual", color="black", linewidth=1.2)
    sarimax_rmse = sarimax_mae = np.nan
    if sarimax_insample is not None:
        sarimax_insample["Date"] = pd.to_datetime(sarimax_insample["Date"])
        sarimax_rmse = _rmse(sarimax_insample["y_true"].values, sarimax_insample["y_fitted"].values)
        sarimax_mae = _mae(sarimax_insample["y_true"].values, sarimax_insample["y_fitted"].values)
        ax1.plot(sarimax_insample["Date"], sarimax_insample["y_fitted"], label="SARIMAX (in-sample)", color="#ff7f0e")
    ax1.set_title("SARIMAX: Actual vs Prediction")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Log return")
    ax1.legend()

    # Panel 3: Residual distribution
    ax2 = ax[2]
    if prophet_insample is not None:
        prophet_resid = prophet_insample["y"] - prophet_insample["yhat"]
        ax2.hist(prophet_resid, bins=30, alpha=0.6, label="Prophet", color="#1f77b4")
    if sarimax_insample is not None:
        sarimax_resid = sarimax_insample["y_true"] - sarimax_insample["y_fitted"]
        ax2.hist(sarimax_resid, bins=30, alpha=0.6, label="SARIMAX", color="#ff7f0e")
    ax2.set_title("Residual Distribution")
    ax2.set_xlabel("Residual")
    ax2.legend()

    # Panel 4: Rolling correlation
    ax3 = ax[3]
    if exog_col in prepared.columns:
        rolling_corr = prepared["y"].rolling(window=rolling_window).corr(prepared[exog_col])
        ax3.plot(prepared["Date"], rolling_corr, color="#2ca02c")
        ax3.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax3.set_title(f"Rolling correlation (window={rolling_window})")
        ax3.set_ylabel("Correlation")
        ax3.set_xlabel("Date")
    else:
        ax3.text(0.5, 0.5, f"{exog_col} not found", ha="center", va="center", transform=ax3.transAxes)
        ax3.axis("off")

    # Panel 5: SARIMAX lag RMSE
    ax4 = ax[4]
    if sarimax_cv is not None and "lag" in sarimax_cv.columns:
        lag_scores = sarimax_cv.groupby("lag")["rmse"].mean()
        ax4.bar(lag_scores.index.astype(str), lag_scores.values, color="#ff7f0e")
        ax4.set_title("CV RMSE by lag")
        ax4.set_xlabel("Lag")
        ax4.set_ylabel("RMSE")
    else:
        ax4.text(0.5, 0.5, "sarimax_cv_metrics.csv not found", ha="center", va="center", transform=ax4.transAxes)
        ax4.axis("off")

    # Panel 6: Summary text
    ax5 = ax[5]
    ax5.axis("off")
    summary_lines = [
        "Summary",
        "",
        f"Prophet RMSE: {prophet_rmse:.4f}" if not np.isnan(prophet_rmse) else "Prophet RMSE: n/a",
        f"Prophet MAE : {prophet_mae:.4f}" if not np.isnan(prophet_mae) else "Prophet MAE : n/a",
        "",
        f"SARIMAX RMSE: {sarimax_rmse:.4f}" if not np.isnan(sarimax_rmse) else "SARIMAX RMSE: n/a",
        f"SARIMAX MAE : {sarimax_mae:.4f}" if not np.isnan(sarimax_mae) else "SARIMAX MAE : n/a",
        "",
        f"Records: {len(prepared)} rows",
    ]
    ax5.text(0.05, 0.95, "\n".join(summary_lines), va="top", ha="left", fontsize=11)

    fig.suptitle(f"Analysis dashboard: {analysis_dir.name}", fontsize=16)
    fig.tight_layout(rect=[0, 0.01, 1, 0.97])
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize analysis outputs.")
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--exog-col", type=str, default="n225_ret")
    parser.add_argument("--rolling-window", type=int, default=24)
    parser.add_argument("--out-file", type=Path, default=None, help="Output image path (default: analysis_dir/dashboard.png)")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    out_file = args.out_file or analysis_dir / "summary_dashboard.png"
    visualize(analysis_dir, args.exog_col, args.rolling_window, out_file, args.show)
    print(f"Dashboard saved to {out_file}")


if __name__ == "__main__":
    main()
