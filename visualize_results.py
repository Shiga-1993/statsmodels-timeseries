#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate consolidated visualizations for analysis.py outputs.

Usage:
    python visualize_results.py --analysis-dir outputs/prototype_default
"""

from __future__ import annotations

import argparse
import re
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


def _lag_correlations(y: pd.Series, exog: pd.Series, max_lag: int) -> pd.Series:
    corrs = []
    for lag in range(max_lag + 1):
        shifted = exog.shift(lag)
        combined = pd.concat([y, shifted], axis=1).dropna()
        if combined.empty:
            corrs.append(np.nan)
        else:
            corrs.append(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
    return pd.Series(corrs, index=range(max_lag + 1), name="corr")


def _load_prophet_coef(analysis_dir: Path, exog_col: str) -> Optional[dict]:
    coeff_path = analysis_dir / "prophet_coefficients.csv"
    if not coeff_path.exists():
        return None
    coeffs = pd.read_csv(coeff_path)
    if "regressor" not in coeffs.columns or "coef" not in coeffs.columns:
        return None
    row = coeffs.loc[coeffs["regressor"] == exog_col]
    if row.empty:
        return None
    entry = row.iloc[0]
    return {
        "coef": float(entry["coef"]),
        "lower": float(entry["coef_lower"]) if "coef_lower" in entry else np.nan,
        "upper": float(entry["coef_upper"]) if "coef_upper" in entry else np.nan,
    }


def _parse_sarimax_coef(summary_path: Path) -> Optional[dict]:
    if not summary_path.exists():
        return None
    pattern = re.compile(
        r"exog_lag\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)"
    )
    for line in summary_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            coef, std_err, z, pvalue, ci_low, ci_high = map(float, match.groups())
            return {
                "coef": coef,
                "std_err": std_err,
                "z": z,
                "pvalue": pvalue,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
    return None


def visualize(
    analysis_dir: Path,
    exog_col: str,
    rolling_window: int,
    max_lag: int,
    out_file: Path,
    show: bool,
) -> None:
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

    fig, axes = plt.subplots(4, 2, figsize=(15, 16))
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

    lag_scores = None
    best_lag = None
    if sarimax_cv is not None and "lag" in sarimax_cv.columns:
        lag_scores = sarimax_cv.groupby("lag")["rmse"].mean()
        if not lag_scores.empty:
            best_lag = int(lag_scores.idxmin())

    # Panel 5: Cross-correlation vs lag
    ax4 = ax[4]
    lag_corr = None
    if exog_col in prepared.columns:
        lag_corr = _lag_correlations(prepared["y"], prepared[exog_col], max_lag)
        ax4.bar(lag_corr.index, lag_corr.values, color="#9467bd")
        ax4.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        if best_lag is not None:
            ax4.axvline(best_lag, color="#ff7f0e", linestyle="--", linewidth=1.2, label="SARIMAX best lag")
            ax4.legend()
        ax4.set_title("Cross-correlation vs lag")
        ax4.set_xlabel("Lag (months)")
        ax4.set_ylabel("Correlation (y vs exog lag)")
    else:
        ax4.text(0.5, 0.5, f"{exog_col} not found", ha="center", va="center", transform=ax4.transAxes)
        ax4.axis("off")

    prophet_coef = _load_prophet_coef(analysis_dir, exog_col)
    sarimax_coef = _parse_sarimax_coef(analysis_dir / "sarimax_summary.txt")
    corr_at_best = None
    if lag_corr is not None and best_lag is not None and best_lag in lag_corr.index:
        corr_at_best = float(lag_corr.loc[best_lag])

    # Panel 6: Summary text (best-parameter metrics)
    ax5 = ax[5]
    ax5.axis("off")
    summary_lines = [
        "Summary (best configurations)",
        "",
        f"Prophet RMSE: {prophet_rmse:.4f}" if not np.isnan(prophet_rmse) else "Prophet RMSE: n/a",
        f"Prophet MAE : {prophet_mae:.4f}" if not np.isnan(prophet_mae) else "Prophet MAE : n/a",
        "",
        f"SARIMAX RMSE: {sarimax_rmse:.4f}" if not np.isnan(sarimax_rmse) else "SARIMAX RMSE: n/a",
        f"SARIMAX MAE : {sarimax_mae:.4f}" if not np.isnan(sarimax_mae) else "SARIMAX MAE : n/a",
        "",
        f"Prophet β ({exog_col}): {prophet_coef['coef']:.3f}" if prophet_coef else "Prophet β: n/a",
        (
            "SARIMAX β: "
            + f"{sarimax_coef['coef']:.3f} (p={sarimax_coef['pvalue']:.3g})"
            if sarimax_coef
            else "SARIMAX β: n/a"
        ),
        (
            f"Corr(y, {exog_col} lag {best_lag}): {corr_at_best:.3f}"
            if corr_at_best is not None and best_lag is not None
            else "Corr(y, exog lag): n/a"
        ),
        "",
        f"SARIMAX CV best lag: {best_lag}" if best_lag is not None else "SARIMAX CV best lag: n/a",
        f"Records: {len(prepared)} rows",
    ]
    ax5.text(0.05, 0.95, "\n".join(summary_lines), va="top", ha="left", fontsize=11)

    # Panel 7: Lag & coefficient insights (textual)
    ax6 = ax[6]
    ax6.axis("off")
    interpret_lines = [
        "Regressor impact summary",
        "",
    ]
    if best_lag is not None:
        interpret_lines.append(f"SARIMAX CV best lag: {best_lag}")
        if corr_at_best is not None:
            interpret_lines.append(f"Corr(y, {exog_col} lag {best_lag}): {corr_at_best:.3f}")
    else:
        interpret_lines.append("SARIMAX CV best lag: n/a")
    if lag_scores is not None and best_lag is not None:
        interpret_lines.append(f"CV RMSE @ lag {best_lag}: {lag_scores.loc[best_lag]:.4f}")
    if prophet_coef:
        ci_text = ""
        lower = prophet_coef.get("lower")
        upper = prophet_coef.get("upper")
        if lower is not None and upper is not None and np.isfinite(lower) and np.isfinite(upper):
            ci_text = f" (CI [{lower:.3f}, {upper:.3f}])"
        interpret_lines.append(f"Prophet β ({exog_col}): {prophet_coef['coef']:.3f}{ci_text}")
    else:
        interpret_lines.append("Prophet β: n/a")
    if sarimax_coef:
        interpret_lines.append(
            f"SARIMAX β: {sarimax_coef['coef']:.3f} (p={sarimax_coef['pvalue']:.3g}, se={sarimax_coef['std_err']:.3f})"
        )
    else:
        interpret_lines.append("SARIMAX β: n/a")
    if corr_at_best is not None and prophet_coef:
        relation = (
            "aligned" if np.sign(prophet_coef["coef"]) == np.sign(corr_at_best) else "opposite"
        )
        interpret_lines.append(f"Prophet β vs corr: {relation}")
    if corr_at_best is not None and sarimax_coef:
        relation = (
            "aligned" if np.sign(sarimax_coef["coef"]) == np.sign(corr_at_best) else "opposite"
        )
        interpret_lines.append(f"SARIMAX β vs corr: {relation}")
    if lag_scores is not None and not lag_scores.empty:
        spread = float(lag_scores.max() - lag_scores.min())
        if spread < 0.01:
            interpret_lines.append(
                "Note: CV RMSE difference across lags <0.01 -> lag effect likely absorbed by AR/MA terms."
            )
        interpret_lines.append(f"CV RMSE spread: {spread:.4f}")
    ax6.text(
        0.05,
        0.95,
        "\n".join(interpret_lines),
        va="top",
        ha="left",
        fontsize=10,
    )

    # Panel 8: unused placeholder
    ax7 = ax[7]
    ax7.axis("off")

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
    parser.add_argument("--max-lag", type=int, default=12, help="Maximum lag in months for cross-correlation panel.")
    parser.add_argument("--out-file", type=Path, default=None, help="Output image path (default: analysis_dir/dashboard.png)")
    parser.add_argument("--show", action="store_true", help="Display the figure after saving.")
    args = parser.parse_args()

    analysis_dir = args.analysis_dir
    out_file = args.out_file or analysis_dir / "summary_dashboard.png"
    visualize(analysis_dir, args.exog_col, args.rolling_window, args.max_lag, out_file, args.show)
    print(f"Dashboard saved to {out_file}")


if __name__ == "__main__":
    main()
