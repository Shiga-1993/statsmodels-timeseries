#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run residual diagnostics (Ljung-Box, ARCH) for best SARIMAX configs recorded in an experiment directory.
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.statespace.sarimax import SARIMAX

from analysis import prepare_dataset
from experiments import make_lagged

try:
    from arch import arch_model
except ImportError:  # pragma: no cover
    arch_model = None


def parse_seasonal(text: str | Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    if isinstance(text, tuple):
        return text  # already parsed
    if isinstance(text, str):
        return tuple(ast.literal_eval(text))
    raise ValueError(f"Unsupported seasonal_order type: {text}")


def parse_order(text) -> Tuple[int, int, int]:
    if isinstance(text, tuple):
        return text
    if isinstance(text, str):
        return tuple(ast.literal_eval(text))
    return tuple(text)


def load_best_configs(exp_dir: Path, companies: Iterable[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(exp_dir / "sarimax_experiments.csv")
    if companies is not None:
        df = df[df["company"].isin(companies)]
    grouped = []
    for company, block in df.groupby("company"):
        block = block.sort_values("cv_rmse")
        if "converged" in block.columns:
            conv = block[block["converged"] == True]
            if not conv.empty:
                block = conv
        grouped.append(block.iloc[0])
    return pd.DataFrame(grouped)


def standardize_columns(df: pd.DataFrame, target: bool, exog_col: str, standardize_exog: bool) -> pd.DataFrame:
    df = df.copy()
    if target:
        mean = df["y"].mean()
        std = df["y"].std()
        if std == 0 or np.isnan(std):
            raise ValueError("Target std=0; cannot standardize.")
        df["y"] = (df["y"] - mean) / std
    if standardize_exog:
        mean = df[exog_col].mean()
        std = df[exog_col].std()
        if std == 0 or np.isnan(std):
            raise ValueError("Exogenous std=0; cannot standardize.")
        df[exog_col] = (df[exog_col] - mean) / std
    return df


def run_diagnostics(
    df: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    lag: int,
    ljung_lags: Iterable[int],
    exog_col: str,
    fit_garch: bool,
) -> Tuple[float, dict]:
    lagged = make_lagged(df, lag, exog_col)
    model = SARIMAX(
        lagged["y"],
        order=order,
        seasonal_order=seasonal_order,
        exog=lagged[["exog_lag"]],
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    res = model.fit(disp=False)
    residuals = res.resid.dropna()
    lb = acorr_ljungbox(residuals, lags=list(ljung_lags), return_df=True)
    arch = het_arch(residuals, nlags=max(ljung_lags))
    diag = {
        "aic": res.aic,
        "bic": res.bic,
        "ljungbox_p": lb["lb_pvalue"].to_dict(),
        "arch_lm_pvalue": arch[1],
        "arch_f_pvalue": arch[3],
    }
    if fit_garch and arch_model is not None:
        try:
            gm = arch_model(residuals, vol="Garch", p=1, q=1, rescale=False, dist="normal").fit(disp="off")
            diag.update(
                {
                    "garch_aic": gm.aic,
                    "garch_bic": gm.bic,
                    "garch_params": gm.params.to_dict(),
                }
            )
        except Exception as exc:  # pragma: no cover
            diag["garch_error"] = str(exc)
    return res, diag


def diagnostics_pass(diag: dict, significance: float, ljung_lags: Iterable[int]) -> bool:
    lb_ok = all(diag["ljungbox_p"].get(lag, 1.0) >= significance for lag in ljung_lags)
    arch_ok = diag.get("arch_lm_pvalue", 1.0) >= significance
    return lb_ok and arch_ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SARIMAX residual diagnostics for experiment best configs.")
    parser.add_argument("--exp-dir", type=Path, required=True, help="Experiment directory containing sarimax_experiments.csv")
    parser.add_argument("--stock-csv", type=Path, default=Path("DATA/stock_prices_monthly.csv"))
    parser.add_argument("--exog-csv", type=Path, default=Path("DATA/n225_monthly.csv"))
    parser.add_argument("--n225-csv", dest="exog_csv", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--exog-date-col", type=str, default="Date")
    parser.add_argument("--exog-value-col", type=str, default="Close_N225")
    parser.add_argument("--exog-use-level", action="store_true", help="Use raw exogenous values instead of log returns.")
    parser.add_argument("--exog-name", type=str, default="n225_ret")
    parser.add_argument("--companies", nargs="+", default=None)
    parser.add_argument("--min-obs", type=int, default=150)
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--standardize-target", action="store_true")
    parser.add_argument("--standardize-n225", action="store_true")
    parser.add_argument("--level-target", action="store_true")
    parser.add_argument("--ljung-lags", type=int, nargs="+", default=[6, 12])
    parser.add_argument("--significance", type=float, default=0.05, help="Significance level for diagnostics.")
    parser.add_argument("--alt-orders", nargs="+", default=[], help="Alternative SARIMAX orders (e.g., '2,0,2') to try when diagnostics fail.")
    parser.add_argument("--fit-garch", action="store_true", help="Fit GARCH(1,1) on residuals for reference.")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/diagnostics"))
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    std_target = args.standardize or args.standardize_target
    std_exog = args.standardize or args.standardize_n225
    alt_orders = [parse_order(text) for text in args.alt_orders]
    exog_col = args.exog_name

    best = load_best_configs(args.exp_dir, args.companies)
    records = []

    for _, row in best.iterrows():
        company = row["company"]
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
        if len(df) < args.min_obs:
            continue
        df = standardize_columns(df, std_target, exog_col, std_exog)
        order = parse_order(row["order"])
        seasonal_order = parse_seasonal(row.get("seasonal_order", "(0, 0, 0, 0)"))
        res, diag = run_diagnostics(
            df,
            order=order,
            seasonal_order=seasonal_order,
            lag=int(row["lag"]),
            ljung_lags=args.ljung_lags,
            exog_col=exog_col,
            fit_garch=args.fit_garch,
        )
        refit_order = None

        if not diagnostics_pass(diag, args.significance, args.ljung_lags) and alt_orders:
            for alt in alt_orders:
                res_alt, diag_alt = run_diagnostics(
                    df,
                    order=alt,
                    seasonal_order=seasonal_order,
                    lag=int(row["lag"]),
                    ljung_lags=args.ljung_lags,
                    exog_col=exog_col,
                    fit_garch=args.fit_garch,
                )
                if diagnostics_pass(diag_alt, args.significance, args.ljung_lags):
                    res, diag = res_alt, diag_alt
                    refit_order = alt
                    order = alt
                    break

        records.append(
            {
                "company": company,
                "order": order,
                "seasonal_order": seasonal_order,
                "lag": int(row["lag"]),
                "aic": diag["aic"],
                "bic": diag["bic"],
                "arch_lm_pvalue": diag["arch_lm_pvalue"],
                "arch_f_pvalue": diag["arch_f_pvalue"],
                **{f"ljungbox_p_lag{lag}": diag["ljungbox_p"][lag] for lag in args.ljung_lags},
                "diagnostics_passed": diagnostics_pass(diag, args.significance, args.ljung_lags),
                "refit_order": refit_order,
                "garch_aic": diag.get("garch_aic"),
                "garch_bic": diag.get("garch_bic"),
            }
        )

    diag_df = pd.DataFrame(records)
    diag_csv = outdir / "sarimax_diagnostics.csv"
    diag_df.to_csv(diag_csv, index=False)

    lines = [
        "# SARIMAX residual diagnostics",
        "",
        "| Company | Order | Lag | Ljung-Box p (lag6) | Ljung-Box p (lag12) | ARCH LM p | Pass | Refit |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in diag_df.iterrows():
        lines.append(
            f"| {row['company']} | {row['order']} | {row['lag']} | "
            f"{row[f'ljungbox_p_lag{args.ljung_lags[0]}']:.3f} | "
            f"{row[f'ljungbox_p_lag{args.ljung_lags[-1]}']:.3f} | "
            f"{row['arch_lm_pvalue']:.3f} | "
            f"{'✅' if row['diagnostics_passed'] else '❌'} | "
            f"{row['refit_order'] if pd.notna(row['refit_order']) else ''} |"
        )
    (outdir / "sarimax_diagnostics.md").write_text("\n".join(lines))
    print(f"Wrote diagnostics for {len(diag_df)} companies to {diag_csv}")


if __name__ == "__main__":
    main()
