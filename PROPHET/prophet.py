# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.utilities import regressor_coefficients

# === 設定 ==========================================
STOCK_CSV = "stock_prices_monthly.csv"
N225_CSV = "n225_monthly.csv"
TARGET_COMPANY = "明豊エンタープライズ"
USE_LOG_RETURN = True
# ===================================================


def load_stock_series(csv_path: str, company: str, use_log_return: bool) -> pd.Series:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Company"] == company].copy()
    df = df.sort_values("Date").set_index("Date")

    if use_log_return:
        series = df["Log_Return"].astype(float)
    else:
        series = df["Close"].astype(float)

    series = series.dropna().asfreq("MS")
    return series


def load_n225_series(csv_path: str, use_log_return: bool) -> pd.Series:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").set_index("Date")

    if use_log_return:
        close = df["Close_N225"].astype(float)
        log_ret = np.log(close / close.shift(1))
        series = log_ret.dropna()
    else:
        series = df["Close_N225"].astype(float).dropna()

    series = series.asfreq("MS")
    return series


def main():
    stock = load_stock_series(STOCK_CSV, TARGET_COMPANY, USE_LOG_RETURN)
    n225 = load_n225_series(N225_CSV, USE_LOG_RETURN)

    data = pd.concat(
        [stock.rename("stock"), n225.rename("n225")],
        axis=1,
        join="inner"
    ).dropna()

    # Prophet 用フォーマットに変換
    # ds: 日付, y: 目的変数（ここでは株のログリターン）
    df_prophet = data.reset_index().rename(columns={"Date": "ds"}) \
        if "Date" in data.reset_index().columns else data.reset_index()
    df_prophet = df_prophet.rename(columns={"index": "ds"})
    df_prophet = df_prophet[["ds", "stock", "n225"]]
    df_prophet = df_prophet.rename(columns={"stock": "y", "n225": "n225_ret"})

    # Prophet モデル構築
    # デフォルトは加法モデル。必要なら seasonality_mode="multiplicative" などに変更
    m = Prophet()
    # N225 ログリターンを追加の線形回帰成分として使う
    m.add_regressor("n225_ret")
    m.fit(df_prophet)

    # 訓練期間と同じ期間で in-sample 予測（まずは適合の確認用）
    future = df_prophet[["ds", "n225_ret"]].copy()
    forecast = m.predict(future)

    # 追加説明変数の係数を取得
    beta_df = regressor_coefficients(m)
    print("\n=== 追加回帰変数 (N225) の係数推定値 ===")
    print(beta_df)

    # ds, y, yhat, n225_ret をまとめて確認
    out = pd.DataFrame({
        "ds": df_prophet["ds"],
        "y": df_prophet["y"],
        "n225_ret": df_prophet["n225_ret"],
        "yhat": forecast["yhat"],
    })
    print("\n=== 一部サンプル ===")
    print(out.tail(10))

    # 将来の予測例（6ヶ月先）
    # 将来 6 ヶ月分の N225 のシナリオを用意する必要がある
    horizon = 6
    last_date = df_prophet["ds"].max()
    future_dates = pd.date_range(
        last_date + pd.offsets.MonthBegin(1),
        periods=horizon,
        freq="MS"
    )

    # ここでは「将来も直近平均の N225 リターンが続く」という簡単なシナリオ
    avg_n225 = df_prophet["n225_ret"].mean()
    future_n225 = np.repeat(avg_n225, horizon)

    future_forecast = pd.DataFrame({
        "ds": future_dates,
        "n225_ret": future_n225,
    })

    forecast_future = m.predict(future_forecast)
    print("\n=== 将来 6 ヶ月の予測 ===")
    print(forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]])


if __name__ == "__main__":
    main()

