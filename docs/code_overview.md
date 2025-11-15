# 時系列モデル実装ガイド（Prophet / SARIMAX / LSTM ハイブリッド概要）

このドキュメントでは、リポジトリに含まれている Python スクリプトのうち、時系列予測モデルの核となる部分を抜粋しながら、必要なライブラリやコード構造を解説する。`analysis.py` や `experiments.py` では実際にこれらの実装が組み合わされており、ここで紹介する各セクションを理解しておけば、任意のデータセットに容易に適用できる。

---

## 1. Prophet モデル

Prophet は Meta（旧 Facebook）が公開しているオープンソースライブラリで、「トレンド」「季節性」「祝日」「外生回帰変数」を人間が解釈しやすい形で扱える点が特徴だ。

### 1.1 必要ライブラリとデータ形式

| 要素 | Pythonライブラリ/オブジェクト | 備考 |
| --- | --- | --- |
| **ライブラリ** | `prophet` | メインの予測ツール |
| **データ形式** | `pandas.DataFrame` | 列名が `ds`（日付）と `y`（目的変数）である必要がある。`analysis.py` の `run_prophet` 関数では `Date` 列を `ds` にリネームしている。 |

### 1.2 基本的なコードフロー

```python
import pandas as pd
from prophet import Prophet

# 1) DataFrame の整形: ds, y および任意の回帰列
df_prophet = df.reset_index().rename(columns={"Date": "ds"})
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

# 2) 外生変数の追加（analysis.py では N225 などの log return を regressor として追加）
model.add_regressor("macro_series")

# 3) モデル学習
model.fit(df_prophet[["ds", "y", "macro_series"]])

# 4) 将来のデータ作成と予測
future = model.make_future_dataframe(periods=6, freq="MS")
future["macro_series"] = avg_macro  # 予測期間の外生変数をセット
forecast = model.predict(future)

# 5) 結果の可視化
model.plot(forecast)
model.plot_components(forecast)
```

`analysis.py::run_prophet` では、外生回帰変数の列名や平均値の扱いを CLI から指定できるように拡張している。`experiments.py` でも `evaluate_prophet` 内で複数ラグの回帰器を自動で追加している。

---

## 2. SARIMAX モデル

SARIMAX（Seasonal ARIMA with eXogenous regressors）は、`statsmodels` ライブラリで提供されている汎用的な時系列モデルである。リポジトリでは2つのアプローチを示している。

### 2.1 `pmdarima.auto_arima` による自動次数推定

`pmdarima` ライブラリの `auto_arima` はグリッドサーチを行って最適な次数 `(p, d, q)` および季節次数 `(P, D, Q)` を探索する関数である。

```python
import pmdarima as pm

model = pm.auto_arima(
    train_series,
    seasonal=True,
    m=12,
    trace=True,
    n_jobs=-1,
    maxiter=10
)

pred, conf_int = model.predict(n_periods=len(test_series), return_conf_int=True)
```

### 2.2 `statsmodels.SARIMAX` とウォークフォワード検証

`analysis.py::run_sarimax` や `experiments.py` の `evaluate_sarimax` では、`statsmodels.tsa.statespace.sarimax.SARIMAX` を用いてグリッドサーチ＋交差検証を行っている。外生変数は `exog=` 引数で指定できる。

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    y_train,
    order=(1, 0, 1),
    seasonal_order=(0, 0, 0, 0),
    exog=exog_train,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
result = model.fit(disp=False)
forecast = result.forecast(steps=len(y_test), exog=exog_test)
```

#### ウォークフォワード（walk-forward）検証

リポジトリの `evaluate_sarimax` は `sklearn.model_selection.TimeSeriesSplit` を使ってウォークフォワード検証を行い、平均RMSEでベストラグを選ぶ。`analysis.py::run_sarimax` もほぼ同様のロジックを単一銘柄向けに実装している。

```python
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(values):
    y_tr, y_te = values[train_idx], values[test_idx]
    x_tr, x_te = exog[train_idx], exog[test_idx]
    model = SARIMAX(y_tr, order=order, exog=x_tr, ...)
    result = model.fit(disp=False)
    y_pred = result.forecast(steps=len(y_te), exog=x_te)
```

### 2.3 残差診断

`sarimax_diagnostics.py` では、ベストモデルに対して Ljung-Box, ARCH テストを実行し、必要に応じて別の `(p,d,q)` にリフィットする仕組みを備えている。また `--fit-garch` を指定すると `arch` ライブラリを用いて GARCH(1,1) を参考情報としてフィットする。

---

## 3. ハイブリッド ARIMAX–LSTM

リポジトリには、SARIMAXで線形成分（＋外生変数）を予測し、その残差に対して LSTM を当てるハイブリッドアプローチの構造も含まれている。ここでは LSTM 部分を中心に解説する。

### 3.1 LSTM モデルの定義

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError

def build_lstm(params, input_shape):
    model = Sequential()
    model.add(LSTM(units=params["lstm_units"], return_sequences=True, input_shape=(input_shape, 1)))
    model.add(Dropout(rate=params["dropout"]))
    model.add(LSTM(units=params["lstm_units"], return_sequences=False))
    model.add(Dropout(rate=params["dropout"]))
    model.add(Dense(1))
    model.compile(
        optimizer=params["optimizer"],
        loss=params["loss"],
        metrics=[RootMeanSquaredError(), MeanAbsoluteError()],
    )
    return model
```

### 3.2 モデルの学習

```python
model = build_lstm(params, input_shape=lookback)
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=params["epochs"],
    callbacks=[early_stopping_cb, model_checkpoint_cb]
)
```

### 3.3 ハイブリッド統合

1. SARIMAX (ARIMAX) で本系列＋外生変数をモデリングし、残差系列を得る
2. 残差系列を LSTM に入力して非線形成分を学習
3. 最終的な予測値 = SARIMAX の線形予測 + LSTM 残差予測

`analysis.py` 部分ではまだ LSTM を統合していないが、上記構造を用いれば簡単に組み込める。

---

## 4. まとめ

- **Prophet**: `prophet` ライブラリで `ds`, `y` の DataFrame を用意し、`add_regressor` で外生変数を加えるだけで季節性を含むモデルを構築できる。
- **SARIMAX**: `statsmodels.SARIMAX` で ARIMAX/SARIMAX を構築。`experiments.py` では TimeSeriesSplit を使ったベストラグ探索、`sarimax_diagnostics.py` で Ljung-Box/ARCH 診断と GARCH フィットを自動化している。
- **ハイブリッド ARIMAX–LSTM**: 線形構造（SARIMAX）と非線形構造（LSTM）を組み合わせ、残差モデリングに LSTM を使用する。

これらのサンプルコードを参考に、`analysis.py` や `experiments.py` の CLI オプションを活用すれば、N225 に限らず任意の外生系列を組み合わせた予測実験を迅速に実行できる。***
