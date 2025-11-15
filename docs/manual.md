# StatsModels Timeseries Analysis Manual

このマニュアルは、N225を外部変数としたProphet / SARIMAXの試行錯誤を再現するための手順をまとめたものです。すべて `python` コマンドを想定し、パスはリポジトリルートからの相対パスです。

---

## 1. 必要ライブラリの準備

```bash
pip install prophet statsmodels arch
```

- `prophet` / `statsmodels` は既に導入済みですが、新規環境では上記でインストールください。
- 残差診断で GARCH を使う場合は `pip install arch` が必要です。

---

## 2. ベースデータの確認

### N225ダウンロード（初回のみ）

## 2.2 モデルが参照するデータ区間と評価指標

- **Prophet** (`analysis.py::run_prophet`)
  - デフォルトでは末尾24ヶ月 (`--prophet-test-h`) をホールドアウト。訓練データで `model.fit()`、テスト区間で `RMSE/MAE` を計算。
  - 6ヶ月先予測は `model.make_future_dataframe()` で作成し、外生変数には直近12ヶ月の平均値（簡易シナリオ）を適用。
- **SARIMAX** (`analysis.py::run_sarimax`)
  - `TimeSeriesSplit` によるウォークフォワード交差検証で複数ラグを比較。平均CV RMSEが最小となるラグが「ベストラグ」。
  - ベストラグで全期間を再フィットし、`sarimax_fit.png` などの図と in-sample 予測を出力。
  - 将来6ヶ月分の外生変数は直近値をホールドするシナリオで `res.forecast()` を生成。
- **experiments.py**
  - Prophet: ホールドアウト24ヶ月 (デフォルト) で RMSE/MAE を評価。
  - SARIMAX: `TimeSeriesSplit` で `cv_rmse/cv_mae` を計算。AIC も記録。
- **sarimax_diagnostics.py**
  - ベスト構成に対し Ljung-Box (lag6/lag12) と ARCHテストを実施。`--alt-orders` で別の `(p,d,q)` を順次試し、p値が 0.05 以上になる構成が見つかればそちらに差し替え。
  - `--fit-garch` を指定すると GARCH(1,1) を resid 上にフィットし、AIC/BIC とパラメータを参考情報として出力。


```bash
python - <<'PY'
import pandas as pd, numpy as np
import yfinance as yf
from pathlib import Path

hist = yf.download('^N225', period='max', interval='1mo', auto_adjust=True, progress=False)
hist = hist.reset_index()[['Date','Close']]
hist['Date'] = pd.to_datetime(hist['Date']).dt.to_period('M').dt.to_timestamp()
hist = hist.rename(columns={'Close':'Close_N225'})
hist = hist.sort_values('Date')
hist['Log_Return'] = np.log(hist['Close_N225'] / hist['Close_N225'].shift(1))
Path('DATA').mkdir(exist_ok=True)
hist.to_csv('DATA/n225_monthly.csv', index=False)
PY
```

### 銘柄データの概要

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv('DATA/stock_prices_monthly.csv')
print(df['Company'].value_counts())
PY
```

上記で13銘柄が各310行前後あることを確認します。

### 2.1 外生変数を差し替える

`analysis.py`, `experiments.py`, `sarimax_diagnostics.py` はすべて `--exog-*` 系オプションで任意の時系列を外生変数として利用できます。

- `--exog-csv`: CSV パス（既定: `DATA/n225_monthly.csv`）
- `--exog-date-col`: 日付列名（既定: `Date`）
- `--exog-value-col`: 値列名（既定: `Close_N225`）
- `--exog-use-level`: 指定するとログリターンではなく値そのものを使用
- `--exog-name`: マージ後の列名 (既定: `n225_ret`)
- `--exog-label`: レポート出力で使用される説明テキスト

サンプルとして `ALT_DATA/macro_index.csv`（日付列 `Month`, 値列 `Macro_Index`）を用意しています。例えば:

```bash
python analysis.py \
  --company 明豊エンタープライズ \
  --exog-csv ALT_DATA/macro_index.csv \
  --exog-date-col Month \
  --exog-value-col Macro_Index \
  --exog-use-level \
  --exog-name macro_series \
  --exog-label "シンセティック景気指数" \
  --outdir outputs/prototype_macro
```

---

## 3. 単一銘柄のプロトタイプ分析

`analysis.py` は Prophet + SARIMAX を1銘柄単位で実行し、図やCSVを `outputs/prototype/` に出力します。

```bash
python analysis.py --company "明豊エンタープライズ" --outdir outputs/prototype
```

> 別の外生変数を使う場合は `--exog-csv`, `--exog-date-col`, `--exog-value-col`, `--exog-use-level`, `--exog-name`, `--exog-label` を適宜セットしてください。

アウトプット例: `prophet_fit.png`, `sarimax_fit.png`, `report.md` 等。

---

## 4. 複数銘柄のハイパーパラメータ探索

共通パラメータ:
- `--companies`: 13銘柄を一覧指定
- `--standardize-n225`: N225リターンのみZスコア化
- `--sarimax-orders`: `(1,0,1)` と `(2,0,1)` の組合せ
- `--sarimax-lags`: `{0,1,2,3,6}` など

### 4.1 ベースライン（lag=0）

```bash
python experiments.py \
  --companies 明豊エンタープライズ 明和地所 ゴールドクレスト エスリード プロパスト アーバネットコーポレーション セントラル総合開発 ディア・ライフ コーセーアールイー コスモスイニシア 日神グループホールディングス シーラホールディングス 日本エスコン \
  --min-obs 160 \
  --standardize-n225 \
  --prophet-reg-lag 0 \
  --sarimax-orders 1,0,1 2,0,1 \
  --sarimax-lags 0 1 2 3 6 \
  --outdir outputs/experiments_stdN_reglag0
```

> `--exog-*` オプションを追加すれば、N225 以外の外生変数でも同じループを実行できます。

### 4.2 Prophetラグ探索（lag=1〜6）

同じコマンドで `--prophet-reg-lag` を `1,2,3,4,5,6` に変更し、それぞれ `outputs/experiments_stdN_reglag{n}` へ出力してください。SARIMAXラグはlag=1のみで十分です。

例:
```bash
python experiments.py ... --prophet-reg-lag 3 --sarimax-lags 1 --outdir outputs/experiments_stdN_reglag3
```

最終的なラグ比較表は
```bash
python - <<'PY'
import pandas as pd
from pathlib import Path

lag_dirs = {lag: Path(f'outputs/experiments_stdN_reglag{lag}') for lag in range(7)}
frames = {}
for lag, path in lag_dirs.items():
    df = pd.read_csv(path/'prophet_experiments.csv')
    idx = df.groupby('company')['test_rmse'].idxmin()
    frame = df.loc[idx, ['company','test_rmse']]
    if 'exog_coef' in df.columns:
        frame['exog_coef'] = df.loc[idx]['exog_coef'].values
    else:
        frame['exog_coef'] = df.loc[idx]['n225_coef'].values
    frames[lag] = frame.set_index('company')

rows = []
for comp in frames[0].index:
    record = {'company': comp}
    best_rmse = float('inf')
    best_lag = None
    for lag, frame in frames.items():
        rmse = frame.loc[comp,'test_rmse']
        record[f'rmse_lag{lag}'] = rmse
        if rmse < best_rmse:
            best_rmse = rmse
            best_lag = lag
    record['best_lag'] = best_lag
    rows.append(record)

pd.DataFrame(rows).to_csv('outputs/prophet_lag_summary.csv', index=False)
PY
```

`outputs/prophet_lag_summary.md` に結果を出力しておけば後で参照しやすくなります。

---

## 5. SARIMAX残差診断と代替モデル

### 5.1 診断ツールの実行

```bash
python sarimax_diagnostics.py \
  --exp-dir outputs/experiments_std_n225 \
  --companies 明豊エンタープライズ 明和地所 ゴールドクレスト エスリード プロパスト アーバネットコーポレーション セントラル総合開発 ディア・ライフ コーセーアールイー コスモスイニシア 日神グループホールディングス シーラホールディングス 日本エスコン \
  --standardize-n225 \
  --significance 0.05 \
  --alt-orders 2,0,2 1,0,2 2,0,1 \
  --fit-garch \
  --outdir outputs/diagnostics_stdN
```

出力:
- `sarimax_diagnostics.csv` / `.md` … Ljung-Box, ARCH, GARCH AIC/BIC を含む表
- `diagnostics_passed`, `refit_order` 列で代替モデルに切り替えたかを確認

### 5.2 収束しない場合の再試行

`--alt-orders` リストに `(p,d,q)` を追加すると、診断失敗時に順番に再フィットします。

---

## 6. 結論のまとめ

### 6.1 Prophet ラグ結論
- `outputs/prophet_lag_summary.md` で最良ラグの分布を確認。lag0〜2 で概ね十分、lag>2 は改善幅がごく小さい。

### 6.2 SARIMAX ラグ & 残差
- `outputs/experiments_std_n225/summary.md` で lag=1 の優位性と係数有意性を確認。
- `outputs/diagnostics_stdN/sarimax_diagnostics.md` で Ljung-Box/ARCH をチェックし、必要に応じて `sarimax_diagnostics.py --alt-orders ...` を再実行。

---

## 7. よく使うコマンド集

| 目的 | コマンド例 |
| --- | --- |
| 単一銘柄分析 | `python analysis.py --company "明和地所" --outdir outputs/prototype_meiwa` |
| ベースマルチスイープ | `python experiments.py ... --standardize-n225 --prophet-reg-lag 0 --sarimax-lags 0 1 2 3 6` |
| Prophetラグ実験 | `python experiments.py ... --prophet-reg-lag 3 --sarimax-lags 1 --outdir outputs/experiments_stdN_reglag3` |
| ラグ比較表出力 | 上述の Python スニペットで `outputs/prophet_lag_summary.csv` を生成 |
| SARIMAX診断 | `python sarimax_diagnostics.py --exp-dir outputs/experiments_std_n225 ... --alt-orders 2,0,2 1,0,2 2,0,1 --fit-garch` |

---

## 8. 結論の要点（2025-08-14時点）

- SARIMAX: lag=1（N225の1ヶ月遅行）が13銘柄中10で最適。係数は0.6〜1.3で p≪0.05。残差診断ではなお改善余地あり。
- Prophet: lag0〜6を網羅してもRMSE差は小さく、lag0〜2で十分。長期ラグは係数が不安定になりやすい。
- N225のみZスコア化すると係数解釈が容易で、性能影響は軽微（`outputs/experiments_std_n225/summary.md`）。
- Seasonality や価格レベルを直接予測する案は大幅劣化（`experiments_seasonal/`, `experiments_level_stdN225/`）。

これらを踏まえ、新しい銘柄への適用や追加要件の際は上記マニュアルに沿って実験を再現してください。
