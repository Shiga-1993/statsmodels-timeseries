# FAQ: StatsModels 時系列ツールの使い方

本ファイルは `analysis.py` / `experiments.py` / `sarimax_diagnostics.py` を使う際によくある質問をまとめたものです。`docs/manual.md` と合わせて参照してください。

---

## Q1. 解析区間（期間）を任意の範囲にしたい

スクリプトは渡された CSV をそのまま使うだけなので、**先に CSV 側をフィルタ**すれば任意期間で解析できます。例:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("DATA/stock_prices_monthly.csv")
df = df[df["Date"] >= "2010-01-01"]          # 2010年以降を抽出
df.to_csv("DATA/stock_prices_since2010.csv", index=False)
PY

# 以降はこのCSVを指定
python analysis.py --company 明豊エンタープライズ \
  --stock-csv DATA/stock_prices_since2010.csv \
  --outdir outputs/prototype_since2010
```

外生変数 (N225など) も同じようにフィルタ済CSVを `--exog-csv` で指定してください。

---

## Q2. テスト期間や将来予測の長さを変えたい

| 対象 | 変更方法 |
| --- | --- |
| **Prophet（experiments.py）** | `--prophet-test-h` でホールドアウト長を指定（既定: 24ヶ月） |
| **Prophet（analysis.py）** | 現状24ヶ月固定。`run_prophet` 内部の `test_horizon`（デフォルト24）を変えれば対応可能 |
| **将来予測 Horizon** | `analysis.py::run_prophet` / `run_sarimax` の `horizon=6` を直接変える（将来的にCLI化予定） |

---

## Q3. 外生変数を差し替えるには？

すべてのスクリプトで `--exog-*` オプションを使います。N225以外の系列を使う場合も同じです。

- `--exog-csv`: 外生データCSV（既定: `DATA/n225_monthly.csv`）
- `--exog-date-col`: 日付列名 (既定: `Date`)
- `--exog-value-col`: 値列名 (既定: `Close_N225`)
- `--exog-use-level`: 指定するとログリターンではなく値そのものを使用
- `--exog-name`: 内部で参照する列名 (既定: `n225_ret`)
- `--exog-label`: レポートで表示するラベル

---

## Q4. スピード重視で検証したい（時間短縮策は？）

最初は小さく試すのがオススメです。

| スクリプト | 小規模テスト例 |
| --- | --- |
| `experiments.py` | `--companies` に 2〜3 銘柄だけ指定し、`--sarimax-orders 1,0,1 --sarimax-lags 1` などグリッドを最小限にする |
| `analysis.py` | `--company` を1銘柄、`--outdir outputs/test_run` など小さな出力先 |
| `sarimax_diagnostics.py` | まずは `--companies` で数銘柄に絞って診断する |

グリッドが固まったら `--companies` や `--sarimax-orders` をフルスペックに戻すと効率的です。

---

## Q5. 欠損値や `N/A` が多いときはどうする？

- `analysis.py` / `experiments.py` は `dropna()` を使って欠損行を自動的に除外します。欠損が連続して長い場合は、CSV 側で補間 (`fillna(method='ffill')` など) してから流し込むほうが安定します。
- `TimeSeriesSplit` は連続サンプル数が十分でないとエラーになるため、`--min-obs` を調整するか補間・削除でサンプル数を確保してください。

---

## Q6. ConvergenceWarning や Ljung-Box/ARCH で失敗したら？

- SARIMAX ではよくある警告です。モデル自体は計算されていますが、診断が NG の場合は `sarimax_diagnostics.py --alt-orders` を使って別の `(p,d,q)` 組を順番に試してください。
- それでも失敗する場合は、外生変数の標準化 (`--standardize-n225`) や差分 (`--exog-use-level` を外す) を試したり、モデルの次数やラグを変えるなどして安定化を図ります。

---

## Q7. LSTM/ハイブリッド構成はどこで使える？

`analysis.py` にはまだ LSTM を組み込んでいませんが、`docs/code_overview.md` にあるように
1. SARIMAX で線形成分を学習 → 残差を保存
2. LSTM (`tensorflow.keras`) で残差の非線形を学習
3. 両者を合成して最終予測

という流れで簡単にハイブリッド化できます。必要に応じて `analysis.py` をコピーし、残差を LSTM に渡す処理を追加してください。
