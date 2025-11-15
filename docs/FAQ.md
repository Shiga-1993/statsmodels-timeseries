# FAQ: StatsModels 時系列ツールの使い方

本ファイルは `analysis.py` / `experiments.py` / `sarimax_diagnostics.py` を扱う際によくある質問をまとめたドキュメントだ。詳細手順は `docs/manual.md` と併せて参照してほしい。

---

## Q1. 解析区間（期間）を任意の範囲にしたい

スクリプトは与えられた CSV をそのまま利用する設計であり、**事前に CSV をフィルタ**すれば任意区間で解析できる。例を示す。

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

外生変数 (N225など) も同様にフィルタ済み CSV を `--exog-csv` で指定する。

---

## Q2. テスト期間や将来予測の長さを変えたい

| 対象 | 変更方法 |
| --- | --- |
| **Prophet（experiments.py）** | `--prophet-test-h` でホールドアウト長を指定（既定: 24ヶ月） |
| **Prophet（analysis.py）** | 現状24ヶ月固定。`run_prophet` 内部の `test_horizon`（デフォルト24）を書き換える |
| **将来予測 Horizon** | `analysis.py::run_prophet` / `run_sarimax` の `horizon=6` を直接変更する（将来的にCLI化予定） |

---

## Q3. 外生変数を差し替えるには？

すべてのスクリプトは `--exog-*` オプションを実装している。N225以外の系列も同じ手順で投入できる。

- `--exog-csv`: 外生データCSV（既定: `DATA/n225_monthly.csv`）
- `--exog-date-col`: 日付列名 (既定: `Date`)
- `--exog-value-col`: 値列名 (既定: `Close_N225`)
- `--exog-use-level`: 指定するとログリターンではなく値そのものを使用
- `--exog-name`: 内部で参照する列名 (既定: `n225_ret`)
- `--exog-label`: レポートで表示するラベル

---

## Q4. スピード重視で検証したい（時間短縮策は？）

まずは小規模な条件で動作確認し、設定が固まってからフルグリッドに拡張するのが安全だ。

| スクリプト | 小規模テスト例 |
| --- | --- |
| `experiments.py` | `--companies` に 2〜3 銘柄だけ指定し、`--sarimax-orders 1,0,1 --sarimax-lags 1` などグリッドを最小限にする |
| `analysis.py` | `--company` を1銘柄、`--outdir outputs/test_run` など小さな出力先 |
| `sarimax_diagnostics.py` | まずは `--companies` で数銘柄に絞って診断する |

十分に検証できたら `--companies` や `--sarimax-orders` をフルスペックに戻して本番のスイープを行う。

---

## Q5. 欠損値や `N/A` が多いときはどうする？

- `analysis.py` / `experiments.py` は `dropna()` で欠損行を除外する。欠損が長く続く場合は CSV を前処理して補間 (`fillna(method='ffill')` など) しておくと安定する。
- `TimeSeriesSplit` は連続サンプル数が不足すると失敗するため、`--min-obs` を増減させるか、欠損区間を補完・削除して学習期間を確保する。

---

## Q6. ConvergenceWarning や Ljung-Box/ARCH で失敗したら？

- SARIMAX ではよく出る警告であり、推定自体は完了しているケースが多い。診断が NG なら `sarimax_diagnostics.py --alt-orders` で別の `(p,d,q)` を順番に試す。
- それでも収束しない場合は外生変数の標準化 (`--standardize-n225`) や原系列利用 (`--exog-use-level` を外す) を検討し、次数・ラグも変更しながら安定性を探る。

---

## Q7. LSTM/ハイブリッド構成はどこで使える？

`analysis.py` にはまだ LSTM を統合していないが、`docs/code_overview.md` に示す通り
1. SARIMAX で線形成分を学習 → 残差を保存
2. LSTM (`tensorflow.keras`) で残差の非線形を学習
3. 両者を合成して最終予測

という流れで簡単にハイブリッド化できる。必要に応じて `analysis.py` をコピーし、残差を LSTM に渡す処理を追加する。

---

## Q8. 可視化ダッシュボードで見る「in-sample」「Rolling correlation」「Residual distribution」とは？

- **In-sample 可視化**: `visualize_results.py` の Prophet/SARIMAX パネルは、学習区間（= in-sample）の実績と予測値を同じプロットで示す。ホールドアウト前に過学習や追従度を直感的に把握できる。
- **Rolling correlation**: 目的変数 `y` と外生変数（例: N225）の相関係数を、`--rolling-window`（既定24ヶ月）で移動計算した指標である。相関が時間でどう変化したか、強弱がいつ現れたかを一目で確認できる。
- **Residual distribution**: Prophet と SARIMAX の残差（実績 − 予測）のヒストグラムを重ねたもの。残差の歪みや裾の重さ、平均が0付近かどうかを比較し、バイアスや外れ値傾向を把握できる。

---

## Q9. クロス相関パネルの棒グラフはどう読む？ `corr ≈ 0.3` は強いのか？

- `visualize_results.py` の「Cross-correlation vs lag」パネルは、外生変数をラグさせたときの相関係数を棒グラフ化したものだ。破線は SARIMAX CV で最良だったラグを示す。
- |corr| のおおまかな目安:
  - **≲0.2** … 弱い結び付き。ノイズの可能性が高く、他ラグや別共変量を確認する。
  - **0.2〜0.4** … 中程度。`corr ≈ 0.3` なら「影響はあるが支配的ではない」と解釈し、係数や p 値と併せて判断する。
  - **≳0.4** … 強い結び付き。モデル化で積極的に利用したいサイン。
- 相関値だけで可否を決めず、`sarimax_summary.txt` の z / p 値や Prophet/SARIMAX の係数の符号と大きさも同時に確認すると、ラグ選択の理由が明確になる。
