# Visualization Guide

時系列分析の結果を俯瞰するための可視化方法をまとめたドキュメントです。`visualize_results.py` を使うと、`analysis.py` の出力（Prophet/SARIMAX の予測や診断）を 1 枚のダッシュボードにまとめて確認できます。

---

## 1. 依存関係と前提

- `analysis.py` を実行して、対象ディレクトリ（例: `outputs/prototype_default`）に以下のファイルが生成されていること:
  - `prepared_dataset.csv`
  - `prophet_in_sample.csv`
  - `prophet_forecast.csv`
  - `sarimax_in_sample.csv`
  - `sarimax_cv_metrics.csv`
- これらは `analysis.py` の既定出力なので、特別な設定なく生成されます。
- 使用ライブラリ: `pandas`, `numpy`, `matplotlib`（`visualize_results.py` 内部で読み込み）

---

## 2. 実行方法

```bash
python visualize_results.py \
  --analysis-dir outputs/prototype_default \
  --exog-col n225_ret \
  --rolling-window 24 \
  --out-file outputs/prototype_default/dashboard.png
```

| オプション | 説明 |
| --- | --- |
| `--analysis-dir` | `analysis.py` の出力ディレクトリ |
| `--exog-col` | 外生変数の列名（既定: `n225_ret`）|
| `--rolling-window` | ローリング相関の窓サイズ |
| `--out-file` | 保存先 PNG（省略時は `<analysis-dir>/summary_dashboard.png`）|
| `--show` | 保存後に matplotlib のウィンドウ表示を行う |

複数のディレクトリに対して実行すれば、モデリング結果や外生変数別の効果を一覧で比較できます。

---

## 3. 出力ダッシュボードの構成

ダッシュボードは 3×2 のサブプロット構成です。

1. **Prophet: 実績 vs 予測**  
   - 黒: 実績  
   - 青: Prophet in-sample  
   - 破線: 6か月先の Prophet 予測（信頼区間付き）

2. **SARIMAX: 実績 vs 予測**  
   - 黒: 実績  
   - 橙: SARIMAX in-sample

3. **残差分布**  
   - Prophet / SARIMAX の残差ヒストグラムを重ねて表示

4. **ローリング相関**  
   - 目的変数 `y` と外生変数 `exog_col` の相関を `rolling_window` で計算

5. **SARIMAX lag 別 CV RMSE**  
   - `sarimax_cv_metrics.csv` の平均RMSEを棒グラフで表示

6. **サマリー**  
   - Prophet/SARIMAX の RMSE/MAE、サンプル数など

---

## 4. 応用例

- **複数ディレクトリ比較**: `outputs/prototype_default`, `outputs/prototype_macro` それぞれにダッシュボードを作成して、N225 vs 合成マクロ指数の影響を比較。
- **外生変数別の相関確認**: `--exog-col macro_series` などで列名を切り替えれば、ローリング相関の曲線を簡単に比較できます。
- **レポートへの貼り付け**: 生成された `summary_dashboard.png` を、そのままレポートやスライドに貼り付ければ、モデルの挙動を一枚で説明できます。

---

## 5. 既存の図との使い分け

`analysis.py` が出力する `prophet_fit.png`, `sarimax_fit.png` はモデル個別の可視化に特化しています。一方 `visualize_results.py` は:

- Prophet / SARIMAX を同じ図面で比較
- 残差や相関、lag といった診断情報も同時に確認

といった特徴があります。レポート作成やモデル間比較の際には `visualize_results.py` で生成したダッシュボードが効率的です。
