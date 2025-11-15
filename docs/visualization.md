# Visualization Guide

時系列分析の結果を俯瞰するための可視化方法をまとめたドキュメントだ。`visualize_results.py` を使うと、`analysis.py` の出力（Prophet/SARIMAX の予測や診断）を 1 枚のダッシュボードに統合して確認できる。

---

## 1. 依存関係と前提

- `analysis.py` を実行して、対象ディレクトリ（例: `outputs/prototype_default`）に以下のファイルが生成されていること:
  - `prepared_dataset.csv`
  - `prophet_in_sample.csv`
  - `prophet_forecast.csv`
  - `sarimax_in_sample.csv`
  - `sarimax_cv_metrics.csv`
- これらは `analysis.py` の既定出力なので、特別な設定なく生成される。
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
| `--max-lag` | クロス相関パネルで評価するラグ数（既定: 12ヶ月）|
| `--show` | 保存後に matplotlib のウィンドウ表示を行う |

複数のディレクトリに対して実行すれば、モデリング結果や外生変数別の効果を一覧で比較できる。

---

## 3. 出力ダッシュボードの構成

ダッシュボードは 4×2 のサブプロット構成だ。

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

5. **クロス相関 (`y` vs 外生変数のラグ)**  
   - `--max-lag` までのラグごとに相関係数を棒グラフ化し、CVで選ばれたベストラグを破線で表示する。経験則として |corr|≲0.2 は弱、0.2〜0.4 は中程度、0.4 以上は強めの結び付きとみなせるので、たとえば 0.3 前後なら「無視できないが支配的でもない」影響度と判断できる。統計的有意性を確認したい場合は `sarimax_summary.txt` の z, p 値と合わせて解釈する。

6. **SARIMAX lag 別 CV RMSE**  
    - `sarimax_cv_metrics.csv` の平均RMSEを棒グラフで表示する。SARIMAX のラグ探索（`analysis.py::run_sarimax` のウォークフォワードCV）の結果なので、どのラグで外生変数を遅行させると CV RMSE が最小だったかを比較できる。

7. **サマリー**  
   - Prophet/SARIMAX の RMSE/MAE、サンプル数、CVベストラグなどを一覧化。`analysis.py` で得られたベスト設定（Prophet/SARIMAXとも CV や検証で選ばれた構成）のスコアである。

8. **Lag & Coefficient Insights**  
   - ベストラグの `corr(y, exog)`、Prophet β（信頼区間付き）、SARIMAX β（p 値・標準誤差）をテキストで列挙し、符号や大きさが生相関と整合しているかを示す。CV RMSE の差がごく小さい場合は、その旨のメモを付記し、「ラグを変えても性能差が僅少」という解釈をサポートする。
   - `Prophet β` は Prophet の `add_regressor` 係数、`SARIMAX β` は CVで選ばれたラグの外生係数、`Corr(y, exog lag k)` は目的変数と `k` ヶ月遅行させた外生変数との相関である。β はモデル内部の推定（正則化や他要素の影響を受ける）、`corr` はあくまで生データの連動度という違いがある。

**読み解きヒント**
- クロス相関の棒は「前月の値がどれくらい y に効いているか」の直感指標だ。|corr|≲0.2 なら弱め、0.2〜0.4 なら中程度、0.4 以上なら強い影響。0.3 付近は「無視できないが支配的でもない」強さなので、係数や p 値と合わせて判断する。
- Lag & Coefficient Insights のテキストは、相関に加えて係数の符号や大きさ、p 値の有意性をまとめたものだ。クロス相関でピークがあるラグと、CV ベストラグ（Summary に表示）が一致するかを見れば、ラグ設定の根拠が明確になる。
- `Corr(y, exog lag k)` は「目的変数と `k` ヶ月遅らせた外生変数の相関」を意味する。N225 なら `corr(y_t, n225_{t-k})` を指すので、正の値なら「N225 が上がるほど k ヶ月後の銘柄リターンも上がる」傾向を示す。
- Lag & Coefficient Insights ではタイトルにも “β=model coefficient, corr=plain corr(y, exog lag)” と表示しており、`corr` はモデル推定ではなく生の相関値であることを明示している。
- `CV RMSE by lag` は SARIMAX のウォークフォワードCV結果そのものであり、ラグ1がわずかに最良でも棒が横並び気味に見えることがある。これは「他のラグもほぼ同等のRMSE」で、AR/MA項や他のパラメータが遅行効果を吸収しているサインであり、クロス相関のピークと直ちに矛盾するものではない。
- SARIMAX のベストラグとクロス相関のピークが一致しない場合は、(1) 生データではそのラグの相関が最大でも、モデル全体では別ラグのほうが残差が安定する、あるいは (2) ピーク差が僅少でどのラグでも CV RMSE がほぼ同じ、といった状況を示す。モデル精度と生相関の両方を見て総合的に解釈する。
- もし `corr` が Prophet/SARIMAX の β よりも大きい場合は、「生データ上では比較的強い連動だが、モデルは他の要因や正則化により抑えている（＝完全には回収していない）」可能性がある。逆に β のほうが大きい場合は、モデルが相関を増幅して説明に使っているサインだ。Impact summary はどれが“良い”かを決める図ではなく、調和/不一致を見つけて次の解析（例: ラグの見直し、別レジームの検討）につなげる補助ツールと位置付けている。

---

## 4. 応用例

- **複数ディレクトリ比較**: `outputs/prototype_default`, `outputs/prototype_macro` それぞれにダッシュボードを作成して、N225 vs 合成マクロ指数の影響を比較する。
- **外生変数別の相関確認**: `--exog-col macro_series` などで列名を切り替えれば、ローリング相関の曲線を簡単に比較できる。
- **レポートへの貼り付け**: 生成された `summary_dashboard.png` をそのままレポートやスライドに貼り付ければ、モデルの挙動を一枚で説明できる。

---

## 5. 既存の図との使い分け

`analysis.py` が出力する `prophet_fit.png`, `sarimax_fit.png` はモデル個別の可視化に特化している。一方 `visualize_results.py` は:

- Prophet / SARIMAX を同じ図面で比較
- 残差や相関、lag といった診断情報も同時に確認

といった特徴がある。レポート作成やモデル間比較の際には `visualize_results.py` で生成したダッシュボードが効率的だ。
