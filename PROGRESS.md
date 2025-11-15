# StatsModels Investigation Progress

## Overview
- Target: 月次株価 (`DATA/stock_prices_monthly.csv`) のログリターンを目的変数とし、N225 (`DATA/n225_monthly.csv`) の変動を外生説明変数として Prophet / SARIMAX のプロトタイピングを進行中。
- `analysis.py` … 単一銘柄向けのProphet+SARIMAX分析、図・CSV・Markdownレポートを自動生成。
- `experiments.py` … 複数銘柄×ハイパーパラメータの一括検証ツール。RegressorラグやSARIMAXラグ、価格レベル / ログの切り替え等をCLIフラグで制御可能。

## 実行済み実験 (ログリターン目的)
1. **Prototype (`analysis.py`)**
   - 銘柄: 明豊エンタープライズ。
   - 結果: Prophet/SARIMAXとも RMSE ≈0.178、N225係数は正で有意 (SARIMAX)。アウトプットは `outputs/prototype/`.
2. **Multi-company sweeps (`experiments.py`)**
   - 対象: 2000年代前半から十分な月次観測がある13銘柄。
   - グリッド: Prophet (seasonality_mode ∈ {additive,multiplicative}, changepoint_prior_scale ∈ {0.05,0.2}, seasonality_prior_scale ∈ {5,15}); SARIMAX (orders (1,0,1)/(1,1,1)/(2,0,1), lags ∈ {0,1,2,3,6}).
   - 出力: `outputs/experiments_all_loglag0/`。
3. **Lag stress test**
   - Prophet: N225を1ヶ月遅行 (`--prophet-reg-lag 1`) に変更。
   - SARIMAX: lag grid を {0,1,2,3,6,9,12} に拡張。
   - 出力: `outputs/experiments_all_reglag1/`.
   - Aggregated markdown: `outputs/experiments_all_summary.md`.

## 観測まとめ
- **Prophet**
  - 13銘柄のうち9銘柄で乗法季節性が最良 (lag 0)。
  - N225を1ヶ月遅行させても RMSE 改善は 3/13 銘柄のみで、多くは係数の絶対値が急増し安定性を欠いた。
  - 提案: リターンを標準化する/レベルターゲットを試す/追加共変量を導入するなど、スケーリングと仕様見直しが優先。
- **SARIMAX**
  - ベストモデルの 10/13 銘柄が lag=1 を選択し、係数も有意 (p<0.05)。
  - lag 6 or 12 が効いたのは一部 (プロパスト、セントラル総合開発、明豊) で、係数は非有意もしくは小さい。
  - ConvergenceWarning が散見されたため、`mle_retvals['converged']` チェックやBICによる選別が推奨。

## 引き継ぎガイド
1. **環境**
   - 必須ライブラリ: `prophet`, `statsmodels`, `yfinance`, `scikit-learn`.
   - `DATA/n225_monthly.csv` は `experiments.py`/`analysis.py` の期待フォーマット済み。
2. **スクリプトの使い方**
   - 単一銘柄の詳細分析: `python analysis.py --company "銘柄名" [--level-target] [--outdir outputs/...]`.
   - パラメータスイープ:  
     `python experiments.py --companies ... --prophet-reg-lag {0|1|...} --sarimax-lags "0 1 2 3 6 9 12" --outdir outputs/...`
   - 低データ銘柄を含める場合は `--min-obs` を調整。
3. **レポート資源**
- `outputs/experiments_all_summary.md`: 現時点のベスト設定表および所見。
- 各 `outputs/experiments*/` ディレクトリ: 生CSVと `summary.json`、必要に応じて `experiment_logs.csv`.

### Prototype Runs (単体分析まとめ)
- `outputs/prototype`: 初期ベースライン。N225ログリターンを外生変数に設定し、Prophet/SARIMAXとも RMSE ≈0.178。SARIMAXのCVベストラグは1で、N225係数は≈1.34と強い正の影響を確認。
- `outputs/prototype_default`: 最新 `analysis.py` で再実行したベースライン。Prophet RMSEが ≈0.186（スケーリング変更の影響）、SARIMAXは依然 RMSE ≈0.178・lag=1。刷新後も挙動が一致することを確認。
- `outputs/prototype_default_explicit`: `--exog-use-level` でN225水準を外生変数に投入。RMSEは変わらず（Prophet/SARIMAXとも baseline と同水準）で、モデルが内部で差分を取り N225レベルを実質的にリターンへ変換していることを示唆。
- `outputs/prototype_macro`: `ALT_DATA/macro_index.csv` を外生変数に置換。RMSEが僅かに悪化し、macro指標ではN225ほどの説明力が得られない。現状ではN225ログリターンが最も安定的な外生変数。
- 以上から「SARIMAX + N225ログリターン (lag=1)」が単一銘柄分析のデフォルト構成として妥当。Prophetのラグは `experiments_stdN_reglag*` を使って別途比較するのが良い。

## 今後のTODO案
- Prophetの入力を標準化/追加共変量で再評価 (`--level-target` フラグも検証)。
- SARIMAXで季節次数 `(P,D,Q, m=12)` も探索し、Ljung–Box / ARCH など残差検定を自動化。
- 係数有意性を報告しやすくするため、`experiments.py` 出力に信頼区間や収束フラグを追記。
- 図の生成が必要になった際は、`analysis.py` を活用してベストパラメータを可視化・レポート化。

## 追加検証: 標準化の影響
- コマンド例: `python experiments.py --companies ... --standardize --outdir outputs/experiments_std_loglag0`.
- 結果: y と N225 を同時にZスコア化したところ、Prophet / SARIMAX とも 13銘柄すべてで RMSE が悪化 (`outputs/experiments_std_loglag0/summary.md`)。SARIMAX の係数有意性は概ね維持したが、値の解釈が難しくなった。
- 次の一手: N225のみを正規化する、または他の正則化（例: prior強化、ridge的制約）を検討。

### 追加入力カスタマイズ
- `--standardize-n225`: N225のみをZスコア化。`outputs/experiments_std_n225/summary.md` によると、13銘柄中6銘柄でSARIMAX CV RMSEが僅かに改善し、Prophetでも係数のスケール安定化に効果。性能への影響はごく小さいが、係数解釈がしやすいので今後のベース候補。
- `--level-target --standardize-n225`: 価格レベルを直接予測する実験 (`outputs/experiments_level_stdN225/summary.md`) はRMSEが桁違いに悪化し、SARIMAXも発散傾向。結論として、月次ログリターンを目的変数とする方針を継続。
- `--sarimax-seasonal`: 季節次数 `(0,1,1,12)` を付与したSARIMAXを `outputs/experiments_seasonal/` で検証。全銘柄で最良モデルは非季節 `(0,0,0,0)` のままで、季節成分導入時は「サンプル不足/収束せず」警告が多発しRMSEも悪化。`converged` フラグを記録して非収束解を除外可能にした。
- 今後の試行錯誤ログは各 `outputs/experiments_*/summary.md` に詳細記述済み。特に `outputs/experiments_std_n225/summary.md`（N225のみ標準化）, `outputs/experiments_seasonal/summary.md`（季節拡張）, `outputs/experiments_level_stdN225/summary.md`（レベルターゲット）を参照すると、これまでの試行錯誤の結果・改善可否が一目で追える。
- `outputs/experiments_stdN_reglag0-6/summary.md` と `outputs/prophet_lag_summary.md`: Prophetの `add_regressor` ラグを0〜6ヶ月で網羅。結果は `best_lag` が0:2銘柄, 2:3, 3:2, 4:1, 5:2, 6:3と散らばるが、RMSE差は最大でも約0.01、実質的な改善はごく小さい。N225効果は数ヶ月遅行で一部に現れるが、係数の安定性を優先すると lag=0〜2範囲で十分。
- `sarimax_diagnostics.py` + `outputs/diagnostics_stdN/`: ベストSARIMAXの残差について Ljung-Box(6/12)・ARCHテストを実施。lag=1中心のモデルでも複数銘柄で p<0.05 が出るため、さらなるAR/MA項やGARCH系アプローチ等による残差改善余地があることを確認。
- `outputs/experiments_stdN_reglag{0,1,2}/summary.md` と `outputs/experiments_prophet_reglags.md`: N225標準化を維持したまま Prophet の `add_regressor` ラグを {0,1,2} で総当たり。8/13銘柄で lag=2 が最良、ただし改善幅は0.002未満。SARIMAX側は依然 lag=1 (一部6)が優勢。
- `sarimax_diagnostics.py` + `outputs/diagnostics_stdN/`: 実験結果からベストSARIMAXを再フィットし、Ljung-Box(6,12)・ARCHテストを実施。多くの銘柄で p<0.05（残差に自己相関/条件付分散あり）が確認でき、診断用のCSV/MDを出力。
