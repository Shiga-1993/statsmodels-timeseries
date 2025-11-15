# Parameter Sweep Summary

## Prophet Best Settings per Company
| Company | Mode | CPS | SPS | Test RMSE | Test MAE | N225 coef |
| --- | --- | --- | --- | --- | --- | --- |
| 日本エスコン | multiplicative | 0.05 | 15.0 | 0.0420 | 0.0376 | 0.0318 |
| 日神グループホールディングス | multiplicative | 0.2 | 5.0 | 0.0411 | 0.0357 | 0.0038 |
| 明豊エンタープライズ | multiplicative | 0.05 | 5.0 | 0.1111 | 0.0857 | -0.0978 |

## SARIMAX Best Settings per Company
| Company | Order | Lag | CV RMSE | In-sample RMSE | N225 coef | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| 日本エスコン | (1, 0, 1) | 1 | 0.3094 | 0.5824 | 1.3153 | 1.270e-01 |
| 日神グループホールディングス | (2, 0, 1) | 1 | 0.0869 | 0.0934 | 0.9925 | 9.803e-26 |
| 明豊エンタープライズ | (2, 0, 1) | 1 | 0.1742 | 0.1783 | 1.3718 | 8.095e-25 |

## Aggregate SARIMAX Diagnostics
- Positive N225 coefficients: 88.9% of tested combos
- Statistically significant (p<0.05): 37.8% of combos
- CV RMSE by lag:
  - Lag 0: 0.2213
  - Lag 1: 0.1935
  - Lag 2: 0.2110
  - Lag 3: 0.1999
  - Lag 6: 0.2086

## Notes
- Prophet sweeps consistently preferred multiplicative seasonality; changepoint prior 0.05-0.2 had little effect on error except for volatile series.
- SARIMAX best-performing models used lag=1 for N225 across all companies, reinforcing a one-month transmission effect.
- Smaller, newer listings lacked enough history (under 150 months) for stable estimation; see experiment_logs.csv for skips.