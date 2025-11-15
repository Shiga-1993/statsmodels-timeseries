# Multi-company Parameter Sweep (Log-return target)

## Prophet (N225 lag 0) Best Configurations
| Company | Mode | CPS | SPS | Test RMSE | N225 coef |
| --- | --- | --- | --- | --- | --- |
| アーバネットコーポレーション | multiplicative | 0.2 | 5.0 | 0.0531 | -1.8900 |
| エスリード | additive | 0.2 | 15.0 | 0.0528 | 0.1658 |
| コスモスイニシア | multiplicative | 0.2 | 5.0 | 0.0934 | -13.6905 |
| コーセーアールイー | additive | 0.05 | 15.0 | 0.0460 | 0.0234 |
| ゴールドクレスト | multiplicative | 0.2 | 5.0 | 0.0544 | 0.5731 |
| シーラホールディングス | additive | 0.2 | 5.0 | 0.1139 | 0.0056 |
| セントラル総合開発 | multiplicative | 0.2 | 5.0 | 0.0463 | -35.9634 |
| ディア・ライフ | multiplicative | 0.2 | 15.0 | 0.0925 | 9.5182 |
| プロパスト | additive | 0.05 | 5.0 | 0.0817 | 0.4936 |
| 日本エスコン | multiplicative | 0.05 | 15.0 | 0.0420 | 0.0318 |
| 日神グループホールディングス | multiplicative | 0.2 | 5.0 | 0.0411 | 0.0038 |
| 明和地所 | multiplicative | 0.2 | 15.0 | 0.0607 | -9.1303 |
| 明豊エンタープライズ | multiplicative | 0.05 | 5.0 | 0.1111 | -0.0978 |

## Prophet (N225 lag 1) Best Configurations
| Company | Mode | CPS | SPS | Test RMSE | N225 coef |
| --- | --- | --- | --- | --- | --- |
| アーバネットコーポレーション | multiplicative | 0.05 | 15.0 | 0.0528 | -30.4701 |
| エスリード | multiplicative | 0.05 | 5.0 | 0.0565 | 0.4464 |
| コスモスイニシア | multiplicative | 0.2 | 5.0 | 0.0930 | -57.5085 |
| コーセーアールイー | additive | 0.2 | 15.0 | 0.0596 | 0.9343 |
| ゴールドクレスト | multiplicative | 0.05 | 5.0 | 0.0560 | 0.5346 |
| シーラホールディングス | additive | 0.2 | 5.0 | 0.1145 | 0.7527 |
| セントラル総合開発 | multiplicative | 0.05 | 5.0 | 0.0607 | -0.4293 |
| ディア・ライフ | multiplicative | 0.05 | 5.0 | 0.0958 | -16.5203 |
| プロパスト | multiplicative | 0.2 | 5.0 | 0.0875 | -20.8912 |
| 日本エスコン | multiplicative | 0.05 | 5.0 | 0.0495 | 0.0757 |
| 日神グループホールディングス | multiplicative | 0.05 | 5.0 | 0.0411 | 1.1432 |
| 明和地所 | multiplicative | 0.05 | 15.0 | 0.0720 | 0.1203 |
| 明豊エンタープライズ | additive | 0.05 | 5.0 | 0.1014 | 1.1929 |

## SARIMAX (lags 0/1/2/3/6/9/12) Best Configurations per Company
| Company | Order | Lag | CV RMSE | N225 coef | p-value |
| --- | --- | --- | --- | --- | --- |
| アーバネットコーポレーション | (1, 0, 1) | 1 | 0.0814 | 1.0522 | 3.812e-14 |
| エスリード | (1, 0, 1) | 1 | 0.0856 | 0.9737 | 1.127e-19 |
| コスモスイニシア | (2, 0, 1) | 1 | 0.1574 | 1.3452 | 1.470e-15 |
| コーセーアールイー | (1, 1, 1) | 1 | 0.0953 | 0.8723 | 5.173e-06 |
| ゴールドクレスト | (2, 0, 1) | 1 | 0.0948 | 0.8266 | 6.838e-32 |
| シーラホールディングス | (1, 0, 1) | 1 | 0.0794 | 0.6619 | 1.014e-10 |
| セントラル総合開発 | (1, 0, 1) | 12 | 0.1083 | -0.1964 | 3.483e-01 |
| ディア・ライフ | (1, 1, 1) | 1 | 0.1123 | 1.2019 | 1.176e-18 |
| プロパスト | (2, 0, 1) | 6 | 0.1553 | -0.2110 | 5.266e-01 |
| 日本エスコン | (1, 0, 1) | 1 | 0.3094 | 1.3153 | 1.270e-01 |
| 日神グループホールディングス | (2, 0, 1) | 1 | 0.0869 | 0.9925 | 9.803e-26 |
| 明和地所 | (1, 1, 1) | 1 | 0.0890 | 0.8927 | 6.052e-29 |
| 明豊エンタープライズ | (1, 0, 1) | 12 | 0.1605 | 0.0960 | 6.945e-01 |

## Aggregate Observations
- Prophet lag 0 preferred multiplicative mode in 9 / 13 companies; lag 1 kept the same mode distribution but improved RMSE for only 3 companies.
- SARIMAX best lags: {1: 10, 12: 2, 6: 1}. One-month lag dominates; longer lags (6 or 12) only helped isolated names.
- Among SARIMAX best fits, 9 / 13 coefficients are statistically significant (p<0.05).
- Detailed experiment CSVs: `outputs/experiments_all_loglag0/` and `outputs/experiments_all_reglag1/`.