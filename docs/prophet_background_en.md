# Prophet Model Structure (English Summary)

Prophet is a decomposable time-series model consisting of trend, seasonality, holiday, and error components:

$$
y(t) = g(t) + s(t) + h(t) + \epsilon_t
$$

where $\epsilon_t \sim N(0,\sigma^2)$. Unlike ARIMA, Prophet does not require differencing and is robust to outliers/missing data.

## 1. Trend Component $g(t)$
Prophet uses piecewise functions to allow trend changes at predefined changepoints $s_j$.

### 1.1 Piecewise growth models
1. **Piecewise logistic growth** (supports capacity $C(t)$ for saturation).
2. **Piecewise linear trend** (default when no capacity is specified).

Changepoint effects $\delta$ are regularized via `changepoint_prior_scale` to control flexibility.

## 2. Seasonality Component $s(t)$
Modeled by Fourier series with order $N$:
$$
s(t) = X(t)\beta
$$
Higher $N$ increases complexity; `seasonality_prior_scale` controls the strength via a Normal prior. Seasonality can be additive or multiplicative relative to the trend.

## 3. Holidays/Event Component $h(t)$
Modeled with indicator functions:
$$
h(t) = \sum_{i=1}^L \kappa_i \mathbf{1}(t \in D_i)
$$
Each holiday $i$ has its own effect $\kappa_i$ with prior `holidays_prior_scale`. Window parameters (`lower_window`, `upper_window`) extend effects before/after the holiday.

## Optimization
After specifying the “shapes” (changepoints, Fourier order, holiday windows), Prophet estimates coefficients ($\delta, \beta, \kappa$) using MAP with the chosen priors. This yields interpretable components while handling trend shifts and multiple seasonalities.
