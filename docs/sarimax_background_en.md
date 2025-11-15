# SARIMAX Model Structure (English Summary)

SARIMAX (Seasonal ARIMA with eXogenous regressors) extends ARIMA to incorporate both seasonal structure and external regressors.

## 1. Components
SARIMAX is specified as $(p,d,q)(P,D,Q)_m$, where:
- $p,d,q$: non-seasonal AR, differencing, MA orders
- $P,D,Q$: seasonal AR, differencing, MA orders
- $m$: seasonal period (e.g., 12 for monthly data)

## 2. Operator Form
Using lag operator $L$:
$$
\phi_p(L)\tilde{\phi}_P(L^m)\nabla^d \nabla_m^D X_t = A(t) + \theta_q(L)\tilde{\theta}_Q(L^m)\epsilon_t
$$
where:
- $\phi_p(L)$, $\theta_q(L)$ are non-seasonal AR/MA polynomials
- $\tilde{\phi}_P(L^m)$, $\tilde{\theta}_Q(L^m)$ are seasonal AR/MA polynomials
- $\nabla^d = (1-L)^d$, $\nabla_m^D = (1-L^m)^D$
- $A(t)$ includes deterministic terms (intercept/trend)
- $\epsilon_t$ is white noise

## 3. Exogenous Regressors
With external regressors $x_t$:
$$
Y_t = \beta^T x_t + U_t,\quad U_t \text{ follows SARIMA}(p,d,q)(P,D,Q)_m
$$
This allows linear effects from exogenous series while residuals are modeled by SARIMA.

## 4. Order Selection & Estimation
- Differencing orders ($d,D$) often chosen via stationarity tests (ADF) or prior knowledge.
- AR/MA orders ($p,q,P,Q$) identified via ACF/PACF or automated search (AIC/BIC, `auto_arima`).
- Coefficients $(\varphi,\theta,\Phi,\Theta,\beta)$ estimated via maximum likelihood (state-space / Kalman filter implementation in `statsmodels`), optionally enforcing stationarity/invertibility.

## Interpretation
SARIMAX retains statistical rigor: MAC-level AR/MA structure handles autocorrelation, while exogenous regressors explain additional variation. Seasonal differencing captures periodicity, and regular differencing stabilizes trends. This unified framework provides interpretable coefficients and out-of-sample forecasts with external drivers.
