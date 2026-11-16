import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252

def optimize_portfolio(returns, expected_daily_returns, objective="Maximum Sharpe Ratio",
                       risk_free_rate=0.065, max_weight=0.25):
    clean = returns.dropna().tail(756)
    tickers = list(clean.columns)
    n = len(tickers)
    if n < 2:
        raise ValueError("At least two assets are required.")
    mu = expected_daily_returns.reindex(tickers).fillna(clean.mean()).to_numpy() * TRADING_DAYS
    cov = LedoitWolf().fit(clean.to_numpy()).covariance_ * TRADING_DAYS
    cov = cov + np.eye(n)*1e-9
    def stats(w):
        ret = float(w @ mu)
        vol = float(np.sqrt(max(w @ cov @ w, 1e-12)))
        sharpe = (ret-risk_free_rate)/vol
        return ret, vol, sharpe
    if objective == "Minimum Volatility":
        fun = lambda w: np.sqrt(max(w @ cov @ w, 1e-12))
    else:
        fun = lambda w: -((w @ mu-risk_free_rate)/np.sqrt(max(w @ cov @ w,1e-12)))
    x0 = np.repeat(1/n,n)
    bounds = [(0.0, float(max_weight)) for _ in range(n)]
    result = minimize(fun, x0, method="SLSQP", bounds=bounds,
                      constraints=[{"type":"eq","fun":lambda w: np.sum(w)-1}],
                      options={"maxiter":1000,"ftol":1e-10})
    if not result.success:
        raise ValueError(f"Optimizer did not converge: {result.message}")
    ret, vol, sharpe = stats(result.x)
    return {"weights":pd.Series(result.x,index=tickers),"return":ret,
            "volatility":vol,"sharpe":sharpe}
