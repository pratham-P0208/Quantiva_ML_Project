import numpy as np
import pandas as pd

TRADING_DAYS = 252

def portfolio_history(returns, weights):
    aligned = returns[weights.index].dropna()
    w = weights.reindex(aligned.columns).to_numpy(dtype=float)
    return aligned @ w

def calculate_risk(portfolio_returns, confidence=0.95):
    r = pd.Series(portfolio_returns).dropna()
    if r.empty:
        raise ValueError("No returns available for risk analysis.")
    wealth = (1+r).cumprod()
    peak = wealth.cummax()
    drawdown = wealth/peak - 1
    quantile = float(r.quantile(1-confidence))
    tail = r[r <= quantile]
    return {
        "volatility":float(r.std(ddof=1)*np.sqrt(TRADING_DAYS)),
        "var_95":float(max(0,-quantile)),
        "cvar_95":float(max(0,-tail.mean())) if len(tail) else float(max(0,-quantile)),
        "max_drawdown":float(drawdown.min())
    }
