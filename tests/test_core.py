import numpy as np
import pandas as pd
from src.features import build_feature_frame
from src.risk import calculate_risk, portfolio_history

def test_risk_outputs_are_finite():
    r = pd.Series(np.random.default_rng(1).normal(0.0003,0.01,500))
    out = calculate_risk(r)
    assert all(np.isfinite(v) for v in out.values())

def test_portfolio_history_weights():
    df = pd.DataFrame({"A":[.01,.02],"B":[-.01,.01]})
    w = pd.Series({"A":.5,"B":.5})
    result = portfolio_history(df,w)
    assert np.allclose(result.to_numpy(),[0,.015])
