import numpy as np
import pandas as pd
from backend.server import app, data
from src.risk import portfolio_history, calculate_risk
from src.optimizer import optimize_portfolio

def test_dashboard_matches_data():
    _,returns,_=data()
    with app.test_client() as c:
        r=c.get('/api/dashboard')
    assert r.status_code==200
    assert r.json['stats']['stocks']==len(returns.columns)
    assert r.json['stats']['observations']==len(returns)

def test_risk_matches_original_functions():
    _,returns,_=data()
    tickers=list(returns.columns[:3])
    weights=pd.Series(1/len(tickers),index=tickers)
    expected=calculate_risk(portfolio_history(returns[tickers],weights))
    with app.test_client() as c:
        r=c.post('/api/risk',json={'tickers':tickers})
    assert r.status_code==200
    for k,v in expected.items():
        assert np.isclose(r.json['metrics'][k],v)

def test_optimization_endpoint_uses_original_function(monkeypatch):
    import backend.server as api
    _,returns,_=data()
    tickers=list(returns.columns[:5])
    pred=pd.Series(0.001,index=returns.columns)
    info=pd.DataFrame([{'Model':'test'}])
    monkeypatch.setattr(api,'predictions',lambda:(pred,info))
    expected=optimize_portfolio(returns[tickers],pred.reindex(tickers),objective='Minimum Volatility',max_weight=.25)
    with app.test_client() as c:
        r=c.post('/api/optimize',json={'tickers':tickers,'objective':'Minimum Volatility','max_weight':.25})
    assert r.status_code==200,r.json
    assert np.isclose(r.json['volatility'],expected['volatility'])
    assert np.isclose(r.json['return'],expected['return'])
