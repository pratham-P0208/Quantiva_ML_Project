"""Flask REST adapter. Original src/ models, features, risk, optimizer and CSVs are untouched."""
import sys
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import load_data, get_asset_columns
from src.features import build_feature_frame
from src.model import train_and_predict
from src.selected_model import predict_selected, ARTIFACT
from src.optimizer import optimize_portfolio
from src.risk import calculate_risk, portfolio_history

app = Flask(__name__)
CORS(app, origins=[v.strip() for v in __import__('os').environ.get('FRONTEND_ORIGINS','http://localhost:3000').split(',')])

@lru_cache(maxsize=1)
def data():
    return load_data(ROOT/'data')

@lru_cache(maxsize=1)
def predictions():
    df, returns, _ = data()
    features = build_feature_frame(df, list(returns.columns))
    selected = predict_selected(features, returns)
    return selected if selected is not None else train_and_predict(features, returns)

def clean_number(v):
    try:
        n=float(v)
        return n if np.isfinite(n) else None
    except (ValueError,TypeError):
        return None

def error(message, status=400):
    return jsonify({'error':message}),status

def selected_assets(payload, minimum=1):
    _,returns,_=data()
    items=payload.get('tickers',[])
    if not isinstance(items,list) or len(items)<minimum or len(items)>len(returns.columns) or len(items)!=len(set(items)):
        raise ValueError(f'Select between {minimum} and {len(returns.columns)} distinct stocks.')
    if any(not isinstance(x,str) or x not in returns.columns for x in items):
        raise ValueError('Unknown stock ticker in selection.')
    return items

@app.get('/api/health')
def health():
    return jsonify({'status':'ok','saved_model_present':ARTIFACT.exists()})

@app.get('/api/dashboard')
def dashboard():
    _,returns,metadata=data()
    growth=(1+returns.fillna(0).mean(axis=1)).cumprod()
    return jsonify({
      'stats':{'stocks':len(returns.columns),'observations':len(returns),'first_date':returns.index.min().strftime('%d %b %Y'),'last_date':returns.index.max().strftime('%d %b %Y')},
      'assets':metadata.replace({np.nan:None}).to_dict(orient='records'),
      'growth':[{'date':d.strftime('%Y-%m-%d'),'value':clean_number(v)} for d,v in growth.items()],
      'saved_model_present':ARTIFACT.exists()
    })

@app.post('/api/optimize')
def optimize():
    try:
        payload=request.get_json(silent=True) or {}
        tickers=selected_assets(payload,2)
        objective=payload.get('objective','Maximum Sharpe Ratio')
        if objective not in ('Maximum Sharpe Ratio','Minimum Volatility'):
            return error('Invalid optimization objective.')
        cap=float(payload.get('max_weight',0.25))
        rf=float(payload.get('risk_free_rate',0.065))
        if not (0.05<=cap<=1 and 0<=rf<=0.2) or len(tickers)*cap<1-1e-9:
            return error('Invalid risk-free rate or infeasible allocation cap.')
        _,returns,metadata=data()
        forecast,info=predictions()
        result=optimize_portfolio(returns[tickers],forecast.reindex(tickers),objective=objective,risk_free_rate=rf,max_weight=cap)
        names=metadata.set_index('Ticker')['Company'].to_dict()
        weights=result['weights'].sort_values(ascending=False)
        allocation=[{'ticker':t,'company':names.get(t,t),'weight':float(w),'percentage':round(float(w)*100,2)} for t,w in weights.items() if w>0.0001]
        model_info=json_safe_records(info)
        return jsonify({'return':result['return'],'volatility':result['volatility'],'sharpe':result['sharpe'],'allocation':allocation,'model_info':model_info,'model_source':'saved' if ARTIFACT.exists() else 'Random Forest fallback'})
    except (ValueError,TypeError,KeyError) as exc:
        return error(str(exc))
    except Exception:
        app.logger.exception('Optimization failed')
        return error('Optimization failed. Check backend logs.',500)

def json_safe_records(df):
    return json.loads(df.to_json(orient='records'))

@app.post('/api/risk')
def risk():
    try:
        payload=request.get_json(silent=True) or {}
        tickers=selected_assets(payload)
        _,returns,_=data()
        weights=pd.Series(1/len(tickers),index=tickers)
        hist=portfolio_history(returns[tickers],weights)
        metrics=calculate_risk(hist)
        wealth=(1+hist).cumprod()
        return jsonify({'metrics':metrics,'growth':[{'date':d.strftime('%Y-%m-%d'),'value':clean_number(v)} for d,v in wealth.items()], 'distribution':[clean_number(v) for v in hist.values]})
    except (ValueError,TypeError,KeyError) as exc:
        return error(str(exc))

if __name__=='__main__':
    app.run(host='127.0.0.1',port=5000,debug=False)
