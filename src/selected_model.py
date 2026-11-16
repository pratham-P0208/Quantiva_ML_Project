from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ARTIFACT = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"

def predict_selected(feature_frames, returns):
    if not ARTIFACT.exists():
        return None
    bundle = joblib.load(ARTIFACT)
    rows = []
    for ticker, frame in feature_frames.items():
        x = frame.drop(columns=["target", "ticker"], errors="ignore").dropna()
        if x.empty:
            continue
        last = x.iloc[[-1]].copy()
        last["ticker"] = ticker
        rows.append(last)
    if not rows:
        return None
    latest = pd.concat(rows)
    cols = bundle["features"]
    latest = latest.reindex(columns=cols)
    values = bundle["pipeline"].predict(latest)
    predictions = pd.Series(values, index=latest["ticker"], dtype=float).reindex(returns.columns)
    predictions = predictions.fillna(returns.tail(60).mean())
    metrics = bundle["metrics"]
    info = pd.DataFrame([{"Model": bundle["name"], **metrics}])
    return predictions, info
