import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def train_and_predict(feature_frames, returns, test_fraction=0.20):
    """
    Fit one Random Forest model per ticker using chronological split.
    Forecast is for the next observation after the final available feature row.
    """
    predictions = {}
    diagnostics = []
    for ticker, frame in feature_frames.items():
        feature_cols = [c for c in frame.columns if c not in ("target","ticker")]
        labelled = frame.dropna(subset=["target"] + feature_cols)
        latest_features = frame.dropna(subset=feature_cols)
        if len(labelled) < 250 or latest_features.empty:
            predictions[ticker] = float(returns[ticker].tail(60).mean())
            diagnostics.append({"Ticker":ticker,"Rows":len(labelled),"MAE":np.nan,"RMSE":np.nan,"Status":"Fallback: insufficient rows"})
            continue
        split = int(len(labelled)*(1-test_fraction))
        train, test = labelled.iloc[:split], labelled.iloc[split:]
        model = RandomForestRegressor(
            n_estimators=120, max_depth=7, min_samples_leaf=8,
            max_features=0.8, random_state=42, n_jobs=-1
        )
        model.fit(train[feature_cols], train["target"])
        pred_test = model.predict(test[feature_cols])
        mae = mean_absolute_error(test["target"], pred_test)
        rmse = mean_squared_error(test["target"], pred_test) ** 0.5
        # Refit on all available labelled observations, then forecast next session.
        final_model = RandomForestRegressor(
            n_estimators=120, max_depth=7, min_samples_leaf=8,
            max_features=0.8, random_state=42, n_jobs=-1
        )
        final_model.fit(labelled[feature_cols], labelled["target"])
        latest_x = latest_features[feature_cols].iloc[[-1]]
        raw_prediction = float(final_model.predict(latest_x)[0])
        # Shrink noisy one-day point forecast toward recent realized mean.
        recent_mean = float(returns[ticker].tail(60).mean())
        predictions[ticker] = 0.5*raw_prediction + 0.5*recent_mean
        diagnostics.append({"Ticker":ticker,"Rows":len(frame),"MAE":mae,"RMSE":rmse,"Status":"Trained"})
    info = pd.DataFrame(diagnostics)
    if not info.empty:
        info[["MAE","RMSE"]] = info[["MAE","RMSE"]].round(6)
    return pd.Series(predictions, dtype=float), info
