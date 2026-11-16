import numpy as np
import pandas as pd
from .data import MACRO_COLUMNS

def build_feature_frame(modeling_df, asset_columns):
    """
    Build strictly historical features at date t to predict the return at t+1.
    All rolling features use values available through t; target is shifted -1.
    """
    df = modeling_df.sort_index().copy()
    frames = {}
    macro_cols = [c for c in MACRO_COLUMNS if c in df.columns]
    for ticker in asset_columns:
        s = df[ticker].astype(float)
        x = pd.DataFrame(index=df.index)
        x["lag_1"] = s
        x["lag_5"] = s.shift(5)
        x["lag_10"] = s.shift(10)
        x["mean_5"] = s.rolling(5).mean()
        x["mean_20"] = s.rolling(20).mean()
        x["mean_60"] = s.rolling(60).mean()
        x["vol_5"] = s.rolling(5).std()
        x["vol_20"] = s.rolling(20).std()
        x["vol_60"] = s.rolling(60).std()
        x["momentum_20"] = (1+s).rolling(20).apply(np.prod, raw=True)-1
        x["momentum_5"] = (1+s).rolling(5).apply(np.prod, raw=True)-1
        x["momentum_60"] = (1+s).rolling(60).apply(np.prod, raw=True)-1
        x["mean_10"] = s.rolling(10).mean()
        x["vol_10"] = s.rolling(10).std()
        x["vol_ratio_5_20"] = x["vol_5"] / (x["vol_20"] + 1e-8)
        x["return_zscore_20"] = (s-x["mean_20"]) / (x["vol_20"] + 1e-8)
        x["positive_days_20"] = s.gt(0).rolling(20).mean()
        x["downside_vol_20"] = s.clip(upper=0).pow(2).rolling(20).mean().pow(0.5)
        x["market_relative_20"] = x["mean_20"] - df["NIFTY50_Return"].rolling(20).mean() if "NIFTY50_Return" in df else x["mean_20"]
        for col in macro_cols:
            x[f"macro_{col}"] = df[col].astype(float)
        x["target"] = s.shift(-1)
        x["ticker"] = ticker
        frames[ticker] = x.replace([np.inf,-np.inf], np.nan)
    return frames
