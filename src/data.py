from pathlib import Path
import pandas as pd

MACRO_COLUMNS = [
    "USDINR_Return", "Brent_Return", "US_10Y_Change_bps",
    "NIFTY50_Return", "INDIA_VIX_Change_Points", "SP500_Return",
    "RBI_Repo_Rate_Change_bps", "RBI_Repo_Change_6M_bps"
]

def load_data(data_dir):
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "portfolio_modeling_dataset.csv", parse_dates=["Date"])
    returns = pd.read_csv(data_dir / "daily_stock_returns.csv", parse_dates=["Date"])
    metadata = pd.read_csv(data_dir / "asset_metadata.csv")
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    returns = returns.sort_values("Date").drop_duplicates("Date").set_index("Date")
    returns = returns.apply(pd.to_numeric, errors="coerce")
    returns = returns.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    return df, returns, metadata

def get_asset_columns(returns):
    return list(returns.columns)
