
from pathlib import Path
import joblib

from src.data import load_data
from src.features import build_feature_frame
from src.model import train_and_predict

ROOT = Path(__file__).resolve().parent

# Use exactly the same data and feature engineering as Flask
df, returns, _ = load_data(ROOT / "data")
features = build_feature_frame(df, list(returns.columns))

# Run the original Random Forest implementation unchanged
forecasts, diagnostics = train_and_predict(features, returns)

# Save the exact outputs used by the optimizer
output = ROOT / "models" / "rf_forecasts.joblib"
output.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "forecasts": forecasts,
    "diagnostics": diagnostics,
}, output)

print("Saved:", output)
print("Number of stocks:", len(forecasts))
print("Forecasts:")
print(forecasts)
