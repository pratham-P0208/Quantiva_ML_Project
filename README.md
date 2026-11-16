# Smart Portfolio Optimizer
### Machine Learning-Based Portfolio Optimization and Risk Analysis

A Streamlit academic project using the supplied 2018–2026 daily returns and lagged macroeconomic dataset for 48 Indian equity assets.

## 1. Features
- Four-page interface: Dashboard, Optimize Portfolio, Risk Analysis, About Project.
- Per-stock Random Forest regression for next-observation return estimation.
- Lagged return, rolling mean, rolling volatility and momentum features.
- Lagged macroeconomic features.
- Chronological train/test split.
- Ledoit-Wolf covariance shrinkage.
- Long-only portfolio optimization: maximum Sharpe ratio and minimum volatility.
- Historical annualized volatility, 95% VaR, 95% CVaR and maximum drawdown.

## 2. Project structure
```
smart_portfolio_optimizer/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── asset_metadata.csv
│   ├── daily_stock_returns.csv
│   ├── data_quality_summary.csv
│   ├── macro_derived_features_lagged.csv
│   ├── macro_features_lagged.csv
│   ├── macro_source_age_audit.csv
│   └── portfolio_modeling_dataset.csv
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── features.py
│   ├── model.py
│   ├── optimizer.py
│   └── risk.py
└── tests/
    └── test_core.py
```

## 3. Setup (Windows)
1. Install Python 3.10 or later.
2. Extract the ZIP folder.
3. Open Command Prompt / VS Code terminal inside the project folder.
4. Create a virtual environment:
   `python -m venv .venv`
5. Activate it:
   `.venv\\Scripts\\activate`
6. Install dependencies:
   `pip install -r requirements.txt`
7. Run the website:
   `streamlit run app.py`
8. Open the local URL shown in the terminal (usually `http://localhost:8501`).

## 4. Run tests
`pytest -q`

## 5. Feature engineering
At date t, the model uses:
- Daily return at t (lag_1)
- Return at t-5 and t-10
- 5, 20 and 60 observation rolling means
- 5, 20 and 60 observation rolling standard deviations
- 20-observation compounded momentum
- Supplied lagged macro derived features

Target: stock return at t+1. All rolling calculations use information through t. The first 60 observations per asset are removed because long-window features are not yet available.

## 6. Model design
A separate RandomForestRegressor is trained for each ticker. The final 20% of each ticker's labelled feature rows is used as a chronological holdout for MAE and RMSE diagnostics. The model is then refitted on all available labelled rows and used to forecast the next observation. The one-day forecast is blended 50:50 with the trailing 60-observation mean to reduce forecast extremity.

The model's MAE and RMSE are diagnostic measures only. They do not establish profitability.

## 7. Optimization design
- Expected daily ML forecasts are annualized by multiplying by 252.
- Covariance is estimated from up to the latest 756 daily observations using Ledoit-Wolf shrinkage and annualized by 252.
- Weights are long-only, sum to 100%, and each selected asset is capped at the chosen maximum weight.
- Maximum Sharpe uses the user-entered annual risk-free rate.
- Minimum Volatility minimizes annualized portfolio standard deviation.

## 8. Risk definitions
- Annualized volatility = daily standard deviation × sqrt(252).
- Historical VaR 95% = negative 5th percentile of daily portfolio returns, floored at zero.
- Historical CVaR 95% = negative mean of returns at or below the 5th percentile, floored at zero.
- Maximum drawdown = minimum decline of cumulative wealth from its previous peak.

## 9. Important data and research limitations
- The provided data quality report has an internal discrepancy: its summary says maximum macro source age is 11 days, while the row-level audit maximum is 7 days. Both supplied files are retained unchanged.
- The dataset contains daily returns, not raw adjusted prices. The dashboard's equal-weighted growth series is a return-index illustration.
- The model estimates the next observation, not a dependable long-term return.
- A stronger research evaluation should use rolling walk-forward backtesting, transaction costs, turnover, benchmark comparison, and a fully untouched final test period.
- Results are educational and are not investment advice or trading recommendations.

## 10. Suggested project report chapters
1. Abstract
2. Introduction
3. Problem Statement and Objectives
4. Literature Review
5. Dataset Description
6. Data Preprocessing
7. Feature Engineering
8. Machine Learning Model
9. Portfolio Optimization Methodology
10. Risk Metrics
11. System Architecture and Interface
12. Results and Discussion
13. Limitations
14. Conclusion and Future Scope
15. References


## UI update
The updated version uses a high-contrast light theme, clear optimization steps, a visible optimization button, a readable allocation table (without Pandas Styler/Jinja2), and consistent chart colors. After replacing the project folder, restart Streamlit with `Ctrl+C` and `streamlit run app.py`.

## NEW: Advanced feature engineering and 5-model comparison
Run `jupyter notebook Feature_Engineering_Model_Comparison.ipynb` from the project folder, then **Run All**. The notebook builds additional momentum, volatility, relative-return and downside-risk features; selects up to 22 numeric features using training-only F-tests; compares Ridge, Elastic Net, Random Forest, Extra Trees and Histogram Gradient Boosting; selects the lowest validation MAE; and reports final untouched test MAE, RMSE, R² and directional accuracy. It saves `models/best_model.joblib` for automatic use by Streamlit. If this artifact is missing, the app uses its original per-stock Random Forest fallback. This is a holdout evaluation, not a full walk-forward portfolio backtest. The model is a research prototype and positive investment returns are not guaranteed.
