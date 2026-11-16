import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.data import load_data, get_asset_columns
from src.features import build_feature_frame
from src.model import train_and_predict
from src.selected_model import predict_selected
from src.optimizer import optimize_portfolio
from src.risk import calculate_risk, portfolio_history

st.set_page_config(
    page_title="Smart Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-contrast, simple light theme
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #F5F7FA; color: #172B4D; }
[data-testid="stHeader"] { background: #F5F7FA; }
[data-testid="stSidebar"] { background: #17324D; }
[data-testid="stSidebar"] * { color: #FFFFFF !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { color: #FFFFFF !important; }
.block-container { padding-top: 1.7rem; padding-bottom: 2.5rem; max-width: 1500px; }
h1, h2, h3, p, label, .stMarkdown { color: #172B4D; }
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DCE3EC;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(20,40,70,.05);
}
div[data-testid="stMetric"] label { color: #52647A !important; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #17324D !important; }
.stButton > button[kind="primary"] {
    background: #1769AA;
    border: 1px solid #1769AA;
    color: white;
    font-weight: 650;
    border-radius: 8px;
    min-height: 2.8rem;
}
.stButton > button[kind="primary"]:hover { background: #0F5289; border-color: #0F5289; }
div[data-testid="stAlert"] { border-radius: 10px; }
[data-testid="stDataFrame"] { background: white; }
.small-note { color: #52647A; font-size: .9rem; }
.section-card {
    background: white; border: 1px solid #DCE3EC;
    border-radius: 12px; padding: 16px 18px; margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_data(ROOT / "data")

@st.cache_resource(show_spinner=False)
def get_model_outputs():
    df, returns, metadata = get_data()
    features = build_feature_frame(df, returns.columns.tolist())
    selected = predict_selected(features, returns)
    return selected if selected is not None else train_and_predict(features, returns)

df, returns, metadata = get_data()
assets = get_asset_columns(returns)
companies = metadata.set_index("Ticker")["Company"].to_dict()

st.sidebar.title("📈 SMART PORTFOLIO")
st.sidebar.caption("Machine Learning • Optimization • Risk")
st.sidebar.divider()
page = st.sidebar.radio(
    "MAIN MENU",
    ["Dashboard", "Optimize Portfolio", "Risk Analysis", "About Project"],
    label_visibility="visible"
)
st.sidebar.divider()
st.sidebar.caption("DATASET")
st.sidebar.caption(f"{len(assets)} stocks")
st.sidebar.caption(f"{len(returns):,} daily observations")
st.sidebar.caption(
    f"{returns.index.min():%d %b %Y} – {returns.index.max():%d %b %Y}"
)

if page == "Dashboard":
    st.title("Portfolio Dashboard")
    st.write("Overview of the historical stock universe used by the machine-learning system.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stocks available", f"{len(assets)}")
    c2.metric("Trading observations", f"{len(returns):,}")
    c3.metric("First date", returns.index.min().strftime("%d %b %Y"))
    c4.metric("Latest date", returns.index.max().strftime("%d %b %Y"))

    st.subheader("Historical stock universe")
    display = metadata.copy()
    display.columns = ["Ticker", "Company", "Sector", "First price date", "Last price date"]
    st.dataframe(display, use_container_width=True, hide_index=True)

    st.subheader("Equal-weighted historical growth")
    growth = (1 + returns.fillna(0).mean(axis=1)).cumprod()
    chart = pd.DataFrame({"Date": growth.index, "Growth of ₹1": growth.values})
    fig = px.line(chart, x="Date", y="Growth of ₹1", template="plotly_white")
    fig.update_traces(line=dict(color="#1769AA", width=2.5))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=25, b=10),
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(color="#172B4D"))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Historical illustration only. It is not a backtested strategy or a guarantee of future returns.")

elif page == "Optimize Portfolio":
    st.title("Optimize Your Portfolio")
    st.write("Select stocks, choose what you want to optimize, and click the button to calculate portfolio weights.")

    st.markdown("""
    <div class="section-card">
      <b>How to use this page</b><br>
      <span style="color:#52647A">1. Select at least two stocks &nbsp; → &nbsp; 2. Choose an optimization goal
      &nbsp; → &nbsp; 3. Set the maximum allocation per stock &nbsp; → &nbsp; 4. Click Optimize Portfolio.</span>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.subheader("Step 1 — Choose your stocks")
        selected = st.multiselect(
            "Stocks to include",
            options=assets,
            default=assets[:10],
            format_func=lambda x: f"{x} — {companies.get(x, x)}",
            help="Choose the companies that the optimizer is allowed to include."
        )

        st.subheader("Step 2 — Choose your optimization goal")
        objective = st.radio(
            "Optimization method",
            ["Maximum Sharpe Ratio", "Minimum Volatility"],
            horizontal=True,
            help="Maximum Sharpe seeks a stronger estimated return per unit of risk. Minimum Volatility seeks the lowest estimated volatility."
        )
        if objective == "Maximum Sharpe Ratio":
            st.info("Maximum Sharpe Ratio: the optimizer balances estimated return against portfolio volatility.")
        else:
            st.info("Minimum Volatility: the optimizer searches for the lowest-volatility allocation among your selected stocks.")

        x, y = st.columns(2)
        with x:
            max_weight = st.slider("Maximum weight in any one stock", 5, 100, 25, 5) / 100
        with y:
            risk_free = st.number_input(
                "Annual risk-free rate (%)", min_value=0.0, max_value=20.0,
                value=6.5, step=0.25
            ) / 100

        st.caption(f"Selected stocks: {len(selected)}. Allocation is long-only (no short selling) and totals 100%.")
        run = st.button("⚙️  OPTIMIZE PORTFOLIO", type="primary", use_container_width=True)

    if run:
        if len(selected) < 2:
            st.error("Please select at least two stocks.")
        elif len(selected) * max_weight < 1 - 1e-9:
            st.error("The maximum weight is too low to allocate 100%. Increase the cap or select more stocks.")
        else:
            try:
                with st.spinner("Training the ML models and calculating the optimal allocation. The first run may take a few minutes..."):
                    predictions, model_info = get_model_outputs()
                    result = optimize_portfolio(
                        returns[selected],
                        predictions.reindex(selected),
                        objective=objective,
                        risk_free_rate=risk_free,
                        max_weight=max_weight
                    )

                st.success("Optimization completed. Your portfolio allocation is shown below.")
                st.subheader("Your optimized portfolio")
                m1, m2, m3 = st.columns(3)
                m1.metric("Expected annual return", f"{result['return']*100:.2f}%")
                m2.metric("Estimated annual risk", f"{result['volatility']*100:.2f}%")
                m3.metric("Estimated Sharpe ratio", f"{result['sharpe']:.2f}")

                weights = result["weights"].sort_values(ascending=False)
                allocation = weights[weights > 0.0001].rename("Weight").reset_index()
                allocation.columns = ["Ticker", "Weight"]
                allocation["Company"] = allocation["Ticker"].map(companies)
                allocation["Allocation (%)"] = (allocation["Weight"] * 100).round(2)
                allocation = allocation[["Ticker", "Company", "Allocation (%)"]]

                st.markdown("#### 1. Where your money is allocated")
                chart_col, table_col = st.columns([1.1, 1])
                with chart_col:
                    fig = px.pie(
                        allocation, names="Ticker", values="Allocation (%)",
                        hole=0.48, template="plotly_white",
                        color_discrete_sequence=[
                            "#1769AA", "#16A085", "#F39C3D", "#8E63CE",
                            "#E45B68", "#36A6B8", "#6C8E3D", "#D18BBD",
                            "#7B8794", "#E0B44C"
                        ]
                    )
                    fig.update_traces(textposition="inside", textinfo="percent",
                                      marker=dict(line=dict(color="white", width=2)))
                    fig.update_layout(
                        height=390, margin=dict(l=5, r=5, t=15, b=5),
                        paper_bgcolor="white", plot_bgcolor="white",
                        font=dict(color="#172B4D"),
                        legend=dict(font=dict(size=11))
                    )
                    st.plotly_chart(fig, use_container_width=True)
                with table_col:
                    st.markdown("#### 2. Exact stock weights")
                    st.dataframe(
                        allocation,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Allocation (%)": st.column_config.NumberColumn(
                                "Allocation (%)", format="%.2f%%"
                            )
                        }
                    )
                    st.caption("The percentages show the share of the total portfolio assigned to each selected stock.")
                    st.download_button(
                        "⬇️ Download allocation CSV",
                        data=allocation.to_csv(index=False).encode("utf-8"),
                        file_name="optimized_portfolio_allocation.csv",
                        mime="text/csv"
                    )

                with st.expander("Model evaluation details"):
                    st.write("Selected model evaluation (or Random Forest fallback):")
                    st.dataframe(model_info, use_container_width=True, hide_index=True)
                    st.caption("MAE and RMSE measure prediction error. They do not establish that a strategy will be profitable.")

                st.warning("These are model-based estimates for an academic project, not guaranteed returns or personal investment advice.")
            except Exception as exc:
                st.error(f"Optimization could not be completed: {exc}")
                st.info("Check the selected stocks and allocation cap. If the error continues, copy the full terminal traceback and share it.")

elif page == "Risk Analysis":
    st.title("Portfolio Risk Analysis")
    st.write("Measure historical risk for an equally weighted portfolio of the stocks you select.")
    selected = st.multiselect(
        "Choose stocks for risk analysis",
        assets, default=assets[:8],
        format_func=lambda x: f"{x} — {companies.get(x, x)}"
    )
    if len(selected) >= 1:
        weights = pd.Series(1 / len(selected), index=selected)
        hist = portfolio_history(returns[selected], weights)
        risk = calculate_risk(hist)
        a, b, c, d = st.columns(4)
        a.metric("Annualized volatility", f"{risk['volatility']*100:.2f}%")
        b.metric("Historical VaR (95%)", f"{risk['var_95']*100:.2f}%")
        c.metric("Historical CVaR (95%)", f"{risk['cvar_95']*100:.2f}%")
        d.metric("Maximum drawdown", f"{risk['max_drawdown']*100:.2f}%")

        st.subheader("Cumulative portfolio growth")
        wealth = (1 + hist).cumprod()
        chart = pd.DataFrame({"Date": wealth.index, "Growth of ₹1": wealth.values})
        fig = px.line(chart, x="Date", y="Growth of ₹1", template="plotly_white")
        fig.update_traces(line=dict(color="#1769AA", width=2.5))
        fig.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="white",
                          font=dict(color="#172B4D"))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Daily return distribution")
        fig2 = px.histogram(hist, nbins=60, template="plotly_white",
                            labels={"value": "Daily return"})
        fig2.update_traces(marker_color="#16A085")
        fig2.update_layout(height=300, paper_bgcolor="white", plot_bgcolor="white",
                           font=dict(color="#172B4D"))
        st.plotly_chart(fig2, use_container_width=True)
        st.caption("VaR and CVaR are historical estimates based on the selected sample. VaR is shown as a positive loss magnitude.")
    else:
        st.info("Select at least one stock to calculate risk.")

else:
    st.title("About the Project")
    st.write("""
    **Smart Portfolio Optimizer** is an academic machine-learning application for portfolio allocation and historical risk analysis.

    ### Problem statement
    Investors need a systematic way to compare assets, estimate returns, allocate capital and understand portfolio risk. This project combines supervised learning with quantitative portfolio optimization.

    ### Machine-learning workflow
    1. Load and validate historical daily stock returns and lagged macroeconomic variables.
    2. Engineer lagged returns, rolling means, momentum and volatility features.
    3. Compare multiple regression algorithms in the feature engineering notebook; use the best saved model when available, otherwise fall back to Random Forest.
    4. Estimate the covariance matrix with Ledoit-Wolf shrinkage.
    5. Optimize long-only weights using maximum Sharpe ratio or minimum volatility.
    6. Evaluate historical volatility, Value at Risk, Conditional Value at Risk and maximum drawdown.

    ### Limitations
    This is an educational research prototype, not investment advice. A next-day point forecast is noisy and should not be interpreted as a reliable long-term return. The interface does not execute trades.
    """)
    st.subheader("Technology")
    st.write("Python • Pandas • NumPy • Scikit-learn • SciPy • Streamlit • Plotly")
    st.subheader("Data quality note")
    st.warning("The supplied data quality summary reports maximum macro source age as 11 days, while the row-level audit reports a maximum of 7 days. Both original files are retained unchanged.")
