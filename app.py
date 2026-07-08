import streamlit as st
import pandas as pd
import plotly.express as px
from backtest_engine import VolatilityBacktester

# Setup UI Layout
st.set_page_config(page_title="Derivatives Odyssey", layout="wide")
st.title("Volatility Trading Backtest Engine")
st.markdown("Test the XGBoost Volatility model against naive strategies using historical Black-Scholes pricing.")

# Sidebar Controls
st.sidebar.header("Backtest Parameters")
strategy = st.sidebar.selectbox("Select Strategy", ["ML_Directional", "Always_Long", "Always_Short"])
trade_size = st.sidebar.slider("Contracts per Trade", min_value=1, max_value=50, value=10)
capital = st.sidebar.number_input("Initial Capital ($)", value=100000)

# Run Button
if st.sidebar.button("Run Backtest"):
    with st.spinner("Pricing options and running simulation..."):
        engine = VolatilityBacktester(initial_capital=capital)
        trades_df, equity_df, metrics = engine.run_backtest(strategy=strategy, trade_size=trade_size)
        
        # Top Row: KPIs
        cols = st.columns(4)
        cols[0].metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
        cols[1].metric("Win Rate", f"{metrics['Win Rate (%)']:.2f}%")
        cols[2].metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
        cols[3].metric("Max Drawdown", f"{metrics['Max Drawdown (%)']:.2f}%")
        
        st.markdown("---")
        
        # Equity Curve Chart
        st.subheader(f"Equity Curve: {strategy}")
        fig = px.line(equity_df, x='Date', y='Equity', template='plotly_dark')
        fig.update_layout(yaxis_title="Account Balance ($)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Trade Log Table
        st.subheader("Trade Log (21-Day ATM Straddles)")
        st.dataframe(trades_df.style.map(lambda x: 'color: #00ff00' if x > 0 else 'color: #ff0000', subset=['PnL']))