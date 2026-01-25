"""
Streamlit Dashboard for Stock Paper Trading
Run: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os

# Page config
st.set_page_config(
    page_title="Stock Paper Trader",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Paper Trading Dashboard")

# Load trading state
STATE_FILE = "stock_trading_state.json"
TRADES_FILE = "trades.csv"
PARAMS_FILE = "report_stocks/best_params_overall.csv"

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return None

def load_trades():
    if os.path.exists(TRADES_FILE):
        return pd.read_csv(TRADES_FILE)
    return None

def load_params():
    if os.path.exists(PARAMS_FILE):
        return pd.read_csv(PARAMS_FILE, sep=';')
    return None

# Sidebar
st.sidebar.header("Settings")
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
if auto_refresh:
    st.rerun()

# Load data
state = load_state()
trades_df = load_trades()
params_df = load_params()

# Main metrics
col1, col2, col3, col4 = st.columns(4)

if state:
    portfolio = state.get('portfolio', {})
    cash = portfolio.get('cash', 100000)
    initial = portfolio.get('initial_capital', 100000)
    positions = portfolio.get('positions', {})
    closed_trades = portfolio.get('closed_trades', [])

    # Calculate total value
    total_value = cash
    for sym, pos in positions.items():
        total_value += pos.get('shares', 0) * pos.get('entry_price', 0)

    pnl = total_value - initial
    pnl_pct = (pnl / initial) * 100

    with col1:
        st.metric("Portfolio Value", f"${total_value:,.2f}", f"{pnl_pct:+.2f}%")
    with col2:
        st.metric("Cash", f"${cash:,.2f}")
    with col3:
        st.metric("Open Positions", len(positions))
    with col4:
        win_trades = [t for t in closed_trades if t.get('pnl', 0) > 0]
        win_rate = len(win_trades) / len(closed_trades) * 100 if closed_trades else 0
        st.metric("Win Rate", f"{win_rate:.1f}%", f"{len(closed_trades)} trades")
else:
    with col1:
        st.metric("Portfolio Value", "$100,000.00", "0%")
    with col2:
        st.metric("Cash", "$100,000.00")
    with col3:
        st.metric("Open Positions", "0")
    with col4:
        st.metric("Win Rate", "N/A", "0 trades")

st.divider()

# Two columns layout
left_col, right_col = st.columns(2)

# Open Positions
with left_col:
    st.subheader("📊 Open Positions")
    if state and positions:
        pos_data = []
        for sym, pos in positions.items():
            pos_data.append({
                'Symbol': sym,
                'Direction': pos.get('direction', 'long').upper(),
                'Shares': pos.get('shares', 0),
                'Entry': f"${pos.get('entry_price', 0):.2f}",
                'Bars Held': pos.get('bars_held', 0)
            })
        st.dataframe(pd.DataFrame(pos_data), use_container_width=True, hide_index=True)
    else:
        st.info("No open positions")

# Strategy Parameters
with right_col:
    st.subheader("⚙️ Strategy Parameters")
    if params_df is not None:
        # Show first 10 symbols
        display_df = params_df.head(10)[['Symbol', 'Indicator', 'ParamA', 'ParamB']].copy()
        display_df.columns = ['Symbol', 'Indicator', 'Fast', 'Slow']
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No params file found")

st.divider()

# Trade History
st.subheader("📜 Trade History")
if state and closed_trades:
    trades_data = []
    for t in closed_trades[-20:]:  # Last 20 trades
        trades_data.append({
            'Symbol': t.get('symbol', ''),
            'Direction': t.get('direction', '').upper(),
            'Entry': f"${t.get('entry_price', 0):.2f}",
            'Exit': f"${t.get('exit_price', 0):.2f}",
            'PnL': f"${t.get('pnl', 0):+.2f}",
            'Reason': t.get('exit_reason', '')
        })
    st.dataframe(pd.DataFrame(trades_data), use_container_width=True, hide_index=True)
elif trades_df is not None:
    st.dataframe(trades_df.tail(20), use_container_width=True, hide_index=True)
else:
    st.info("No trades yet")

# PnL Chart
st.subheader("📈 Cumulative P&L")
if state and closed_trades:
    cumulative = []
    running = 0
    for i, t in enumerate(closed_trades):
        running += t.get('pnl', 0)
        cumulative.append({'Trade': i+1, 'Cumulative PnL': running})

    if cumulative:
        fig = px.line(pd.DataFrame(cumulative), x='Trade', y='Cumulative PnL')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
elif trades_df is not None and 'PnL' in trades_df.columns:
    trades_df['Cumulative'] = trades_df['PnL'].cumsum()
    fig = px.line(trades_df, y='Cumulative')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No PnL data to display")

# Footer
st.divider()
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
