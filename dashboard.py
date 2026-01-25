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
import time

# IB Connection
try:
    from ib_insync import IB, Stock
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

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

@st.cache_resource
def get_ib_connection():
    """Get or create IB connection."""
    if not IB_AVAILABLE:
        return None
    try:
        ib = IB()
        ib.connect('127.0.0.1', 7497, clientId=99, timeout=20)
        return ib
    except Exception as e:
        return None

def get_ib_portfolio(ib):
    """Fetch portfolio from IB."""
    if ib is None or not ib.isConnected():
        return None
    try:
        portfolio = ib.portfolio()
        data = []
        for item in portfolio:
            data.append({
                'Symbol': item.contract.symbol,
                'Shares': item.position,
                'Price': item.marketPrice,
                'Value': item.marketValue,
                'Avg Cost': item.averageCost,
                'Unrealized PnL': item.unrealizedPNL,
                'Realized PnL': item.realizedPNL
            })
        return pd.DataFrame(data)
    except:
        return None

def get_ib_account(ib):
    """Fetch account summary from IB."""
    if ib is None or not ib.isConnected():
        return None
    try:
        ib.reqAccountSummary()
        time.sleep(0.5)
        summary = ib.accountSummary()
        account = {}
        for item in summary:
            account[item.tag] = item.value
        return account
    except:
        return None

# Sidebar
st.sidebar.header("Settings")
use_ib = st.sidebar.checkbox("Connect to IB", value=True)
auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=False)
if st.sidebar.button("🔄 Refresh Now"):
    st.cache_resource.clear()
    st.rerun()

# IB Connection
ib = None
ib_portfolio = None
ib_account = None

if use_ib and IB_AVAILABLE:
    ib = get_ib_connection()
    if ib and ib.isConnected():
        st.sidebar.success("IB Connected")
        ib_portfolio = get_ib_portfolio(ib)
        ib_account = get_ib_account(ib)
    else:
        st.sidebar.warning("IB not connected")
elif not IB_AVAILABLE:
    st.sidebar.info("ib_insync not installed")

if auto_refresh:
    time.sleep(30)
    st.rerun()

# Load data
state = load_state()
trades_df = load_trades()
params_df = load_params()

# Main metrics
col1, col2, col3, col4 = st.columns(4)

# Use IB data if available
if ib_portfolio is not None and len(ib_portfolio) > 0:
    total_value = ib_portfolio['Value'].sum()
    total_unrealized = ib_portfolio['Unrealized PnL'].sum()
    total_realized = ib_portfolio['Realized PnL'].sum()

    # Get cash from account
    cash = float(ib_account.get('TotalCashValue', 0)) if ib_account else 0
    net_liq = float(ib_account.get('NetLiquidation', total_value + cash)) if ib_account else total_value + cash

    with col1:
        st.metric("Net Liquidation", f"${net_liq:,.2f}")
    with col2:
        st.metric("Cash", f"${cash:,.2f}")
    with col3:
        st.metric("Positions", len(ib_portfolio))
    with col4:
        st.metric("Unrealized PnL", f"${total_unrealized:+,.2f}")

elif state:
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
    if ib_portfolio is not None and len(ib_portfolio) > 0:
        display_df = ib_portfolio.copy()
        display_df['Price'] = display_df['Price'].apply(lambda x: f"${x:.2f}")
        display_df['Value'] = display_df['Value'].apply(lambda x: f"${x:,.2f}")
        display_df['Avg Cost'] = display_df['Avg Cost'].apply(lambda x: f"${x:.2f}")
        display_df['Unrealized PnL'] = display_df['Unrealized PnL'].apply(lambda x: f"${x:+,.2f}")
        st.dataframe(display_df[['Symbol', 'Shares', 'Price', 'Value', 'Avg Cost', 'Unrealized PnL']],
                     use_container_width=True, hide_index=True)
    elif state and positions:
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

# PnL by Symbol Chart (IB)
if ib_portfolio is not None and len(ib_portfolio) > 0:
    st.subheader("📊 Unrealized P&L by Symbol")
    fig = px.bar(ib_portfolio, x='Symbol', y='Unrealized PnL',
                 color='Unrealized PnL',
                 color_continuous_scale=['red', 'gray', 'green'],
                 color_continuous_midpoint=0)
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

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
