#!/usr/bin/env python3
"""
Performance Analyzer for Stock Trading Bot
Analyzes trade history and generates performance reports with visualizations.

Usage:
    python performance_analyzer.py --trades trades.csv
    python performance_analyzer.py --trades trades.csv --plot
    python performance_analyzer.py --trades trades.csv --html report.html
    python performance_analyzer.py --trades trades.csv --positions positions.csv --html report.html
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from html import escape

import numpy as np
import pandas as pd

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================
# DATA LOADING
# ============================================
def parse_datetime_column(series: pd.Series) -> pd.Series:
    """
    Parse datetime column with multiple format attempts.
    Handles various datetime formats including ISO, with/without timezone.
    """
    # First try standard pandas parsing
    result = pd.to_datetime(series, errors='coerce')

    # Check if we have many NaT values that might be parseable with different formats
    nat_mask = result.isna() & series.notna() & (series != '') & (series != 'NaT')

    if nat_mask.any():
        # Try additional formats for unparsed values
        formats_to_try = [
            '%Y-%m-%d %H:%M:%S%z',      # ISO with timezone
            '%Y-%m-%dT%H:%M:%S%z',      # ISO T-separator with timezone
            '%Y-%m-%d %H:%M:%S',         # Without timezone
            '%Y-%m-%dT%H:%M:%S',         # ISO T-separator
            '%m/%d/%Y %H:%M:%S',         # US format
            '%d/%m/%Y %H:%M:%S',         # EU format
            '%Y-%m-%d',                  # Date only
        ]

        for fmt in formats_to_try:
            try:
                parsed = pd.to_datetime(series[nat_mask], format=fmt, errors='coerce')
                # Update only the values that were successfully parsed
                newly_parsed = parsed.notna()
                if newly_parsed.any():
                    result.loc[nat_mask] = result.loc[nat_mask].fillna(parsed)
                    nat_mask = result.isna() & series.notna() & (series != '') & (series != 'NaT')
                    if not nat_mask.any():
                        break
            except Exception:
                continue

    return result


def load_trades(filepath: str) -> pd.DataFrame:
    """
    Load trades from CSV file with proper datetime parsing.

    Handles multiple column naming conventions:
    - PascalCase: EntryTime, ExitTime, EntryPrice, etc.
    - snake_case: entry_time, exit_time, entry_price, etc.
    """
    # Try to detect separator
    with open(filepath, 'r') as f:
        first_line = f.readline()
        sep = ';' if ';' in first_line else ','

    # Load CSV
    df = pd.read_csv(filepath, sep=sep)

    # Normalize column names - mapping lowercase to standard names
    column_mapping = {
        'symbol': 'Symbol',
        'direction': 'Direction',
        'entry_price': 'EntryPrice',
        'exit_price': 'ExitPrice',
        'shares': 'Shares',
        'quantity': 'Shares',
        'stake': 'Stake',
        'pnl': 'PnL',
        'pnl_pct': 'PnLPct',
        'pnl_percent': 'PnLPct',
        'entry_time': 'EntryTime',
        'exit_time': 'ExitTime',
        'bars_held': 'BarsHeld',
        'reason': 'Reason',
        'exit_reason': 'Reason',
        'indicator': 'Indicator',
        'htf': 'HTF',
    }

    # Apply column mapping (case-insensitive)
    new_columns = {}
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if col_lower in column_mapping:
            new_columns[col] = column_mapping[col_lower]

    df = df.rename(columns=new_columns)

    # Parse datetime columns with enhanced parsing
    datetime_columns = ['EntryTime', 'ExitTime']
    for col in datetime_columns:
        if col in df.columns:
            df[col] = parse_datetime_column(df[col])

    # Normalize direction column
    if 'Direction' in df.columns:
        df['Direction'] = df['Direction'].str.lower().str.strip()

    # Sort by EntryTime (trades with valid times first, then NaT at the end)
    if 'EntryTime' in df.columns:
        df = df.sort_values('EntryTime', na_position='last').reset_index(drop=True)

    return df


def load_positions(filepath: str) -> pd.DataFrame:
    """Load open positions from CSV or JSON file."""
    if filepath.endswith('.json'):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        # Handle different JSON structures
        if isinstance(data, dict):
            if 'positions' in data:
                positions = data['positions']
                if isinstance(positions, dict):
                    # Convert dict of positions to list
                    positions = list(positions.values())
            else:
                positions = [data]
        else:
            positions = data
        df = pd.DataFrame(positions)
    else:
        # CSV file
        with open(filepath, 'r') as f:
            first_line = f.readline()
            sep = ';' if ';' in first_line else ','
        df = pd.DataFrame(pd.read_csv(filepath, sep=sep))

    # Normalize column names
    column_mapping = {
        'symbol': 'Symbol',
        'direction': 'Direction',
        'entry_price': 'EntryPrice',
        'entry_time': 'EntryTime',
        'shares': 'Shares',
        'quantity': 'Shares',
        'stake': 'Stake',
        'bars_held': 'BarsHeld',
        'indicator': 'Indicator',
        'htf': 'HTF',
        'current_price': 'CurrentPrice',
        'unrealized_pnl': 'UnrealizedPnL',
    }

    new_columns = {}
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if col_lower in column_mapping:
            new_columns[col] = column_mapping[col_lower]

    df = df.rename(columns=new_columns)

    # Parse datetime columns
    if 'EntryTime' in df.columns:
        df['EntryTime'] = pd.to_datetime(df['EntryTime'], errors='coerce')

    return df


# ============================================
# TRADE ANALYSIS
# ============================================
def analyze_trades(trades: pd.DataFrame, direction_filter: str = None) -> Dict:
    """
    Comprehensive trade analysis.

    Args:
        trades: DataFrame with trade data
        direction_filter: 'long', 'short', or None for all

    Returns dictionary with all analysis metrics.
    """
    if trades.empty:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
        }

    # Filter by direction if specified
    if direction_filter and 'Direction' in trades.columns:
        trades = trades[trades['Direction'] == direction_filter.lower()].copy()
        if trades.empty:
            return {
                'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
                'win_rate': 0, 'total_pnl': 0, 'avg_pnl': 0,
            }

    # Calculate duration if datetime columns exist
    if 'EntryTime' in trades.columns and 'ExitTime' in trades.columns:
        trades['EntryTime'] = pd.to_datetime(trades['EntryTime'], errors='coerce')
        trades['ExitTime'] = pd.to_datetime(trades['ExitTime'], errors='coerce')
        trades['Duration'] = (trades['ExitTime'] - trades['EntryTime']).dt.total_seconds() / 3600

    total_trades = len(trades)

    # PnL column detection
    pnl_col = 'PnL' if 'PnL' in trades.columns else 'pnl' if 'pnl' in trades.columns else None

    if pnl_col is None:
        if all(col in trades.columns for col in ['EntryPrice', 'ExitPrice', 'Shares']):
            trades['PnL'] = (trades['ExitPrice'] - trades['EntryPrice']) * trades['Shares']
            pnl_col = 'PnL'
        elif all(col in trades.columns for col in ['EntryPrice', 'ExitPrice', 'Stake']):
            trades['PnL'] = (trades['ExitPrice'] - trades['EntryPrice']) / trades['EntryPrice'] * trades['Stake']
            pnl_col = 'PnL'
        else:
            return {'error': 'Cannot calculate PnL - missing required columns'}

    # Win/Loss analysis
    winning_trades = trades[trades[pnl_col] > 0]
    losing_trades = trades[trades[pnl_col] < 0]

    total_pnl = trades[pnl_col].sum()
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    avg_win = winning_trades[pnl_col].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades[pnl_col].mean() if len(losing_trades) > 0 else 0

    # Profit factor
    gross_profit = winning_trades[pnl_col].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades[pnl_col].sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Maximum drawdown
    trades_sorted = trades.sort_values('ExitTime') if 'ExitTime' in trades.columns else trades
    cumulative_pnl = trades_sorted[pnl_col].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()

    return {
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl': trades[pnl_col].mean() if total_trades > 0 else 0,
        'max_win': trades[pnl_col].max() if total_trades > 0 else 0,
        'max_loss': trades[pnl_col].min() if total_trades > 0 else 0,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
    }


def analyze_by_symbol(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze trades grouped by symbol."""
    if trades.empty or 'Symbol' not in trades.columns:
        return pd.DataFrame()

    pnl_col = 'PnL' if 'PnL' in trades.columns else 'pnl'

    results = []
    for symbol in trades['Symbol'].unique():
        sym_trades = trades[trades['Symbol'] == symbol]
        sym_wins = sym_trades[sym_trades[pnl_col] > 0]
        sym_losses = sym_trades[sym_trades[pnl_col] <= 0]

        # Long/Short breakdown
        long_trades = sym_trades[sym_trades['Direction'] == 'long'] if 'Direction' in sym_trades.columns else sym_trades
        short_trades = sym_trades[sym_trades['Direction'] == 'short'] if 'Direction' in sym_trades.columns else pd.DataFrame()

        long_pnl = long_trades[pnl_col].sum() if len(long_trades) > 0 else 0
        short_pnl = short_trades[pnl_col].sum() if len(short_trades) > 0 else 0

        # Max drawdown for symbol
        sym_sorted = sym_trades.sort_values('ExitTime') if 'ExitTime' in sym_trades.columns else sym_trades
        cum_pnl = sym_sorted[pnl_col].cumsum()
        running_max = cum_pnl.cummax()
        dd = running_max - cum_pnl
        max_dd = dd.max()

        # Profit factor
        gross_profit = sym_wins[pnl_col].sum() if len(sym_wins) > 0 else 0
        gross_loss = abs(sym_losses[pnl_col].sum()) if len(sym_losses) > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        results.append({
            'Symbol': symbol,
            'Trades': len(sym_trades),
            'Win': len(sym_wins),
            'Loss': len(sym_losses),
            'Win%': f"{len(sym_wins) / len(sym_trades) * 100:.1f}%" if len(sym_trades) > 0 else "0.0%",
            'Total PnL': sym_trades[pnl_col].sum(),
            'Avg PnL': sym_trades[pnl_col].mean(),
            'Best': sym_trades[pnl_col].max(),
            'Worst': sym_trades[pnl_col].min(),
            'Max DD': max_dd,
            'PF': pf if pf != float('inf') else 999.99,
            'Long': len(long_trades),
            'Short': len(short_trades),
            'Long PnL': long_pnl,
            'Short PnL': short_pnl,
        })

    df = pd.DataFrame(results)
    df = df.sort_values('Total PnL', ascending=False)
    return df


def calculate_open_equity(positions: pd.DataFrame, current_prices: Dict[str, float] = None) -> float:
    """Calculate unrealized PnL from open positions."""
    if positions.empty:
        return 0.0

    if 'UnrealizedPnL' in positions.columns:
        return positions['UnrealizedPnL'].sum()

    if current_prices is None:
        return 0.0

    total = 0.0
    for _, pos in positions.iterrows():
        symbol = pos.get('Symbol', '')
        if symbol in current_prices:
            entry = pos.get('EntryPrice', 0)
            current = current_prices[symbol]
            stake = pos.get('Stake', pos.get('Shares', 0) * entry)
            direction = pos.get('Direction', 'long').lower()

            if direction == 'long':
                pnl = (current - entry) / entry * stake if entry > 0 else 0
            else:
                pnl = (entry - current) / entry * stake if entry > 0 else 0
            total += pnl

    return total


def max_consecutive(arr: np.ndarray, value: int) -> int:
    """Find maximum consecutive occurrences of a value in array."""
    max_count = 0
    current_count = 0
    for v in arr:
        if v == value:
            current_count += 1
            max_count = max(max_count, current_count)
        else:
            current_count = 0
    return max_count


# ============================================
# HTML REPORT GENERATION
# ============================================
def generate_html_report(
    trades: pd.DataFrame,
    positions: pd.DataFrame = None,
    initial_capital: float = 100000,
    output_path: str = "performance_report.html"
) -> str:
    """Generate comprehensive HTML report."""

    pnl_col = 'PnL' if 'PnL' in trades.columns else 'pnl'

    # Calculate stats
    overall_stats = analyze_trades(trades)
    long_stats = analyze_trades(trades, 'long')
    short_stats = analyze_trades(trades, 'short')
    symbol_stats = analyze_by_symbol(trades)

    # Time range
    start_time = trades['EntryTime'].min() if 'EntryTime' in trades.columns else datetime.now()
    end_time = trades['ExitTime'].max() if 'ExitTime' in trades.columns else datetime.now()

    # Open positions stats
    open_count = len(positions) if positions is not None else 0
    open_long = len(positions[positions['Direction'] == 'long']) if positions is not None and 'Direction' in positions.columns else open_count
    open_short = len(positions[positions['Direction'] == 'short']) if positions is not None and 'Direction' in positions.columns else 0
    open_equity = positions['UnrealizedPnL'].sum() if positions is not None and 'UnrealizedPnL' in positions.columns else 0

    # Final capital
    final_capital = initial_capital + overall_stats['total_pnl'] + open_equity

    # CSS styles
    css = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
            color: #333;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #bdc3c7;
            padding-bottom: 5px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px 12px;
            text-align: right;
        }
        th {
            background-color: #3498db;
            color: white;
            font-weight: 600;
        }
        td:first-child, th:first-child {
            text-align: left;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f1f1f1;
        }
        .positive { color: #27ae60; font-weight: 600; }
        .negative { color: #e74c3c; font-weight: 600; }
        .summary-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .stat-label {
            color: #7f8c8d;
            font-size: 12px;
            text-transform: uppercase;
        }
        .trade-table {
            font-size: 13px;
        }
        .trade-table td, .trade-table th {
            padding: 6px 10px;
        }
        .collapsible {
            cursor: pointer;
            padding: 10px;
            background-color: #3498db;
            color: white;
            border: none;
            width: 100%;
            text-align: left;
            font-size: 16px;
            font-weight: 600;
            border-radius: 5px;
            margin-top: 20px;
        }
        .collapsible:hover {
            background-color: #2980b9;
        }
        .content {
            max-height: 500px;
            overflow-y: auto;
            background-color: white;
        }
    </style>
    """

    # Helper function for formatting
    def fmt_pnl(val):
        if pd.isna(val):
            return "0.00"
        cls = "positive" if val >= 0 else "negative"
        return f'<span class="{cls}">{val:,.2f}</span>'

    def fmt_pct(val):
        if pd.isna(val) or val == 0:
            return "0.00"
        cls = "positive" if val >= 0 else "negative"
        return f'<span class="{cls}">{val:.2f}%</span>'

    def fmt_pf(val):
        if val == float('inf') or val > 100:
            return "∞"
        return f"{val:.2f}"

    # Build HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading Performance Report</title>
    {css}
</head>
<body>
    <div class="summary-box">
        <h1 style="color: white; border: none; margin: 0;">Simulation Summary</h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">{start_time} → {end_time}</p>
    </div>

    <h2>Statistics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Overall</th>
            <th>Long</th>
            <th>Short</th>
        </tr>
        <tr>
            <td>Closed trades</td>
            <td>{overall_stats['total_trades']}</td>
            <td>{long_stats['total_trades']}</td>
            <td>{short_stats['total_trades']}</td>
        </tr>
        <tr>
            <td>Open positions</td>
            <td>{open_count}</td>
            <td>{open_long}</td>
            <td>{open_short}</td>
        </tr>
        <tr>
            <td>PnL (USD)</td>
            <td>{fmt_pnl(overall_stats['total_pnl'])}</td>
            <td>{fmt_pnl(long_stats['total_pnl'])}</td>
            <td>{fmt_pnl(short_stats['total_pnl'])}</td>
        </tr>
        <tr>
            <td>Avg PnL (USD)</td>
            <td>{fmt_pnl(overall_stats['avg_pnl'])}</td>
            <td>{fmt_pnl(long_stats['avg_pnl'])}</td>
            <td>{fmt_pnl(short_stats['avg_pnl'])}</td>
        </tr>
        <tr>
            <td>Win rate (%)</td>
            <td>{fmt_pct(overall_stats['win_rate'])}</td>
            <td>{fmt_pct(long_stats['win_rate'])}</td>
            <td>{fmt_pct(short_stats['win_rate'])}</td>
        </tr>
        <tr>
            <td>Winners</td>
            <td>{overall_stats['winning_trades']}</td>
            <td>{long_stats['winning_trades']}</td>
            <td>{short_stats['winning_trades']}</td>
        </tr>
        <tr>
            <td>Losers</td>
            <td>{overall_stats['losing_trades']}</td>
            <td>{long_stats['losing_trades']}</td>
            <td>{short_stats['losing_trades']}</td>
        </tr>
        <tr>
            <td>Open equity (USD)</td>
            <td>{fmt_pnl(open_equity)}</td>
            <td>{fmt_pnl(open_equity if open_short == 0 else 0)}</td>
            <td>{fmt_pnl(0 if open_short == 0 else open_equity)}</td>
        </tr>
        <tr>
            <td><strong>Final capital (USD)</strong></td>
            <td><strong>{fmt_pnl(final_capital)}</strong></td>
            <td>-</td>
            <td>-</td>
        </tr>
    </table>

    <h2>Statistics by Symbol</h2>
    <table>
        <tr>
            <th>Symbol</th>
            <th>Trades</th>
            <th>Win</th>
            <th>Loss</th>
            <th>Win%</th>
            <th>Total PnL</th>
            <th>Avg PnL</th>
            <th>Best</th>
            <th>Worst</th>
            <th>Max DD</th>
            <th>PF</th>
            <th>Long</th>
            <th>Short</th>
            <th>Long PnL</th>
            <th>Short PnL</th>
        </tr>
"""

    # Add symbol rows
    for _, row in symbol_stats.iterrows():
        html += f"""        <tr>
            <td>{escape(str(row['Symbol']))}</td>
            <td>{row['Trades']}</td>
            <td>{row['Win']}</td>
            <td>{row['Loss']}</td>
            <td>{row['Win%']}</td>
            <td>{fmt_pnl(row['Total PnL'])}</td>
            <td>{fmt_pnl(row['Avg PnL'])}</td>
            <td>{fmt_pnl(row['Best'])}</td>
            <td>{fmt_pnl(row['Worst'])}</td>
            <td>{fmt_pnl(row['Max DD'])}</td>
            <td>{fmt_pf(row['PF'])}</td>
            <td>{row['Long']}</td>
            <td>{row['Short']}</td>
            <td>{fmt_pnl(row['Long PnL'])}</td>
            <td>{fmt_pnl(row['Short PnL'])}</td>
        </tr>
"""

    html += "    </table>\n"

    # Open Positions Table
    if positions is not None and not positions.empty:
        html += f"""
    <h2>Open Positions ({len(positions)} positions)</h2>
    <table class="trade-table">
        <tr>
            <th>Symbol</th>
            <th>Direction</th>
            <th>Entry Time</th>
            <th>Entry Price</th>
            <th>Stake</th>
            <th>Bars Held</th>
            <th>Unrealized PnL</th>
        </tr>
"""
        for _, pos in positions.iterrows():
            entry_time = pos.get('EntryTime', pos.get('entry_time', ''))
            if pd.notna(entry_time):
                entry_time = pd.to_datetime(entry_time)

            html += f"""        <tr>
            <td>{escape(str(pos.get('Symbol', pos.get('symbol', ''))))}</td>
            <td>{escape(str(pos.get('Direction', pos.get('direction', 'Long'))).title())}</td>
            <td>{entry_time}</td>
            <td>{pos.get('EntryPrice', pos.get('entry_price', 0)):.8f}</td>
            <td>{pos.get('Stake', pos.get('stake', 0)):.2f}</td>
            <td>{pos.get('BarsHeld', pos.get('bars_held', 0))}</td>
            <td>{fmt_pnl(pos.get('UnrealizedPnL', pos.get('unrealized_pnl', 0)))}</td>
        </tr>
"""
        html += "    </table>\n"

    # Long Trades Table
    long_trades = trades[trades['Direction'] == 'long'] if 'Direction' in trades.columns else trades
    if 'EntryTime' in long_trades.columns:
        long_trades = long_trades.sort_values('EntryTime', na_position='last')
    if not long_trades.empty:
        long_total_pnl = long_trades[pnl_col].sum()
        html += f"""
    <button class="collapsible">Long Trades ({len(long_trades)} trades, PnL: {long_total_pnl:,.2f} USD)</button>
    <div class="content">
    <table class="trade-table">
        <tr>
            <th>Symbol</th>
            <th>Direction</th>
            <th>Indicator</th>
            <th>HTF</th>
            <th>Entry Time</th>
            <th>Entry Price</th>
            <th>Exit Time</th>
            <th>Exit Price</th>
            <th>Stake</th>
            <th>PnL</th>
            <th>Reason</th>
        </tr>
"""
        for _, trade in long_trades.iterrows():
            html += f"""        <tr>
            <td>{escape(str(trade.get('Symbol', '')))}</td>
            <td>Long</td>
            <td>{escape(str(trade.get('Indicator', trade.get('indicator', ''))))}</td>
            <td>{escape(str(trade.get('HTF', trade.get('htf', ''))))}</td>
            <td>{trade.get('EntryTime', '')}</td>
            <td>{trade.get('EntryPrice', 0):.8f}</td>
            <td>{trade.get('ExitTime', '')}</td>
            <td>{trade.get('ExitPrice', 0):.8f}</td>
            <td>{trade.get('Stake', trade.get('Shares', 0) * trade.get('EntryPrice', 1)):.8f}</td>
            <td>{fmt_pnl(trade.get(pnl_col, 0))}</td>
            <td>{escape(str(trade.get('Reason', '')))}</td>
        </tr>
"""
        html += """    </table>
    </div>
"""

    # Short Trades Table
    short_trades = trades[trades['Direction'] == 'short'] if 'Direction' in trades.columns else pd.DataFrame()
    if 'EntryTime' in short_trades.columns and not short_trades.empty:
        short_trades = short_trades.sort_values('EntryTime', na_position='last')
    if not short_trades.empty:
        short_total_pnl = short_trades[pnl_col].sum()
        html += f"""
    <button class="collapsible">Short Trades ({len(short_trades)} trades, PnL: {short_total_pnl:,.2f} USD)</button>
    <div class="content">
    <table class="trade-table">
        <tr>
            <th>Symbol</th>
            <th>Direction</th>
            <th>Indicator</th>
            <th>HTF</th>
            <th>Entry Time</th>
            <th>Entry Price</th>
            <th>Exit Time</th>
            <th>Exit Price</th>
            <th>Stake</th>
            <th>PnL</th>
            <th>Reason</th>
        </tr>
"""
        for _, trade in short_trades.iterrows():
            html += f"""        <tr>
            <td>{escape(str(trade.get('Symbol', '')))}</td>
            <td>Short</td>
            <td>{escape(str(trade.get('Indicator', trade.get('indicator', ''))))}</td>
            <td>{escape(str(trade.get('HTF', trade.get('htf', ''))))}</td>
            <td>{trade.get('EntryTime', '')}</td>
            <td>{trade.get('EntryPrice', 0):.8f}</td>
            <td>{trade.get('ExitTime', '')}</td>
            <td>{trade.get('ExitPrice', 0):.8f}</td>
            <td>{trade.get('Stake', trade.get('Shares', 0) * trade.get('EntryPrice', 1)):.8f}</td>
            <td>{fmt_pnl(trade.get(pnl_col, 0))}</td>
            <td>{escape(str(trade.get('Reason', '')))}</td>
        </tr>
"""
        html += """    </table>
    </div>
"""

    # JavaScript for collapsible sections
    html += """
    <script>
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.maxHeight) {
                    content.style.maxHeight = null;
                } else {
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            });
        }
    </script>
</body>
</html>
"""

    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path


# ============================================
# CONSOLE REPORTING
# ============================================
def print_analysis_report(trades: pd.DataFrame, initial_capital: float = 100000):
    """Print comprehensive analysis report to console."""
    analysis = analyze_trades(trades)

    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return

    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-" * 42)
    print(f"{'Total Trades':<25} {analysis['total_trades']:>15}")
    print(f"{'Winning Trades':<25} {analysis['winning_trades']:>15}")
    print(f"{'Losing Trades':<25} {analysis['losing_trades']:>15}")
    print(f"{'Win Rate':<25} {analysis['win_rate']:>14.2f}%")
    print(f"{'Total PnL':<25} ${analysis['total_pnl']:>14,.2f}")
    print(f"{'Average PnL':<25} ${analysis['avg_pnl']:>14,.2f}")
    print(f"{'Best Trade':<25} ${analysis['max_win']:>14,.2f}")
    print(f"{'Worst Trade':<25} ${analysis['max_loss']:>14,.2f}")
    print(f"{'Max Drawdown':<25} ${analysis['max_drawdown']:>14,.2f}")

    pf_str = f"{analysis['profit_factor']:.2f}" if analysis['profit_factor'] != float('inf') else "∞"
    print(f"{'Profit Factor':<25} {pf_str:>15}")

    roi = (analysis['total_pnl'] / initial_capital) * 100
    print(f"{'Return on Capital':<25} {roi:>14.2f}%")

    print("\n" + "=" * 80)


# ============================================
# PLOTTING
# ============================================
def plot_analysis(trades: pd.DataFrame, initial_capital: float = 100000, save_path: str = None):
    """Generate analysis plots."""
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    pnl_col = 'PnL' if 'PnL' in trades.columns else 'pnl'

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Trading Performance Analysis', fontsize=14, fontweight='bold')

    # 1. Equity Curve
    ax1 = axes[0, 0]

    # Filter out rows with missing ExitTime for time-based plotting
    if 'ExitTime' in trades.columns:
        # Drop NaT values and sort
        trades_with_time = trades.dropna(subset=['ExitTime']).copy()
        trades_with_time = trades_with_time.sort_values('ExitTime')

        # Convert timezone-aware to timezone-naive for matplotlib
        if len(trades_with_time) > 0:
            exit_times = trades_with_time['ExitTime']
            if hasattr(exit_times.dt, 'tz') and exit_times.dt.tz is not None:
                trades_with_time['ExitTime'] = exit_times.dt.tz_localize(None)

            cumulative_pnl = trades_with_time[pnl_col].cumsum()
            equity = initial_capital + cumulative_pnl

            ax1.plot(trades_with_time['ExitTime'].values, equity.values, 'b-', linewidth=1.5)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        else:
            # Fallback to trade number
            cumulative_pnl = trades[pnl_col].cumsum()
            equity = initial_capital + cumulative_pnl
            ax1.plot(range(len(equity)), equity.values, 'b-', linewidth=1.5)
    else:
        cumulative_pnl = trades[pnl_col].cumsum()
        equity = initial_capital + cumulative_pnl
        ax1.plot(range(len(equity)), equity.values, 'b-', linewidth=1.5)

    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)

    # 2. PnL Distribution
    ax2 = axes[0, 1]
    ax2.hist(trades[pnl_col].dropna(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('PnL Distribution')
    ax2.set_xlabel('PnL ($)')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative PnL (by trade number, using all trades)
    ax3 = axes[1, 0]
    all_cumulative_pnl = trades[pnl_col].cumsum()
    ax3.fill_between(range(len(all_cumulative_pnl)), all_cumulative_pnl, 0,
                     where=all_cumulative_pnl >= 0, color='green', alpha=0.3)
    ax3.fill_between(range(len(all_cumulative_pnl)), all_cumulative_pnl, 0,
                     where=all_cumulative_pnl < 0, color='red', alpha=0.3)
    ax3.plot(all_cumulative_pnl.values, 'b-', linewidth=1.5)
    ax3.set_title('Cumulative PnL')
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Cumulative PnL ($)')
    ax3.grid(True, alpha=0.3)

    # 4. PnL by Symbol
    ax4 = axes[1, 1]
    if 'Symbol' in trades.columns:
        symbol_pnl = trades.groupby('Symbol')[pnl_col].sum().sort_values()
        colors = ['green' if x >= 0 else 'red' for x in symbol_pnl.values]
        symbol_pnl.tail(15).plot(kind='barh', ax=ax4, color=colors[-15:])
        ax4.set_title('PnL by Symbol (Top 15)')
        ax4.set_xlabel('Total PnL ($)')

    ax4.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Analyze trading performance from CSV')
    parser.add_argument('--trades', required=True, help='Path to trades CSV file')
    parser.add_argument('--positions', help='Path to open positions CSV/JSON file')
    parser.add_argument('--capital', type=float, default=20000, help='Initial capital (default: 20000)')
    parser.add_argument('--plot', action='store_true', help='Show performance plots')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
    parser.add_argument('--html', type=str, help='Generate HTML report at specified path')
    parser.add_argument('--export', type=str, help='Export analysis to JSON file')

    args = parser.parse_args()

    # Load trades
    print("Loading trades...")
    try:
        trades = load_trades(args.trades)
        print(f"Loaded {len(trades)} trades")
    except Exception as e:
        print(f"Error loading trades: {e}")
        sys.exit(1)

    # Load positions if provided
    positions = None
    if args.positions:
        try:
            positions = load_positions(args.positions)
            print(f"Loaded {len(positions)} open positions")
        except Exception as e:
            print(f"Warning: Could not load positions: {e}")

    # Print analysis report
    print_analysis_report(trades, args.capital)

    # Generate HTML report if requested
    if args.html:
        output_path = generate_html_report(trades, positions, args.capital, args.html)
        print(f"\nHTML report generated: {output_path}")

    # Export to JSON if requested
    if args.export:
        import json
        analysis = analyze_trades(trades)
        for key, value in analysis.items():
            if isinstance(value, float) and (np.isinf(value) or np.isnan(value)):
                analysis[key] = str(value)
        with open(args.export, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\nAnalysis exported to {args.export}")

    # Show plots if requested
    if args.plot or args.save_plot:
        plot_analysis(trades, args.capital, args.save_plot)


if __name__ == "__main__":
    main()
