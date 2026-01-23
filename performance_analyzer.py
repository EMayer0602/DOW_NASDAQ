#!/usr/bin/env python3
"""
Performance Analyzer for Stock Trading Bot
Analyzes trade history and generates performance reports with visualizations.

Usage:
    python performance_analyzer.py --trades trades.csv
    python performance_analyzer.py --trades trades.csv --plot
    python performance_analyzer.py --trades trades.csv --plot --capital 100000
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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

    # Normalize column names to PascalCase for consistency
    column_mapping = {
        'symbol': 'Symbol',
        'direction': 'Direction',
        'entry_price': 'EntryPrice',
        'exit_price': 'ExitPrice',
        'shares': 'Shares',
        'quantity': 'Shares',
        'pnl': 'PnL',
        'pnl_pct': 'PnLPct',
        'pnl_percent': 'PnLPct',
        'entry_time': 'EntryTime',
        'exit_time': 'ExitTime',
        'bars_held': 'BarsHeld',
        'reason': 'Reason',
        'exit_reason': 'Reason',
    }

    # Apply column mapping (case-insensitive)
    new_columns = {}
    for col in df.columns:
        col_lower = col.lower().replace(' ', '_')
        if col_lower in column_mapping:
            new_columns[col] = column_mapping[col_lower]
        elif col not in column_mapping.values():
            # Keep original if already in correct format
            pass

    df = df.rename(columns=new_columns)

    # Parse datetime columns
    datetime_columns = ['EntryTime', 'ExitTime']
    for col in datetime_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    return df


# ============================================
# TRADE ANALYSIS
# ============================================
def analyze_trades(trades: pd.DataFrame) -> Dict:
    """
    Comprehensive trade analysis.

    Returns dictionary with all analysis metrics.
    """
    if trades.empty:
        return {'error': 'No trades to analyze'}

    # Calculate duration if datetime columns exist
    if 'EntryTime' in trades.columns and 'ExitTime' in trades.columns:
        # Ensure datetime types
        trades['EntryTime'] = pd.to_datetime(trades['EntryTime'], errors='coerce')
        trades['ExitTime'] = pd.to_datetime(trades['ExitTime'], errors='coerce')
        trades['Duration'] = (trades['ExitTime'] - trades['EntryTime']).dt.total_seconds() / 3600  # hours

    # Basic stats
    total_trades = len(trades)

    # PnL column detection
    pnl_col = 'PnL' if 'PnL' in trades.columns else 'pnl' if 'pnl' in trades.columns else None

    if pnl_col is None:
        # Calculate PnL if not present
        if all(col in trades.columns for col in ['EntryPrice', 'ExitPrice', 'Shares']):
            trades['PnL'] = (trades['ExitPrice'] - trades['EntryPrice']) * trades['Shares']
            pnl_col = 'PnL'
        else:
            return {'error': 'Cannot calculate PnL - missing required columns'}

    # Win/Loss analysis
    winning_trades = trades[trades[pnl_col] > 0]
    losing_trades = trades[trades[pnl_col] < 0]
    breakeven_trades = trades[trades[pnl_col] == 0]

    total_pnl = trades[pnl_col].sum()

    # Calculate metrics
    win_count = len(winning_trades)
    loss_count = len(losing_trades)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0

    avg_win = winning_trades[pnl_col].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades[pnl_col].mean() if len(losing_trades) > 0 else 0

    # Profit factor
    gross_profit = winning_trades[pnl_col].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades[pnl_col].sum()) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Risk/Reward ratio
    risk_reward = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Expectancy
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)

    # Maximum drawdown calculation
    trades_sorted = trades.sort_values('ExitTime') if 'ExitTime' in trades.columns else trades
    cumulative_pnl = trades_sorted[pnl_col].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()
    max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() > 0 else 0

    # Consecutive wins/losses
    pnl_signs = np.sign(trades_sorted[pnl_col].values)
    max_consecutive_wins = max_consecutive(pnl_signs, 1)
    max_consecutive_losses = max_consecutive(pnl_signs, -1)

    # Duration stats
    duration_stats = {}
    if 'Duration' in trades.columns:
        duration_stats = {
            'avg_duration_hours': trades['Duration'].mean(),
            'min_duration_hours': trades['Duration'].min(),
            'max_duration_hours': trades['Duration'].max(),
        }

    # Per-symbol analysis
    symbol_stats = {}
    if 'Symbol' in trades.columns:
        for symbol in trades['Symbol'].unique():
            sym_trades = trades[trades['Symbol'] == symbol]
            sym_wins = sym_trades[sym_trades[pnl_col] > 0]
            symbol_stats[symbol] = {
                'trades': len(sym_trades),
                'win_rate': len(sym_wins) / len(sym_trades) * 100 if len(sym_trades) > 0 else 0,
                'total_pnl': sym_trades[pnl_col].sum(),
                'avg_pnl': sym_trades[pnl_col].mean(),
            }

    # Monthly breakdown
    monthly_stats = {}
    if 'ExitTime' in trades.columns:
        trades['Month'] = trades['ExitTime'].dt.to_period('M')
        for month in trades['Month'].dropna().unique():
            month_trades = trades[trades['Month'] == month]
            month_wins = month_trades[month_trades[pnl_col] > 0]
            monthly_stats[str(month)] = {
                'trades': len(month_trades),
                'win_rate': len(month_wins) / len(month_trades) * 100 if len(month_trades) > 0 else 0,
                'total_pnl': month_trades[pnl_col].sum(),
            }

    return {
        # Basic stats
        'total_trades': total_trades,
        'winning_trades': win_count,
        'losing_trades': loss_count,
        'breakeven_trades': len(breakeven_trades),
        'win_rate': win_rate,

        # PnL stats
        'total_pnl': total_pnl,
        'avg_pnl': trades[pnl_col].mean(),
        'median_pnl': trades[pnl_col].median(),
        'std_pnl': trades[pnl_col].std(),
        'max_win': trades[pnl_col].max(),
        'max_loss': trades[pnl_col].min(),

        # Win/Loss analysis
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,

        # Ratios
        'profit_factor': profit_factor,
        'risk_reward': risk_reward,
        'expectancy': expectancy,

        # Drawdown
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,

        # Streaks
        'max_consecutive_wins': max_consecutive_wins,
        'max_consecutive_losses': max_consecutive_losses,

        # Duration
        **duration_stats,

        # Detailed breakdowns
        'symbol_stats': symbol_stats,
        'monthly_stats': monthly_stats,
    }


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
# EQUITY CURVE
# ============================================
def calculate_equity_curve(trades: pd.DataFrame, initial_capital: float = 100000) -> pd.DataFrame:
    """Calculate equity curve from trades."""
    pnl_col = 'PnL' if 'PnL' in trades.columns else 'pnl'

    if 'ExitTime' in trades.columns:
        trades_sorted = trades.sort_values('ExitTime').copy()
        trades_sorted['CumulativePnL'] = trades_sorted[pnl_col].cumsum()
        trades_sorted['Equity'] = initial_capital + trades_sorted['CumulativePnL']
        return trades_sorted[['ExitTime', 'CumulativePnL', 'Equity', pnl_col]]
    else:
        trades_sorted = trades.copy()
        trades_sorted['CumulativePnL'] = trades_sorted[pnl_col].cumsum()
        trades_sorted['Equity'] = initial_capital + trades_sorted['CumulativePnL']
        trades_sorted['TradeNum'] = range(1, len(trades_sorted) + 1)
        return trades_sorted[['TradeNum', 'CumulativePnL', 'Equity', pnl_col]]


# ============================================
# REPORTING
# ============================================
def print_analysis_report(trades: pd.DataFrame, initial_capital: float = 100000):
    """Print comprehensive analysis report."""
    analysis = analyze_trades(trades)

    if 'error' in analysis:
        print(f"Error: {analysis['error']}")
        return

    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS REPORT")
    print("=" * 80)

    # Overview
    print("\n--- OVERVIEW ---")
    print(f"Total Trades:        {analysis['total_trades']}")
    print(f"Winning Trades:      {analysis['winning_trades']}")
    print(f"Losing Trades:       {analysis['losing_trades']}")
    print(f"Breakeven Trades:    {analysis['breakeven_trades']}")
    print(f"Win Rate:            {analysis['win_rate']:.2f}%")

    # PnL Summary
    print("\n--- PROFIT & LOSS ---")
    print(f"Total PnL:           ${analysis['total_pnl']:,.2f}")
    print(f"Average PnL:         ${analysis['avg_pnl']:,.2f}")
    print(f"Median PnL:          ${analysis['median_pnl']:,.2f}")
    print(f"Std Dev PnL:         ${analysis['std_pnl']:,.2f}")
    print(f"Best Trade:          ${analysis['max_win']:,.2f}")
    print(f"Worst Trade:         ${analysis['max_loss']:,.2f}")

    # Win/Loss Details
    print("\n--- WIN/LOSS ANALYSIS ---")
    print(f"Average Win:         ${analysis['avg_win']:,.2f}")
    print(f"Average Loss:        ${analysis['avg_loss']:,.2f}")
    print(f"Gross Profit:        ${analysis['gross_profit']:,.2f}")
    print(f"Gross Loss:          ${analysis['gross_loss']:,.2f}")

    # Ratios
    print("\n--- KEY RATIOS ---")
    pf_str = f"{analysis['profit_factor']:.2f}" if analysis['profit_factor'] != float('inf') else "∞"
    rr_str = f"{analysis['risk_reward']:.2f}" if analysis['risk_reward'] != float('inf') else "∞"
    print(f"Profit Factor:       {pf_str}")
    print(f"Risk/Reward Ratio:   {rr_str}")
    print(f"Expectancy:          ${analysis['expectancy']:,.2f}")

    # Drawdown
    print("\n--- RISK METRICS ---")
    print(f"Max Drawdown:        ${analysis['max_drawdown']:,.2f}")
    print(f"Max Drawdown %:      {analysis['max_drawdown_pct']:.2f}%")
    print(f"Max Consecutive Wins:   {analysis['max_consecutive_wins']}")
    print(f"Max Consecutive Losses: {analysis['max_consecutive_losses']}")

    # Return on Capital
    roi = (analysis['total_pnl'] / initial_capital) * 100
    print(f"\nReturn on Capital:   {roi:.2f}% (on ${initial_capital:,.0f})")

    # Duration stats
    if 'avg_duration_hours' in analysis:
        print("\n--- TRADE DURATION ---")
        print(f"Average Duration:    {analysis['avg_duration_hours']:.1f} hours")
        print(f"Min Duration:        {analysis['min_duration_hours']:.1f} hours")
        print(f"Max Duration:        {analysis['max_duration_hours']:.1f} hours")

    # Top/Bottom symbols
    if analysis['symbol_stats']:
        print("\n--- TOP SYMBOLS (by PnL) ---")
        sorted_symbols = sorted(
            analysis['symbol_stats'].items(),
            key=lambda x: x[1]['total_pnl'],
            reverse=True
        )

        # Top 5
        for symbol, stats in sorted_symbols[:5]:
            print(f"  {symbol:8s}  Trades: {stats['trades']:4d}  "
                  f"WinRate: {stats['win_rate']:5.1f}%  "
                  f"PnL: ${stats['total_pnl']:>10,.2f}")

        if len(sorted_symbols) > 5:
            print("\n--- BOTTOM SYMBOLS (by PnL) ---")
            for symbol, stats in sorted_symbols[-5:]:
                print(f"  {symbol:8s}  Trades: {stats['trades']:4d}  "
                      f"WinRate: {stats['win_rate']:5.1f}%  "
                      f"PnL: ${stats['total_pnl']:>10,.2f}")

    # Monthly breakdown
    if analysis['monthly_stats']:
        print("\n--- MONTHLY PERFORMANCE ---")
        for month, stats in sorted(analysis['monthly_stats'].items()):
            print(f"  {month}  Trades: {stats['trades']:4d}  "
                  f"WinRate: {stats['win_rate']:5.1f}%  "
                  f"PnL: ${stats['total_pnl']:>10,.2f}")

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
    equity = calculate_equity_curve(trades, initial_capital)

    if 'ExitTime' in equity.columns:
        ax1.plot(equity['ExitTime'], equity['Equity'], 'b-', linewidth=1.5)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    else:
        ax1.plot(equity['TradeNum'], equity['Equity'], 'b-', linewidth=1.5)
        ax1.set_xlabel('Trade Number')

    ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    ax1.set_title('Equity Curve')
    ax1.set_ylabel('Equity ($)')
    ax1.grid(True, alpha=0.3)

    # 2. PnL Distribution
    ax2 = axes[0, 1]
    pnl_values = trades[pnl_col].dropna()
    ax2.hist(pnl_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.axvline(x=pnl_values.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: ${pnl_values.mean():.2f}')
    ax2.set_title('PnL Distribution')
    ax2.set_xlabel('PnL ($)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Cumulative PnL
    ax3 = axes[1, 0]
    cumulative_pnl = trades[pnl_col].cumsum()

    # Color based on positive/negative
    colors = ['green' if x >= 0 else 'red' for x in cumulative_pnl]
    ax3.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                     where=cumulative_pnl >= 0, color='green', alpha=0.3)
    ax3.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                     where=cumulative_pnl < 0, color='red', alpha=0.3)
    ax3.plot(cumulative_pnl, 'b-', linewidth=1.5)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_title('Cumulative PnL')
    ax3.set_xlabel('Trade Number')
    ax3.set_ylabel('Cumulative PnL ($)')
    ax3.grid(True, alpha=0.3)

    # 4. Win Rate by Symbol (if available)
    ax4 = axes[1, 1]
    if 'Symbol' in trades.columns:
        symbol_stats = trades.groupby('Symbol').agg({
            pnl_col: ['count', lambda x: (x > 0).sum() / len(x) * 100, 'sum']
        }).round(2)
        symbol_stats.columns = ['Trades', 'WinRate', 'TotalPnL']
        symbol_stats = symbol_stats.sort_values('TotalPnL', ascending=True).tail(15)

        colors = ['green' if x >= 0 else 'red' for x in symbol_stats['TotalPnL']]
        bars = ax4.barh(symbol_stats.index, symbol_stats['TotalPnL'], color=colors, alpha=0.7)
        ax4.set_title('PnL by Symbol (Top 15)')
        ax4.set_xlabel('Total PnL ($)')
        ax4.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    else:
        # Win rate pie chart as fallback
        wins = len(trades[trades[pnl_col] > 0])
        losses = len(trades[trades[pnl_col] < 0])
        breakeven = len(trades[trades[pnl_col] == 0])

        sizes = [wins, losses, breakeven]
        labels = [f'Wins ({wins})', f'Losses ({losses})', f'Breakeven ({breakeven})']
        colors = ['green', 'red', 'gray']
        explode = (0.05, 0, 0)

        ax4.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=90)
        ax4.set_title('Win/Loss Distribution')

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
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--plot', action='store_true', help='Show performance plots')
    parser.add_argument('--save-plot', type=str, help='Save plot to file')
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

    # Print analysis report
    print_analysis_report(trades, args.capital)

    # Export to JSON if requested
    if args.export:
        import json
        analysis = analyze_trades(trades)
        # Convert non-serializable types
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
