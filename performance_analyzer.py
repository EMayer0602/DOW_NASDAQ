#!/usr/bin/env python3
"""
Performance Analyzer for DOW/NASDAQ Trading Strategy
Analyzes backtest results and generates detailed reports.

Usage:
    python performance_analyzer.py --trades trades.csv
    python performance_analyzer.py --results results.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def load_trades(filepath: str) -> pd.DataFrame:
    """Load trades from CSV."""
    df = pd.read_csv(filepath)
    df['EntryTime'] = pd.to_datetime(df['EntryTime'])
    df['ExitTime'] = pd.to_datetime(df['ExitTime'])
    return df


def load_results(filepath: str) -> pd.DataFrame:
    """Load results from CSV."""
    return pd.read_csv(filepath)


# ============================================
# TRADE ANALYSIS
# ============================================
def analyze_trades(trades: pd.DataFrame) -> Dict:
    """Comprehensive trade analysis."""

    if trades.empty:
        return {"error": "No trades to analyze"}

    # Basic stats
    total_trades = len(trades)
    winners = trades[trades['PnL'] > 0]
    losers = trades[trades['PnL'] <= 0]

    # Direction breakdown
    long_trades = trades[trades['Direction'] == 'long']
    short_trades = trades[trades['Direction'] == 'short']

    # Time analysis
    trades['Duration'] = (trades['ExitTime'] - trades['EntryTime']).dt.total_seconds() / 3600  # hours

    # Results
    analysis = {
        # Overall
        'total_trades': total_trades,
        'winning_trades': len(winners),
        'losing_trades': len(losers),
        'win_rate': len(winners) / total_trades * 100 if total_trades > 0 else 0,

        # PnL
        'total_pnl': trades['PnL'].sum(),
        'avg_pnl': trades['PnL'].mean(),
        'median_pnl': trades['PnL'].median(),
        'std_pnl': trades['PnL'].std(),

        # Winners
        'avg_winner': winners['PnL'].mean() if len(winners) > 0 else 0,
        'max_winner': winners['PnL'].max() if len(winners) > 0 else 0,
        'avg_winner_pct': winners['PnLPct'].mean() if len(winners) > 0 else 0,

        # Losers
        'avg_loser': losers['PnL'].mean() if len(losers) > 0 else 0,
        'max_loser': losers['PnL'].min() if len(losers) > 0 else 0,
        'avg_loser_pct': losers['PnLPct'].mean() if len(losers) > 0 else 0,

        # Risk/Reward
        'profit_factor': abs(winners['PnL'].sum() / losers['PnL'].sum()) if len(losers) > 0 and losers['PnL'].sum() != 0 else 0,
        'avg_rr_ratio': abs(winners['PnL'].mean() / losers['PnL'].mean()) if len(losers) > 0 and losers['PnL'].mean() != 0 else 0,

        # Duration
        'avg_duration_hours': trades['Duration'].mean(),
        'avg_bars_held': trades['BarsHeld'].mean(),

        # By direction
        'long_trades': len(long_trades),
        'long_win_rate': len(long_trades[long_trades['PnL'] > 0]) / len(long_trades) * 100 if len(long_trades) > 0 else 0,
        'long_pnl': long_trades['PnL'].sum() if len(long_trades) > 0 else 0,

        'short_trades': len(short_trades),
        'short_win_rate': len(short_trades[short_trades['PnL'] > 0]) / len(short_trades) * 100 if len(short_trades) > 0 else 0,
        'short_pnl': short_trades['PnL'].sum() if len(short_trades) > 0 else 0,
    }

    return analysis


def analyze_by_symbol(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by symbol."""
    results = []

    for symbol in trades['Symbol'].unique():
        sym_trades = trades[trades['Symbol'] == symbol]
        winners = sym_trades[sym_trades['PnL'] > 0]

        results.append({
            'Symbol': symbol,
            'Trades': len(sym_trades),
            'Winners': len(winners),
            'WinRate': len(winners) / len(sym_trades) * 100 if len(sym_trades) > 0 else 0,
            'TotalPnL': sym_trades['PnL'].sum(),
            'AvgPnL': sym_trades['PnL'].mean(),
            'AvgBarsHeld': sym_trades['BarsHeld'].mean()
        })

    return pd.DataFrame(results).sort_values('TotalPnL', ascending=False)


def analyze_by_exit_reason(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by exit reason."""
    results = []

    for reason in trades['ExitReason'].unique():
        reason_trades = trades[trades['ExitReason'] == reason]
        winners = reason_trades[reason_trades['PnL'] > 0]

        results.append({
            'ExitReason': reason,
            'Trades': len(reason_trades),
            'WinRate': len(winners) / len(reason_trades) * 100 if len(reason_trades) > 0 else 0,
            'TotalPnL': reason_trades['PnL'].sum(),
            'AvgPnL': reason_trades['PnL'].mean()
        })

    return pd.DataFrame(results).sort_values('Trades', ascending=False)


def analyze_by_day_of_week(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by day of week."""
    trades = trades.copy()
    trades['DayOfWeek'] = trades['EntryTime'].dt.day_name()

    results = []
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
        day_trades = trades[trades['DayOfWeek'] == day]
        if len(day_trades) == 0:
            continue

        winners = day_trades[day_trades['PnL'] > 0]
        results.append({
            'Day': day,
            'Trades': len(day_trades),
            'WinRate': len(winners) / len(day_trades) * 100,
            'TotalPnL': day_trades['PnL'].sum(),
            'AvgPnL': day_trades['PnL'].mean()
        })

    return pd.DataFrame(results)


def analyze_by_hour(trades: pd.DataFrame) -> pd.DataFrame:
    """Analyze performance by entry hour."""
    trades = trades.copy()
    trades['Hour'] = trades['EntryTime'].dt.hour

    results = []
    for hour in sorted(trades['Hour'].unique()):
        hour_trades = trades[trades['Hour'] == hour]
        winners = hour_trades[hour_trades['PnL'] > 0]

        results.append({
            'Hour': f"{hour:02d}:00",
            'Trades': len(hour_trades),
            'WinRate': len(winners) / len(hour_trades) * 100 if len(hour_trades) > 0 else 0,
            'TotalPnL': hour_trades['PnL'].sum(),
            'AvgPnL': hour_trades['PnL'].mean()
        })

    return pd.DataFrame(results)


def calculate_streaks(trades: pd.DataFrame) -> Dict:
    """Calculate winning/losing streaks."""
    if trades.empty:
        return {'max_win_streak': 0, 'max_lose_streak': 0}

    # Sort by exit time
    trades = trades.sort_values('ExitTime')
    results = (trades['PnL'] > 0).astype(int).values

    # Calculate streaks
    max_win = 0
    max_lose = 0
    current_win = 0
    current_lose = 0

    for r in results:
        if r == 1:
            current_win += 1
            current_lose = 0
            max_win = max(max_win, current_win)
        else:
            current_lose += 1
            current_win = 0
            max_lose = max(max_lose, current_lose)

    return {
        'max_win_streak': max_win,
        'max_lose_streak': max_lose,
        'current_streak': current_win if results[-1] == 1 else -current_lose
    }


def calculate_monthly_returns(trades: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly returns."""
    if trades.empty:
        return pd.DataFrame()

    trades = trades.copy()
    trades['Month'] = trades['ExitTime'].dt.to_period('M')

    monthly = trades.groupby('Month').agg({
        'PnL': ['sum', 'count'],
        'Symbol': 'nunique'
    }).reset_index()

    monthly.columns = ['Month', 'PnL', 'Trades', 'Symbols']
    monthly['Month'] = monthly['Month'].astype(str)

    return monthly


# ============================================
# RISK METRICS
# ============================================
def calculate_risk_metrics(trades: pd.DataFrame, initial_capital: float = 100000) -> Dict:
    """Calculate advanced risk metrics."""
    if trades.empty:
        return {}

    # Sort by exit time
    trades = trades.sort_values('ExitTime')

    # Build equity curve
    equity = [initial_capital]
    for pnl in trades['PnL']:
        equity.append(equity[-1] + pnl)

    equity = np.array(equity)

    # Drawdown
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd = np.max(drawdown)
    max_dd_pct = max_dd / np.max(peak) * 100 if np.max(peak) > 0 else 0

    # Returns
    returns = np.diff(equity) / equity[:-1]

    # Sharpe (annualized, assuming ~250 trading days, 7 bars/day)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(250 * 7)
    else:
        sharpe = 0

    # Sortino (downside deviation)
    negative_returns = returns[returns < 0]
    if len(negative_returns) > 0 and np.std(negative_returns) > 0:
        sortino = np.mean(returns) / np.std(negative_returns) * np.sqrt(250 * 7)
    else:
        sortino = 0

    # Calmar ratio
    final_return = (equity[-1] - initial_capital) / initial_capital
    calmar = final_return / (max_dd_pct / 100) if max_dd_pct > 0 else 0

    return {
        'final_equity': equity[-1],
        'total_return': final_return * 100,
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd_pct,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'avg_return_per_trade': np.mean(returns) * 100,
        'volatility': np.std(returns) * 100
    }


# ============================================
# REPORTING
# ============================================
def print_analysis_report(trades: pd.DataFrame, initial_capital: float = 100000):
    """Print comprehensive analysis report."""

    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS REPORT")
    print("="*80)

    # Basic analysis
    analysis = analyze_trades(trades)

    print("\n--- OVERALL STATISTICS ---")
    print(f"Total Trades: {analysis['total_trades']}")
    print(f"Winners: {analysis['winning_trades']} ({analysis['win_rate']:.1f}%)")
    print(f"Losers: {analysis['losing_trades']}")
    print(f"Total PnL: ${analysis['total_pnl']:+,.2f}")
    print(f"Average PnL: ${analysis['avg_pnl']:+,.2f}")
    print(f"Profit Factor: {analysis['profit_factor']:.2f}")

    print("\n--- WINNER/LOSER ANALYSIS ---")
    print(f"Average Winner: ${analysis['avg_winner']:+,.2f} ({analysis['avg_winner_pct']:+.2f}%)")
    print(f"Average Loser: ${analysis['avg_loser']:+,.2f} ({analysis['avg_loser_pct']:+.2f}%)")
    print(f"Largest Winner: ${analysis['max_winner']:+,.2f}")
    print(f"Largest Loser: ${analysis['max_loser']:+,.2f}")
    print(f"Avg R:R Ratio: {analysis['avg_rr_ratio']:.2f}")

    print("\n--- DURATION ---")
    print(f"Avg Duration: {analysis['avg_duration_hours']:.1f} hours")
    print(f"Avg Bars Held: {analysis['avg_bars_held']:.1f}")

    print("\n--- BY DIRECTION ---")
    print(f"Long Trades: {analysis['long_trades']} | Win Rate: {analysis['long_win_rate']:.1f}% | PnL: ${analysis['long_pnl']:+,.2f}")
    print(f"Short Trades: {analysis['short_trades']} | Win Rate: {analysis['short_win_rate']:.1f}% | PnL: ${analysis['short_pnl']:+,.2f}")

    # Streaks
    streaks = calculate_streaks(trades)
    print("\n--- STREAKS ---")
    print(f"Max Winning Streak: {streaks['max_win_streak']}")
    print(f"Max Losing Streak: {streaks['max_lose_streak']}")

    # Risk metrics
    risk = calculate_risk_metrics(trades, initial_capital)
    print("\n--- RISK METRICS ---")
    print(f"Final Equity: ${risk['final_equity']:,.2f}")
    print(f"Total Return: {risk['total_return']:.2f}%")
    print(f"Max Drawdown: ${risk['max_drawdown']:,.2f} ({risk['max_drawdown_pct']:.2f}%)")
    print(f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {risk['sortino_ratio']:.2f}")
    print(f"Calmar Ratio: {risk['calmar_ratio']:.2f}")

    # By symbol
    print("\n--- BY SYMBOL ---")
    symbol_df = analyze_by_symbol(trades)
    print(symbol_df.to_string(index=False))

    # By exit reason
    print("\n--- BY EXIT REASON ---")
    exit_df = analyze_by_exit_reason(trades)
    print(exit_df.to_string(index=False))

    # By day
    print("\n--- BY DAY OF WEEK ---")
    day_df = analyze_by_day_of_week(trades)
    if not day_df.empty:
        print(day_df.to_string(index=False))

    # Monthly
    print("\n--- MONTHLY RETURNS ---")
    monthly_df = calculate_monthly_returns(trades)
    if not monthly_df.empty:
        print(monthly_df.to_string(index=False))

    print("\n" + "="*80)


def plot_equity_curve(trades: pd.DataFrame, initial_capital: float = 100000, save_path: str = None):
    """Plot equity curve."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available for plotting")
        return

    trades = trades.sort_values('ExitTime')

    # Build equity curve
    equity = [initial_capital]
    dates = [trades['EntryTime'].min()]

    for _, trade in trades.iterrows():
        equity.append(equity[-1] + trade['PnL'])
        dates.append(trade['ExitTime'])

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Equity curve
    axes[0].plot(dates, equity, 'b-', linewidth=1)
    axes[0].axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5)
    axes[0].set_title('Equity Curve')
    axes[0].set_ylabel('Portfolio Value ($)')
    axes[0].grid(True, alpha=0.3)

    # Drawdown
    equity_arr = np.array(equity)
    peak = np.maximum.accumulate(equity_arr)
    drawdown = (peak - equity_arr) / peak * 100

    axes[1].fill_between(dates, 0, drawdown, color='red', alpha=0.3)
    axes[1].plot(dates, drawdown, 'r-', linewidth=1)
    axes[1].set_title('Drawdown')
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_xlabel('Date')
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Chart saved to {save_path}")
    else:
        plt.show()


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Analyze Trading Performance')
    parser.add_argument('--trades', required=True, help='Trades CSV file')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--plot', action='store_true', help='Show equity curve plot')
    parser.add_argument('--save-plot', default=None, help='Save plot to file')

    args = parser.parse_args()

    print("Loading trades...")
    trades = load_trades(args.trades)
    print(f"Loaded {len(trades)} trades")

    print_analysis_report(trades, args.capital)

    if args.plot or args.save_plot:
        plot_equity_curve(trades, args.capital, args.save_plot)


if __name__ == "__main__":
    main()
