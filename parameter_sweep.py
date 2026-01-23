#!/usr/bin/env python3
"""
Parameter Sweep for DOW/NASDAQ Trading Strategy
Finds optimal Supertrend parameters for each symbol.

Usage:
    python parameter_sweep.py                     # Sweep all default symbols
    python parameter_sweep.py --symbols AAPL      # Sweep single symbol
    python parameter_sweep.py --quick             # Quick sweep (fewer params)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from backtester import Backtester, fetch_historical_data, BacktestResult
from stock_settings import SYMBOLS, START_TOTAL_CAPITAL, POSITION_SIZE_USD


@dataclass
class SweepResult:
    """Result from parameter sweep."""
    symbol: str
    atr_period: int
    atr_multiplier: float
    total_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    score: float  # Combined optimization score


def calculate_score(result: BacktestResult) -> float:
    """
    Calculate optimization score from backtest result.
    Higher is better. Balances profitability with risk.
    """
    if result.total_trades < 3:
        return -999  # Not enough trades

    # Components (all normalized roughly 0-1)
    profit_score = min(result.total_return_pct / 50, 1.0)  # Cap at 50% return
    win_rate_score = result.win_rate / 100
    pf_score = min(result.profit_factor / 3, 1.0)  # Cap at PF=3
    dd_penalty = result.max_drawdown_pct / 100  # Lower is better
    sharpe_score = min(max(result.sharpe_ratio, 0) / 2, 1.0)  # Cap at Sharpe=2

    # Weighted combination
    score = (
        0.30 * profit_score +
        0.20 * win_rate_score +
        0.25 * pf_score +
        0.15 * sharpe_score -
        0.10 * dd_penalty
    )

    return score


def sweep_symbol(
    symbol: str,
    df: pd.DataFrame,
    atr_periods: List[int],
    atr_multipliers: List[float],
    initial_capital: float = START_TOTAL_CAPITAL,
    position_size: float = POSITION_SIZE_USD
) -> List[SweepResult]:
    """Run parameter sweep for single symbol."""
    results = []

    for period in atr_periods:
        for mult in atr_multipliers:
            backtester = Backtester(
                initial_capital=initial_capital,
                position_size=position_size,
                atr_period=period,
                atr_multiplier=mult
            )

            bt_result = backtester.run_backtest(symbol, df.copy())
            score = calculate_score(bt_result)

            results.append(SweepResult(
                symbol=symbol,
                atr_period=period,
                atr_multiplier=mult,
                total_trades=bt_result.total_trades,
                win_rate=bt_result.win_rate,
                total_pnl=bt_result.total_pnl,
                profit_factor=bt_result.profit_factor,
                max_drawdown_pct=bt_result.max_drawdown_pct,
                sharpe_ratio=bt_result.sharpe_ratio,
                score=score
            ))

    return results


def run_parameter_sweep(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1h",
    atr_periods: List[int] = None,
    atr_multipliers: List[float] = None,
    parallel: bool = True
) -> Dict[str, List[SweepResult]]:
    """Run parameter sweep on multiple symbols."""

    if atr_periods is None:
        atr_periods = [7, 10, 14, 20]
    if atr_multipliers is None:
        atr_multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]

    total_combos = len(atr_periods) * len(atr_multipliers)
    print(f"Testing {total_combos} parameter combinations per symbol")
    print(f"ATR Periods: {atr_periods}")
    print(f"ATR Multipliers: {atr_multipliers}")
    print()

    all_results = {}

    for symbol in symbols:
        print(f"Sweeping {symbol}...", end=" ", flush=True)

        df = fetch_historical_data(symbol, period, interval)
        if df is None or len(df) < 50:
            print("SKIPPED (insufficient data)")
            continue

        results = sweep_symbol(symbol, df, atr_periods, atr_multipliers)
        all_results[symbol] = results

        # Find best
        best = max(results, key=lambda r: r.score)
        print(f"Best: ATR({best.atr_period}, {best.atr_multiplier}) "
              f"Score={best.score:.3f}, PnL=${best.total_pnl:+,.0f}")

    return all_results


def find_best_params(results: Dict[str, List[SweepResult]]) -> pd.DataFrame:
    """Find best parameters for each symbol."""
    rows = []

    for symbol, sweep_results in results.items():
        if not sweep_results:
            continue

        # Best by score
        best = max(sweep_results, key=lambda r: r.score)

        rows.append({
            'Symbol': symbol,
            'Direction': 'long',
            'Indicator': 'supertrend',
            'ParamA': best.atr_period,
            'ParamB': best.atr_multiplier,
            'Trades': best.total_trades,
            'WinRate': best.win_rate,
            'TotalPnL': best.total_pnl,
            'ProfitFactor': best.profit_factor,
            'MaxDD%': best.max_drawdown_pct,
            'Sharpe': best.sharpe_ratio,
            'Score': best.score
        })

    return pd.DataFrame(rows)


def print_sweep_summary(results: Dict[str, List[SweepResult]]):
    """Print sweep summary."""
    print("\n" + "="*90)
    print("PARAMETER SWEEP SUMMARY")
    print("="*90)

    df = find_best_params(results)
    if df.empty:
        print("No results")
        return

    print(f"{'Symbol':<8} {'ATR':>10} {'Trades':>7} {'Win%':>7} {'PnL':>12} "
          f"{'PF':>6} {'MaxDD%':>8} {'Score':>7}")
    print("-"*90)

    for _, row in df.iterrows():
        atr_str = f"({row['ParamA']}, {row['ParamB']})"
        pnl_str = f"${row['TotalPnL']:+,.0f}"
        print(f"{row['Symbol']:<8} {atr_str:>10} {row['Trades']:>7} {row['WinRate']:>6.1f}% "
              f"{pnl_str:>12} {row['ProfitFactor']:>6.2f} {row['MaxDD%']:>7.2f}% {row['Score']:>7.3f}")

    print("="*90)


def save_best_params(df: pd.DataFrame, filepath: str):
    """Save best parameters to CSV (compatible with stock_paper_trader)."""
    # Format for stock_paper_trader
    output_df = df[['Symbol', 'Direction', 'Indicator', 'ParamA', 'ParamB']].copy()
    output_df.to_csv(filepath, sep=';', index=False)
    print(f"\nBest parameters saved to {filepath}")


def create_heatmap_data(results: List[SweepResult]) -> pd.DataFrame:
    """Create heatmap data from sweep results."""
    data = {}
    for r in results:
        mult = r.atr_multiplier
        period = r.atr_period
        if mult not in data:
            data[mult] = {}
        data[mult][period] = r.score

    return pd.DataFrame(data).T


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Parameter Sweep for Trading Strategy')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to sweep')
    parser.add_argument('--all-dow', action='store_true', help='Use all DOW 30 symbols')
    parser.add_argument('--all-nasdaq', action='store_true', help='Use all NASDAQ 100 symbols')
    parser.add_argument('--period', default='1y', help='Historical period')
    parser.add_argument('--interval', default='1h', help='Bar interval')
    parser.add_argument('--quick', action='store_true', help='Quick sweep (fewer params)')
    parser.add_argument('--thorough', action='store_true', help='Thorough sweep (more params)')
    parser.add_argument('--output', default='report_stocks/best_params_overall.csv',
                        help='Output file for best params')

    args = parser.parse_args()

    # Determine symbols
    if args.all_nasdaq:
        from stock_symbols import NASDAQ_100_TOP
        symbols = NASDAQ_100_TOP
    elif args.all_dow:
        from stock_symbols import DOW_30
        symbols = DOW_30
    else:
        symbols = args.symbols

    # Parameter ranges
    if args.quick:
        atr_periods = [10, 14]
        atr_multipliers = [2.5, 3.0, 3.5]
    elif args.thorough:
        atr_periods = [5, 7, 10, 12, 14, 17, 20, 25]
        atr_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    else:
        atr_periods = [7, 10, 14, 20]
        atr_multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]

    print("="*60)
    print("PARAMETER SWEEP - DOW/NASDAQ")
    print("="*60)
    print(f"Symbols: {len(symbols)} stocks")
    print(f"Period: {args.period}, Interval: {args.interval}")
    print(f"Mode: {'Quick' if args.quick else 'Thorough' if args.thorough else 'Standard'}")
    print("="*60 + "\n")

    results = run_parameter_sweep(
        symbols=symbols,
        period=args.period,
        interval=args.interval,
        atr_periods=atr_periods,
        atr_multipliers=atr_multipliers
    )

    print_sweep_summary(results)

    # Save results
    df = find_best_params(results)
    if not df.empty:
        # Ensure directory exists
        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        save_best_params(df, args.output)

        # Also save full results
        full_output = args.output.replace('.csv', '_full.csv')
        df.to_csv(full_output, index=False)
        print(f"Full results saved to {full_output}")


if __name__ == "__main__":
    main()
