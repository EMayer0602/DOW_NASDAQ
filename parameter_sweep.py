#!/usr/bin/env python3
"""
Parameter Sweep for Stock Backtesting
Finds optimal Supertrend parameters for each symbol.

Usage:
    python parameter_sweep.py --symbols AAPL MSFT      # Specific symbols
    python parameter_sweep.py --all-dow                 # All DOW 30
    python parameter_sweep.py --all-nasdaq              # All NASDAQ 100
    python parameter_sweep.py --output best_params.csv  # Save results
"""

from __future__ import annotations

import argparse
import itertools
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

from backtester import Backtester, fetch_historical_data, BacktestResult
from stock_settings import SYMBOLS, START_TOTAL_CAPITAL, POSITION_SIZE_USD
from stock_symbols import DOW_30, NASDAQ_100_TOP


def run_parameter_sweep(
    symbol: str,
    df: pd.DataFrame,
    atr_periods: List[int],
    atr_multipliers: List[float],
    initial_capital: float = START_TOTAL_CAPITAL,
    position_size: float = POSITION_SIZE_USD
) -> List[Dict]:
    """
    Run parameter sweep for a single symbol.

    Returns list of results for each parameter combination.
    """
    results = []

    for period, mult in itertools.product(atr_periods, atr_multipliers):
        try:
            bt = Backtester(
                initial_capital=initial_capital,
                position_size=position_size,
                atr_period=period,
                atr_multiplier=mult
            )
            result = bt.run_backtest(symbol, df.copy())

            results.append({
                'Symbol': symbol,
                'ATR_Period': period,
                'ATR_Mult': mult,
                'Trades': result.total_trades,
                'WinRate': result.win_rate,
                'PnL': result.total_pnl,
                'Return': result.total_return_pct,
                'ProfitFactor': result.profit_factor,
                'MaxDD': result.max_drawdown_pct,
                'Sharpe': result.sharpe_ratio
            })
        except Exception as e:
            print(f"  Error with ATR({period},{mult}): {e}")

    return results


def find_best_params(results: List[Dict], metric: str = 'PnL') -> Dict:
    """Find best parameters based on specified metric."""
    if not results:
        return None

    # Filter out results with too few trades
    valid = [r for r in results if r['Trades'] >= 5]
    if not valid:
        valid = results

    # Sort by metric (descending)
    sorted_results = sorted(valid, key=lambda x: x.get(metric, 0), reverse=True)
    return sorted_results[0] if sorted_results else None


def main():
    parser = argparse.ArgumentParser(description='Parameter Sweep for Stock Backtesting')
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to test')
    parser.add_argument('--all-dow', action='store_true', help='Use all DOW 30 symbols')
    parser.add_argument('--all-nasdaq', action='store_true', help='Use all NASDAQ 100 symbols')
    parser.add_argument('--period', default='1y', help='Historical period (1mo, 3mo, 6mo, 1y)')
    parser.add_argument('--interval', default='1h', help='Bar interval (1h, 1d)')
    parser.add_argument('--output', default='best_params_stocks.csv', help='Output CSV file')
    parser.add_argument('--metric', default='PnL', choices=['PnL', 'WinRate', 'Sharpe', 'ProfitFactor'],
                        help='Metric to optimize')
    parser.add_argument('--quick', action='store_true', help='Quick sweep with fewer parameters')

    args = parser.parse_args()

    # Determine symbols
    if args.all_nasdaq:
        symbols = NASDAQ_100_TOP
    elif args.all_dow:
        symbols = DOW_30
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = SYMBOLS[:10]  # Default to first 10

    # Parameter ranges
    if args.quick:
        atr_periods = [7, 10, 14, 21]
        atr_multipliers = [2.0, 2.5, 3.0]
    else:
        atr_periods = [5, 7, 10, 12, 14, 18, 21, 25]
        atr_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    total_combinations = len(atr_periods) * len(atr_multipliers)

    print("=" * 60)
    print("PARAMETER SWEEP - STOCK SUPERTREND")
    print("=" * 60)
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {args.period}, Interval: {args.interval}")
    print(f"ATR Periods: {atr_periods}")
    print(f"ATR Multipliers: {atr_multipliers}")
    print(f"Total combinations per symbol: {total_combinations}")
    print(f"Optimizing for: {args.metric}")
    print("=" * 60 + "\n")

    all_results = []
    best_params = []

    for i, symbol in enumerate(symbols, 1):
        print(f"[{i}/{len(symbols)}] {symbol}...", end=" ", flush=True)

        df = fetch_historical_data(symbol, args.period, args.interval)
        if df is None or len(df) < 50:
            print("SKIPPED (insufficient data)")
            continue

        results = run_parameter_sweep(
            symbol, df, atr_periods, atr_multipliers
        )
        all_results.extend(results)

        best = find_best_params(results, args.metric)
        if best:
            best_params.append(best)
            print(f"Best: ATR({best['ATR_Period']},{best['ATR_Mult']}) "
                  f"-> {best['Trades']} trades, {best['WinRate']:.1f}% win, ${best['PnL']:+,.2f}")
        else:
            print("No valid results")

    # Summary
    print("\n" + "=" * 60)
    print("BEST PARAMETERS BY SYMBOL")
    print("=" * 60)
    print(f"{'Symbol':<8} {'ATR':>10} {'Trades':>7} {'Win%':>7} {'PnL':>12} {'PF':>6}")
    print("-" * 60)

    total_pnl = 0
    for p in sorted(best_params, key=lambda x: x['PnL'], reverse=True):
        print(f"{p['Symbol']:<8} ({p['ATR_Period']:>2},{p['ATR_Mult']:.1f}) "
              f"{p['Trades']:>7} {p['WinRate']:>6.1f}% ${p['PnL']:>10,.2f} {p['ProfitFactor']:>6.2f}")
        total_pnl += p['PnL']

    print("-" * 60)
    print(f"{'TOTAL':<8} {'':<10} {sum(p['Trades'] for p in best_params):>7} "
          f"{'':<7} ${total_pnl:>10,.2f}")

    # Save best params in format compatible with backtester
    if best_params:
        output_df = pd.DataFrame([{
            'Symbol': p['Symbol'],
            'Direction': 'long',
            'Indicator': 'supertrend',
            'ParamA': p['ATR_Period'],
            'ParamB': p['ATR_Mult'],
            'MinHoldBars': 5,
            'HTF': '1d',
            'WinRate': p['WinRate'],
            'PnL': p['PnL'],
            'ProfitFactor': p['ProfitFactor']
        } for p in best_params])

        output_df.to_csv(args.output, sep=';', index=False)
        print(f"\nBest parameters saved to {args.output}")

    # Also save all results for analysis
    if all_results:
        all_df = pd.DataFrame(all_results)
        all_file = args.output.replace('.csv', '_all.csv')
        all_df.to_csv(all_file, index=False)
        print(f"All results saved to {all_file}")


if __name__ == "__main__":
    main()
