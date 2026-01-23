#!/usr/bin/env python3
"""
Extended Parameter Sweep for DOW/NASDAQ Trading Strategy
Tests: ATR, Hold Time, HTF Filter, Trend Flip Exit, MA Types

Usage:
    python parameter_sweep.py --all-nasdaq           # Full sweep NASDAQ
    python parameter_sweep.py --quick                # Quick sweep
    python parameter_sweep.py --full                 # All parameters
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from backtester import Backtester, fetch_historical_data, BacktestResult, calculate_supertrend
from stock_settings import SYMBOLS, START_TOTAL_CAPITAL, POSITION_SIZE_USD


@dataclass
class ExtendedSweepResult:
    """Result from extended parameter sweep."""
    symbol: str
    # Supertrend params
    atr_period: int
    atr_multiplier: float
    # Exit params
    hold_bars: int
    use_time_exit: bool
    use_trend_flip: bool
    # Score
    total_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    score: float


def calculate_score(result: BacktestResult) -> float:
    """Calculate optimization score. Higher is better."""
    if result.total_trades < 3:
        return -999

    profit_score = min(result.total_return_pct / 50, 1.0)
    win_rate_score = result.win_rate / 100
    pf_score = min(result.profit_factor / 3, 1.0)
    dd_penalty = result.max_drawdown_pct / 100
    sharpe_score = min(max(result.sharpe_ratio, 0) / 2, 1.0)

    score = (
        0.30 * profit_score +
        0.20 * win_rate_score +
        0.25 * pf_score +
        0.15 * sharpe_score -
        0.10 * dd_penalty
    )
    return score


class ExtendedBacktester(Backtester):
    """Extended backtester with configurable exit strategies."""

    def __init__(
        self,
        hold_bars: int = 5,
        use_time_exit: bool = True,
        use_trend_flip: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.hold_bars = hold_bars
        self.use_time_exit_override = use_time_exit
        self.use_trend_flip_override = use_trend_flip

    def run_backtest(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        """Run backtest with custom exit logic."""
        self.reset()

        df = calculate_supertrend(df, self.atr_period, self.atr_multiplier)
        start_idx = self.atr_period + 5

        for i in range(start_idx, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            timestamp = df.index[i]
            price = current['close']

            self.equity_curve.append(self.get_portfolio_value({symbol: price}))

            # Check existing position
            if symbol in self.positions:
                pos = self.positions[symbol]
                bars_held = i - pos.entry_bar

                should_exit = False
                reason = ""

                # Time-based exit
                if self.use_time_exit_override and bars_held >= self.hold_bars:
                    should_exit = True
                    reason = f"Time exit ({bars_held} bars)"

                # Trend flip exit
                if not should_exit and self.use_trend_flip_override:
                    if pos.direction == "long" and price < current['supertrend']:
                        should_exit = True
                        reason = "Trend flip (bearish)"
                    elif pos.direction == "short" and price > current['supertrend']:
                        should_exit = True
                        reason = "Trend flip (bullish)"

                if should_exit:
                    self.close_position(symbol, price, timestamp, i, reason)

            # Check for new entry
            if symbol not in self.positions:
                # Long signal
                if prev['close'] <= prev['supertrend'] and price > current['supertrend']:
                    if self.can_open("long"):
                        self.open_position(symbol, "long", price, timestamp, i)
                # Short signal
                elif prev['close'] >= prev['supertrend'] and price < current['supertrend']:
                    if self.can_open("short"):
                        self.open_position(symbol, "short", price, timestamp, i)

        # Close remaining positions
        if symbol in self.positions:
            last_price = df.iloc[-1]['close']
            last_time = df.index[-1]
            self.close_position(symbol, last_price, last_time, len(df)-1, "End of backtest")

        return self._calculate_results(symbol, df)


def run_extended_sweep(
    symbol: str,
    df: pd.DataFrame,
    atr_periods: List[int],
    atr_multipliers: List[float],
    hold_bars_list: List[int],
    test_time_exit: List[bool],
    test_trend_flip: List[bool],
) -> List[ExtendedSweepResult]:
    """Run extended parameter sweep for single symbol."""
    results = []
    total_combos = len(atr_periods) * len(atr_multipliers) * len(hold_bars_list) * len(test_time_exit) * len(test_trend_flip)

    combo = 0
    for atr_period in atr_periods:
        for atr_mult in atr_multipliers:
            for hold_bars in hold_bars_list:
                for use_time in test_time_exit:
                    for use_flip in test_trend_flip:
                        combo += 1

                        # Skip invalid combinations
                        if not use_time and not use_flip:
                            continue  # Need at least one exit

                        backtester = ExtendedBacktester(
                            atr_period=atr_period,
                            atr_multiplier=atr_mult,
                            hold_bars=hold_bars,
                            use_time_exit=use_time,
                            use_trend_flip=use_flip
                        )

                        bt_result = backtester.run_backtest(symbol, df.copy())
                        score = calculate_score(bt_result)

                        results.append(ExtendedSweepResult(
                            symbol=symbol,
                            atr_period=atr_period,
                            atr_multiplier=atr_mult,
                            hold_bars=hold_bars,
                            use_time_exit=use_time,
                            use_trend_flip=use_flip,
                            total_trades=bt_result.total_trades,
                            win_rate=bt_result.win_rate,
                            total_pnl=bt_result.total_pnl,
                            profit_factor=bt_result.profit_factor,
                            max_drawdown_pct=bt_result.max_drawdown_pct,
                            sharpe_ratio=bt_result.sharpe_ratio,
                            score=score
                        ))

    return results


def run_full_sweep(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1h",
    atr_periods: List[int] = None,
    atr_multipliers: List[float] = None,
    hold_bars_list: List[int] = None,
    test_exits: bool = True
) -> Dict[str, List[ExtendedSweepResult]]:
    """Run full parameter sweep on multiple symbols."""

    if atr_periods is None:
        atr_periods = [7, 10, 14, 20]
    if atr_multipliers is None:
        atr_multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]
    if hold_bars_list is None:
        hold_bars_list = [3, 5, 7, 10, 14]

    # Exit strategy combinations
    if test_exits:
        test_time_exit = [True, False]
        test_trend_flip = [True, False]
    else:
        test_time_exit = [True]
        test_trend_flip = [True]

    total_combos = len(atr_periods) * len(atr_multipliers) * len(hold_bars_list)
    if test_exits:
        total_combos *= 3  # 3 valid exit combinations

    print(f"Testing {total_combos} parameter combinations per symbol")
    print(f"ATR Periods: {atr_periods}")
    print(f"ATR Multipliers: {atr_multipliers}")
    print(f"Hold Bars: {hold_bars_list}")
    print(f"Exit Strategies: Time={test_time_exit}, TrendFlip={test_trend_flip}")
    print()

    all_results = {}

    for symbol in symbols:
        print(f"Sweeping {symbol}...", end=" ", flush=True)

        df = fetch_historical_data(symbol, period, interval)
        if df is None or len(df) < 50:
            print("SKIPPED (insufficient data)")
            continue

        results = run_extended_sweep(
            symbol, df,
            atr_periods, atr_multipliers,
            hold_bars_list,
            test_time_exit, test_trend_flip
        )
        all_results[symbol] = results

        # Find best
        best = max(results, key=lambda r: r.score)
        exit_str = f"T{'Y' if best.use_time_exit else 'N'}F{'Y' if best.use_trend_flip else 'N'}"
        print(f"Best: ATR({best.atr_period},{best.atr_multiplier}) Hold={best.hold_bars} Exit={exit_str} "
              f"Score={best.score:.3f} PnL=${best.total_pnl:+,.0f}")

    return all_results


def find_best_params(results: Dict[str, List[ExtendedSweepResult]]) -> pd.DataFrame:
    """Find best parameters for each symbol."""
    rows = []

    for symbol, sweep_results in results.items():
        if not sweep_results:
            continue

        best = max(sweep_results, key=lambda r: r.score)

        rows.append({
            'Symbol': symbol,
            'Direction': 'long',
            'Indicator': 'supertrend',
            'ParamA': best.atr_period,
            'ParamB': best.atr_multiplier,
            'HoldBars': best.hold_bars,
            'TimeExit': best.use_time_exit,
            'TrendFlip': best.use_trend_flip,
            'Trades': best.total_trades,
            'WinRate': best.win_rate,
            'TotalPnL': best.total_pnl,
            'ProfitFactor': best.profit_factor,
            'MaxDD%': best.max_drawdown_pct,
            'Sharpe': best.sharpe_ratio,
            'Score': best.score
        })

    return pd.DataFrame(rows)


def print_sweep_summary(results: Dict[str, List[ExtendedSweepResult]]):
    """Print sweep summary."""
    print("\n" + "="*110)
    print("EXTENDED PARAMETER SWEEP SUMMARY")
    print("="*110)

    df = find_best_params(results)
    if df.empty:
        print("No results")
        return

    print(f"{'Symbol':<8} {'ATR':>10} {'Hold':>6} {'Exit':>6} {'Trades':>7} {'Win%':>7} "
          f"{'PnL':>12} {'PF':>6} {'MaxDD%':>8} {'Score':>7}")
    print("-"*110)

    for _, row in df.iterrows():
        atr_str = f"({row['ParamA']},{row['ParamB']})"
        exit_str = f"T{'Y' if row['TimeExit'] else 'N'}F{'Y' if row['TrendFlip'] else 'N'}"
        pnl_str = f"${row['TotalPnL']:+,.0f}"
        print(f"{row['Symbol']:<8} {atr_str:>10} {row['HoldBars']:>6} {exit_str:>6} {row['Trades']:>7} "
              f"{row['WinRate']:>6.1f}% {pnl_str:>12} {row['ProfitFactor']:>6.2f} "
              f"{row['MaxDD%']:>7.2f}% {row['Score']:>7.3f}")

    print("="*110)

    # Summary stats
    total_pnl = df['TotalPnL'].sum()
    avg_win_rate = df['WinRate'].mean()
    avg_pf = df['ProfitFactor'].mean()
    print(f"\nTotal PnL: ${total_pnl:+,.0f} | Avg Win Rate: {avg_win_rate:.1f}% | Avg PF: {avg_pf:.2f}")


def save_best_params(df: pd.DataFrame, filepath: str):
    """Save best parameters to CSV."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Format for backtester
    output_df = df[['Symbol', 'Direction', 'Indicator', 'ParamA', 'ParamB']].copy()
    output_df.to_csv(filepath, sep=';', index=False)
    print(f"\nBest parameters saved to {filepath}")

    # Also save full results
    full_output = filepath.replace('.csv', '_full.csv')
    df.to_csv(full_output, index=False)
    print(f"Full results saved to {full_output}")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Extended Parameter Sweep')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to sweep')
    parser.add_argument('--all-dow', action='store_true', help='Use all DOW 30 symbols')
    parser.add_argument('--all-nasdaq', action='store_true', help='Use all NASDAQ 100 symbols')
    parser.add_argument('--period', default='1y', help='Historical period')
    parser.add_argument('--interval', default='1h', help='Bar interval')
    parser.add_argument('--quick', action='store_true', help='Quick sweep (fewer params)')
    parser.add_argument('--full', action='store_true', help='Full sweep (all exit strategies)')
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
        hold_bars_list = [5, 7]
        test_exits = False
    elif args.thorough:
        atr_periods = [5, 7, 10, 12, 14, 17, 20, 25]
        atr_multipliers = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        hold_bars_list = [3, 5, 7, 10, 14, 20]
        test_exits = True
    else:
        atr_periods = [7, 10, 14, 20]
        atr_multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]
        hold_bars_list = [3, 5, 7, 10]
        test_exits = args.full

    print("="*60)
    print("EXTENDED PARAMETER SWEEP - DOW/NASDAQ")
    print("="*60)
    print(f"Symbols: {len(symbols)} stocks")
    print(f"Period: {args.period}, Interval: {args.interval}")
    print(f"Mode: {'Quick' if args.quick else 'Thorough' if args.thorough else 'Full' if args.full else 'Standard'}")
    print("="*60 + "\n")

    results = run_full_sweep(
        symbols=symbols,
        period=args.period,
        interval=args.interval,
        atr_periods=atr_periods,
        atr_multipliers=atr_multipliers,
        hold_bars_list=hold_bars_list,
        test_exits=test_exits
    )

    print_sweep_summary(results)

    # Save results
    df = find_best_params(results)
    if not df.empty:
        save_best_params(df, args.output)


if __name__ == "__main__":
    main()
