#!/usr/bin/env python3
"""
Daily Bar Parameter Sweep for DOW/NASDAQ Stocks
Supertrend Strategy Optimization - Uses Interactive Brokers data

Usage:
    python stock_sweep_daily.py                          # Sweep all default symbols
    python stock_sweep_daily.py --symbols AAPL MSFT      # Specific symbols
    python stock_sweep_daily.py --quick                  # Quick sweep (fewer params)

Requirements:
    - TWS or IB Gateway running with API enabled
    - Port 7497 (paper) or 7496 (live)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd

# Import IB connector
try:
    from ib_connector import IBConnector
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("ERROR: ib_connector not found")

from stock_symbols import DOW_30, NASDAQ_100_TOP, DEFAULT_TRADING_SYMBOLS, ALL_STOCKS

# Global IB connector (initialized in main)
_ib_connector: Optional[IBConnector] = None


# ============================================
# SWEEP CONFIGURATION
# ============================================
# Parameter ranges for sweep
ATR_PERIODS = [7, 10, 14]
ATR_MULTIPLIERS = [2.0, 2.5, 3.0, 3.5]
MIN_HOLD_BARS = [3, 4, 5, 6, 7, 8]  # Days

# Quick sweep (fewer combinations)
ATR_PERIODS_QUICK = [7, 10, 14]
ATR_MULTIPLIERS_QUICK = [2.0, 3.0]
MIN_HOLD_BARS_QUICK = [4, 6, 8]

# Backtest settings
INITIAL_CAPITAL = 10_000.0  # Per symbol backtest
POSITION_SIZE = 10_000.0    # Full position per signal
COMMISSION_PER_TRADE = 5.0  # $5 minimum per trade (round trip = $10)

# Data settings
YEARS_OF_DATA = 2
MONTHS_TO_EXCLUDE = 1  # Last month for out-of-sample testing

# Output
REPORT_DIR = "report_stocks"
OUTPUT_CSV = "best_params_daily.csv"


# ============================================
# SUPERTREND CALCULATION
# ============================================
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate Supertrend indicator."""
    df = df.copy()

    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Basic bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Final bands
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)

    for i in range(period, len(df)):
        # Upper band
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        # Lower band
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    # Supertrend
    for i in range(period, len(df)):
        if supertrend.iloc[i-1] == final_upper.iloc[i-1]:
            if close.iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1]:
            if close.iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
        else:
            supertrend.iloc[i] = final_lower.iloc[i]

    df['supertrend'] = supertrend
    df['trend'] = np.where(close > supertrend, 1, -1)

    return df


# ============================================
# DATA FETCHING (IB)
# ============================================
def fetch_daily_data(symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV data from Interactive Brokers."""
    global _ib_connector

    if _ib_connector is None or not _ib_connector.connected:
        print(f"[{symbol}] IB not connected")
        return None

    try:
        # IB duration string for 2 years
        duration = f"{years} Y"

        df = _ib_connector.get_ohlcv(symbol, "1 day", duration)

        if df is None or df.empty:
            print(f"[{symbol}] No data returned from IB")
            return None

        # Normalize columns (should already be correct from ib_connector)
        df.columns = [c.lower() for c in df.columns]

        # Ensure we have the required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            print(f"[{symbol}] Missing columns in IB data")
            return None

        df = df[required]

        # Remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        print(f"[{symbol}] Fetched {len(df)} daily bars from IB")
        return df

    except Exception as e:
        print(f"[{symbol}] Error fetching IB data: {e}")
        return None


def split_data(df: pd.DataFrame, months_exclude: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training (sweep) and test (out-of-sample) sets."""
    if df is None or len(df) < 100:
        return None, None

    # Calculate split date (exclude last N months)
    end_date = df.index[-1]
    split_date = end_date - timedelta(days=months_exclude * 30)

    train_df = df[df.index < split_date].copy()
    test_df = df[df.index >= split_date].copy()

    return train_df, test_df


# ============================================
# BACKTEST ENGINE
# ============================================
@dataclass
class BacktestResult:
    """Results from a single backtest run."""
    total_pnl: float
    total_pnl_pct: float
    trades: int
    wins: int
    losses: int
    win_rate: float
    max_drawdown: float
    max_drawdown_pct: float
    avg_trade_pnl: float
    profit_factor: float
    final_equity: float


def run_backtest(
    df: pd.DataFrame,
    atr_period: int,
    atr_multiplier: float,
    min_hold_bars: int,
    initial_capital: float = INITIAL_CAPITAL,
    position_size: float = POSITION_SIZE,
    commission: float = COMMISSION_PER_TRADE
) -> Optional[BacktestResult]:
    """
    Run backtest with given parameters.
    Long-only strategy with time-based and trend-flip exits.
    """
    if df is None or len(df) < atr_period + 20:
        return None

    # Calculate Supertrend
    df = calculate_supertrend(df, period=atr_period, multiplier=atr_multiplier)

    capital = initial_capital
    position = None  # {'entry_price', 'entry_idx', 'shares'}
    trades = []
    equity_curve = [initial_capital]
    peak_equity = initial_capital
    max_drawdown = 0.0

    for i in range(atr_period + 1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        close = current['close']
        st = current['supertrend']
        prev_close = prev['close']
        prev_st = prev['supertrend']

        # Check exit if in position
        if position is not None:
            bars_held = i - position['entry_idx']
            should_exit = False

            # Time-based exit
            if bars_held >= min_hold_bars:
                should_exit = True

            # Trend flip exit (price crosses below supertrend)
            if close < st:
                should_exit = True

            if should_exit:
                # Close position
                exit_price = close
                shares = position['shares']
                pnl = shares * (exit_price - position['entry_price']) - commission
                capital += shares * exit_price - commission

                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'shares': shares,
                    'pnl': pnl,
                    'bars_held': bars_held
                })
                position = None

        # Check entry if no position
        if position is None:
            # Long signal: price crosses above supertrend
            if prev_close <= prev_st and close > st:
                # Calculate shares
                shares = int(position_size / close)
                if shares > 0 and shares * close <= capital:
                    entry_cost = shares * close + commission
                    capital -= entry_cost
                    position = {
                        'entry_price': close,
                        'entry_idx': i,
                        'shares': shares
                    }

        # Update equity
        if position is not None:
            current_equity = capital + position['shares'] * close
        else:
            current_equity = capital

        equity_curve.append(current_equity)

        # Update max drawdown
        if current_equity > peak_equity:
            peak_equity = current_equity
        drawdown = peak_equity - current_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    # Close any remaining position
    if position is not None:
        exit_price = df.iloc[-1]['close']
        shares = position['shares']
        pnl = shares * (exit_price - position['entry_price']) - commission
        capital += shares * exit_price - commission
        trades.append({
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'bars_held': len(df) - position['entry_idx']
        })

    # Calculate statistics
    if len(trades) == 0:
        return BacktestResult(
            total_pnl=0, total_pnl_pct=0, trades=0, wins=0, losses=0,
            win_rate=0, max_drawdown=0, max_drawdown_pct=0,
            avg_trade_pnl=0, profit_factor=0, final_equity=initial_capital
        )

    total_pnl = sum(t['pnl'] for t in trades)
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

    return BacktestResult(
        total_pnl=total_pnl,
        total_pnl_pct=(total_pnl / initial_capital) * 100,
        trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        win_rate=(len(wins) / len(trades)) * 100 if trades else 0,
        max_drawdown=max_drawdown,
        max_drawdown_pct=(max_drawdown / peak_equity) * 100 if peak_equity > 0 else 0,
        avg_trade_pnl=total_pnl / len(trades),
        profit_factor=profit_factor,
        final_equity=capital
    )


# ============================================
# PARAMETER SWEEP
# ============================================
def sweep_symbol(
    symbol: str,
    df: pd.DataFrame,
    atr_periods: List[int],
    atr_multipliers: List[float],
    hold_bars: List[int],
    verbose: bool = False
) -> Optional[Dict]:
    """
    Run parameter sweep for a single symbol.
    Returns best parameters based on profit factor and PnL.
    """
    if df is None or len(df) < 100:
        print(f"[{symbol}] Insufficient data for sweep")
        return None

    best_result = None
    best_params = None
    best_score = -float('inf')

    total_combos = len(atr_periods) * len(atr_multipliers) * len(hold_bars)
    tested = 0

    for period, mult, hold in itertools.product(atr_periods, atr_multipliers, hold_bars):
        result = run_backtest(df, period, mult, hold)
        tested += 1

        if result is None or result.trades < 5:
            continue

        # Score: prioritize profit factor, then PnL, penalize drawdown
        score = (
            result.profit_factor * 100 +
            result.total_pnl_pct * 2 -
            result.max_drawdown_pct * 0.5
        )

        if score > best_score:
            best_score = score
            best_result = result
            best_params = {
                'atr_period': period,
                'atr_multiplier': mult,
                'min_hold_bars': hold
            }

        if verbose and tested % 10 == 0:
            print(f"  [{symbol}] Tested {tested}/{total_combos} combinations...")

    if best_result is None:
        print(f"[{symbol}] No valid results found")
        return None

    return {
        'symbol': symbol,
        'params': best_params,
        'result': best_result
    }


def run_sweep(
    symbols: List[str],
    quick: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """Run parameter sweep for all symbols."""

    # Select parameter ranges
    if quick:
        atr_periods = ATR_PERIODS_QUICK
        atr_multipliers = ATR_MULTIPLIERS_QUICK
        hold_bars = MIN_HOLD_BARS_QUICK
        print("Running QUICK sweep (fewer parameters)")
    else:
        atr_periods = ATR_PERIODS
        atr_multipliers = ATR_MULTIPLIERS
        hold_bars = MIN_HOLD_BARS
        print("Running FULL sweep")

    total_combos = len(atr_periods) * len(atr_multipliers) * len(hold_bars)
    print(f"Parameter combinations per symbol: {total_combos}")
    print(f"  ATR Periods: {atr_periods}")
    print(f"  ATR Multipliers: {atr_multipliers}")
    print(f"  Hold Bars (days): {hold_bars}")
    print(f"\nSymbols to sweep: {len(symbols)}")
    print("="*60)

    results = []

    for idx, symbol in enumerate(symbols, 1):
        print(f"\n[{idx}/{len(symbols)}] Processing {symbol}...")

        # Fetch data
        df = fetch_daily_data(symbol, years=YEARS_OF_DATA)
        if df is None:
            continue

        # Split into train/test
        train_df, test_df = split_data(df, months_exclude=MONTHS_TO_EXCLUDE)
        if train_df is None or len(train_df) < 100:
            print(f"  Insufficient training data for {symbol}")
            continue

        print(f"  Data: {len(df)} days total, {len(train_df)} train, {len(test_df)} test")

        # Run sweep on training data
        sweep_result = sweep_symbol(
            symbol, train_df,
            atr_periods, atr_multipliers, hold_bars,
            verbose=verbose
        )

        if sweep_result is None:
            continue

        # Validate on test data (out-of-sample)
        params = sweep_result['params']
        test_result = run_backtest(
            test_df,
            params['atr_period'],
            params['atr_multiplier'],
            params['min_hold_bars']
        )

        sweep_result['test_result'] = test_result
        results.append(sweep_result)

        # Print summary
        train_res = sweep_result['result']
        print(f"  TRAIN: PnL=${train_res.total_pnl:+.2f} ({train_res.total_pnl_pct:+.1f}%), "
              f"WR={train_res.win_rate:.1f}%, Trades={train_res.trades}, "
              f"PF={train_res.profit_factor:.2f}")

        if test_result:
            print(f"  TEST:  PnL=${test_result.total_pnl:+.2f} ({test_result.total_pnl_pct:+.1f}%), "
                  f"WR={test_result.win_rate:.1f}%, Trades={test_result.trades}")

        print(f"  Best params: ATR({params['atr_period']}), "
              f"Mult({params['atr_multiplier']}), Hold({params['min_hold_bars']} days)")

    return results


def save_results(results: List[Dict], output_path: str):
    """Save sweep results to CSV."""
    if not results:
        print("No results to save")
        return

    rows = []
    for r in results:
        params = r['params']
        train = r['result']
        test = r.get('test_result')

        rows.append({
            'Symbol': r['symbol'],
            'Direction': 'Long',
            'Indicator': 'supertrend',
            'IndicatorDisplay': 'Supertrend',
            'ParamA': params['atr_period'],
            'ParamB': str(params['atr_multiplier']).replace('.', ','),
            'Length': str(float(params['atr_period'])).replace('.', ','),
            'Factor': str(params['atr_multiplier']).replace('.', ','),
            'ATRStopMult': 'None',
            'ATRStopMultValue': '',
            'MinHoldBars': params['min_hold_bars'],
            'HTF': '1d',
            'FinalEquity': str(train.final_equity).replace('.', ','),
            'Trades': train.trades,
            'WinRate': str(round(train.win_rate / 100, 2)).replace('.', ','),
            'MaxDrawdown': str(round(train.max_drawdown, 1)).replace('.', ','),
            'ProfitFactor': str(round(train.profit_factor, 2)).replace('.', ','),
            'TestPnL': str(round(test.total_pnl, 2)).replace('.', ',') if test else '',
            'TestWinRate': str(round(test.win_rate / 100, 2)).replace('.', ',') if test else '',
            'Phase': '',
            'SlowLength': ''
        })

    df = pd.DataFrame(rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    df.to_csv(output_path, sep=';', index=False)
    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[Dict]):
    """Print summary of sweep results."""
    if not results:
        return

    print("\n" + "="*70)
    print("SWEEP SUMMARY - DAILY BARS")
    print("="*70)
    print(f"{'Symbol':<8} {'ATR':<5} {'Mult':<5} {'Hold':<5} {'Trades':<7} {'WinRate':<8} {'PnL%':<10} {'TestPnL%':<10}")
    print("-"*70)

    total_train_pnl = 0
    total_test_pnl = 0

    for r in results:
        params = r['params']
        train = r['result']
        test = r.get('test_result')

        test_pnl = f"{test.total_pnl_pct:+.1f}%" if test else "N/A"

        print(f"{r['symbol']:<8} {params['atr_period']:<5} {params['atr_multiplier']:<5} "
              f"{params['min_hold_bars']:<5} {train.trades:<7} {train.win_rate:<7.1f}% "
              f"{train.total_pnl_pct:+.1f}%{'':<4} {test_pnl:<10}")

        total_train_pnl += train.total_pnl_pct
        if test:
            total_test_pnl += test.total_pnl_pct

    print("-"*70)
    avg_train = total_train_pnl / len(results) if results else 0
    avg_test = total_test_pnl / len(results) if results else 0
    print(f"{'AVERAGE':<8} {'':<5} {'':<5} {'':<5} {'':<7} {'':<8} {avg_train:+.1f}%{'':<4} {avg_test:+.1f}%")
    print("="*70)


# ============================================
# MAIN
# ============================================
def main():
    global _ib_connector

    parser = argparse.ArgumentParser(description='Daily Bar Parameter Sweep for Stocks (IB)')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to sweep (default: DEFAULT_TRADING_SYMBOLS)')
    parser.add_argument('--all', action='store_true',
                        help='Sweep all DOW 30 + NASDAQ 100 stocks')
    parser.add_argument('--dow', action='store_true',
                        help='Sweep DOW 30 stocks only')
    parser.add_argument('--quick', action='store_true',
                        help='Quick sweep with fewer parameter combinations')
    parser.add_argument('--output', default=None,
                        help='Output CSV path')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')
    parser.add_argument('--live', action='store_true',
                        help='Connect to live IB (default: paper)')

    args = parser.parse_args()

    # Check IB availability
    if not IB_AVAILABLE:
        print("ERROR: ib_connector module not available")
        print("Make sure ib_insync is installed: pip install ib_insync")
        sys.exit(1)

    # Select symbols
    if args.symbols:
        symbols = args.symbols
    elif args.all:
        symbols = ALL_STOCKS
    elif args.dow:
        symbols = DOW_30
    else:
        symbols = DEFAULT_TRADING_SYMBOLS

    # Output path
    output_path = args.output or os.path.join(REPORT_DIR, OUTPUT_CSV)

    print("="*60)
    print("DAILY BAR PARAMETER SWEEP (IB)")
    print("Supertrend Strategy Optimization")
    print("="*60)
    print(f"Data source: Interactive Brokers")
    print(f"Data period: {YEARS_OF_DATA} years")
    print(f"Out-of-sample: Last {MONTHS_TO_EXCLUDE} month(s)")
    print(f"Commission: ${COMMISSION_PER_TRADE} per trade")
    print(f"Output: {output_path}")
    print("="*60)

    # Connect to IB
    print("\nConnecting to Interactive Brokers...")
    _ib_connector = IBConnector(paper_trading=not args.live)

    if not _ib_connector.connect():
        print("ERROR: Could not connect to IB")
        print("Make sure TWS or IB Gateway is running with API enabled")
        print(f"  Paper trading: port 7497 (TWS) or 4002 (Gateway)")
        print(f"  Live trading: port 7496 (TWS) or 4001 (Gateway)")
        sys.exit(1)

    print("Connected to IB successfully!\n")

    try:
        # Run sweep
        results = run_sweep(symbols, quick=args.quick, verbose=args.verbose)

        # Print summary
        print_summary(results)

        # Save results
        save_results(results, output_path)

        print(f"\nSweep complete! {len(results)} symbols processed.")
        print(f"Best parameters saved to: {output_path}")

    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
    finally:
        # Disconnect from IB
        if _ib_connector:
            _ib_connector.disconnect()
            print("Disconnected from IB")


if __name__ == "__main__":
    main()
