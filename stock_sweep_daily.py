#!/usr/bin/env python3
"""
Daily Bar Parameter Sweep for DOW/NASDAQ Stocks
Multi-Indicator Strategy Optimization - Uses Interactive Brokers data

Indicators tested:
    - Supertrend (ATR-based trend following)
    - EMA (Exponential Moving Average crossover)
    - KAMA (Kaufman Adaptive Moving Average)
    - JMA (Jurik Moving Average approximation)

Usage:
    python stock_sweep_daily.py                          # Sweep all default symbols
    python stock_sweep_daily.py --symbols AAPL MSFT      # Specific symbols
    python stock_sweep_daily.py --quick                  # Quick sweep (fewer params)
    python stock_sweep_daily.py --indicator supertrend   # Single indicator
    python stock_sweep_daily.py --all --quick            # All DOW+NASDAQ, quick mode

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
# Available indicators
INDICATORS = ['supertrend', 'ema', 'kama', 'jma']

# Supertrend parameters
ATR_PERIODS = [7, 10, 14]
ATR_MULTIPLIERS = [2.0, 2.5, 3.0, 3.5]

# EMA parameters (fast/slow periods)
EMA_FAST_PERIODS = [8, 12, 20]
EMA_SLOW_PERIODS = [21, 50, 100]

# KAMA parameters (efficiency ratio period, fast/slow constants)
KAMA_PERIODS = [10, 14, 21]
KAMA_FAST = [2, 3]
KAMA_SLOW = [30, 40]

# JMA parameters (period, phase, power)
JMA_PERIODS = [7, 14, 21]
JMA_PHASES = [0, 50, 100]  # -100 to +100, 0 = balanced
JMA_POWERS = [1, 2]  # 1 = linear, 2 = squared smoothing

# Hold bars for all indicators
MIN_HOLD_BARS = [3, 4, 5, 6, 7, 8]  # Days

# Quick sweep (fewer combinations)
ATR_PERIODS_QUICK = [7, 10, 14]
ATR_MULTIPLIERS_QUICK = [2.0, 3.0]
EMA_FAST_PERIODS_QUICK = [12, 20]
EMA_SLOW_PERIODS_QUICK = [50]
KAMA_PERIODS_QUICK = [10, 14]
KAMA_FAST_QUICK = [2]
KAMA_SLOW_QUICK = [30]
JMA_PERIODS_QUICK = [7, 14]
JMA_PHASES_QUICK = [0, 50]
JMA_POWERS_QUICK = [2]
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
# EMA CALCULATION
# ============================================
def calculate_ema(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 50) -> pd.DataFrame:
    """Calculate EMA crossover indicator."""
    df = df.copy()
    close = df['close']

    df['ema_fast'] = close.ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = close.ewm(span=slow_period, adjust=False).mean()

    # Trend: 1 when fast > slow, -1 otherwise
    df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)

    return df


# ============================================
# KAMA CALCULATION (Kaufman Adaptive Moving Average)
# ============================================
def calculate_kama(df: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30) -> pd.DataFrame:
    """
    Calculate Kaufman Adaptive Moving Average.
    Adapts to market volatility using efficiency ratio.
    """
    df = df.copy()
    close = df['close']

    # Efficiency Ratio (ER)
    change = abs(close - close.shift(period))
    volatility = abs(close - close.shift(1)).rolling(period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)

    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)

    # Scaled smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # KAMA calculation
    kama = pd.Series(index=df.index, dtype=float)
    kama.iloc[period] = close.iloc[period]

    for i in range(period + 1, len(df)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])

    df['kama'] = kama

    # Trend: 1 when price > KAMA, -1 otherwise
    df['trend'] = np.where(close > kama, 1, -1)

    return df


# ============================================
# JMA CALCULATION (Jurik Moving Average approximation)
# ============================================
def calculate_jma(df: pd.DataFrame, period: int = 14, phase: int = 0, power: int = 2) -> pd.DataFrame:
    """
    Calculate Jurik Moving Average (approximation).
    JMA is a proprietary indicator; this is an approximation using adaptive smoothing.

    Args:
        period: Lookback period
        phase: Phase shift (-100 to +100), 0 = no shift
        power: Smoothing power (1=linear, 2=squared)
    """
    df = df.copy()
    close = df['close']

    # Phase adjustment (-100 to +100 mapped to -0.5 to +0.5)
    phase_ratio = phase / 100 * 0.5

    # Beta calculation
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)

    # Alpha with phase adjustment
    alpha = beta ** power

    # Initialize JMA
    jma = pd.Series(index=df.index, dtype=float)
    e0 = pd.Series(0.0, index=df.index)
    e1 = pd.Series(0.0, index=df.index)
    e2 = pd.Series(0.0, index=df.index)

    jma.iloc[0] = close.iloc[0]
    e0.iloc[0] = close.iloc[0]
    e1.iloc[0] = 0
    e2.iloc[0] = 0

    for i in range(1, len(df)):
        price = close.iloc[i]

        e0.iloc[i] = (1 - alpha) * price + alpha * e0.iloc[i-1]
        e1.iloc[i] = (price - e0.iloc[i]) * (1 - beta) + beta * e1.iloc[i-1]
        e2.iloc[i] = (e0.iloc[i] + phase_ratio * e1.iloc[i] - jma.iloc[i-1]) * ((1 - alpha) ** 2) + (alpha ** 2) * e2.iloc[i-1]
        jma.iloc[i] = e2.iloc[i] + jma.iloc[i-1]

    df['jma'] = jma

    # Trend: 1 when price > JMA, -1 otherwise
    df['trend'] = np.where(close > jma, 1, -1)

    return df


# ============================================
# UNIFIED INDICATOR CALCULATION
# ============================================
def calculate_indicator(df: pd.DataFrame, indicator: str, params: Dict) -> pd.DataFrame:
    """Calculate specified indicator with given parameters."""
    if indicator == 'supertrend':
        return calculate_supertrend(df, params['period'], params['multiplier'])
    elif indicator == 'ema':
        return calculate_ema(df, params['fast_period'], params['slow_period'])
    elif indicator == 'kama':
        return calculate_kama(df, params['period'], params['fast'], params['slow'])
    elif indicator == 'jma':
        return calculate_jma(df, params['period'], params['phase'], params['power'])
    else:
        raise ValueError(f"Unknown indicator: {indicator}")


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
    indicator: str
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
    indicator: str,
    params: Dict,
    min_hold_bars: int,
    initial_capital: float = INITIAL_CAPITAL,
    position_size: float = POSITION_SIZE,
    commission: float = COMMISSION_PER_TRADE
) -> Optional[BacktestResult]:
    """
    Run backtest with given indicator and parameters.
    Long-only strategy with time-based and trend-flip exits.

    All indicators use trend crossover signals:
    - Entry: trend changes from -1 to 1 (bullish crossover)
    - Exit: trend changes to -1 OR time-based exit
    """
    # Determine warmup period
    if indicator == 'supertrend':
        warmup = params.get('period', 14) + 5
    elif indicator == 'ema':
        warmup = params.get('slow_period', 50) + 5
    elif indicator == 'kama':
        warmup = params.get('period', 14) + 5
    elif indicator == 'jma':
        warmup = params.get('period', 14) + 5
    else:
        warmup = 20

    if df is None or len(df) < warmup + 20:
        return None

    # Calculate indicator
    try:
        df = calculate_indicator(df, indicator, params)
    except Exception as e:
        return None

    capital = initial_capital
    position = None  # {'entry_price', 'entry_idx', 'shares'}
    trades = []
    equity_curve = [initial_capital]
    peak_equity = initial_capital
    max_drawdown = 0.0

    for i in range(warmup + 1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        close = current['close']
        trend = current['trend']
        prev_trend = prev['trend']

        # Check exit if in position
        if position is not None:
            bars_held = i - position['entry_idx']
            should_exit = False

            # Time-based exit
            if bars_held >= min_hold_bars:
                should_exit = True

            # Trend flip exit (trend goes bearish)
            if trend == -1:
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
            # Long signal: trend flips from bearish to bullish
            if prev_trend == -1 and trend == 1:
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
            indicator=indicator,
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
        indicator=indicator,
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
# PARAMETER GENERATORS
# ============================================
def get_param_combinations(indicator: str, quick: bool = False) -> List[Dict]:
    """Generate parameter combinations for an indicator."""
    combos = []

    if indicator == 'supertrend':
        periods = ATR_PERIODS_QUICK if quick else ATR_PERIODS
        multipliers = ATR_MULTIPLIERS_QUICK if quick else ATR_MULTIPLIERS
        for period, mult in itertools.product(periods, multipliers):
            combos.append({'period': period, 'multiplier': mult})

    elif indicator == 'ema':
        fast_periods = EMA_FAST_PERIODS_QUICK if quick else EMA_FAST_PERIODS
        slow_periods = EMA_SLOW_PERIODS_QUICK if quick else EMA_SLOW_PERIODS
        for fast, slow in itertools.product(fast_periods, slow_periods):
            if fast < slow:  # Fast must be less than slow
                combos.append({'fast_period': fast, 'slow_period': slow})

    elif indicator == 'kama':
        periods = KAMA_PERIODS_QUICK if quick else KAMA_PERIODS
        fasts = KAMA_FAST_QUICK if quick else KAMA_FAST
        slows = KAMA_SLOW_QUICK if quick else KAMA_SLOW
        for period, fast, slow in itertools.product(periods, fasts, slows):
            combos.append({'period': period, 'fast': fast, 'slow': slow})

    elif indicator == 'jma':
        periods = JMA_PERIODS_QUICK if quick else JMA_PERIODS
        phases = JMA_PHASES_QUICK if quick else JMA_PHASES
        powers = JMA_POWERS_QUICK if quick else JMA_POWERS
        for period, phase, power in itertools.product(periods, phases, powers):
            combos.append({'period': period, 'phase': phase, 'power': power})

    return combos


# ============================================
# PARAMETER SWEEP
# ============================================
def sweep_symbol(
    symbol: str,
    df: pd.DataFrame,
    indicators: List[str],
    hold_bars: List[int],
    quick: bool = False,
    verbose: bool = False
) -> Optional[Dict]:
    """
    Run parameter sweep for a single symbol across all indicators.
    Returns best parameters based on profit factor and PnL.
    """
    if df is None or len(df) < 100:
        print(f"[{symbol}] Insufficient data for sweep")
        return None

    best_result = None
    best_params = None
    best_indicator = None
    best_score = -float('inf')
    tested = 0

    for indicator in indicators:
        param_combos = get_param_combinations(indicator, quick)

        for params in param_combos:
            for hold in hold_bars:
                result = run_backtest(df, indicator, params, hold)
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
                    best_indicator = indicator
                    best_params = {
                        'indicator': indicator,
                        **params,
                        'min_hold_bars': hold
                    }

        if verbose:
            print(f"  [{symbol}] Tested {indicator}: {len(param_combos) * len(hold_bars)} combinations")

    if best_result is None:
        print(f"[{symbol}] No valid results found")
        return None

    return {
        'symbol': symbol,
        'indicator': best_indicator,
        'params': best_params,
        'result': best_result
    }


def run_sweep(
    symbols: List[str],
    indicators: List[str] = None,
    quick: bool = False,
    verbose: bool = True
) -> List[Dict]:
    """Run parameter sweep for all symbols across all indicators."""

    # Default to all indicators
    if indicators is None:
        indicators = INDICATORS

    # Select hold bars
    hold_bars = MIN_HOLD_BARS_QUICK if quick else MIN_HOLD_BARS

    # Count total combinations
    total_combos = 0
    for ind in indicators:
        combos = get_param_combinations(ind, quick)
        total_combos += len(combos) * len(hold_bars)

    print("Running", "QUICK" if quick else "FULL", "sweep")
    print(f"Indicators: {', '.join(indicators)}")
    print(f"Parameter combinations per symbol: {total_combos}")
    print(f"Hold Bars (days): {hold_bars}")
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
            indicators, hold_bars,
            quick=quick, verbose=verbose
        )

        if sweep_result is None:
            continue

        # Validate on test data (out-of-sample)
        params = sweep_result['params']
        indicator = sweep_result['indicator']

        # Extract indicator-specific params (without 'indicator' and 'min_hold_bars')
        ind_params = {k: v for k, v in params.items() if k not in ['indicator', 'min_hold_bars']}

        test_result = run_backtest(
            test_df,
            indicator,
            ind_params,
            params['min_hold_bars']
        )

        sweep_result['test_result'] = test_result
        results.append(sweep_result)

        # Print summary
        train_res = sweep_result['result']
        print(f"  BEST: {indicator.upper()}")
        print(f"  TRAIN: PnL=${train_res.total_pnl:+.2f} ({train_res.total_pnl_pct:+.1f}%), "
              f"WR={train_res.win_rate:.1f}%, Trades={train_res.trades}, "
              f"PF={train_res.profit_factor:.2f}")

        if test_result:
            print(f"  TEST:  PnL=${test_result.total_pnl:+.2f} ({test_result.total_pnl_pct:+.1f}%), "
                  f"WR={test_result.win_rate:.1f}%, Trades={test_result.trades}")

        # Print params based on indicator type
        param_str = format_params(indicator, params)
        print(f"  Params: {param_str}")

    return results


def format_params(indicator: str, params: Dict) -> str:
    """Format parameters for display."""
    if indicator == 'supertrend':
        return f"Period={params.get('period')}, Mult={params.get('multiplier')}, Hold={params.get('min_hold_bars')}d"
    elif indicator == 'ema':
        return f"Fast={params.get('fast_period')}, Slow={params.get('slow_period')}, Hold={params.get('min_hold_bars')}d"
    elif indicator == 'kama':
        return f"Period={params.get('period')}, Fast={params.get('fast')}, Slow={params.get('slow')}, Hold={params.get('min_hold_bars')}d"
    elif indicator == 'jma':
        return f"Period={params.get('period')}, Phase={params.get('phase')}, Power={params.get('power')}, Hold={params.get('min_hold_bars')}d"
    return str(params)


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
        indicator = r.get('indicator', 'supertrend')

        # Build row with indicator-specific params
        row = {
            'Symbol': r['symbol'],
            'Direction': 'Long',
            'Indicator': indicator,
            'IndicatorDisplay': indicator.upper(),
            'MinHoldBars': params.get('min_hold_bars', 5),
            'HTF': '1d',
            'FinalEquity': str(round(train.final_equity, 2)).replace('.', ','),
            'Trades': train.trades,
            'WinRate': str(round(train.win_rate / 100, 2)).replace('.', ','),
            'MaxDrawdown': str(round(train.max_drawdown, 1)).replace('.', ','),
            'ProfitFactor': str(round(train.profit_factor, 2)).replace('.', ','),
            'TestPnL': str(round(test.total_pnl, 2)).replace('.', ',') if test else '',
            'TestWinRate': str(round(test.win_rate / 100, 2)).replace('.', ',') if test else '',
        }

        # Indicator-specific parameters
        if indicator == 'supertrend':
            row['ParamA'] = params.get('period', 10)
            row['ParamB'] = str(params.get('multiplier', 3.0)).replace('.', ',')
            row['Length'] = str(float(params.get('period', 10))).replace('.', ',')
            row['Factor'] = str(params.get('multiplier', 3.0)).replace('.', ',')
        elif indicator == 'ema':
            row['ParamA'] = params.get('fast_period', 12)
            row['ParamB'] = params.get('slow_period', 50)
            row['FastPeriod'] = params.get('fast_period', 12)
            row['SlowPeriod'] = params.get('slow_period', 50)
        elif indicator == 'kama':
            row['ParamA'] = params.get('period', 10)
            row['ParamB'] = params.get('fast', 2)
            row['ParamC'] = params.get('slow', 30)
            row['KamaPeriod'] = params.get('period', 10)
            row['KamaFast'] = params.get('fast', 2)
            row['KamaSlow'] = params.get('slow', 30)
        elif indicator == 'jma':
            row['ParamA'] = params.get('period', 14)
            row['ParamB'] = params.get('phase', 0)
            row['ParamC'] = params.get('power', 2)
            row['JmaPeriod'] = params.get('period', 14)
            row['JmaPhase'] = params.get('phase', 0)
            row['JmaPower'] = params.get('power', 2)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    df.to_csv(output_path, sep=';', index=False)
    print(f"\nResults saved to: {output_path}")


def print_summary(results: List[Dict]):
    """Print summary of sweep results."""
    if not results:
        return

    print("\n" + "="*85)
    print("SWEEP SUMMARY - DAILY BARS - MULTI-INDICATOR")
    print("="*85)
    print(f"{'Symbol':<8} {'Indicator':<12} {'Hold':<5} {'Trades':<7} {'WinRate':<8} {'PnL%':<10} {'TestPnL%':<10}")
    print("-"*85)

    total_train_pnl = 0
    total_test_pnl = 0

    # Group by indicator
    by_indicator = {}
    for r in results:
        ind = r.get('indicator', 'supertrend')
        if ind not in by_indicator:
            by_indicator[ind] = []
        by_indicator[ind].append(r)

    for r in results:
        params = r['params']
        train = r['result']
        test = r.get('test_result')
        indicator = r.get('indicator', 'supertrend')

        test_pnl = f"{test.total_pnl_pct:+.1f}%" if test else "N/A"

        print(f"{r['symbol']:<8} {indicator:<12} "
              f"{params.get('min_hold_bars', 5):<5} {train.trades:<7} {train.win_rate:<7.1f}% "
              f"{train.total_pnl_pct:+.1f}%{'':<4} {test_pnl:<10}")

        total_train_pnl += train.total_pnl_pct
        if test:
            total_test_pnl += test.total_pnl_pct

    print("-"*85)
    avg_train = total_train_pnl / len(results) if results else 0
    avg_test = total_test_pnl / len(results) if results else 0
    print(f"{'AVERAGE':<8} {'':<12} {'':<5} {'':<7} {'':<8} {avg_train:+.1f}%{'':<4} {avg_test:+.1f}%")

    # Print indicator breakdown
    print("\n" + "-"*50)
    print("INDICATOR BREAKDOWN:")
    for ind, ind_results in by_indicator.items():
        count = len(ind_results)
        avg_pnl = sum(r['result'].total_pnl_pct for r in ind_results) / count if count > 0 else 0
        print(f"  {ind.upper()}: {count} symbols, avg PnL: {avg_pnl:+.1f}%")

    print("="*85)


# ============================================
# MAIN
# ============================================
def main():
    global _ib_connector

    parser = argparse.ArgumentParser(description='Daily Bar Multi-Indicator Sweep (IB)')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to sweep (default: DEFAULT_TRADING_SYMBOLS)')
    parser.add_argument('--all', action='store_true',
                        help='Sweep all DOW 30 + NASDAQ 100 stocks')
    parser.add_argument('--dow', action='store_true',
                        help='Sweep DOW 30 stocks only')
    parser.add_argument('--indicator', nargs='+', default=None,
                        choices=['supertrend', 'ema', 'kama', 'jma', 'all'],
                        help='Indicators to test (default: all)')
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

    # Select indicators
    if args.indicator is None or 'all' in args.indicator:
        indicators = INDICATORS  # All indicators
    else:
        indicators = args.indicator

    # Output path
    output_path = args.output or os.path.join(REPORT_DIR, OUTPUT_CSV)

    print("="*60)
    print("DAILY BAR MULTI-INDICATOR SWEEP (IB)")
    print("Supertrend, EMA, KAMA, JMA Optimization")
    print("="*60)
    print(f"Data source: Interactive Brokers")
    print(f"Indicators: {', '.join([i.upper() for i in indicators])}")
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
        results = run_sweep(symbols, indicators=indicators, quick=args.quick, verbose=args.verbose)

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
