#!/usr/bin/env python3
"""
Portfolio Simulator with HTML Report Generator
Simulates multi-symbol portfolio with Long/Short trades
Generates comprehensive HTML report with all statistics

Usage:
    python stock_portfolio_simulator.py                    # Run with defaults
    python stock_portfolio_simulator.py --params report_stocks/best_params_daily.csv
    python stock_portfolio_simulator.py --months 24 --capital 20000
    python stock_portfolio_simulator.py --allow-short      # Enable short trades
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json

import numpy as np
import pandas as pd

# Import IB connector
try:
    from ib_connector import IBConnector
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("ERROR: ib_connector not found")

from stock_symbols import DEFAULT_TRADING_SYMBOLS, ALL_STOCKS

# Global IB connector
_ib_connector: Optional[IBConnector] = None


# ============================================
# PORTFOLIO SETTINGS
# ============================================
INITIAL_CAPITAL = 20_000.0
MAX_POSITIONS = 10
POSITION_SIZE_PCT = 0.10  # 10% of equity per position
COMMISSION_PER_TRADE = 5.0  # $5 per trade

# Data settings
YEARS_OF_DATA = 2
MONTHS_TO_SIMULATE = 24  # Full 2 years

# Params file
DEFAULT_PARAMS_CSV = "report_stocks/best_params_daily.csv"
REPORT_DIR = "report_stocks"

# HTF settings (from sweep)
HTF_ATR_PERIOD = 10
HTF_ATR_MULTIPLIER = 3.0


# ============================================
# INDICATOR CALCULATIONS
# ============================================
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate Supertrend indicator."""
    df = df.copy()
    high, low, close = df['high'], df['low'], df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)

    for i in range(period, len(df)):
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]

        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]

    for i in range(period, len(df)):
        if supertrend.iloc[i-1] == final_upper.iloc[i-1]:
            supertrend.iloc[i] = final_upper.iloc[i] if close.iloc[i] <= final_upper.iloc[i] else final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1]:
            supertrend.iloc[i] = final_lower.iloc[i] if close.iloc[i] >= final_lower.iloc[i] else final_upper.iloc[i]
        else:
            supertrend.iloc[i] = final_lower.iloc[i]

    df['supertrend'] = supertrend
    df['trend'] = np.where(close > supertrend, 1, -1)
    return df


def calculate_ema(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 50) -> pd.DataFrame:
    """Calculate EMA crossover indicator."""
    df = df.copy()
    df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
    df['trend'] = np.where(df['ema_fast'] > df['ema_slow'], 1, -1)
    return df


def calculate_kama(df: pd.DataFrame, period: int = 10, fast: int = 2, slow: int = 30) -> pd.DataFrame:
    """Calculate KAMA indicator."""
    df = df.copy()
    close = df['close']

    change = abs(close - close.shift(period))
    volatility = abs(close - close.shift(1)).rolling(period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)

    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    kama = pd.Series(index=df.index, dtype=float)
    kama.iloc[period] = close.iloc[period]

    for i in range(period + 1, len(df)):
        kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (close.iloc[i] - kama.iloc[i-1])

    df['kama'] = kama
    df['trend'] = np.where(close > kama, 1, -1)
    return df


def calculate_jma(df: pd.DataFrame, period: int = 14, phase: int = 0, power: int = 2) -> pd.DataFrame:
    """Calculate JMA indicator."""
    df = df.copy()
    close = df['close']

    phase_ratio = phase / 100 * 0.5
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)
    alpha = beta ** power

    jma = pd.Series(index=df.index, dtype=float)
    e0 = pd.Series(0.0, index=df.index)
    e1 = pd.Series(0.0, index=df.index)
    e2 = pd.Series(0.0, index=df.index)

    jma.iloc[0] = close.iloc[0]
    e0.iloc[0] = close.iloc[0]

    for i in range(1, len(df)):
        price = close.iloc[i]
        e0.iloc[i] = (1 - alpha) * price + alpha * e0.iloc[i-1]
        e1.iloc[i] = (price - e0.iloc[i]) * (1 - beta) + beta * e1.iloc[i-1]
        e2.iloc[i] = (e0.iloc[i] + phase_ratio * e1.iloc[i] - jma.iloc[i-1]) * ((1 - alpha) ** 2) + (alpha ** 2) * e2.iloc[i-1]
        jma.iloc[i] = e2.iloc[i] + jma.iloc[i-1]

    df['jma'] = jma
    df['trend'] = np.where(close > jma, 1, -1)
    return df


def calculate_indicator(df: pd.DataFrame, indicator: str, params: Dict) -> pd.DataFrame:
    """Calculate specified indicator."""
    if indicator == 'supertrend':
        return calculate_supertrend(df, params.get('period', 10), params.get('multiplier', 3.0))
    elif indicator == 'ema':
        return calculate_ema(df, params.get('fast_period', 12), params.get('slow_period', 50))
    elif indicator == 'kama':
        return calculate_kama(df, params.get('period', 10), params.get('fast', 2), params.get('slow', 30))
    elif indicator == 'jma':
        return calculate_jma(df, params.get('period', 14), params.get('phase', 0), params.get('power', 2))
    return calculate_supertrend(df)


# ============================================
# DATA FETCHING
# ============================================
def fetch_daily_data(symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch daily data from IB."""
    global _ib_connector
    if _ib_connector is None or not _ib_connector.connected:
        return None

    try:
        df = _ib_connector.get_ohlcv(symbol, "1 day", f"{years} Y")
        if df is None or df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df
    except Exception as e:
        print(f"[{symbol}] Error: {e}")
        return None


def fetch_weekly_data(symbol: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch weekly data from IB for HTF filter."""
    global _ib_connector
    if _ib_connector is None or not _ib_connector.connected:
        return None

    try:
        df = _ib_connector.get_ohlcv(symbol, "1 week", f"{years} Y")
        if df is None or df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df
    except:
        return None


def get_htf_trend(weekly_df: pd.DataFrame, date: datetime) -> int:
    """Get HTF trend for date."""
    if weekly_df is None or len(weekly_df) < 20:
        return 0

    weekly_df = calculate_supertrend(weekly_df.copy(), HTF_ATR_PERIOD, HTF_ATR_MULTIPLIER)
    valid_bars = weekly_df[weekly_df.index <= date]

    if len(valid_bars) == 0:
        return 0

    return int(valid_bars.iloc[-1]['trend'])


# ============================================
# PARAMETER LOADING
# ============================================
def load_params(csv_path: str) -> Dict[str, Dict]:
    """Load best parameters from sweep CSV."""
    params = {}

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found")
        return params

    try:
        df = pd.read_csv(csv_path, sep=';')
        for _, row in df.iterrows():
            symbol = row['Symbol']

            def parse_float(val):
                if pd.isna(val):
                    return None
                if isinstance(val, str):
                    val = val.replace(',', '.')
                return float(val)

            params[symbol] = {
                'indicator': row.get('Indicator', 'supertrend'),
                'period': int(parse_float(row.get('ParamA', 10)) or 10),
                'multiplier': parse_float(row.get('ParamB', 3.0)) or 3.0,
                'fast_period': int(parse_float(row.get('ParamA', 12)) or 12),
                'slow_period': int(parse_float(row.get('ParamB', 50)) or 50),
                'fast': int(parse_float(row.get('ParamB', 2)) or 2),
                'slow': int(parse_float(row.get('ParamC', 30)) or 30),
                'phase': int(parse_float(row.get('ParamB', 0)) or 0),
                'power': int(parse_float(row.get('ParamC', 2)) or 2),
                'min_hold_bars': int(parse_float(row.get('MinHoldBars', 5)) or 5),
                'use_htf_filter': row.get('UseHTF', 'No') == 'Yes'
            }

    except Exception as e:
        print(f"Error loading params: {e}")

    return params


# ============================================
# TRADE & POSITION CLASSES
# ============================================
@dataclass
class Position:
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    entry_date: datetime
    shares: int
    min_hold_bars: int
    bars_held: int = 0
    unrealized_pnl: float = 0.0


@dataclass
class Trade:
    id: int
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    entry_date: datetime
    exit_date: datetime
    shares: int
    pnl: float
    pnl_pct: float
    bars_held: int
    exit_reason: str
    commission: float


# ============================================
# PORTFOLIO SIMULATOR
# ============================================
class PortfolioSimulator:
    """Multi-symbol portfolio simulator with Long/Short support."""

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        max_positions: int = MAX_POSITIONS,
        position_size_pct: float = POSITION_SIZE_PCT,
        commission: float = COMMISSION_PER_TRADE,
        allow_short: bool = False
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.allow_short = allow_short

        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: List[float] = []

        # Statistics
        self.max_equity = initial_capital
        self.max_drawdown = 0.0
        self.max_drawdown_pct = 0.0

    def get_equity(self, prices: Dict[str, float]) -> float:
        """Calculate total equity."""
        position_value = 0.0
        for pos in self.positions.values():
            price = prices.get(pos.symbol, pos.entry_price)
            if pos.direction == 'long':
                position_value += pos.shares * price
            else:  # short
                position_value += pos.shares * (2 * pos.entry_price - price)

        return self.cash + position_value

    def get_position_size(self, prices: Dict[str, float]) -> float:
        """Calculate position size based on equity."""
        equity = self.get_equity(prices)
        return equity * self.position_size_pct

    def can_open_position(self, direction: str) -> bool:
        """Check if can open new position."""
        if len(self.positions) >= self.max_positions:
            return False
        if direction == 'short' and not self.allow_short:
            return False
        return True

    def open_position(
        self,
        symbol: str,
        direction: str,
        price: float,
        date: datetime,
        min_hold_bars: int,
        prices: Dict[str, float]
    ) -> bool:
        """Open new position."""
        if symbol in self.positions:
            return False
        if not self.can_open_position(direction):
            return False

        position_value = self.get_position_size(prices)
        shares = int(position_value / price)
        cost = shares * price + self.commission

        if shares <= 0 or cost > self.cash:
            return False

        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            entry_date=date,
            shares=shares,
            min_hold_bars=min_hold_bars
        )
        return True

    def close_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        reason: str
    ) -> Optional[Trade]:
        """Close existing position."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        proceeds = pos.shares * price - self.commission
        self.cash += proceeds

        if pos.direction == 'long':
            pnl = pos.shares * (price - pos.entry_price) - 2 * self.commission
        else:  # short
            pnl = pos.shares * (pos.entry_price - price) - 2 * self.commission

        pnl_pct = (pnl / (pos.shares * pos.entry_price)) * 100

        self.trade_counter += 1
        trade = Trade(
            id=self.trade_counter,
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            entry_date=pos.entry_date,
            exit_date=date,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            bars_held=pos.bars_held,
            exit_reason=reason,
            commission=2 * self.commission
        )

        self.closed_trades.append(trade)
        del self.positions[symbol]
        return trade

    def update_drawdown(self, equity: float):
        """Update max drawdown tracking."""
        if equity > self.max_equity:
            self.max_equity = equity

        drawdown = self.max_equity - equity
        drawdown_pct = (drawdown / self.max_equity) * 100 if self.max_equity > 0 else 0

        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown
            self.max_drawdown_pct = drawdown_pct

    def get_statistics(self) -> Dict:
        """Calculate all portfolio statistics."""
        if not self.closed_trades:
            return self._empty_stats()

        # Separate long and short trades
        long_trades = [t for t in self.closed_trades if t.direction == 'long']
        short_trades = [t for t in self.closed_trades if t.direction == 'short']

        long_wins = [t for t in long_trades if t.pnl > 0]
        long_losses = [t for t in long_trades if t.pnl <= 0]
        short_wins = [t for t in short_trades if t.pnl > 0]
        short_losses = [t for t in short_trades if t.pnl <= 0]

        all_wins = long_wins + short_wins
        all_losses = long_losses + short_losses

        total_pnl = sum(t.pnl for t in self.closed_trades)
        long_pnl = sum(t.pnl for t in long_trades)
        short_pnl = sum(t.pnl for t in short_trades)

        gross_profit = sum(t.pnl for t in all_wins) if all_wins else 0
        gross_loss = abs(sum(t.pnl for t in all_losses)) if all_losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999

        final_equity = self.equity_curve[-1][1] if self.equity_curve else self.initial_capital

        # Calculate Sharpe ratio
        if len(self.daily_returns) > 1:
            avg_return = np.mean(self.daily_returns)
            std_return = np.std(self.daily_returns)
            sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            sharpe = 0

        return {
            'initial_capital': self.initial_capital,
            'final_equity': final_equity,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / self.initial_capital) * 100,
            'total_trades': len(self.closed_trades),
            'long_trades': len(long_trades),
            'short_trades': len(short_trades),
            'winning_trades': len(all_wins),
            'losing_trades': len(all_losses),
            'win_rate': (len(all_wins) / len(self.closed_trades)) * 100,
            'long_win_rate': (len(long_wins) / len(long_trades)) * 100 if long_trades else 0,
            'short_win_rate': (len(short_wins) / len(short_trades)) * 100 if short_trades else 0,
            'long_pnl': long_pnl,
            'short_pnl': short_pnl,
            'avg_pnl': total_pnl / len(self.closed_trades),
            'avg_win': sum(t.pnl for t in all_wins) / len(all_wins) if all_wins else 0,
            'avg_loss': sum(t.pnl for t in all_losses) / len(all_losses) if all_losses else 0,
            'largest_win': max((t.pnl for t in all_wins), default=0),
            'largest_loss': min((t.pnl for t in all_losses), default=0),
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_pct': self.max_drawdown_pct,
            'sharpe_ratio': sharpe,
            'avg_bars_held': np.mean([t.bars_held for t in self.closed_trades]),
            'total_commission': sum(t.commission for t in self.closed_trades)
        }

    def _empty_stats(self) -> Dict:
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.initial_capital,
            'total_pnl': 0, 'total_pnl_pct': 0, 'total_trades': 0,
            'long_trades': 0, 'short_trades': 0, 'winning_trades': 0,
            'losing_trades': 0, 'win_rate': 0, 'long_win_rate': 0,
            'short_win_rate': 0, 'long_pnl': 0, 'short_pnl': 0,
            'avg_pnl': 0, 'avg_win': 0, 'avg_loss': 0,
            'largest_win': 0, 'largest_loss': 0, 'profit_factor': 0,
            'max_drawdown': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0,
            'avg_bars_held': 0, 'total_commission': 0
        }

    def get_open_positions(self, prices: Dict[str, float]) -> List[Dict]:
        """Get list of open positions with current values."""
        positions = []
        for pos in self.positions.values():
            price = prices.get(pos.symbol, pos.entry_price)
            if pos.direction == 'long':
                unrealized_pnl = pos.shares * (price - pos.entry_price)
            else:
                unrealized_pnl = pos.shares * (pos.entry_price - price)

            positions.append({
                'symbol': pos.symbol,
                'direction': pos.direction,
                'entry_price': pos.entry_price,
                'current_price': price,
                'shares': pos.shares,
                'entry_date': pos.entry_date.strftime('%Y-%m-%d'),
                'bars_held': pos.bars_held,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / (pos.shares * pos.entry_price)) * 100
            })

        return positions


# ============================================
# SIMULATION RUNNER
# ============================================
def run_simulation(
    symbols: List[str],
    params: Dict[str, Dict],
    months: int = MONTHS_TO_SIMULATE,
    initial_capital: float = INITIAL_CAPITAL,
    allow_short: bool = False,
    verbose: bool = True
) -> PortfolioSimulator:
    """Run portfolio simulation."""

    print(f"\nFetching data for {len(symbols)} symbols...")

    # Fetch all data
    all_data = {}
    weekly_data = {}

    for symbol in symbols:
        df = fetch_daily_data(symbol, years=YEARS_OF_DATA)
        if df is not None and len(df) > 100:
            all_data[symbol] = df
            weekly_data[symbol] = fetch_weekly_data(symbol, years=YEARS_OF_DATA)
            if verbose:
                print(f"  {symbol}: {len(df)} days")

    if not all_data:
        print("No data fetched!")
        return None

    # Determine simulation period
    end_date = min(df.index[-1] for df in all_data.values())
    start_date = end_date - timedelta(days=months * 30)

    earliest = max(df.index[0] for df in all_data.values())
    warmup_date = earliest + timedelta(days=60)
    if start_date < warmup_date:
        start_date = warmup_date

    print(f"\nSimulation: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Short trades: {'ENABLED' if allow_short else 'DISABLED'}")

    # Prepare data with indicators
    test_data = {}
    for symbol, df in all_data.items():
        sym_params = params.get(symbol, {'indicator': 'supertrend', 'period': 10, 'multiplier': 3.0, 'min_hold_bars': 5})
        indicator = sym_params.get('indicator', 'supertrend')

        df = calculate_indicator(df.copy(), indicator, sym_params)
        test_data[symbol] = df[df.index >= start_date].copy()

    # Get all dates
    all_dates = sorted(set(date for df in test_data.values() for date in df.index))
    print(f"Trading days: {len(all_dates)}")

    # Initialize simulator
    simulator = PortfolioSimulator(
        initial_capital=initial_capital,
        allow_short=allow_short
    )

    prev_equity = initial_capital

    # Run simulation day by day
    for i, date in enumerate(all_dates):
        if i == 0:
            continue

        current_prices = {}

        # Collect prices
        for symbol, df in test_data.items():
            if date in df.index:
                current_prices[symbol] = df.loc[date, 'close']

        # Process each symbol
        for symbol, df in test_data.items():
            if date not in df.index:
                continue

            idx = df.index.get_loc(date)
            if idx < 1:
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            close = current['close']
            trend = current['trend']
            prev_trend = prev['trend']

            sym_params = params.get(symbol, {'min_hold_bars': 5, 'use_htf_filter': False})
            min_hold = sym_params.get('min_hold_bars', 5)
            use_htf = sym_params.get('use_htf_filter', False)

            # Check exits first
            if symbol in simulator.positions:
                pos = simulator.positions[symbol]
                pos.bars_held += 1

                should_exit = False
                reason = ""

                # Time-based exit
                if pos.bars_held >= min_hold:
                    should_exit = True
                    reason = f"Time ({pos.bars_held}d)"

                # Trend flip exit
                if pos.direction == 'long' and trend == -1:
                    should_exit = True
                    reason = "Trend flip"
                elif pos.direction == 'short' and trend == 1:
                    should_exit = True
                    reason = "Trend flip"

                if should_exit:
                    trade = simulator.close_position(symbol, close, date, reason)
                    if trade and verbose:
                        pnl_str = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
                        print(f"  [{date.strftime('%Y-%m-%d')}] CLOSE {trade.direction.upper()} {symbol}: {pnl_str} ({reason})")

            # Check entries
            else:
                # Check HTF filter
                htf_ok = True
                if use_htf and symbol in weekly_data and weekly_data[symbol] is not None:
                    htf_trend = get_htf_trend(weekly_data[symbol].copy(), date)
                    htf_ok = (htf_trend >= 0)

                # Long signal
                if prev_trend == -1 and trend == 1 and htf_ok:
                    if simulator.open_position(symbol, 'long', close, date, min_hold, current_prices):
                        if verbose:
                            equity = simulator.get_equity(current_prices)
                            print(f"  [{date.strftime('%Y-%m-%d')}] OPEN LONG {symbol} @ ${close:.2f} (equity: ${equity:,.0f})")

                # Short signal (if enabled)
                elif allow_short and prev_trend == 1 and trend == -1:
                    htf_short_ok = True
                    if use_htf and symbol in weekly_data and weekly_data[symbol] is not None:
                        htf_trend = get_htf_trend(weekly_data[symbol].copy(), date)
                        htf_short_ok = (htf_trend <= 0)

                    if htf_short_ok and simulator.open_position(symbol, 'short', close, date, min_hold, current_prices):
                        if verbose:
                            equity = simulator.get_equity(current_prices)
                            print(f"  [{date.strftime('%Y-%m-%d')}] OPEN SHORT {symbol} @ ${close:.2f} (equity: ${equity:,.0f})")

        # Record equity
        equity = simulator.get_equity(current_prices)
        simulator.equity_curve.append((date, equity))
        simulator.update_drawdown(equity)

        # Calculate daily return
        if prev_equity > 0:
            daily_return = (equity - prev_equity) / prev_equity
            simulator.daily_returns.append(daily_return)
        prev_equity = equity

    # Close remaining positions
    for symbol in list(simulator.positions.keys()):
        if symbol in test_data:
            df = test_data[symbol]
            if len(df) > 0:
                price = df.iloc[-1]['close']
                simulator.close_position(symbol, price, all_dates[-1], "End of simulation")

    return simulator


# ============================================
# HTML REPORT GENERATOR
# ============================================
def generate_html_report(simulator: PortfolioSimulator, output_path: str, params: Dict):
    """Generate comprehensive HTML report."""

    stats = simulator.get_statistics()
    open_positions = simulator.get_open_positions({})

    # Prepare equity curve data
    equity_dates = [e[0].strftime('%Y-%m-%d') for e in simulator.equity_curve]
    equity_values = [e[1] for e in simulator.equity_curve]

    # Prepare trades table
    trades_html = ""
    for t in simulator.closed_trades:
        pnl_class = "profit" if t.pnl >= 0 else "loss"
        trades_html += f"""
        <tr class="{pnl_class}">
            <td>{t.id}</td>
            <td>{t.symbol}</td>
            <td class="direction-{t.direction}">{t.direction.upper()}</td>
            <td>{t.entry_date.strftime('%Y-%m-%d')}</td>
            <td>{t.exit_date.strftime('%Y-%m-%d')}</td>
            <td>${t.entry_price:.2f}</td>
            <td>${t.exit_price:.2f}</td>
            <td>{t.shares}</td>
            <td class="{pnl_class}">${t.pnl:+.2f}</td>
            <td class="{pnl_class}">{t.pnl_pct:+.2f}%</td>
            <td>{t.bars_held}</td>
            <td>{t.exit_reason}</td>
        </tr>
        """

    # Open positions table
    open_html = ""
    for p in open_positions:
        pnl_class = "profit" if p['unrealized_pnl'] >= 0 else "loss"
        open_html += f"""
        <tr class="{pnl_class}">
            <td>{p['symbol']}</td>
            <td class="direction-{p['direction']}">{p['direction'].upper()}</td>
            <td>{p['entry_date']}</td>
            <td>${p['entry_price']:.2f}</td>
            <td>${p['current_price']:.2f}</td>
            <td>{p['shares']}</td>
            <td class="{pnl_class}">${p['unrealized_pnl']:+.2f}</td>
            <td class="{pnl_class}">{p['unrealized_pnl_pct']:+.2f}%</td>
            <td>{p['bars_held']}</td>
        </tr>
        """

    # Symbol breakdown
    symbol_stats = {}
    for t in simulator.closed_trades:
        if t.symbol not in symbol_stats:
            symbol_stats[t.symbol] = {'trades': 0, 'pnl': 0, 'wins': 0, 'long': 0, 'short': 0}
        symbol_stats[t.symbol]['trades'] += 1
        symbol_stats[t.symbol]['pnl'] += t.pnl
        if t.pnl > 0:
            symbol_stats[t.symbol]['wins'] += 1
        if t.direction == 'long':
            symbol_stats[t.symbol]['long'] += 1
        else:
            symbol_stats[t.symbol]['short'] += 1

    symbols_html = ""
    for sym in sorted(symbol_stats.keys()):
        s = symbol_stats[sym]
        wr = (s['wins'] / s['trades']) * 100 if s['trades'] > 0 else 0
        pnl_class = "profit" if s['pnl'] >= 0 else "loss"
        symbols_html += f"""
        <tr>
            <td>{sym}</td>
            <td>{s['trades']}</td>
            <td>{s['long']}</td>
            <td>{s['short']}</td>
            <td>{wr:.1f}%</td>
            <td class="{pnl_class}">${s['pnl']:+.2f}</td>
        </tr>
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Portfolio Simulation Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }}
        h1, h2, h3 {{ color: #00d4ff; margin-bottom: 15px; }}
        h1 {{ text-align: center; font-size: 2em; margin-bottom: 30px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .stat-card {{ background: #16213e; padding: 20px; border-radius: 10px; text-align: center; }}
        .stat-value {{ font-size: 1.8em; font-weight: bold; color: #00d4ff; }}
        .stat-value.profit {{ color: #00ff88; }}
        .stat-value.loss {{ color: #ff4444; }}
        .stat-label {{ color: #888; font-size: 0.9em; margin-top: 5px; }}
        .section {{ background: #16213e; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
        .chart-container {{ height: 300px; margin: 20px 0; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #333; }}
        th {{ background: #0f3460; color: #00d4ff; }}
        tr:hover {{ background: #1f4068; }}
        .profit {{ color: #00ff88; }}
        .loss {{ color: #ff4444; }}
        .direction-long {{ color: #00d4ff; font-weight: bold; }}
        .direction-short {{ color: #ff9900; font-weight: bold; }}
        .summary-row {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .summary-col {{ flex: 1; min-width: 300px; }}
        .timestamp {{ text-align: center; color: #666; margin-top: 20px; font-size: 0.8em; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Portfolio Simulation Report</h1>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">${stats['initial_capital']:,.0f}</div>
                <div class="stat-label">Initial Capital</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'profit' if stats['total_pnl'] >= 0 else 'loss'}">${stats['final_equity']:,.0f}</div>
                <div class="stat-label">Final Equity</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'profit' if stats['total_pnl'] >= 0 else 'loss'}">${stats['total_pnl']:+,.0f}</div>
                <div class="stat-label">Total P&L ({stats['total_pnl_pct']:+.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['total_trades']}</div>
                <div class="stat-label">Total Trades</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['win_rate']:.1f}%</div>
                <div class="stat-label">Win Rate</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['profit_factor']:.2f}</div>
                <div class="stat-label">Profit Factor</div>
            </div>
            <div class="stat-card">
                <div class="stat-value loss">{stats['max_drawdown_pct']:.1f}%</div>
                <div class="stat-label">Max Drawdown</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['sharpe_ratio']:.2f}</div>
                <div class="stat-label">Sharpe Ratio</div>
            </div>
        </div>

        <div class="section">
            <h2>Equity Curve</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>

        <div class="summary-row">
            <div class="summary-col section">
                <h3>Long Trades</h3>
                <table>
                    <tr><td>Total Trades</td><td><strong>{stats['long_trades']}</strong></td></tr>
                    <tr><td>Win Rate</td><td><strong>{stats['long_win_rate']:.1f}%</strong></td></tr>
                    <tr><td>Total P&L</td><td class="{'profit' if stats['long_pnl'] >= 0 else 'loss'}"><strong>${stats['long_pnl']:+,.2f}</strong></td></tr>
                </table>
            </div>
            <div class="summary-col section">
                <h3>Short Trades</h3>
                <table>
                    <tr><td>Total Trades</td><td><strong>{stats['short_trades']}</strong></td></tr>
                    <tr><td>Win Rate</td><td><strong>{stats['short_win_rate']:.1f}%</strong></td></tr>
                    <tr><td>Total P&L</td><td class="{'profit' if stats['short_pnl'] >= 0 else 'loss'}"><strong>${stats['short_pnl']:+,.2f}</strong></td></tr>
                </table>
            </div>
            <div class="summary-col section">
                <h3>Trade Statistics</h3>
                <table>
                    <tr><td>Average P&L</td><td class="{'profit' if stats['avg_pnl'] >= 0 else 'loss'}"><strong>${stats['avg_pnl']:+.2f}</strong></td></tr>
                    <tr><td>Average Win</td><td class="profit"><strong>${stats['avg_win']:+.2f}</strong></td></tr>
                    <tr><td>Average Loss</td><td class="loss"><strong>${stats['avg_loss']:.2f}</strong></td></tr>
                    <tr><td>Largest Win</td><td class="profit"><strong>${stats['largest_win']:+.2f}</strong></td></tr>
                    <tr><td>Largest Loss</td><td class="loss"><strong>${stats['largest_loss']:.2f}</strong></td></tr>
                    <tr><td>Avg Bars Held</td><td><strong>{stats['avg_bars_held']:.1f}</strong></td></tr>
                    <tr><td>Total Commission</td><td><strong>${stats['total_commission']:.2f}</strong></td></tr>
                </table>
            </div>
        </div>

        <div class="section">
            <h2>Open Positions ({len(open_positions)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Entry Date</th>
                        <th>Entry Price</th>
                        <th>Current Price</th>
                        <th>Shares</th>
                        <th>Unrealized P&L</th>
                        <th>P&L %</th>
                        <th>Bars Held</th>
                    </tr>
                </thead>
                <tbody>
                    {open_html if open_html else '<tr><td colspan="9" style="text-align:center;">No open positions</td></tr>'}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Closed Trades ({len(simulator.closed_trades)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Symbol</th>
                        <th>Direction</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>Entry $</th>
                        <th>Exit $</th>
                        <th>Shares</th>
                        <th>P&L</th>
                        <th>P&L %</th>
                        <th>Bars</th>
                        <th>Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {trades_html}
                </tbody>
            </table>
        </div>

        <div class="section">
            <h2>Per-Symbol Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Total Trades</th>
                        <th>Long</th>
                        <th>Short</th>
                        <th>Win Rate</th>
                        <th>Total P&L</th>
                    </tr>
                </thead>
                <tbody>
                    {symbols_html}
                </tbody>
            </table>
        </div>

        <div class="timestamp">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>

    <script>
        const ctx = document.getElementById('equityChart').getContext('2d');
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {json.dumps(equity_dates)},
                datasets: [{{
                    label: 'Equity',
                    data: {json.dumps(equity_values)},
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    fill: true,
                    tension: 0.1,
                    pointRadius: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }} }},
                scales: {{
                    x: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#888', maxTicksLimit: 10 }} }},
                    y: {{ grid: {{ color: '#333' }}, ticks: {{ color: '#888', callback: v => '$' + v.toLocaleString() }} }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nHTML Report saved: {output_path}")


# ============================================
# MAIN
# ============================================
def main():
    global _ib_connector

    parser = argparse.ArgumentParser(description='Portfolio Simulator with HTML Report')
    parser.add_argument('--symbols', nargs='+', default=None)
    parser.add_argument('--params', default=DEFAULT_PARAMS_CSV)
    parser.add_argument('--months', type=int, default=MONTHS_TO_SIMULATE)
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL)
    parser.add_argument('--allow-short', action='store_true', help='Enable short trades')
    parser.add_argument('--quiet', '-q', action='store_true')
    parser.add_argument('--output', default='report_stocks/simulation_report.html')
    parser.add_argument('--live', action='store_true')

    args = parser.parse_args()

    if not IB_AVAILABLE:
        print("ERROR: ib_connector not available")
        sys.exit(1)

    # Load parameters
    print(f"Loading parameters: {args.params}")
    params = load_params(args.params)

    # Select symbols
    symbols = args.symbols or list(params.keys()) or DEFAULT_TRADING_SYMBOLS

    print("="*70)
    print("PORTFOLIO SIMULATOR WITH HTML REPORT")
    print("="*70)
    print(f"Symbols: {len(symbols)}")
    print(f"Period: {args.months} months")
    print(f"Capital: ${args.capital:,.0f}")
    print(f"Short trades: {'ENABLED' if args.allow_short else 'DISABLED'}")
    print("="*70)

    # Connect to IB
    print("\nConnecting to IB...")
    _ib_connector = IBConnector(paper_trading=not args.live)

    if not _ib_connector.connect():
        print("ERROR: Could not connect to IB")
        sys.exit(1)

    print("Connected!\n")

    try:
        # Run simulation
        simulator = run_simulation(
            symbols, params,
            months=args.months,
            initial_capital=args.capital,
            allow_short=args.allow_short,
            verbose=not args.quiet
        )

        if simulator:
            # Print summary
            stats = simulator.get_statistics()
            print("\n" + "="*70)
            print("SIMULATION RESULTS")
            print("="*70)
            print(f"Final Equity: ${stats['final_equity']:,.2f}")
            print(f"Total P&L: ${stats['total_pnl']:+,.2f} ({stats['total_pnl_pct']:+.2f}%)")
            print(f"Total Trades: {stats['total_trades']} (Long: {stats['long_trades']}, Short: {stats['short_trades']})")
            print(f"Win Rate: {stats['win_rate']:.1f}%")
            print(f"Profit Factor: {stats['profit_factor']:.2f}")
            print(f"Max Drawdown: {stats['max_drawdown_pct']:.1f}%")
            print("="*70)

            # Generate HTML report
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            generate_html_report(simulator, args.output, params)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if _ib_connector:
            _ib_connector.disconnect()
            print("Disconnected from IB")


if __name__ == "__main__":
    main()
