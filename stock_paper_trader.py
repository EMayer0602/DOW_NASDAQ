#!/usr/bin/env python3
"""
Stock Paper Trader for DOW/NASDAQ
Adapted from Crypto9 paper_trader.py for Interactive Brokers

Usage:
    python stock_paper_trader.py                    # Simulation only (yfinance)
    python stock_paper_trader.py --ib               # IB Paper Trading
    python stock_paper_trader.py --ib --live        # IB Live Trading (CAREFUL!)
    python stock_paper_trader.py --symbols AAPL MSFT NVDA  # Specific symbols
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd

# Try to import yfinance for data fallback
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not installed. Run: pip install yfinance")

# Import IB connector
try:
    from ib_connector import IBConnector, fetch_ohlcv_ib
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False

# Import indicators
from ta.indicators import calculate_jma_crossover, calculate_supertrend as calc_supertrend_indicator

# Import stock settings
from stock_settings import (
    SYMBOLS, TIMEFRAME, HTF_TIMEFRAME,
    START_TOTAL_CAPITAL, MAX_OPEN_POSITIONS, STAKE_DIVISOR,
    MAX_LONG_POSITIONS, POSITION_SIZE_USD,
    USE_TIME_BASED_EXIT, DISABLE_TREND_FLIP_EXIT,
    REPORT_DIR, BEST_PARAMS_CSV,
    get_bars_per_day, is_market_open, RESPECT_MARKET_HOURS
)

from optimal_hold_times_defaults import get_optimal_hold_bars

# ============================================
# SUPERTREND CALCULATION
# ============================================
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Calculate Supertrend indicator.

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
        period: ATR period
        multiplier: ATR multiplier

    Returns:
        DataFrame with added Supertrend columns
    """
    df = df.copy()

    # Calculate ATR
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    # Calculate basic bands
    hl2 = (high + low) / 2
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)

    # Initialize final bands
    final_upper = pd.Series(0.0, index=df.index)
    final_lower = pd.Series(0.0, index=df.index)
    supertrend = pd.Series(0.0, index=df.index)

    # Calculate final bands
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

    # Calculate Supertrend
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
            supertrend.iloc[i] = final_lower.iloc[i]  # Default to bullish

    df['supertrend'] = supertrend
    df['supertrend_upper'] = final_upper
    df['supertrend_lower'] = final_lower
    df['trend'] = np.where(close > supertrend, 1, -1)  # 1 = bullish, -1 = bearish

    return df


# ============================================
# DATA FETCHING
# ============================================
def fetch_ohlcv_yfinance(symbol: str, period: str = "3mo", interval: str = "1h") -> Optional[pd.DataFrame]:
    """Fetch OHLCV data using yfinance."""
    if not YFINANCE_AVAILABLE:
        return None

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return None

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })

        return df[['open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"Error fetching {symbol} from yfinance: {e}")
        return None


def fetch_ohlcv(symbol: str, connector: Optional[IBConnector] = None,
                timeframe: str = "1h", limit: int = 500) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV data from IB or yfinance.

    Args:
        symbol: Stock symbol
        connector: IB connector (optional)
        timeframe: Timeframe string
        limit: Number of bars

    Returns:
        DataFrame with OHLCV data
    """
    # Try IB first
    if connector and connector.connected:
        try:
            df = connector.get_ohlcv(symbol, timeframe, "3 M")
            if df is not None and len(df) > 0:
                return df.tail(limit)
        except Exception as e:
            print(f"IB fetch failed for {symbol}: {e}")

    # Fallback to yfinance
    if YFINANCE_AVAILABLE:
        # Map timeframe to yfinance interval
        tf_map = {"1h": "1h", "1d": "1d", "4h": "1h", "6h": "1h"}
        interval = tf_map.get(timeframe, "1h")
        df = fetch_ohlcv_yfinance(symbol, period="3mo", interval=interval)
        if df is not None:
            return df.tail(limit)

    return None


# ============================================
# SIGNAL DETECTION
# ============================================
def detect_signal(df: pd.DataFrame, symbol: str, indicator: str = "supertrend") -> Optional[str]:
    """
    Detect trading signal based on indicator.

    Returns:
        "long" for buy signal, "short" for sell signal, None if no signal
    """
    if len(df) < 3:
        return None

    current = df.iloc[-1]
    prev = df.iloc[-2]

    if indicator == "jma":
        # JMA crossover signal
        if 'jma_signal' not in df.columns:
            return None
        sig_now = current.get('jma_signal', 0)
        sig_prev = prev.get('jma_signal', 0)

        # Long: signal changes to 1 (fast crosses above slow)
        if sig_prev != 1 and sig_now == 1:
            return "long"
        # Short: signal changes to -1 (fast crosses below slow)
        if sig_prev != -1 and sig_now == -1:
            return "short"
    else:
        # Supertrend signal
        close_now = current['close']
        close_prev = prev['close']
        st_now = current.get('supertrend', current['close'])
        st_prev = prev.get('supertrend', prev['close'])

        # Long signal: price crosses above Supertrend
        if close_prev <= st_prev and close_now > st_now:
            return "long"
        # Short signal: price crosses below Supertrend
        if close_prev >= st_prev and close_now < st_now:
            return "short"

    return None


def check_exit_signal(df: pd.DataFrame, position: Dict, bars_held: int) -> Tuple[bool, str, float]:
    """
    Check if position should be exited.

    Returns:
        (should_exit, reason, exit_price)
    """
    current = df.iloc[-1]
    close_now = current['close']
    st_now = current['supertrend']
    direction = position['direction']
    symbol = position['symbol']

    # Get optimal hold bars
    optimal_bars = get_optimal_hold_bars(symbol, direction)

    # Time-based exit
    if USE_TIME_BASED_EXIT and bars_held >= optimal_bars:
        return True, f"Time exit ({bars_held} bars, optimal={optimal_bars})", close_now

    # Trend flip exit
    if not DISABLE_TREND_FLIP_EXIT:
        if direction == "long" and close_now < st_now:
            return True, "Trend flip (bearish)", close_now
        elif direction == "short" and close_now > st_now:
            return True, "Trend flip (bullish)", close_now

    return False, "", 0.0


# ============================================
# POSITION MANAGEMENT
# ============================================
class StockPortfolio:
    """Manages stock positions and capital."""

    def __init__(self, initial_capital: float = START_TOTAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        self.closed_trades: List[Dict] = []
        self.trade_log: List[Dict] = []

    def get_position_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def can_open_position(self) -> bool:
        return self.get_position_count() < MAX_OPEN_POSITIONS

    def get_stake(self) -> float:
        """Calculate stake for new position."""
        return min(POSITION_SIZE_USD, self.cash / max(1, MAX_OPEN_POSITIONS - self.get_position_count()))

    def open_position(self, symbol: str, direction: str, price: float,
                      shares: int, timestamp: datetime) -> bool:
        """Open a new position."""
        if self.has_position(symbol):
            print(f"[Portfolio] Already have position in {symbol}")
            return False

        cost = shares * price
        if cost > self.cash:
            print(f"[Portfolio] Insufficient cash for {symbol}: need ${cost:.2f}, have ${self.cash:.2f}")
            return False

        self.cash -= cost
        self.positions[symbol] = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': price,
            'shares': shares,
            'entry_time': timestamp,
            'bars_held': 0
        }

        print(f"[OPEN] {direction.upper()} {shares} {symbol} @ ${price:.2f} (cost: ${cost:.2f})")
        return True

    def close_position(self, symbol: str, price: float, timestamp: datetime,
                       reason: str = "") -> Optional[Dict]:
        """Close an existing position."""
        if not self.has_position(symbol):
            return None

        pos = self.positions[symbol]
        shares = pos['shares']
        entry_price = pos['entry_price']
        direction = pos['direction']

        # Calculate PnL
        if direction == "long":
            pnl = shares * (price - entry_price)
        else:
            pnl = shares * (entry_price - price)

        proceeds = shares * price
        self.cash += proceeds

        # Record closed trade
        trade = {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry_price,
            'exit_price': price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': (pnl / (shares * entry_price)) * 100,
            'entry_time': pos['entry_time'].isoformat() if isinstance(pos['entry_time'], datetime) else pos['entry_time'],
            'exit_time': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'bars_held': pos['bars_held'],
            'reason': reason
        }

        self.closed_trades.append(trade)
        del self.positions[symbol]

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        print(f"[CLOSE] {symbol} @ ${price:.2f} | PnL: {pnl_str} ({trade['pnl_pct']:.1f}%) | {reason}")

        return trade

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = sum(
            pos['shares'] * current_prices.get(pos['symbol'], pos['entry_price'])
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def get_stats(self) -> Dict:
        """Get portfolio statistics."""
        if not self.closed_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0
            }

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        total_pnl = sum(t['pnl'] for t in self.closed_trades)

        return {
            'total_trades': len(self.closed_trades),
            'winning_trades': len(wins),
            'losing_trades': len(self.closed_trades) - len(wins),
            'win_rate': len(wins) / len(self.closed_trades) * 100,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(self.closed_trades)
        }

    def save_state(self, filepath: str):
        """Save portfolio state to JSON."""
        state = {
            'cash': self.cash,
            'positions': self.positions,
            'closed_trades': self.closed_trades,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, filepath: str) -> bool:
        """Load portfolio state from JSON."""
        if not os.path.exists(filepath):
            return False
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            self.cash = state.get('cash', self.initial_capital)
            self.positions = state.get('positions', {})
            self.closed_trades = state.get('closed_trades', [])
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False


# ============================================
# TRADING PARAMETERS
# ============================================
def load_trading_params(csv_path: str = BEST_PARAMS_CSV) -> Dict[str, Dict]:
    """Load trading parameters from CSV."""
    params = {}

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, using defaults")
        return params

    try:
        df = pd.read_csv(csv_path, sep=';')
        for _, row in df.iterrows():
            symbol = row['Symbol']
            direction = row['Direction'].lower()

            # Parse parameters
            def parse_float(val):
                if pd.isna(val):
                    return None
                if isinstance(val, str):
                    val = val.replace(',', '.')
                return float(val)

            params[(symbol, direction)] = {
                'indicator': row.get('Indicator', 'supertrend'),
                'atr_period': int(parse_float(row.get('ParamA', 10)) or 10),
                'atr_multiplier': parse_float(row.get('ParamB', 3.0)) or 3.0,
                'min_hold_bars': int(parse_float(row.get('MinHoldBars', 5)) or 5),
                'htf': row.get('HTF', '1d')
            }

    except Exception as e:
        print(f"Error loading params: {e}")

    return params


# ============================================
# MAIN TRADING LOOP
# ============================================
def run_trading_cycle(symbols: List[str], connector: Optional[IBConnector] = None,
                      portfolio: Optional[StockPortfolio] = None,
                      params: Optional[Dict] = None):
    """Run one trading cycle for all symbols."""

    if portfolio is None:
        portfolio = StockPortfolio()

    if params is None:
        params = load_trading_params()

    now = datetime.now()
    print(f"\n{'='*60}")
    print(f"Trading Cycle: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cash: ${portfolio.cash:,.2f} | Positions: {portfolio.get_position_count()}/{MAX_OPEN_POSITIONS}")
    print(f"{'='*60}")

    # Check market hours
    if RESPECT_MARKET_HOURS and not is_market_open():
        print("Market closed. Skipping cycle.")
        return portfolio

    current_prices = {}

    for symbol in symbols:
        try:
            # Fetch data
            df = fetch_ohlcv(symbol, connector, TIMEFRAME)
            if df is None or len(df) < 20:
                print(f"[{symbol}] Insufficient data")
                continue

            # Get parameters
            sym_params = params.get((symbol, 'long'), {
                'indicator': 'supertrend',
                'atr_period': 10,
                'atr_multiplier': 3.0
            })
            indicator = sym_params.get('indicator', 'supertrend')

            # Calculate indicator
            if indicator == 'jma':
                fast_period = int(sym_params.get('atr_period', 7))
                slow_period = int(sym_params.get('atr_multiplier', 21))
                df = calculate_jma_crossover(df, fast_period, slow_period)
            else:
                df = calculate_supertrend(
                    df,
                    period=sym_params.get('atr_period', 10),
                    multiplier=sym_params.get('atr_multiplier', 3.0)
                )

            current_price = df.iloc[-1]['close']
            current_prices[symbol] = current_price

            # Check existing position
            if portfolio.has_position(symbol):
                pos = portfolio.positions[symbol]
                pos['bars_held'] += 1

                should_exit, reason, exit_price = check_exit_signal(df, pos, pos['bars_held'])
                if should_exit:
                    portfolio.close_position(symbol, exit_price, now, reason)

            # Check for new entry
            elif portfolio.can_open_position():
                signal = detect_signal(df, symbol, indicator)
                if signal in ["long", "short"]:
                    stake = portfolio.get_stake()
                    shares = int(stake / current_price)
                    if shares > 0:
                        portfolio.open_position(symbol, signal, current_price, shares, now)
                        print(f"[{symbol}] {signal.upper()} entry @ ${current_price:.2f} ({indicator})")

        except Exception as e:
            print(f"[{symbol}] Error: {e}")

    # Update portfolio value
    total_value = portfolio.get_total_value(current_prices)
    print(f"\nPortfolio Value: ${total_value:,.2f} (PnL: ${total_value - portfolio.initial_capital:+,.2f})")

    # Print stats
    stats = portfolio.get_stats()
    if stats['total_trades'] > 0:
        print(f"Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.1f}% | Total PnL: ${stats['total_pnl']:+,.2f}")

    return portfolio


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Stock Paper Trader for DOW/NASDAQ')
    parser.add_argument('--ib', action='store_true', help='Use Interactive Brokers')
    parser.add_argument('--live', action='store_true', help='Live trading (default: paper)')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to trade')
    parser.add_argument('--loop', action='store_true', help='Run continuously')
    parser.add_argument('--interval', type=int, default=3600, help='Loop interval in seconds')
    parser.add_argument('--state', default='stock_trading_state.json', help='State file path')

    args = parser.parse_args()

    print("="*60)
    print("STOCK PAPER TRADER - DOW/NASDAQ")
    print("="*60)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Mode: {'IB ' + ('LIVE' if args.live else 'Paper') if args.ib else 'Simulation (yfinance)'}")
    print(f"Loop: {'Yes' if args.loop else 'No'}")
    print("="*60)

    # Initialize IB connection if requested
    connector = None
    if args.ib:
        if not IB_AVAILABLE:
            print("ERROR: ib_insync not installed. Run: pip install ib_insync")
            sys.exit(1)

        connector = IBConnector(paper_trading=not args.live)
        if not connector.connect():
            print("ERROR: Could not connect to IB. Make sure TWS/Gateway is running.")
            sys.exit(1)

    # Initialize portfolio
    portfolio = StockPortfolio()
    if os.path.exists(args.state):
        portfolio.load_state(args.state)
        print(f"Loaded state from {args.state}")

    # Load parameters
    params = load_trading_params()

    try:
        if args.loop:
            while True:
                portfolio = run_trading_cycle(args.symbols, connector, portfolio, params)
                portfolio.save_state(args.state)
                print(f"\nNext cycle in {args.interval} seconds...")
                time.sleep(args.interval)
        else:
            portfolio = run_trading_cycle(args.symbols, connector, portfolio, params)
            portfolio.save_state(args.state)

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if connector:
            connector.disconnect()
        portfolio.save_state(args.state)
        print(f"State saved to {args.state}")


if __name__ == "__main__":
    main()
