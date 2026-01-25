#!/usr/bin/env python3
"""
Daily Bar Stock Trader for DOW/NASDAQ
Runs at market open and market close using daily bars
Uses Interactive Brokers exclusively

Lower trade frequency = lower commission costs!

Usage:
    python stock_daily_trader.py                         # IB Paper Trading
    python stock_daily_trader.py --live                  # IB Live Trading (CAREFUL!)
    python stock_daily_trader.py --symbols AAPL MSFT     # Specific symbols
    python stock_daily_trader.py --loop                  # Run at open & close

Requirements:
    - TWS or IB Gateway running with API enabled
    - Port 7497 (paper) or 7496 (live)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pytz
    PYTZ_AVAILABLE = True
except ImportError:
    PYTZ_AVAILABLE = False
    print("WARNING: pytz not installed. Run: pip install pytz")

# Import IB connector
try:
    from ib_connector import IBConnector
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("ERROR: ib_connector not found")

# Import symbols
from stock_symbols import DEFAULT_TRADING_SYMBOLS


# ============================================
# CONFIGURATION - DAILY BARS
# ============================================
TIMEFRAME = "1d"  # Daily bars

# Position sizing
START_TOTAL_CAPITAL = 100_000.0
MAX_OPEN_POSITIONS = 10
POSITION_SIZE_USD = 10_000.0

# Commission
COMMISSION_PER_TRADE = 5.0  # $5 minimum per trade

# Default Supertrend params (will be overridden by CSV)
DEFAULT_ATR_PERIOD = 10
DEFAULT_ATR_MULTIPLIER = 3.0
DEFAULT_HOLD_DAYS = 5

# Exit settings
USE_TIME_BASED_EXIT = True
USE_TREND_FLIP_EXIT = True

# Market hours (US Eastern)
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Files
STATE_FILE = "daily_trading_state.json"
PARAMS_CSV = "report_stocks/best_params_daily.csv"
PARAMS_CSV_FALLBACK = "report_stocks/best_params_overall.csv"


# ============================================
# SUPERTREND CALCULATION
# ============================================
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate Supertrend indicator."""
    df = df.copy()

    high = df['high']
    low = df['low']
    close = df['close']

    # True Range & ATR
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
    df['supertrend_upper'] = final_upper
    df['supertrend_lower'] = final_lower
    df['trend'] = np.where(close > supertrend, 1, -1)

    return df


# ============================================
# DATA FETCHING (IB only)
# ============================================
def fetch_daily_data(symbol: str, connector: IBConnector, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch daily OHLCV data from Interactive Brokers."""
    if connector is None or not connector.connected:
        print(f"[{symbol}] IB not connected")
        return None

    try:
        # IB duration string
        if days <= 365:
            duration = "1 Y"
        else:
            duration = f"{days // 365 + 1} Y"

        df = connector.get_ohlcv(symbol, "1 day", duration)

        if df is None or df.empty:
            print(f"[{symbol}] No data from IB")
            return None

        # Normalize columns
        df.columns = [c.lower() for c in df.columns]

        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            print(f"[{symbol}] Missing columns in IB data")
            return None

        df = df[required]

        # Remove timezone if present
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"[{symbol}] IB error: {e}")
        return None


# ============================================
# SIGNAL DETECTION
# ============================================
def detect_entry_signal(df: pd.DataFrame) -> Optional[str]:
    """Detect entry signal based on Supertrend crossover."""
    if len(df) < 3:
        return None

    current = df.iloc[-1]
    prev = df.iloc[-2]

    close = current['close']
    st = current['supertrend']
    prev_close = prev['close']
    prev_st = prev['supertrend']

    # Long signal: price crosses above Supertrend
    if prev_close <= prev_st and close > st:
        return "long"

    return None


def check_exit_signal(df: pd.DataFrame, days_held: int, min_hold_days: int) -> Tuple[bool, str]:
    """Check exit conditions."""
    current = df.iloc[-1]
    close = current['close']
    st = current['supertrend']

    # Time-based exit
    if USE_TIME_BASED_EXIT and days_held >= min_hold_days:
        return True, f"Time exit ({days_held} days)"

    # Trend flip exit
    if USE_TREND_FLIP_EXIT and close < st:
        return True, "Trend flip"

    return False, ""


# ============================================
# PORTFOLIO MANAGEMENT
# ============================================
class DailyPortfolio:
    """Manages daily trading positions and capital."""

    def __init__(self, initial_capital: float = START_TOTAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.closed_trades: List[Dict] = []

    def get_position_count(self) -> int:
        return len(self.positions)

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def can_open_position(self) -> bool:
        return self.get_position_count() < MAX_OPEN_POSITIONS

    def open_position(self, symbol: str, price: float, shares: int,
                      timestamp: datetime, min_hold_days: int) -> bool:
        if self.has_position(symbol):
            return False

        cost = shares * price + COMMISSION_PER_TRADE
        if cost > self.cash:
            return False

        self.cash -= cost
        self.positions[symbol] = {
            'symbol': symbol,
            'direction': 'long',
            'entry_price': price,
            'shares': shares,
            'entry_time': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'days_held': 0,
            'min_hold_days': min_hold_days
        }

        print(f"[OPEN] LONG {shares} {symbol} @ ${price:.2f} (cost: ${cost:.2f})")
        return True

    def close_position(self, symbol: str, price: float, timestamp: datetime,
                       reason: str = "") -> Optional[Dict]:
        if not self.has_position(symbol):
            return None

        pos = self.positions[symbol]
        shares = pos['shares']
        entry_price = pos['entry_price']

        # PnL including commission
        pnl = shares * (price - entry_price) - 2 * COMMISSION_PER_TRADE
        proceeds = shares * price - COMMISSION_PER_TRADE
        self.cash += proceeds

        trade = {
            'symbol': symbol,
            'direction': 'long',
            'entry_price': entry_price,
            'exit_price': price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': (pnl / (shares * entry_price)) * 100,
            'entry_time': pos['entry_time'],
            'exit_time': timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
            'days_held': pos['days_held'],
            'reason': reason
        }

        self.closed_trades.append(trade)
        del self.positions[symbol]

        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        print(f"[CLOSE] {symbol} @ ${price:.2f} | PnL: {pnl_str} ({trade['pnl_pct']:.1f}%) | {reason}")

        return trade

    def increment_days_held(self):
        """Increment days held for all positions (call once per day)."""
        for pos in self.positions.values():
            pos['days_held'] += 1

    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        position_value = sum(
            pos['shares'] * current_prices.get(pos['symbol'], pos['entry_price'])
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def get_stats(self) -> Dict:
        if not self.closed_trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0}

        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        total_pnl = sum(t['pnl'] for t in self.closed_trades)

        return {
            'total_trades': len(self.closed_trades),
            'wins': len(wins),
            'losses': len(self.closed_trades) - len(wins),
            'win_rate': len(wins) / len(self.closed_trades) * 100,
            'total_pnl': total_pnl,
            'avg_pnl': total_pnl / len(self.closed_trades)
        }

    def save_state(self, filepath: str):
        state = {
            'cash': self.cash,
            'positions': self.positions,
            'closed_trades': self.closed_trades,
            'timestamp': datetime.now().isoformat()
        }
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, filepath: str) -> bool:
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
# PARAMETER LOADING
# ============================================
def load_trading_params(csv_path: str = None) -> Dict[str, Dict]:
    """Load trading parameters from CSV."""
    params = {}

    # Try daily params first, then fallback
    paths_to_try = []
    if csv_path:
        paths_to_try.append(csv_path)
    paths_to_try.extend([PARAMS_CSV, PARAMS_CSV_FALLBACK])

    csv_used = None
    for path in paths_to_try:
        if os.path.exists(path):
            csv_used = path
            break

    if csv_used is None:
        print("Warning: No params CSV found, using defaults")
        return params

    print(f"Loading params from: {csv_used}")

    try:
        df = pd.read_csv(csv_used, sep=';')
        for _, row in df.iterrows():
            symbol = row['Symbol']

            def parse_float(val):
                if pd.isna(val):
                    return None
                if isinstance(val, str):
                    val = val.replace(',', '.')
                return float(val)

            params[symbol] = {
                'atr_period': int(parse_float(row.get('ParamA', 10)) or 10),
                'atr_multiplier': parse_float(row.get('ParamB', 3.0)) or 3.0,
                'min_hold_days': int(parse_float(row.get('MinHoldBars', 5)) or 5)
            }

    except Exception as e:
        print(f"Error loading params: {e}")

    return params


# ============================================
# MARKET HOURS
# ============================================
def get_eastern_time() -> datetime:
    """Get current time in US Eastern timezone."""
    if PYTZ_AVAILABLE:
        eastern = pytz.timezone('US/Eastern')
        return datetime.now(eastern)
    return datetime.now()


def is_market_open() -> bool:
    """Check if US stock market is currently open."""
    now = get_eastern_time()

    # Weekend check
    if now.weekday() >= 5:
        return False

    # Time check
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)

    return market_open <= now <= market_close


def is_near_market_open(minutes_before: int = 5) -> bool:
    """Check if we're near market open."""
    now = get_eastern_time()
    if now.weekday() >= 5:
        return False

    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)
    delta = (market_open - now).total_seconds() / 60

    return 0 <= delta <= minutes_before or -minutes_before <= delta <= 0


def is_near_market_close(minutes_before: int = 15) -> bool:
    """Check if we're near market close."""
    now = get_eastern_time()
    if now.weekday() >= 5:
        return False

    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    delta = (market_close - now).total_seconds() / 60

    return 0 <= delta <= minutes_before


def seconds_until_market_open() -> int:
    """Get seconds until market open."""
    now = get_eastern_time()

    # If weekend, calculate to Monday
    days_until_monday = (7 - now.weekday()) % 7
    if days_until_monday == 0 and now.weekday() >= 5:
        days_until_monday = 7 - now.weekday()

    target = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0, microsecond=0)

    if now.weekday() < 5 and now < target:
        # Same day, before open
        return int((target - now).total_seconds())
    elif now.weekday() < 5:
        # Same day, after open - next day
        target += timedelta(days=1)
        if target.weekday() >= 5:
            target += timedelta(days=7 - target.weekday())
    else:
        # Weekend
        target += timedelta(days=days_until_monday)

    return int((target - now).total_seconds())


def seconds_until_market_close() -> int:
    """Get seconds until market close."""
    now = get_eastern_time()

    if now.weekday() >= 5:
        return -1  # Weekend

    target = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)

    if now < target:
        return int((target - now).total_seconds())

    return -1  # Already closed


# ============================================
# MAIN TRADING CYCLE
# ============================================
def run_daily_cycle(
    symbols: List[str],
    connector: Optional[IBConnector],
    portfolio: DailyPortfolio,
    params: Dict[str, Dict],
    is_open_cycle: bool = True
) -> DailyPortfolio:
    """Run one daily trading cycle."""
    now = datetime.now()
    cycle_type = "OPEN" if is_open_cycle else "CLOSE"

    print(f"\n{'='*60}")
    print(f"Daily Cycle ({cycle_type}): {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Cash: ${portfolio.cash:,.2f} | Positions: {portfolio.get_position_count()}/{MAX_OPEN_POSITIONS}")
    print(f"{'='*60}")

    # Increment days held at market open
    if is_open_cycle:
        portfolio.increment_days_held()

    current_prices = {}

    for symbol in symbols:
        try:
            # Fetch daily data
            df = fetch_daily_data(symbol, connector)
            if df is None or len(df) < 20:
                print(f"[{symbol}] Insufficient data")
                continue

            # Get params
            sym_params = params.get(symbol, {
                'atr_period': DEFAULT_ATR_PERIOD,
                'atr_multiplier': DEFAULT_ATR_MULTIPLIER,
                'min_hold_days': DEFAULT_HOLD_DAYS
            })

            # Calculate Supertrend
            df = calculate_supertrend(
                df,
                period=sym_params['atr_period'],
                multiplier=sym_params['atr_multiplier']
            )

            current_price = df.iloc[-1]['close']
            current_prices[symbol] = current_price

            # Check existing position for exits
            if portfolio.has_position(symbol):
                pos = portfolio.positions[symbol]

                should_exit, reason = check_exit_signal(
                    df,
                    pos['days_held'],
                    sym_params['min_hold_days']
                )

                if should_exit:
                    portfolio.close_position(symbol, current_price, now, reason)

            # Check for new entries (preferably at market open)
            elif portfolio.can_open_position() and is_open_cycle:
                signal = detect_entry_signal(df)
                if signal == "long":
                    shares = int(POSITION_SIZE_USD / current_price)
                    if shares > 0:
                        portfolio.open_position(
                            symbol,
                            current_price,
                            shares,
                            now,
                            sym_params['min_hold_days']
                        )

        except Exception as e:
            print(f"[{symbol}] Error: {e}")

    # Portfolio summary
    total_value = portfolio.get_total_value(current_prices)
    pnl = total_value - portfolio.initial_capital

    print(f"\nPortfolio: ${total_value:,.2f} (PnL: ${pnl:+,.2f})")

    stats = portfolio.get_stats()
    if stats['total_trades'] > 0:
        print(f"Trades: {stats['total_trades']} | Win Rate: {stats['win_rate']:.1f}% | Total PnL: ${stats['total_pnl']:+,.2f}")

    # List open positions
    if portfolio.positions:
        print("\nOpen Positions:")
        for sym, pos in portfolio.positions.items():
            price = current_prices.get(sym, pos['entry_price'])
            pnl = pos['shares'] * (price - pos['entry_price'])
            print(f"  {sym}: {pos['shares']} shares @ ${pos['entry_price']:.2f} "
                  f"(current: ${price:.2f}, PnL: ${pnl:+.2f}, days: {pos['days_held']})")

    return portfolio


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Daily Bar Stock Trader (IB)')
    parser.add_argument('--live', action='store_true', help='Live trading (default: paper)')
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_TRADING_SYMBOLS, help='Symbols to trade')
    parser.add_argument('--loop', action='store_true', help='Run continuously at open/close')
    parser.add_argument('--state', default=STATE_FILE, help='State file path')
    parser.add_argument('--params', default=None, help='Parameters CSV path')

    args = parser.parse_args()

    # Check IB availability
    if not IB_AVAILABLE:
        print("ERROR: ib_connector not available")
        print("Make sure ib_insync is installed: pip install ib_insync")
        sys.exit(1)

    print("="*60)
    print("DAILY BAR STOCK TRADER (IB)")
    print("Lower frequency = Lower commission costs!")
    print("="*60)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Mode: IB {'LIVE' if args.live else 'Paper'}")
    print(f"Timeframe: DAILY (1d)")
    print(f"Loop: {'Yes (open & close)' if args.loop else 'No (single run)'}")
    print("="*60)

    # Connect to IB (always required)
    print("\nConnecting to Interactive Brokers...")
    connector = IBConnector(paper_trading=not args.live)

    if not connector.connect():
        print("ERROR: Could not connect to IB")
        print("Make sure TWS or IB Gateway is running with API enabled")
        print(f"  Paper trading: port 7497 (TWS) or 4002 (Gateway)")
        print(f"  Live trading: port 7496 (TWS) or 4001 (Gateway)")
        sys.exit(1)

    print("Connected to IB!\n")

    # Initialize portfolio
    portfolio = DailyPortfolio()
    if os.path.exists(args.state):
        portfolio.load_state(args.state)
        print(f"Loaded state from {args.state}")

    # Load params
    params = load_trading_params(args.params)
    print(f"Loaded params for {len(params)} symbols")

    try:
        if args.loop:
            print("\nRunning in loop mode (open & close cycles)")
            print("Press Ctrl+C to stop\n")

            while True:
                now = get_eastern_time()

                if is_market_open():
                    if is_near_market_open(minutes_before=10):
                        # Market open cycle
                        print("\n>>> MARKET OPEN CYCLE <<<")
                        portfolio = run_daily_cycle(args.symbols, connector, portfolio, params, is_open_cycle=True)
                        portfolio.save_state(args.state)

                        # Wait until near close
                        secs = seconds_until_market_close() - 15 * 60  # 15 min before close
                        if secs > 0:
                            print(f"Waiting {secs // 3600}h {(secs % 3600) // 60}m until close cycle...")
                            time.sleep(max(60, secs))

                    elif is_near_market_close(minutes_before=20):
                        # Market close cycle
                        print("\n>>> MARKET CLOSE CYCLE <<<")
                        portfolio = run_daily_cycle(args.symbols, connector, portfolio, params, is_open_cycle=False)
                        portfolio.save_state(args.state)

                        # Wait until next day open
                        secs = seconds_until_market_open()
                        print(f"Waiting {secs // 3600}h {(secs % 3600) // 60}m until next open...")
                        time.sleep(max(60, secs - 300))  # Wake up 5 min early

                    else:
                        # During market hours, wait
                        secs = seconds_until_market_close() - 20 * 60
                        if secs > 60:
                            print(f"Market open. Waiting {secs // 60}m until close cycle...")
                            time.sleep(min(secs, 3600))  # Check every hour max
                        else:
                            time.sleep(60)

                else:
                    # Market closed
                    secs = seconds_until_market_open()
                    if secs > 0:
                        print(f"Market closed. Next open in {secs // 3600}h {(secs % 3600) // 60}m")
                        time.sleep(min(secs - 300, 3600))  # Check hourly, wake up 5 min early
                    else:
                        time.sleep(60)

        else:
            # Single run
            portfolio = run_daily_cycle(args.symbols, connector, portfolio, params, is_open_cycle=True)
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
