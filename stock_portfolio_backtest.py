#!/usr/bin/env python3
"""
Multi-Symbol Portfolio Backtest for DOW/NASDAQ Stocks
Tests Supertrend strategy with best parameters from sweep
Uses Interactive Brokers for data

Usage:
    python stock_portfolio_backtest.py                    # Test with default symbols
    python stock_portfolio_backtest.py --symbols AAPL MSFT NVDA
    python stock_portfolio_backtest.py --params report_stocks/best_params_daily.csv

Requirements:
    - TWS or IB Gateway running with API enabled
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Import IB connector
try:
    from ib_connector import IBConnector
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("ERROR: ib_connector not found")

from stock_symbols import DEFAULT_TRADING_SYMBOLS

# Global IB connector
_ib_connector: Optional[IBConnector] = None


# ============================================
# PORTFOLIO SETTINGS
# ============================================
INITIAL_CAPITAL = 100_000.0
MAX_POSITIONS = 10
POSITION_SIZE_USD = 10_000.0
COMMISSION_PER_TRADE = 5.0  # $5 per trade

# Data settings
YEARS_OF_DATA = 2
MONTHS_TO_TEST = 1  # Test on last N months

# Default params file
DEFAULT_PARAMS_CSV = "report_stocks/best_params_daily.csv"
REPORT_DIR = "report_stocks"


# ============================================
# SUPERTREND CALCULATION
# ============================================
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate Supertrend indicator."""
    df = df.copy()

    high = df['high']
    low = df['low']
    close = df['close']

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
        duration = f"{years} Y"
        df = _ib_connector.get_ohlcv(symbol, "1 day", duration)

        if df is None or df.empty:
            print(f"[{symbol}] No data from IB")
            return None

        df.columns = [c.lower() for c in df.columns]

        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            return None

        df = df[required]

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df

    except Exception as e:
        print(f"[{symbol}] Error: {e}")
        return None


# ============================================
# PARAMETER LOADING
# ============================================
def load_params(csv_path: str) -> Dict[str, Dict]:
    """Load best parameters from CSV."""
    params = {}

    if not os.path.exists(csv_path):
        print(f"Warning: {csv_path} not found, using defaults")
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
                'atr_period': int(parse_float(row.get('ParamA', 10)) or 10),
                'atr_multiplier': parse_float(row.get('ParamB', 3.0)) or 3.0,
                'min_hold_bars': int(parse_float(row.get('MinHoldBars', 5)) or 5)
            }

    except Exception as e:
        print(f"Error loading params: {e}")

    return params


# ============================================
# PORTFOLIO BACKTEST
# ============================================
@dataclass
class Position:
    symbol: str
    entry_price: float
    entry_date: datetime
    shares: int
    min_hold_bars: int
    bars_held: int = 0


@dataclass
class Trade:
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


class PortfolioBacktest:
    """Multi-symbol portfolio backtester."""

    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        max_positions: int = MAX_POSITIONS,
        position_size: float = POSITION_SIZE_USD,
        commission: float = COMMISSION_PER_TRADE
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size = position_size
        self.commission = commission

        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []

    def can_open_position(self) -> bool:
        return len(self.positions) < self.max_positions

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions

    def get_equity(self, prices: Dict[str, float]) -> float:
        position_value = sum(
            pos.shares * prices.get(pos.symbol, pos.entry_price)
            for pos in self.positions.values()
        )
        return self.cash + position_value

    def open_position(
        self,
        symbol: str,
        price: float,
        date: datetime,
        min_hold_bars: int
    ) -> bool:
        if self.has_position(symbol) or not self.can_open_position():
            return False

        shares = int(self.position_size / price)
        cost = shares * price + self.commission

        if shares <= 0 or cost > self.cash:
            return False

        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
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
        if not self.has_position(symbol):
            return None

        pos = self.positions[symbol]
        proceeds = pos.shares * price - self.commission
        self.cash += proceeds

        pnl = pos.shares * (price - pos.entry_price) - 2 * self.commission
        pnl_pct = (pnl / (pos.shares * pos.entry_price)) * 100

        trade = Trade(
            symbol=symbol,
            direction='long',
            entry_price=pos.entry_price,
            exit_price=price,
            entry_date=pos.entry_date,
            exit_date=date,
            shares=pos.shares,
            pnl=pnl,
            pnl_pct=pnl_pct,
            bars_held=pos.bars_held,
            exit_reason=reason
        )

        self.trades.append(trade)
        del self.positions[symbol]
        return trade

    def get_stats(self) -> Dict:
        if not self.trades:
            return {
                'total_trades': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'max_drawdown': 0,
                'max_drawdown_pct': 0
            }

        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in self.trades)

        # Max drawdown from equity curve
        max_dd = 0
        max_dd_pct = 0
        peak = self.initial_capital
        for date, equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak) * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct

        return {
            'total_trades': len(self.trades),
            'wins': len(wins),
            'losses': len(losses),
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / self.initial_capital) * 100,
            'win_rate': (len(wins) / len(self.trades)) * 100,
            'avg_pnl': total_pnl / len(self.trades),
            'avg_win': sum(t.pnl for t in wins) / len(wins) if wins else 0,
            'avg_loss': sum(t.pnl for t in losses) / len(losses) if losses else 0,
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'final_equity': self.equity_curve[-1][1] if self.equity_curve else self.initial_capital
        }


def run_portfolio_backtest(
    symbols: List[str],
    params: Dict[str, Dict],
    test_months: int = MONTHS_TO_TEST,
    verbose: bool = True
) -> PortfolioBacktest:
    """
    Run portfolio backtest on multiple symbols.
    """
    print(f"\nFetching data for {len(symbols)} symbols...")

    # Fetch all data
    all_data = {}
    for symbol in symbols:
        df = fetch_daily_data(symbol, years=YEARS_OF_DATA)
        if df is not None and len(df) > 100:
            all_data[symbol] = df
            if verbose:
                print(f"  {symbol}: {len(df)} days")

    if not all_data:
        print("No data fetched!")
        return None

    # Find common date range for test period
    end_date = min(df.index[-1] for df in all_data.values())
    start_date = end_date - timedelta(days=test_months * 30)

    print(f"\nTest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Prepare data for test period and calculate supertrend
    test_data = {}
    for symbol, df in all_data.items():
        # Get full data for supertrend calculation, then filter to test period
        sym_params = params.get(symbol, {'atr_period': 10, 'atr_multiplier': 3.0, 'min_hold_bars': 5})
        df = calculate_supertrend(df, sym_params['atr_period'], sym_params['atr_multiplier'])
        test_data[symbol] = df[df.index >= start_date].copy()

    # Get all unique dates in test period
    all_dates = sorted(set(
        date for df in test_data.values() for date in df.index
    ))

    print(f"Test days: {len(all_dates)}")

    # Initialize portfolio
    portfolio = PortfolioBacktest()

    # Run backtest day by day
    for i, date in enumerate(all_dates):
        if i == 0:
            continue  # Need previous day for signal

        current_prices = {}

        for symbol, df in test_data.items():
            if date not in df.index:
                continue

            idx = df.index.get_loc(date)
            if idx < 1:
                continue

            current = df.iloc[idx]
            prev = df.iloc[idx - 1]

            close = current['close']
            st = current['supertrend']
            prev_close = prev['close']
            prev_st = prev['supertrend']

            current_prices[symbol] = close

            # Get params
            sym_params = params.get(symbol, {'atr_period': 10, 'atr_multiplier': 3.0, 'min_hold_bars': 5})
            min_hold = sym_params['min_hold_bars']

            # Check exits first
            if portfolio.has_position(symbol):
                pos = portfolio.positions[symbol]
                pos.bars_held += 1

                should_exit = False
                reason = ""

                # Time-based exit
                if pos.bars_held >= min_hold:
                    should_exit = True
                    reason = f"Time ({pos.bars_held} days)"

                # Trend flip exit
                if close < st:
                    should_exit = True
                    reason = "Trend flip"

                if should_exit:
                    trade = portfolio.close_position(symbol, close, date, reason)
                    if trade and verbose:
                        pnl_str = f"+${trade.pnl:.2f}" if trade.pnl >= 0 else f"-${abs(trade.pnl):.2f}"
                        print(f"  [{date.strftime('%Y-%m-%d')}] CLOSE {symbol}: {pnl_str} ({reason})")

            # Check entries
            elif portfolio.can_open_position():
                # Long signal: price crosses above supertrend
                if prev_close <= prev_st and close > st:
                    if portfolio.open_position(symbol, close, date, min_hold):
                        if verbose:
                            print(f"  [{date.strftime('%Y-%m-%d')}] OPEN {symbol} @ ${close:.2f}")

        # Record equity
        equity = portfolio.get_equity(current_prices)
        portfolio.equity_curve.append((date, equity))

    # Close remaining positions at end
    for symbol in list(portfolio.positions.keys()):
        if symbol in test_data:
            df = test_data[symbol]
            if len(df) > 0:
                price = df.iloc[-1]['close']
                portfolio.close_position(symbol, price, all_dates[-1], "End of test")

    return portfolio


def print_results(portfolio: PortfolioBacktest, symbols: List[str]):
    """Print backtest results."""
    stats = portfolio.get_stats()

    print("\n" + "="*70)
    print("PORTFOLIO BACKTEST RESULTS - DAILY BARS")
    print("="*70)
    print(f"Initial Capital:  ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Equity:     ${stats['final_equity']:,.2f}")
    print(f"Total PnL:        ${stats['total_pnl']:+,.2f} ({stats['total_pnl_pct']:+.2f}%)")
    print("-"*70)
    print(f"Total Trades:     {stats['total_trades']}")
    print(f"Winning Trades:   {stats['wins']} ({stats['win_rate']:.1f}%)")
    print(f"Losing Trades:    {stats['losses']}")
    print(f"Average PnL:      ${stats['avg_pnl']:+.2f}")
    print(f"Avg Win:          ${stats['avg_win']:+.2f}")
    print(f"Avg Loss:         ${stats['avg_loss']:.2f}")
    print("-"*70)
    print(f"Max Drawdown:     ${stats['max_drawdown']:,.2f} ({stats['max_drawdown_pct']:.1f}%)")
    print("="*70)

    # Per-symbol breakdown
    if portfolio.trades:
        print("\nPer-Symbol Performance:")
        print("-"*50)

        symbol_stats = {}
        for trade in portfolio.trades:
            if trade.symbol not in symbol_stats:
                symbol_stats[trade.symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
            symbol_stats[trade.symbol]['trades'] += 1
            symbol_stats[trade.symbol]['pnl'] += trade.pnl
            if trade.pnl > 0:
                symbol_stats[trade.symbol]['wins'] += 1

        print(f"{'Symbol':<8} {'Trades':<8} {'PnL':<12} {'Win Rate':<10}")
        print("-"*50)

        for symbol in sorted(symbol_stats.keys()):
            s = symbol_stats[symbol]
            wr = (s['wins'] / s['trades']) * 100 if s['trades'] > 0 else 0
            print(f"{symbol:<8} {s['trades']:<8} ${s['pnl']:+.2f}{'':<4} {wr:.1f}%")

        print("="*70)


def save_trades(portfolio: PortfolioBacktest, output_path: str):
    """Save trades to CSV."""
    if not portfolio.trades:
        return

    rows = []
    for t in portfolio.trades:
        rows.append({
            'Symbol': t.symbol,
            'Direction': t.direction,
            'EntryDate': t.entry_date.strftime('%Y-%m-%d'),
            'ExitDate': t.exit_date.strftime('%Y-%m-%d'),
            'EntryPrice': t.entry_price,
            'ExitPrice': t.exit_price,
            'Shares': t.shares,
            'PnL': round(t.pnl, 2),
            'PnL%': round(t.pnl_pct, 2),
            'BarsHeld': t.bars_held,
            'ExitReason': t.exit_reason
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nTrades saved to: {output_path}")


# ============================================
# MAIN
# ============================================
def main():
    global _ib_connector

    parser = argparse.ArgumentParser(description='Multi-Symbol Portfolio Backtest (IB)')
    parser.add_argument('--symbols', nargs='+', default=None,
                        help='Symbols to test (default: DEFAULT_TRADING_SYMBOLS)')
    parser.add_argument('--params', default=DEFAULT_PARAMS_CSV,
                        help='Parameters CSV file')
    parser.add_argument('--months', type=int, default=MONTHS_TO_TEST,
                        help='Months to test (out-of-sample)')
    parser.add_argument('--capital', type=float, default=INITIAL_CAPITAL,
                        help='Initial capital')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Quiet mode (less output)')
    parser.add_argument('--live', action='store_true',
                        help='Connect to live IB (default: paper)')

    args = parser.parse_args()

    # Check IB
    if not IB_AVAILABLE:
        print("ERROR: ib_connector not available")
        sys.exit(1)

    # Load parameters
    print(f"Loading parameters from: {args.params}")
    params = load_params(args.params)

    if not params:
        print("No parameters loaded, using defaults for all symbols")

    # Select symbols
    symbols = args.symbols or DEFAULT_TRADING_SYMBOLS

    print("="*60)
    print("MULTI-SYMBOL PORTFOLIO BACKTEST (IB)")
    print("="*60)
    print(f"Data source: Interactive Brokers")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Test period: Last {args.months} month(s)")
    print(f"Initial capital: ${args.capital:,.2f}")
    print(f"Position size: ${POSITION_SIZE_USD:,.2f}")
    print(f"Max positions: {MAX_POSITIONS}")
    print(f"Commission: ${COMMISSION_PER_TRADE} per trade")
    print("="*60)

    # Connect to IB
    print("\nConnecting to Interactive Brokers...")
    _ib_connector = IBConnector(paper_trading=not args.live)

    if not _ib_connector.connect():
        print("ERROR: Could not connect to IB")
        print("Make sure TWS/Gateway is running with API enabled")
        sys.exit(1)

    print("Connected to IB!\n")

    try:
        # Run backtest
        portfolio = run_portfolio_backtest(
            symbols,
            params,
            test_months=args.months,
            verbose=not args.quiet
        )

        if portfolio:
            # Print results
            print_results(portfolio, symbols)

            # Save trades
            trades_file = os.path.join(REPORT_DIR, "portfolio_backtest_trades.csv")
            save_trades(portfolio, trades_file)

    except KeyboardInterrupt:
        print("\nBacktest interrupted")
    finally:
        if _ib_connector:
            _ib_connector.disconnect()
            print("Disconnected from IB")


if __name__ == "__main__":
    main()
