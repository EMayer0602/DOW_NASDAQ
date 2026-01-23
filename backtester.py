#!/usr/bin/env python3
"""
Backtesting Module for DOW/NASDAQ Stock Trading
Simulates trading strategy on historical data with detailed metrics.

Usage:
    python backtester.py                          # Backtest default symbols
    python backtester.py --symbols AAPL MSFT      # Specific symbols
    python backtester.py --period 6mo             # 6 months of data
    python backtester.py --output results.csv     # Save results to CSV
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not installed. Run: pip install yfinance")

import os

from stock_settings import (
    SYMBOLS, TIMEFRAME, START_TOTAL_CAPITAL,
    MAX_OPEN_POSITIONS, MAX_LONG_POSITIONS, MAX_SHORT_POSITIONS,
    POSITION_SIZE_USD, USE_TIME_BASED_EXIT, DISABLE_TREND_FLIP_EXIT,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_MULTIPLIER, BEST_PARAMS_CSV
)
from optimal_hold_times_defaults import get_optimal_hold_bars


# ============================================
# OPTIMIZED PARAMETERS LOADER
# ============================================
def load_optimized_params(filepath: str = None) -> Dict[str, Tuple[int, float]]:
    """
    Load optimized parameters from CSV file.
    Returns dict: {symbol: (atr_period, atr_multiplier)}
    """
    if filepath is None:
        filepath = BEST_PARAMS_CSV

    params = {}

    if not os.path.exists(filepath):
        return params

    try:
        # Try semicolon separator first (parameter_sweep format)
        df = pd.read_csv(filepath, sep=';')
        if 'ParamA' not in df.columns:
            # Try comma separator
            df = pd.read_csv(filepath, sep=',')

        for _, row in df.iterrows():
            symbol = row['Symbol']
            atr_period = int(row['ParamA'])
            atr_mult = float(row['ParamB'])
            params[symbol] = (atr_period, atr_mult)

        print(f"Loaded optimized params for {len(params)} symbols from {filepath}")
    except Exception as e:
        print(f"Could not load optimized params: {e}")

    return params


# Global cache for optimized params
_OPTIMIZED_PARAMS: Dict[str, Tuple[int, float]] = {}


# ============================================
# DATA CLASSES
# ============================================
@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    entry_time: datetime
    exit_time: datetime
    bars_held: int
    exit_reason: str
    pnl: float = 0.0
    pnl_pct: float = 0.0

    def __post_init__(self):
        if self.direction == "long":
            self.pnl = self.shares * (self.exit_price - self.entry_price)
        else:
            self.pnl = self.shares * (self.entry_price - self.exit_price)
        self.pnl_pct = (self.pnl / (self.shares * self.entry_price)) * 100


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    direction: str
    entry_price: float
    shares: int
    entry_time: datetime
    entry_bar: int
    bars_held: int = 0


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    symbol: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_return_pct: float
    avg_trade_pnl: float
    avg_winner: float
    avg_loser: float
    profit_factor: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


# ============================================
# SUPERTREND CALCULATION
# ============================================
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate Supertrend indicator."""
    df = df.copy()

    high = df['high']
    low = df['low']
    close = df['close']

    # ATR calculation
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
            supertrend.iloc[i] = final_upper.iloc[i] if close.iloc[i] <= final_upper.iloc[i] else final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1]:
            supertrend.iloc[i] = final_lower.iloc[i] if close.iloc[i] >= final_lower.iloc[i] else final_upper.iloc[i]
        else:
            supertrend.iloc[i] = final_lower.iloc[i]

    df['supertrend'] = supertrend
    df['trend'] = np.where(close > supertrend, 1, -1)

    return df


# ============================================
# DATA FETCHING
# ============================================
def fetch_historical_data(symbol: str, period: str = "1y", interval: str = "1h") -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data using yfinance."""
    if not YFINANCE_AVAILABLE:
        print("yfinance not available")
        return None

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        return df

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


# ============================================
# BACKTESTER CLASS
# ============================================
class Backtester:
    """Backtesting engine for Supertrend strategy."""

    def __init__(
        self,
        initial_capital: float = START_TOTAL_CAPITAL,
        position_size: float = POSITION_SIZE_USD,
        max_positions: int = MAX_OPEN_POSITIONS,
        max_long: int = MAX_LONG_POSITIONS,
        max_short: int = MAX_SHORT_POSITIONS,
        atr_period: int = DEFAULT_ATR_PERIOD,
        atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
        use_time_exit: bool = USE_TIME_BASED_EXIT,
        use_trend_flip_exit: bool = not DISABLE_TREND_FLIP_EXIT,
        commission_per_share: float = 0.005,
        min_commission: float = 1.0
    ):
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.max_positions = max_positions
        self.max_long = max_long
        self.max_short = max_short
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_time_exit = use_time_exit
        self.use_trend_flip_exit = use_trend_flip_exit
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission

        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def reset(self):
        """Reset backtester state."""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []

    def get_long_count(self) -> int:
        return sum(1 for p in self.positions.values() if p.direction == 'long')

    def get_short_count(self) -> int:
        return sum(1 for p in self.positions.values() if p.direction == 'short')

    def can_open(self, direction: str) -> bool:
        if len(self.positions) >= self.max_positions:
            return False
        if direction == "long":
            return self.get_long_count() < self.max_long
        return self.get_short_count() < self.max_short

    def calculate_commission(self, shares: int) -> float:
        """Calculate commission for trade."""
        return max(self.min_commission, shares * self.commission_per_share)

    def open_position(self, symbol: str, direction: str, price: float,
                      timestamp: datetime, bar_idx: int) -> bool:
        """Open a new position."""
        if symbol in self.positions:
            return False

        shares = int(self.position_size / price)
        if shares <= 0:
            return False

        cost = shares * price + self.calculate_commission(shares)
        if cost > self.cash:
            return False

        self.cash -= cost
        self.positions[symbol] = Position(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            shares=shares,
            entry_time=timestamp,
            entry_bar=bar_idx
        )
        return True

    def close_position(self, symbol: str, price: float, timestamp: datetime,
                       bar_idx: int, reason: str) -> Optional[Trade]:
        """Close an existing position."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        commission = self.calculate_commission(pos.shares)

        # Calculate proceeds
        if pos.direction == "long":
            proceeds = pos.shares * price - commission
        else:
            # Short: we initially received cash, now we buy back
            proceeds = pos.shares * (2 * pos.entry_price - price) - commission

        self.cash += proceeds

        trade = Trade(
            symbol=symbol,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=price,
            shares=pos.shares,
            entry_time=pos.entry_time,
            exit_time=timestamp,
            bars_held=bar_idx - pos.entry_bar,
            exit_reason=reason
        )

        self.trades.append(trade)
        del self.positions[symbol]
        return trade

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        position_value = 0.0
        for symbol, pos in self.positions.items():
            price = current_prices.get(symbol, pos.entry_price)
            if pos.direction == "long":
                position_value += pos.shares * price
            else:
                # Short position value
                position_value += pos.shares * (2 * pos.entry_price - price)
        return self.cash + position_value

    def run_backtest(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        """Run backtest on single symbol."""
        self.reset()

        # Calculate indicators
        df = calculate_supertrend(df, self.atr_period, self.atr_multiplier)

        # Skip warmup period
        start_idx = self.atr_period + 5

        for i in range(start_idx, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            timestamp = df.index[i]
            price = current['close']

            # Record equity
            self.equity_curve.append(self.get_portfolio_value({symbol: price}))

            # Check existing position
            if symbol in self.positions:
                pos = self.positions[symbol]
                bars_held = i - pos.entry_bar

                should_exit = False
                reason = ""

                # Time-based exit
                if self.use_time_exit:
                    optimal_bars = get_optimal_hold_bars(symbol, pos.direction)
                    if bars_held >= optimal_bars:
                        should_exit = True
                        reason = f"Time exit ({bars_held} bars)"

                # Trend flip exit
                if not should_exit and self.use_trend_flip_exit:
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

        # Close any remaining positions
        if symbol in self.positions:
            last_price = df.iloc[-1]['close']
            last_time = df.index[-1]
            self.close_position(symbol, last_price, last_time, len(df)-1, "End of backtest")

        return self._calculate_results(symbol, df)

    def _calculate_results(self, symbol: str, df: pd.DataFrame) -> BacktestResult:
        """Calculate backtest statistics."""
        if not self.trades:
            return BacktestResult(
                symbol=symbol,
                start_date=df.index[0],
                end_date=df.index[-1],
                initial_capital=self.initial_capital,
                final_capital=self.cash,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_return_pct=0.0,
                avg_trade_pnl=0.0,
                avg_winner=0.0,
                avg_loser=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                trades=[],
                equity_curve=self.equity_curve
            )

        # Basic stats
        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)

        # Profit factor
        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Drawdown
        equity = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital])
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_drawdown = np.max(drawdown)
        max_drawdown_pct = (max_drawdown / np.max(peak)) * 100 if np.max(peak) > 0 else 0

        # Sharpe ratio (simplified - daily returns)
        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 7) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            symbol=symbol,
            start_date=df.index[0],
            end_date=df.index[-1],
            initial_capital=self.initial_capital,
            final_capital=self.cash,
            total_trades=len(self.trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(self.trades) * 100 if self.trades else 0,
            total_pnl=total_pnl,
            total_return_pct=(self.cash - self.initial_capital) / self.initial_capital * 100,
            avg_trade_pnl=total_pnl / len(self.trades) if self.trades else 0,
            avg_winner=sum(t.pnl for t in winners) / len(winners) if winners else 0,
            avg_loser=sum(t.pnl for t in losers) / len(losers) if losers else 0,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            sharpe_ratio=sharpe,
            trades=self.trades.copy(),
            equity_curve=self.equity_curve.copy()
        )


def run_multi_symbol_backtest(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1h",
    use_optimized_params: bool = True,
    params_file: str = None,
    **kwargs
) -> Dict[str, BacktestResult]:
    """
    Run backtest on multiple symbols.

    Args:
        symbols: List of symbols to backtest
        period: Historical data period
        interval: Bar interval
        use_optimized_params: If True, load params from CSV file per symbol
        params_file: Path to optimized params CSV (default: BEST_PARAMS_CSV)
        **kwargs: Additional Backtester arguments
    """
    results = {}

    # Load optimized parameters if requested
    optimized_params = {}
    if use_optimized_params:
        optimized_params = load_optimized_params(params_file)

    for symbol in symbols:
        print(f"Backtesting {symbol}...", end=" ", flush=True)
        df = fetch_historical_data(symbol, period, interval)

        if df is None or len(df) < 50:
            print("SKIPPED (insufficient data)")
            continue

        # Use optimized params if available, otherwise use defaults/kwargs
        bt_kwargs = kwargs.copy()
        if symbol in optimized_params:
            atr_period, atr_mult = optimized_params[symbol]
            bt_kwargs['atr_period'] = atr_period
            bt_kwargs['atr_multiplier'] = atr_mult
            param_str = f"ATR({atr_period},{atr_mult})"
        else:
            param_str = "default"

        backtester = Backtester(**bt_kwargs)
        result = backtester.run_backtest(symbol, df)
        results[symbol] = result

        pnl_str = f"+${result.total_pnl:.2f}" if result.total_pnl >= 0 else f"-${abs(result.total_pnl):.2f}"
        print(f"[{param_str}] {result.total_trades} trades, {result.win_rate:.1f}% win rate, {pnl_str}")

    return results


def print_summary(results: Dict[str, BacktestResult]):
    """Print backtest summary."""
    if not results:
        print("No results to display")
        return

    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print("="*80)

    # Header
    print(f"{'Symbol':<8} {'Trades':>7} {'Win%':>7} {'PnL':>12} {'Return%':>9} "
          f"{'PF':>6} {'MaxDD%':>8} {'Sharpe':>7}")
    print("-"*80)

    # Results per symbol
    total_trades = 0
    total_pnl = 0

    for symbol, r in sorted(results.items()):
        pnl_str = f"${r.total_pnl:+,.2f}"
        print(f"{symbol:<8} {r.total_trades:>7} {r.win_rate:>6.1f}% {pnl_str:>12} "
              f"{r.total_return_pct:>8.2f}% {r.profit_factor:>6.2f} "
              f"{r.max_drawdown_pct:>7.2f}% {r.sharpe_ratio:>7.2f}")
        total_trades += r.total_trades
        total_pnl += r.total_pnl

    print("-"*80)
    print(f"{'TOTAL':<8} {total_trades:>7} {'':<7} ${total_pnl:>+11,.2f}")
    print("="*80)


def save_results_to_csv(results: Dict[str, BacktestResult], filepath: str):
    """Save results to CSV file."""
    rows = []
    for symbol, r in results.items():
        rows.append({
            'Symbol': symbol,
            'Start': r.start_date,
            'End': r.end_date,
            'Trades': r.total_trades,
            'Winners': r.winning_trades,
            'Losers': r.losing_trades,
            'WinRate': r.win_rate,
            'TotalPnL': r.total_pnl,
            'ReturnPct': r.total_return_pct,
            'AvgTradePnL': r.avg_trade_pnl,
            'AvgWinner': r.avg_winner,
            'AvgLoser': r.avg_loser,
            'ProfitFactor': r.profit_factor,
            'MaxDrawdown': r.max_drawdown,
            'MaxDrawdownPct': r.max_drawdown_pct,
            'SharpeRatio': r.sharpe_ratio
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"\nResults saved to {filepath}")


def save_trades_to_csv(results: Dict[str, BacktestResult], filepath: str):
    """Save all trades to CSV file."""
    rows = []
    for symbol, r in results.items():
        for t in r.trades:
            rows.append({
                'Symbol': t.symbol,
                'Direction': t.direction,
                'EntryPrice': t.entry_price,
                'ExitPrice': t.exit_price,
                'Shares': t.shares,
                'EntryTime': t.entry_time,
                'ExitTime': t.exit_time,
                'BarsHeld': t.bars_held,
                'ExitReason': t.exit_reason,
                'PnL': t.pnl,
                'PnLPct': t.pnl_pct
            })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Trades saved to {filepath}")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Backtest DOW/NASDAQ Trading Strategy')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to backtest')
    parser.add_argument('--period', default='1y', help='Historical period (1mo, 3mo, 6mo, 1y, 2y)')
    parser.add_argument('--interval', default='1h', help='Bar interval (1h, 1d)')
    parser.add_argument('--capital', type=float, default=START_TOTAL_CAPITAL, help='Initial capital')
    parser.add_argument('--position-size', type=float, default=POSITION_SIZE_USD, help='Position size USD')
    parser.add_argument('--atr-period', type=int, default=DEFAULT_ATR_PERIOD, help='ATR period (fallback)')
    parser.add_argument('--atr-mult', type=float, default=DEFAULT_ATR_MULTIPLIER, help='ATR multiplier (fallback)')
    parser.add_argument('--output', default=None, help='Output CSV file for results')
    parser.add_argument('--trades', default=None, help='Output CSV file for trades')
    parser.add_argument('--no-optimized', action='store_true', help='Disable optimized params, use defaults')
    parser.add_argument('--params-file', default=None, help='Custom params CSV file')

    args = parser.parse_args()

    use_optimized = not args.no_optimized

    print("="*60)
    print("STOCK BACKTESTER - DOW/NASDAQ")
    print("="*60)
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Period: {args.period}, Interval: {args.interval}")
    print(f"Capital: ${args.capital:,.0f}, Position Size: ${args.position_size:,.0f}")
    if use_optimized:
        print(f"Parameters: OPTIMIZED (from CSV, fallback ATR({args.atr_period}, {args.atr_mult}))")
    else:
        print(f"Parameters: FIXED ATR({args.atr_period}), Multiplier={args.atr_mult}")
    print("="*60 + "\n")

    results = run_multi_symbol_backtest(
        symbols=args.symbols,
        period=args.period,
        interval=args.interval,
        use_optimized_params=use_optimized,
        params_file=args.params_file,
        initial_capital=args.capital,
        position_size=args.position_size,
        atr_period=args.atr_period,
        atr_multiplier=args.atr_mult
    )

    print_summary(results)

    if args.output:
        save_results_to_csv(results, args.output)

    if args.trades:
        save_trades_to_csv(results, args.trades)


if __name__ == "__main__":
    main()
