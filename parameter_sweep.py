#!/usr/bin/env python3
"""
Multi-Indicator Parameter Sweep for DOW/NASDAQ Trading Strategy
Supports: Supertrend, JMA, KAMA, HTF Filter, Hold Time Optimization

Usage:
    python parameter_sweep.py --all-nasdaq              # NASDAQ with Supertrend
    python parameter_sweep.py --all-nasdaq --indicator jma    # NASDAQ with JMA
    python parameter_sweep.py --all-nasdaq --indicator kama   # NASDAQ with KAMA
    python parameter_sweep.py --all-nasdaq --htf              # With HTF filter
    python parameter_sweep.py --quick                    # Quick sweep
    python parameter_sweep.py --thorough                 # Thorough sweep
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

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from stock_settings import SYMBOLS, START_TOTAL_CAPITAL, POSITION_SIZE_USD, HTF_TIMEFRAME

# Import custom indicators
from ta.indicators import (
    calculate_supertrend, calculate_jma, calculate_kama,
    calculate_jma_crossover, calculate_kama_price_cross,
    calculate_supertrend_htf
)


# ============================================
# DATA CLASSES
# ============================================
@dataclass
class SweepResult:
    """Result from parameter sweep."""
    symbol: str
    indicator: str  # supertrend, jma, kama, supertrend_htf
    # Indicator params
    param_a: float  # ATR period / JMA period / KAMA ER period
    param_b: float  # ATR mult / JMA slow period / KAMA fast period
    param_c: float  # Optional: phase, slow period
    # Exit params
    hold_bars: int
    use_time_exit: bool
    use_trend_flip: bool
    use_htf: bool
    # Results
    total_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    score: float


@dataclass
class Trade:
    """Completed trade."""
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

    def __post_init__(self):
        if self.direction == "long":
            self.pnl = self.shares * (self.exit_price - self.entry_price)
        else:
            self.pnl = self.shares * (self.entry_price - self.exit_price)


@dataclass
class Position:
    """Open position."""
    symbol: str
    direction: str
    entry_price: float
    shares: int
    entry_time: datetime
    entry_bar: int


# ============================================
# DATA FETCHING
# ============================================
def fetch_historical_data(symbol: str, period: str = "1y", interval: str = "1h",
                          start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV data."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        ticker = yf.Ticker(symbol)
        if start_date and end_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        elif start_date:
            df = ticker.history(start=start_date, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]
        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def fetch_htf_data(symbol: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch higher timeframe data."""
    return fetch_historical_data(symbol, period, interval)


# ============================================
# INDICATOR BACKTESTER
# ============================================
class MultiIndicatorBacktester:
    """Backtester supporting multiple indicator types."""

    def __init__(
        self,
        indicator: str = "supertrend",
        param_a: float = 10,
        param_b: float = 3.0,
        param_c: float = 0,
        hold_bars: int = 5,
        use_time_exit: bool = True,
        use_trend_flip: bool = True,
        use_htf: bool = False,
        initial_capital: float = START_TOTAL_CAPITAL,
        position_size: float = POSITION_SIZE_USD,
    ):
        self.indicator = indicator
        self.param_a = int(param_a) if indicator != "supertrend" or param_a == int(param_a) else param_a
        self.param_b = param_b
        self.param_c = param_c
        self.hold_bars = hold_bars
        self.use_time_exit = use_time_exit
        self.use_trend_flip = use_trend_flip
        self.use_htf = use_htf
        self.initial_capital = initial_capital
        self.position_size = position_size

        # State
        self.cash = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def reset(self):
        self.cash = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []

    def calculate_indicators(self, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate indicators based on type."""
        df = df.copy()

        if self.indicator == "supertrend":
            period = int(self.param_a)
            multiplier = self.param_b
            df = calculate_supertrend(df, period, multiplier)
            df['signal_trend'] = df['trend']

        elif self.indicator == "supertrend_htf":
            period = int(self.param_a)
            multiplier = self.param_b
            if htf_df is not None:
                df = calculate_supertrend_htf(df, htf_df, period, multiplier)
                df['signal_trend'] = df['aligned_trend']
            else:
                df = calculate_supertrend(df, period, multiplier)
                df['signal_trend'] = df['trend']

        elif self.indicator == "jma":
            fast_period = int(self.param_a)
            slow_period = int(self.param_b)
            phase = int(self.param_c)
            df = calculate_jma_crossover(df, fast_period, slow_period, phase)
            df['signal_trend'] = df['jma_trend']

        elif self.indicator == "kama":
            er_period = int(self.param_a)
            fast_period = int(self.param_b)
            slow_period = int(self.param_c) if self.param_c > 0 else 30
            df = calculate_kama_price_cross(df, er_period, fast_period, slow_period)
            df['signal_trend'] = df['kama_trend']

        return df

    def run_backtest(self, symbol: str, df: pd.DataFrame, htf_df: Optional[pd.DataFrame] = None) -> Dict:
        """Run backtest."""
        self.reset()

        # Calculate indicators
        df = self.calculate_indicators(df, htf_df)

        # Warmup period
        warmup = max(int(self.param_a), int(self.param_b), 20) + 5
        if len(df) <= warmup:
            return self._empty_result(symbol, df)

        for i in range(warmup, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            timestamp = df.index[i]
            price = current['close']
            trend_now = current.get('signal_trend', 0)
            trend_prev = prev.get('signal_trend', 0)

            # Record equity
            equity = self.cash
            if self.position:
                if self.position.direction == "long":
                    equity += self.position.shares * price
                else:
                    equity += self.position.shares * (2 * self.position.entry_price - price)
            self.equity_curve.append(equity)

            # Check exit
            if self.position:
                bars_held = i - self.position.entry_bar
                should_exit = False
                reason = ""

                # Time exit
                if self.use_time_exit and bars_held >= self.hold_bars:
                    should_exit = True
                    reason = f"Time exit ({bars_held} bars)"

                # Trend flip exit
                if not should_exit and self.use_trend_flip:
                    if self.position.direction == "long" and trend_now == -1:
                        should_exit = True
                        reason = "Trend flip (bearish)"
                    elif self.position.direction == "short" and trend_now == 1:
                        should_exit = True
                        reason = "Trend flip (bullish)"

                if should_exit:
                    self._close_position(price, timestamp, i, reason)

            # Check entry (no position)
            if not self.position:
                # Long signal: trend flipped from -1/0 to 1
                if trend_prev <= 0 and trend_now == 1:
                    self._open_position(symbol, "long", price, timestamp, i)
                # Short signal: trend flipped from 1/0 to -1
                elif trend_prev >= 0 and trend_now == -1:
                    self._open_position(symbol, "short", price, timestamp, i)

        # Close remaining position
        if self.position:
            last_price = df.iloc[-1]['close']
            last_time = df.index[-1]
            self._close_position(last_price, last_time, len(df)-1, "End of backtest")

        return self._calculate_results(symbol, df)

    def _open_position(self, symbol: str, direction: str, price: float, timestamp: datetime, bar_idx: int):
        shares = int(self.position_size / price)
        if shares <= 0 or shares * price > self.cash:
            return
        self.cash -= shares * price
        self.position = Position(symbol, direction, price, shares, timestamp, bar_idx)

    def _close_position(self, price: float, timestamp: datetime, bar_idx: int, reason: str):
        if not self.position:
            return
        pos = self.position
        trade = Trade(
            pos.symbol, pos.direction, pos.entry_price, price,
            pos.shares, pos.entry_time, timestamp, bar_idx - pos.entry_bar, reason
        )
        self.trades.append(trade)
        if pos.direction == "long":
            self.cash += pos.shares * price
        else:
            self.cash += pos.shares * (2 * pos.entry_price - price)
        self.position = None

    def _empty_result(self, symbol: str, df: pd.DataFrame) -> Dict:
        return {
            'symbol': symbol,
            'total_trades': 0, 'win_rate': 0, 'total_pnl': 0,
            'profit_factor': 0, 'max_drawdown_pct': 0, 'sharpe_ratio': 0
        }

    def _calculate_results(self, symbol: str, df: pd.DataFrame) -> Dict:
        if not self.trades:
            return self._empty_result(symbol, df)

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        equity = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital])
        peak = np.maximum.accumulate(equity)
        drawdown = peak - equity
        max_dd = np.max(drawdown)
        max_dd_pct = (max_dd / np.max(peak)) * 100 if np.max(peak) > 0 else 0

        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 7) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return {
            'symbol': symbol,
            'total_trades': len(self.trades),
            'win_rate': len(winners) / len(self.trades) * 100,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_dd_pct,
            'sharpe_ratio': sharpe
        }


# ============================================
# SCORING
# ============================================
def calculate_score(result: Dict) -> float:
    """Calculate optimization score."""
    if result['total_trades'] < 3:
        return -999

    profit_score = min(result['total_pnl'] / 5000, 1.0)
    win_rate_score = result['win_rate'] / 100
    pf_score = min(result['profit_factor'] / 3, 1.0)
    dd_penalty = result['max_drawdown_pct'] / 100
    sharpe_score = min(max(result['sharpe_ratio'], 0) / 2, 1.0)

    return (0.30 * profit_score + 0.20 * win_rate_score + 0.25 * pf_score +
            0.15 * sharpe_score - 0.10 * dd_penalty)


# ============================================
# PARAMETER SWEEP
# ============================================
def run_sweep(
    symbols: List[str],
    indicator: str = "supertrend",
    period: str = "1y",
    interval: str = "1h",
    start_date: str = None,
    end_date: str = None,
    params_a: List[float] = None,
    params_b: List[float] = None,
    params_c: List[float] = None,
    hold_bars_list: List[int] = None,
    test_time_exit: List[bool] = None,
    test_trend_flip: List[bool] = None,
    use_htf: bool = False,
    htf_interval: str = "1d"
) -> Dict[str, List[SweepResult]]:
    """Run full parameter sweep."""

    # Default parameters based on indicator
    if indicator == "supertrend" or indicator == "supertrend_htf":
        if params_a is None:
            params_a = [7, 10, 14, 20]
        if params_b is None:
            params_b = [2.0, 2.5, 3.0, 3.5, 4.0]
        if params_c is None:
            params_c = [0]
    elif indicator == "jma":
        if params_a is None:
            params_a = [5, 7, 10, 14]  # Fast period
        if params_b is None:
            params_b = [14, 21, 30, 50]  # Slow period
        if params_c is None:
            params_c = [-50, 0, 50]  # Phase
    elif indicator == "kama":
        if params_a is None:
            params_a = [5, 10, 15, 20]  # ER period
        if params_b is None:
            params_b = [2, 3, 5]  # Fast EMA
        if params_c is None:
            params_c = [20, 30, 40]  # Slow EMA

    if hold_bars_list is None:
        hold_bars_list = [3, 5, 7, 10, 14]
    if test_time_exit is None:
        test_time_exit = [True]
    if test_trend_flip is None:
        test_trend_flip = [True]

    total_combos = len(params_a) * len(params_b) * len(params_c) * len(hold_bars_list)
    print(f"Indicator: {indicator.upper()}")
    print(f"Testing {total_combos} parameter combinations per symbol")
    print(f"ParamA: {params_a}")
    print(f"ParamB: {params_b}")
    print(f"ParamC: {params_c}")
    print(f"Hold Bars: {hold_bars_list}")
    print(f"HTF Filter: {use_htf}")
    print()

    all_results = {}

    for symbol in symbols:
        print(f"Sweeping {symbol}...", end=" ", flush=True)

        df = fetch_historical_data(symbol, period, interval, start_date, end_date)
        if df is None or len(df) < 50:
            print("SKIPPED (insufficient data)")
            continue

        htf_df = None
        if use_htf or indicator == "supertrend_htf":
            htf_df = fetch_htf_data(symbol, "2y", htf_interval)

        results = []
        for pa in params_a:
            for pb in params_b:
                for pc in params_c:
                    for hold_bars in hold_bars_list:
                        for use_time in test_time_exit:
                            for use_flip in test_trend_flip:
                                if not use_time and not use_flip:
                                    continue

                                backtester = MultiIndicatorBacktester(
                                    indicator=indicator,
                                    param_a=pa, param_b=pb, param_c=pc,
                                    hold_bars=hold_bars,
                                    use_time_exit=use_time,
                                    use_trend_flip=use_flip,
                                    use_htf=use_htf
                                )

                                bt_result = backtester.run_backtest(symbol, df.copy(), htf_df)
                                score = calculate_score(bt_result)

                                results.append(SweepResult(
                                    symbol=symbol,
                                    indicator=indicator,
                                    param_a=pa, param_b=pb, param_c=pc,
                                    hold_bars=hold_bars,
                                    use_time_exit=use_time,
                                    use_trend_flip=use_flip,
                                    use_htf=use_htf,
                                    total_trades=bt_result['total_trades'],
                                    win_rate=bt_result['win_rate'],
                                    total_pnl=bt_result['total_pnl'],
                                    profit_factor=bt_result['profit_factor'],
                                    max_drawdown_pct=bt_result['max_drawdown_pct'],
                                    sharpe_ratio=bt_result['sharpe_ratio'],
                                    score=score
                                ))

        all_results[symbol] = results

        if results:
            best = max(results, key=lambda r: r.score)
            print(f"Best: {indicator}({best.param_a},{best.param_b}) Hold={best.hold_bars} "
                  f"Score={best.score:.3f} PnL=${best.total_pnl:+,.0f}")
        else:
            print("No results")

    return all_results


# ============================================
# OUTPUT
# ============================================
def find_best_params(results: Dict[str, List[SweepResult]]) -> pd.DataFrame:
    """Find best parameters for each symbol."""
    rows = []
    for symbol, sweep_results in results.items():
        if not sweep_results:
            continue
        best = max(sweep_results, key=lambda r: r.score)
        rows.append({
            'Symbol': symbol,
            'Direction': 'long',
            'Indicator': best.indicator,
            'ParamA': best.param_a,
            'ParamB': best.param_b,
            'ParamC': best.param_c,
            'HoldBars': best.hold_bars,
            'TimeExit': best.use_time_exit,
            'TrendFlip': best.use_trend_flip,
            'HTF': best.use_htf,
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
    print("\n" + "="*120)
    print("MULTI-INDICATOR PARAMETER SWEEP SUMMARY")
    print("="*120)

    df = find_best_params(results)
    if df.empty:
        print("No results")
        return

    print(f"{'Symbol':<8} {'Indicator':<12} {'Params':>18} {'Hold':>6} {'HTF':>4} {'Trades':>7} {'Win%':>7} "
          f"{'PnL':>12} {'PF':>6} {'MaxDD%':>8} {'Score':>7}")
    print("-"*120)

    for _, row in df.iterrows():
        params_str = f"({row['ParamA']},{row['ParamB']})"
        htf_str = "Y" if row['HTF'] else "N"
        pnl_str = f"${row['TotalPnL']:+,.0f}"
        print(f"{row['Symbol']:<8} {row['Indicator']:<12} {params_str:>18} {row['HoldBars']:>6} {htf_str:>4} "
              f"{row['Trades']:>7} {row['WinRate']:>6.1f}% {pnl_str:>12} {row['ProfitFactor']:>6.2f} "
              f"{row['MaxDD%']:>7.2f}% {row['Score']:>7.3f}")

    print("="*120)

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

    # Full results
    full_output = filepath.replace('.csv', '_full.csv')
    df.to_csv(full_output, index=False)
    print(f"Full results saved to {full_output}")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(
        description='Multi-Indicator Parameter Sweep (Supertrend, JMA, KAMA, HTF)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python parameter_sweep.py --all-nasdaq                       # Supertrend sweep
  python parameter_sweep.py --all-nasdaq --indicator jma       # JMA sweep
  python parameter_sweep.py --all-nasdaq --indicator kama      # KAMA sweep
  python parameter_sweep.py --all-nasdaq --htf                 # Supertrend with HTF filter
  python parameter_sweep.py --all-nasdaq --indicator supertrend_htf  # HTF-filtered Supertrend
  python parameter_sweep.py --quick                            # Quick sweep
  python parameter_sweep.py --thorough                         # Thorough sweep
        """
    )

    # Symbol selection
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS, help='Symbols to sweep')
    parser.add_argument('--all-dow', action='store_true', help='Use all DOW 30 symbols')
    parser.add_argument('--all-nasdaq', action='store_true', help='Use all NASDAQ 100 symbols')

    # Indicator selection
    parser.add_argument('--indicator', '-i', default='supertrend',
                        choices=['supertrend', 'jma', 'kama', 'supertrend_htf'],
                        help='Indicator type (default: supertrend)')
    parser.add_argument('--htf', action='store_true', help='Enable HTF filter')
    parser.add_argument('--htf-interval', default='1d', help='HTF interval (default: 1d)')

    # Data settings
    parser.add_argument('--period', default='1y', help='Historical period')
    parser.add_argument('--interval', default='1h', help='Bar interval')
    parser.add_argument('--start', default=None, help='Start date: YYYY-MM-DD')
    parser.add_argument('--end', default=None, help='End date: YYYY-MM-DD')

    # Sweep modes
    parser.add_argument('--quick', action='store_true', help='Quick sweep (fewer params)')
    parser.add_argument('--thorough', action='store_true', help='Thorough sweep (more params)')
    parser.add_argument('--full', action='store_true', help='Full sweep (all exit strategies)')

    # Parameter overrides
    parser.add_argument('--param-a', nargs='+', type=float, help='ParamA values (ATR period / JMA fast / KAMA ER)')
    parser.add_argument('--param-b', nargs='+', type=float, help='ParamB values (ATR mult / JMA slow / KAMA fast)')
    parser.add_argument('--param-c', nargs='+', type=float, help='ParamC values (Phase / Slow EMA)')
    parser.add_argument('--hold-bars', nargs='+', type=int, help='Hold bars to test')

    # Output
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

    indicator = args.indicator
    use_htf = args.htf or indicator == 'supertrend_htf'

    # Parameter ranges based on mode and indicator
    if args.param_a:
        params_a = args.param_a
    elif args.quick:
        params_a = [10, 14] if indicator == 'supertrend' else [7, 10]
    elif args.thorough:
        params_a = [5, 7, 10, 12, 14, 17, 20, 25] if indicator == 'supertrend' else [5, 7, 10, 14, 20]
    else:
        params_a = None

    if args.param_b:
        params_b = args.param_b
    elif args.quick:
        params_b = [2.5, 3.0, 3.5] if indicator == 'supertrend' else [14, 21]
    elif args.thorough:
        params_b = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] if indicator == 'supertrend' else [14, 21, 30, 50]
    else:
        params_b = None

    if args.param_c:
        params_c = args.param_c
    elif indicator in ['jma']:
        params_c = [0] if args.quick else [-50, 0, 50]
    elif indicator == 'kama':
        params_c = [30] if args.quick else [20, 30, 40]
    else:
        params_c = [0]

    if args.hold_bars:
        hold_bars_list = args.hold_bars
    elif args.quick:
        hold_bars_list = [5, 7]
    elif args.thorough:
        hold_bars_list = [3, 5, 7, 10, 14, 20]
    else:
        hold_bars_list = [3, 5, 7, 10, 14]

    # Exit strategy testing
    if args.full:
        test_time_exit = [True, False]
        test_trend_flip = [True, False]
    else:
        test_time_exit = [True]
        test_trend_flip = [True]

    print("="*60)
    print("MULTI-INDICATOR PARAMETER SWEEP - DOW/NASDAQ")
    print("="*60)
    print(f"Symbols: {len(symbols)} stocks")
    print(f"Indicator: {indicator.upper()}")
    print(f"HTF Filter: {'Yes' if use_htf else 'No'}")
    print(f"Period: {args.period}, Interval: {args.interval}")
    print(f"Mode: {'Quick' if args.quick else 'Thorough' if args.thorough else 'Full' if args.full else 'Standard'}")
    print("="*60 + "\n")

    results = run_sweep(
        symbols=symbols,
        indicator=indicator,
        period=args.period,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        params_a=params_a,
        params_b=params_b,
        params_c=params_c,
        hold_bars_list=hold_bars_list,
        test_time_exit=test_time_exit,
        test_trend_flip=test_trend_flip,
        use_htf=use_htf,
        htf_interval=args.htf_interval
    )

    print_sweep_summary(results)

    # Save results
    df = find_best_params(results)
    if not df.empty:
        save_best_params(df, args.output)


if __name__ == "__main__":
    main()
