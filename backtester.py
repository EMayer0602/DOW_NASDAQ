#!/usr/bin/env python3
"""
Backtesting Module for DOW/NASDAQ Stock Trading
Uses ALL optimized parameters from parameter sweep (indicator, hold bars, etc.)
Generates HTML report with plots and trade tables.

Usage:
    python backtester.py --all-nasdaq --output results.csv --trades trades.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import warnings
import os
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not installed. Run: pip install yfinance")

from stock_settings import SYMBOLS, BEST_PARAMS_CSV
from ta.indicators import (
    calculate_supertrend, calculate_jma_crossover,
    calculate_kama_price_cross, calculate_supertrend_htf
)

# ============================================
# DEFAULT SETTINGS
# ============================================
DEFAULT_CAPITAL = 20000.0
DEFAULT_MAX_POSITIONS = 10
DEFAULT_MAX_LONG = 10
DEFAULT_MAX_SHORT = 10  # Shorts enabled
DEFAULT_ATR_PERIOD = 10
DEFAULT_ATR_MULT = 3.0
DEFAULT_HOLD_BARS = 5


# ============================================
# DATA CLASSES
# ============================================
@dataclass
class SymbolParams:
    """All parameters for a symbol."""
    symbol: str
    indicator: str = "supertrend"
    param_a: float = 10
    param_b: float = 3.0
    param_c: float = 0
    hold_bars: int = 5
    use_time_exit: bool = True
    use_trend_flip: bool = True
    use_htf: bool = False


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
    pnl_pct: float = 0.0

    def __post_init__(self):
        if self.direction == "long":
            self.pnl = self.shares * (self.exit_price - self.entry_price)
        else:
            self.pnl = self.shares * (self.entry_price - self.exit_price)
        if self.shares > 0 and self.entry_price > 0:
            self.pnl_pct = (self.pnl / (self.shares * self.entry_price)) * 100


@dataclass
class Position:
    """Open position."""
    symbol: str
    direction: str
    entry_price: float
    shares: int
    entry_time: datetime
    entry_bar: int


@dataclass
class BacktestResult:
    """Results from backtest."""
    symbol: str
    indicator: str
    params: str
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    max_drawdown_pct: float
    sharpe_ratio: float
    trades: List[Trade] = field(default_factory=list)


# ============================================
# LOAD OPTIMIZED PARAMETERS
# ============================================
def load_all_params(filepath: str = None) -> Dict[str, SymbolParams]:
    """Load ALL optimized parameters from CSV (not just ATR)."""
    if filepath is None:
        filepath = BEST_PARAMS_CSV

    params = {}

    if not os.path.exists(filepath):
        print(f"No params file found at {filepath}, using defaults")
        return params

    try:
        # Try to load full params file first
        full_path = filepath.replace('.csv', '_full.csv')
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
        else:
            # Try semicolon then comma separator
            try:
                df = pd.read_csv(filepath, sep=';')
                if 'ParamA' not in df.columns:
                    df = pd.read_csv(filepath, sep=',')
            except:
                df = pd.read_csv(filepath)

        for _, row in df.iterrows():
            symbol = row['Symbol']

            # Handle European decimal format
            param_a = float(str(row.get('ParamA', 10)).replace(',', '.'))
            param_b = float(str(row.get('ParamB', 3.0)).replace(',', '.'))
            param_c = float(str(row.get('ParamC', 0)).replace(',', '.')) if 'ParamC' in row else 0

            params[symbol] = SymbolParams(
                symbol=symbol,
                indicator=row.get('Indicator', 'supertrend'),
                param_a=param_a,
                param_b=param_b,
                param_c=param_c,
                hold_bars=int(row.get('HoldBars', DEFAULT_HOLD_BARS)),
                use_time_exit=row.get('TimeExit', True) if 'TimeExit' in row else True,
                use_trend_flip=row.get('TrendFlip', True) if 'TrendFlip' in row else True,
                use_htf=row.get('HTF', False) if 'HTF' in row else False
            )

        print(f"Loaded optimized params for {len(params)} symbols")
    except Exception as e:
        print(f"Error loading params: {e}")

    return params


# ============================================
# DATA FETCHING
# ============================================
def fetch_data(symbol: str, period: str = "1y", interval: str = "1h",
               start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data with proper datetime index."""
    if not YFINANCE_AVAILABLE:
        return None
    try:
        ticker = yf.Ticker(symbol)

        # Use date range if provided, otherwise use period
        if start_date:
            df = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)

        if df.empty:
            return None

        df.columns = [c.lower() for c in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume']]

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')

        # Remove NaT rows
        df = df[df.index.notna()]

        # Make timezone naive for consistency
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return df
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def fetch_htf_data(symbol: str, period: str = "2y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """Fetch higher timeframe data."""
    return fetch_data(symbol, period, interval)


# ============================================
# MULTI-INDICATOR BACKTESTER
# ============================================
class Backtester:
    """Backtester supporting all indicator types."""

    def __init__(
        self,
        initial_capital: float = DEFAULT_CAPITAL,
        max_positions: int = DEFAULT_MAX_POSITIONS,
        max_long: int = DEFAULT_MAX_LONG,
        max_short: int = DEFAULT_MAX_SHORT
    ):
        self.initial_capital = initial_capital
        self.position_size = initial_capital / max_positions  # Dynamic: Capital / max_positions
        self.max_positions = max_positions
        self.max_long = max_long
        self.max_short = max_short

        self.cash = initial_capital
        self.position: Optional[Position] = None
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def reset(self):
        self.cash = self.initial_capital
        self.position = None
        self.trades = []
        self.equity_curve = []

    def calculate_indicators(self, df: pd.DataFrame, params: SymbolParams,
                            htf_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculate indicators based on params."""
        df = df.copy()
        indicator = params.indicator

        if indicator == "supertrend":
            period = int(params.param_a)
            mult = params.param_b
            df = calculate_supertrend(df, period, mult)
            df['signal_trend'] = df['trend']

        elif indicator == "supertrend_htf":
            period = int(params.param_a)
            mult = params.param_b
            if htf_df is not None:
                df = calculate_supertrend_htf(df, htf_df, period, mult)
                df['signal_trend'] = df['aligned_trend']
            else:
                df = calculate_supertrend(df, period, mult)
                df['signal_trend'] = df['trend']

        elif indicator == "jma":
            fast = int(params.param_a)
            slow = int(params.param_b)
            phase = int(params.param_c)
            df = calculate_jma_crossover(df, fast, slow, phase)
            df['signal_trend'] = df['jma_trend']

        elif indicator == "kama":
            er = int(params.param_a)
            fast = int(params.param_b)
            slow = int(params.param_c) if params.param_c > 0 else 30
            df = calculate_kama_price_cross(df, er, fast, slow)
            df['signal_trend'] = df['kama_trend']

        else:
            # Default to supertrend
            df = calculate_supertrend(df, int(params.param_a), params.param_b)
            df['signal_trend'] = df['trend']

        return df

    def run(self, symbol: str, df: pd.DataFrame, params: SymbolParams,
            htf_df: Optional[pd.DataFrame] = None) -> BacktestResult:
        """Run backtest for single symbol."""
        self.reset()

        # Validate datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors='coerce')
        df = df[df.index.notna()]

        if len(df) < 50:
            return self._empty_result(symbol, params, df)

        # Calculate indicators
        df = self.calculate_indicators(df, params, htf_df)

        # Warmup period
        warmup = max(int(params.param_a), int(params.param_b), 25) + 5

        for i in range(warmup, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]
            timestamp = df.index[i]
            price = current['close']
            trend_now = current.get('signal_trend', 0)
            trend_prev = prev.get('signal_trend', 0)

            # Validate timestamp
            if pd.isna(timestamp):
                continue

            # Convert to Python datetime
            if hasattr(timestamp, 'to_pydatetime'):
                timestamp = timestamp.to_pydatetime()
                if hasattr(timestamp, 'replace') and timestamp.tzinfo is not None:
                    timestamp = timestamp.replace(tzinfo=None)

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
                if params.use_time_exit and bars_held >= params.hold_bars:
                    should_exit = True
                    reason = f"Time ({bars_held} bars)"

                # Trend flip exit
                if not should_exit and params.use_trend_flip:
                    if self.position.direction == "long" and trend_now == -1:
                        should_exit = True
                        reason = "Trend flip"
                    elif self.position.direction == "short" and trend_now == 1:
                        should_exit = True
                        reason = "Trend flip"

                if should_exit:
                    self._close(price, timestamp, i, reason)

            # Check entry
            if not self.position:
                # Long signal
                if trend_prev <= 0 and trend_now == 1:
                    self._open(symbol, "long", price, timestamp, i)
                # Short signal
                elif trend_prev >= 0 and trend_now == -1:
                    self._open(symbol, "short", price, timestamp, i)

        # Close remaining
        if self.position and len(df) > 0:
            last_price = df.iloc[-1]['close']
            last_time = df.index[-1]
            if hasattr(last_time, 'to_pydatetime'):
                last_time = last_time.to_pydatetime()
                if hasattr(last_time, 'replace') and last_time.tzinfo is not None:
                    last_time = last_time.replace(tzinfo=None)
            self._close(last_price, last_time, len(df)-1, "End")

        return self._calc_result(symbol, params, df)

    def _open(self, symbol: str, direction: str, price: float, timestamp: datetime, bar_idx: int):
        # Dynamic position size based on current cash
        position_size = self.cash / self.max_positions
        shares = int(position_size / price)
        if shares <= 0 or shares * price > self.cash:
            return
        self.cash -= shares * price
        self.position = Position(symbol, direction, price, shares, timestamp, bar_idx)

    def _close(self, price: float, timestamp: datetime, bar_idx: int, reason: str):
        if not self.position:
            return
        pos = self.position
        trade = Trade(
            pos.symbol, pos.direction, pos.entry_price, price,
            pos.shares, pos.entry_time, timestamp,
            bar_idx - pos.entry_bar, reason
        )
        self.trades.append(trade)
        if pos.direction == "long":
            self.cash += pos.shares * price
        else:
            self.cash += pos.shares * (2 * pos.entry_price - price)
        self.position = None

    def _empty_result(self, symbol: str, params: SymbolParams, df: pd.DataFrame) -> BacktestResult:
        start = df.index[0] if len(df) > 0 else datetime.now()
        end = df.index[-1] if len(df) > 0 else datetime.now()
        return BacktestResult(
            symbol=symbol, indicator=params.indicator,
            params=f"({params.param_a},{params.param_b})",
            start_date=start, end_date=end,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_pnl=0, profit_factor=0,
            max_drawdown_pct=0, sharpe_ratio=0, trades=[]
        )

    def _calc_result(self, symbol: str, params: SymbolParams, df: pd.DataFrame) -> BacktestResult:
        if not self.trades:
            return self._empty_result(symbol, params, df)

        winners = [t for t in self.trades if t.pnl > 0]
        losers = [t for t in self.trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in self.trades)

        gross_profit = sum(t.pnl for t in winners) if winners else 0
        gross_loss = abs(sum(t.pnl for t in losers)) if losers else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else 0

        equity = np.array(self.equity_curve) if self.equity_curve else np.array([self.initial_capital])
        peak = np.maximum.accumulate(equity)
        dd = peak - equity
        max_dd = np.max(dd)
        max_dd_pct = (max_dd / np.max(peak)) * 100 if np.max(peak) > 0 else 0

        if len(equity) > 1:
            returns = np.diff(equity) / equity[:-1]
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252*7) if np.std(returns) > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            symbol=symbol,
            indicator=params.indicator,
            params=f"({params.param_a},{params.param_b})",
            start_date=df.index[0],
            end_date=df.index[-1],
            total_trades=len(self.trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners)/len(self.trades)*100 if self.trades else 0,
            total_pnl=total_pnl,
            profit_factor=pf,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
            trades=self.trades.copy()
        )


# ============================================
# HTML REPORT WITH PLOTLY CHARTS
# ============================================
def generate_html_report(results: Dict[str, BacktestResult], filepath: str,
                         initial_capital: float = DEFAULT_CAPITAL):
    """Generate HTML report with Plotly charts and trade tables."""

    # Load Plotly library for inline embedding
    plotly_js = ""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plotly_path = os.path.join(script_dir, "plotly.min.js")
    if os.path.exists(plotly_path):
        with open(plotly_path, 'r', encoding='utf-8') as f:
            plotly_js = f.read()

    all_trades = []
    for r in results.values():
        all_trades.extend(r.trades)

    # Sort trades by entry time
    all_trades.sort(key=lambda t: t.entry_time if t.entry_time else datetime.min)

    long_trades = [t for t in all_trades if t.direction == "long"]
    short_trades = [t for t in all_trades if t.direction == "short"]

    total_pnl = sum(t.pnl for t in all_trades)
    winners = [t for t in all_trades if t.pnl > 0]
    losers = [t for t in all_trades if t.pnl <= 0]

    win_rate = len(winners)/len(all_trades)*100 if all_trades else 0
    long_win_rate = len([t for t in long_trades if t.pnl > 0])/len(long_trades)*100 if long_trades else 0
    short_win_rate = len([t for t in short_trades if t.pnl > 0])/len(short_trades)*100 if short_trades else 0
    gross_profit = sum(t.pnl for t in winners)
    gross_loss = abs(sum(t.pnl for t in losers))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Calculate cumulative PnL for equity curve (convert to float for JSON/JS)
    cumulative_pnl = []
    running_total = float(initial_capital)
    trade_dates = []
    for t in all_trades:
        running_total += float(t.pnl)
        cumulative_pnl.append(round(running_total, 2))
        trade_dates.append(t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else '')

    # PnL by symbol for bar chart (convert to float for JSON/JS)
    symbol_pnl = {}
    for t in all_trades:
        symbol_pnl[t.symbol] = symbol_pnl.get(t.symbol, 0) + t.pnl
    sorted_symbols = sorted(symbol_pnl.items(), key=lambda x: x[1], reverse=True)

    # Statistics by symbol
    symbol_stats = []
    for symbol, r in sorted(results.items()):
        if r.total_trades > 0:
            symbol_stats.append({
                'Symbol': symbol,
                'Indicator': r.indicator,
                'Params': r.params,
                'Trades': r.total_trades,
                'Win%': f"{r.win_rate:.1f}",
                'PnL': f"${r.total_pnl:+,.2f}",
                'PF': f"{r.profit_factor:.2f}"
            })

    final_capital = initial_capital + total_pnl
    total_return = (total_pnl / initial_capital) * 100

    # Separate stats for long/short
    long_pnl = sum(t.pnl for t in long_trades)
    short_pnl = sum(t.pnl for t in short_trades)
    long_winners = [t for t in long_trades if t.pnl > 0]
    long_losers = [t for t in long_trades if t.pnl <= 0]
    short_winners = [t for t in short_trades if t.pnl > 0]
    short_losers = [t for t in short_trades if t.pnl <= 0]
    long_gross_profit = sum(t.pnl for t in long_winners) if long_winners else 0
    long_gross_loss = abs(sum(t.pnl for t in long_losers)) if long_losers else 1
    short_gross_profit = sum(t.pnl for t in short_winners) if short_winners else 0
    short_gross_loss = abs(sum(t.pnl for t in short_losers)) if short_losers else 1
    long_pf = long_gross_profit / long_gross_loss if long_gross_loss > 0 else 0
    short_pf = short_gross_profit / short_gross_loss if short_gross_loss > 0 else 0

    html_header = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Backtest Report</title>
    <script>""" + plotly_js + """</script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #1a1a2e; color: #eee; }
        h1, h2, h3 { color: #00d4ff; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #444; padding: 8px; text-align: right; }
        th { background: #16213e; color: #00d4ff; }
        tr:nth-child(even) { background: #1f1f3d; }
        tr:hover { background: #2a2a5a; }
        .positive { color: #00ff88; }
        .negative { color: #ff4466; }
        .summary { background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; display: flex; flex-wrap: wrap; justify-content: space-around; }
        .stat-box { margin: 10px 15px; text-align: center; min-width: 100px; }
        .stat-value { font-size: 24px; font-weight: bold; }
        .stat-label { font-size: 12px; color: #888; }
        .chart-container { background: #16213e; border-radius: 10px; padding: 15px; margin: 20px 0; }
        .charts-row { display: flex; flex-wrap: wrap; gap: 20px; }
        .chart-half { flex: 1; min-width: 400px; }
        details { margin: 10px 0; }
        summary { cursor: pointer; padding: 10px; background: #16213e; border-radius: 5px; font-size: 16px; }
        summary:hover { background: #1f1f5f; }
    </style>
</head>
<body>
    <h1>Backtest Report</h1>
"""
    html = html_header + f"""    <p style="color:#888">Period: {all_trades[0].entry_time.strftime('%Y-%m-%d') if all_trades else 'N/A'} to {all_trades[-1].exit_time.strftime('%Y-%m-%d') if all_trades else 'N/A'}</p>

    <div class="summary">
        <div class="stat-box">
            <div class="stat-value">${initial_capital:,.0f}</div>
            <div class="stat-label">Initial Capital</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">${final_capital:,.0f}</div>
            <div class="stat-label">Final Capital</div>
        </div>
        <div class="stat-box">
            <div class="stat-value {'positive' if total_pnl >= 0 else 'negative'}">${total_pnl:+,.2f}</div>
            <div class="stat-label">Total P&L</div>
        </div>
        <div class="stat-box">
            <div class="stat-value {'positive' if total_return >= 0 else 'negative'}">{total_return:+.1f}%</div>
            <div class="stat-label">Return</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(all_trades)}</div>
            <div class="stat-label">Total Trades</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{win_rate:.1f}%</div>
            <div class="stat-label">Win Rate</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{pf:.2f}</div>
            <div class="stat-label">Profit Factor</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(long_trades)}</div>
            <div class="stat-label">Long Trades ({long_win_rate:.0f}%)</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">{len(short_trades)}</div>
            <div class="stat-label">Short Trades ({short_win_rate:.0f}%)</div>
        </div>
    </div>

    <div class="charts-row">
        <div class="chart-half">
            <div class="chart-container">
                <div id="equityCurve"></div>
            </div>
        </div>
        <div class="chart-half">
            <div class="chart-container">
                <div id="pnlBySymbol"></div>
            </div>
        </div>
    </div>

    <div class="charts-row">
        <div class="chart-half">
            <div class="chart-container">
                <div id="winLossPie"></div>
            </div>
        </div>
        <div class="chart-half">
            <div class="chart-container">
                <div id="pnlDistribution"></div>
            </div>
        </div>
    </div>

    <script>
        function renderCharts() {{
            if (typeof Plotly === 'undefined') {{
                setTimeout(renderCharts, 100);
                return;
            }}
        // Equity Curve
        var equityTrace = {{
            x: {trade_dates},
            y: {cumulative_pnl},
            type: 'scatter',
            mode: 'lines',
            name: 'Equity',
            line: {{ color: '#00d4ff', width: 2 }},
            fill: 'tozeroy',
            fillcolor: 'rgba(0,212,255,0.1)'
        }};
        var equityLayout = {{
            title: 'Equity Curve',
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#1a1a2e',
            font: {{ color: '#eee' }},
            xaxis: {{ title: 'Date', gridcolor: '#333', type: 'date' }},
            yaxis: {{ title: 'Portfolio Value ($)', gridcolor: '#333' }},
            shapes: [{{ type: 'line', x0: '{trade_dates[0] if trade_dates else ""}', x1: '{trade_dates[-1] if trade_dates else ""}', y0: {initial_capital}, y1: {initial_capital}, line: {{ color: '#888', dash: 'dash' }} }}]
        }};
        Plotly.newPlot('equityCurve', [equityTrace], equityLayout);

        // PnL by Symbol
        var symbolNames = {[s[0] for s in sorted_symbols[:20]]};
        var symbolPnLs = {[round(float(s[1]), 2) for s in sorted_symbols[:20]]};
        var barColors = symbolPnLs.map(v => v >= 0 ? '#00ff88' : '#ff4466');
        var pnlTrace = {{
            x: symbolNames,
            y: symbolPnLs,
            type: 'bar',
            marker: {{ color: barColors }}
        }};
        var pnlLayout = {{
            title: 'P&L by Symbol (Top 20)',
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#1a1a2e',
            font: {{ color: '#eee' }},
            xaxis: {{ gridcolor: '#333' }},
            yaxis: {{ title: 'P&L ($)', gridcolor: '#333' }}
        }};
        Plotly.newPlot('pnlBySymbol', [pnlTrace], pnlLayout);

        // Win/Loss Pie
        var pieTrace = {{
            values: [{len(winners)}, {len(losers)}],
            labels: ['Winners', 'Losers'],
            type: 'pie',
            marker: {{ colors: ['#00ff88', '#ff4466'] }},
            hole: 0.4,
            textinfo: 'label+percent'
        }};
        var pieLayout = {{
            title: 'Win/Loss Distribution',
            paper_bgcolor: '#16213e',
            font: {{ color: '#eee' }}
        }};
        Plotly.newPlot('winLossPie', [pieTrace], pieLayout);

        // PnL Distribution Histogram
        var pnlValues = {[round(float(t.pnl), 2) for t in all_trades]};
        var histTrace = {{
            x: pnlValues,
            type: 'histogram',
            nbinsx: 30,
            marker: {{ color: '#00d4ff' }}
        }};
        var histLayout = {{
            title: 'P&L Distribution',
            paper_bgcolor: '#16213e',
            plot_bgcolor: '#1a1a2e',
            font: {{ color: '#eee' }},
            xaxis: {{ title: 'P&L ($)', gridcolor: '#333' }},
            yaxis: {{ title: 'Frequency', gridcolor: '#333' }}
        }};
        Plotly.newPlot('pnlDistribution', [histTrace], histLayout);
        }}
        // Initialize charts when DOM is ready
        if (document.readyState === 'loading') {{
            document.addEventListener('DOMContentLoaded', renderCharts);
        }} else {{
            renderCharts();
        }}
    </script>

    <h2>Statistics Overview</h2>
    <table>
        <tr><th>Type</th><th>Trades</th><th>Winners</th><th>Losers</th><th>Win Rate</th><th>P&L</th><th>Profit Factor</th></tr>
        <tr><td style='text-align:left'><b>Overall</b></td><td>{len(all_trades)}</td><td>{len(winners)}</td><td>{len(losers)}</td><td>{win_rate:.1f}%</td><td class='{'positive' if total_pnl >= 0 else 'negative'}'>${total_pnl:+,.2f}</td><td>{pf:.2f}</td></tr>
        <tr><td style='text-align:left'><b>Long</b></td><td>{len(long_trades)}</td><td>{len(long_winners)}</td><td>{len(long_losers)}</td><td>{long_win_rate:.1f}%</td><td class='{'positive' if long_pnl >= 0 else 'negative'}'>${long_pnl:+,.2f}</td><td>{long_pf:.2f}</td></tr>
        <tr><td style='text-align:left'><b>Short</b></td><td>{len(short_trades)}</td><td>{len(short_winners)}</td><td>{len(short_losers)}</td><td>{short_win_rate:.1f}%</td><td class='{'positive' if short_pnl >= 0 else 'negative'}'>${short_pnl:+,.2f}</td><td>{short_pf:.2f}</td></tr>
    </table>

    <h2>Statistics by Symbol</h2>
    <table>
        <tr><th>Symbol</th><th>Indicator</th><th>Params</th><th>Trades</th><th>Win%</th><th>P&L</th><th>PF</th></tr>
"""

    for s in symbol_stats:
        pnl_class = 'positive' if '+' in s['PnL'] else 'negative'
        html += f"<tr><td style='text-align:left'>{s['Symbol']}</td><td>{s['Indicator']}</td><td>{s['Params']}</td>"
        html += f"<td>{s['Trades']}</td><td>{s['Win%']}%</td><td class='{pnl_class}'>{s['PnL']}</td><td>{s['PF']}</td></tr>\n"

    html += "</table>\n"

    # Long trades table
    html += f"""
    <details>
        <summary>Long Trades ({len(long_trades)}) - Win Rate: {long_win_rate:.1f}%</summary>
        <table>
            <tr><th>Symbol</th><th>Entry Time</th><th>Exit Time</th><th>Entry</th><th>Exit</th><th>Shares</th><th>Bars</th><th>P&L</th><th>Reason</th></tr>
"""
    for t in long_trades:
        entry_str = t.entry_time.strftime('%Y-%m-%d %H:%M') if t.entry_time else 'N/A'
        exit_str = t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else 'N/A'
        pnl_class = 'positive' if t.pnl >= 0 else 'negative'
        html += f"<tr><td style='text-align:left'>{t.symbol}</td><td>{entry_str}</td><td>{exit_str}</td>"
        html += f"<td>${t.entry_price:.2f}</td><td>${t.exit_price:.2f}</td><td>{t.shares}</td>"
        html += f"<td>{t.bars_held}</td><td class='{pnl_class}'>${t.pnl:+,.2f}</td><td>{t.exit_reason}</td></tr>\n"

    html += "</table></details>\n"

    # Short trades table
    html += f"""
    <details>
        <summary>Short Trades ({len(short_trades)}) - Win Rate: {short_win_rate:.1f}%</summary>
        <table>
            <tr><th>Symbol</th><th>Entry Time</th><th>Exit Time</th><th>Entry</th><th>Exit</th><th>Shares</th><th>Bars</th><th>P&L</th><th>Reason</th></tr>
"""
    for t in short_trades:
        entry_str = t.entry_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else 'N/A'
        exit_str = t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time else 'N/A'
        pnl_class = 'positive' if t.pnl >= 0 else 'negative'
        html += f"<tr><td style='text-align:left'>{t.symbol}</td><td>{entry_str}</td><td>{exit_str}</td>"
        html += f"<td>${t.entry_price:.2f}</td><td>${t.exit_price:.2f}</td><td>{t.shares}</td>"
        html += f"<td>{t.bars_held}</td><td class='{pnl_class}'>${t.pnl:+,.2f}</td><td>{t.exit_reason}</td></tr>\n"

    html += "</table></details>\n"
    html += "</body></html>"

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"HTML report saved to {filepath}")


# ============================================
# MAIN
# ============================================
def run_backtest(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1h",
    use_optimized: bool = True,
    params_file: str = None,
    capital: float = DEFAULT_CAPITAL,
    max_positions: int = DEFAULT_MAX_POSITIONS,
    start_date: str = None,
    end_date: str = None
) -> Dict[str, BacktestResult]:
    """Run backtest on multiple symbols."""

    # Load optimized params
    all_params = {}
    if use_optimized:
        all_params = load_all_params(params_file)

    results = {}
    position_size = capital / max_positions  # Dynamic position size

    for symbol in symbols:
        print(f"Backtesting {symbol}...", end=" ", flush=True)

        df = fetch_data(symbol, period, interval, start_date, end_date)
        if df is None or len(df) < 50:
            print("SKIPPED")
            continue

        # Get params (or use defaults)
        if symbol in all_params:
            params = all_params[symbol]
        else:
            params = SymbolParams(symbol=symbol)

        # Fetch HTF data if needed
        htf_df = None
        if params.use_htf or params.indicator == "supertrend_htf":
            htf_df = fetch_htf_data(symbol)

        backtester = Backtester(initial_capital=capital)

        result = backtester.run(symbol, df, params, htf_df)
        results[symbol] = result

        pnl_str = f"${result.total_pnl:+,.0f}"
        print(f"[{params.indicator}] {result.total_trades} trades, {result.win_rate:.0f}% win, {pnl_str}")

    return results


def print_summary(results: Dict[str, BacktestResult], capital: float):
    """Print summary."""
    print("\n" + "="*100)
    print("BACKTEST SUMMARY")
    print("="*100)

    print(f"{'Symbol':<8} {'Indicator':<12} {'Params':<12} {'Trades':>7} {'Win%':>7} {'PnL':>12} {'PF':>6}")
    print("-"*100)

    total_pnl = 0
    total_trades = 0

    for symbol, r in sorted(results.items()):
        pnl_str = f"${r.total_pnl:+,.0f}"
        print(f"{symbol:<8} {r.indicator:<12} {r.params:<12} {r.total_trades:>7} {r.win_rate:>6.1f}% {pnl_str:>12} {r.profit_factor:>6.2f}")
        total_pnl += r.total_pnl
        total_trades += r.total_trades

    print("="*100)
    print(f"TOTAL: {total_trades} trades, P&L: ${total_pnl:+,.2f}, Return: {(total_pnl/capital)*100:+.1f}%")


def save_trades_csv(results: Dict[str, BacktestResult], filepath: str):
    """Save trades to CSV."""
    rows = []
    for r in results.values():
        for t in r.trades:
            rows.append({
                'Symbol': t.symbol,
                'Direction': t.direction,
                'EntryTime': t.entry_time,
                'ExitTime': t.exit_time,
                'EntryPrice': t.entry_price,
                'ExitPrice': t.exit_price,
                'Shares': t.shares,
                'BarsHeld': t.bars_held,
                'PnL': t.pnl,
                'PnLPct': t.pnl_pct,
                'ExitReason': t.exit_reason
            })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Trades saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description='Backtest with ALL optimized parameters')
    parser.add_argument('--symbols', nargs='+', default=SYMBOLS)
    parser.add_argument('--all-dow', action='store_true')
    parser.add_argument('--all-nasdaq', action='store_true')
    parser.add_argument('--period', default='1y', help='Period: 1mo, 3mo, 6mo, 1y, 2y')
    parser.add_argument('--start', default=None, help='Start date: YYYY-MM-DD (e.g. 2024-12-01)')
    parser.add_argument('--end', default=None, help='End date: YYYY-MM-DD (e.g. 2025-01-15)')
    parser.add_argument('--interval', default='1h')
    parser.add_argument('--capital', type=float, default=DEFAULT_CAPITAL)
    parser.add_argument('--max-positions', type=int, default=DEFAULT_MAX_POSITIONS, dest='max_positions')
    parser.add_argument('--output', default=None, help='Results CSV')
    parser.add_argument('--trades', default=None, help='Trades CSV')
    parser.add_argument('--html', default=None, help='HTML report')
    parser.add_argument('--no-optimized', action='store_true')
    parser.add_argument('--params-file', default=None)

    args = parser.parse_args()

    if args.all_nasdaq:
        from stock_symbols import NASDAQ_100_TOP
        symbols = NASDAQ_100_TOP
    elif args.all_dow:
        from stock_symbols import DOW_30
        symbols = DOW_30
    else:
        symbols = args.symbols

    position_size = args.capital / args.max_positions  # Dynamic: Capital / max_positions

    print("="*60)
    print("BACKTESTER - DOW/NASDAQ (Multi-Indicator)")
    print("="*60)
    print(f"Symbols: {len(symbols)}")
    print(f"Capital: ${args.capital:,.0f}, Position: ${position_size:,.0f} (Capital/{args.max_positions})")
    if args.start:
        print(f"Date Range: {args.start} to {args.end or 'now'}")
    else:
        print(f"Period: {args.period}, Interval: {args.interval}")
    print(f"Shorts: ENABLED")
    print("="*60 + "\n")

    results = run_backtest(
        symbols=symbols,
        period=args.period,
        interval=args.interval,
        use_optimized=not args.no_optimized,
        params_file=args.params_file,
        capital=args.capital,
        max_positions=args.max_positions,
        start_date=args.start,
        end_date=args.end
    )

    print_summary(results, args.capital)

    # Always save trades CSV and generate HTML report
    trades_file = args.trades or 'trades.csv'
    save_trades_csv(results, trades_file)

    # Always generate report.html with Plotly charts
    html_file = args.html or 'report.html'
    generate_html_report(results, html_file, args.capital)


if __name__ == "__main__":
    main()
