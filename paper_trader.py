"""Paper trading runner for overall-best strategies.

This script is intended to be launched every 30 minutes (e.g., via Windows
Task Scheduler). On each invocation it
  * loads the latest best_params_overall.csv entries
  * filters by the per-symbol long/short configuration
  * evaluates entry/exit signals using the same logic as the backtester
  * records paper trades to CSV and maintains a JSON state file
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
import plotly.graph_objects as go

import numpy as np
import pandas as pd
from ta.volatility import AverageTrueRange

try:
    from twilio.rest import Client as TwilioClient # pyright: ignore[reportMissingImports]
except ImportError:  # Twilio is optional; SMS alerts require pip install twilio
    TwilioClient = None

import Supertrend_5Min as st

# Import optimal hold times configuration
try:
    from optimal_hold_times_defaults import get_optimal_hold_bars
except ImportError:
    # Fallback if file not found
    def get_optimal_hold_bars(symbol: str, direction: str) -> int:
        return 12 if direction.lower() == "long" else 15

# Import EMA slope filter configuration
try:
    from optimal_ema_slope_defaults import (
        USE_EMA_SLOPE_FILTER,
        get_ema_slope_params,
        should_filter_entry_by_ema_slope
    )
    # Import optional advanced filter flags
    try:
        from optimal_ema_slope_defaults import USE_PRICE_ABOVE_EMA_FILTER
    except ImportError:
        USE_PRICE_ABOVE_EMA_FILTER = False
    try:
        from optimal_ema_slope_defaults import USE_DUAL_EMA_FILTER, SECONDARY_EMA_PERIOD, SECONDARY_SLOPE_THRESHOLD
    except ImportError:
        USE_DUAL_EMA_FILTER = False
        SECONDARY_EMA_PERIOD = 50
        SECONDARY_SLOPE_THRESHOLD = 0.0
except ImportError:
    USE_EMA_SLOPE_FILTER = False
    USE_PRICE_ABOVE_EMA_FILTER = False
    USE_DUAL_EMA_FILTER = False
    SECONDARY_EMA_PERIOD = 50
    SECONDARY_SLOPE_THRESHOLD = 0.0
    def get_ema_slope_params(symbol: str, direction: str):
        return None, None
    def should_filter_entry_by_ema_slope(symbol: str, direction: str, slope: float) -> bool:
        return False

# ========== LOT SIZE / STEP SIZE HANDLING ==========
# Cache for lot sizes fetched from Binance API
_LOT_SIZE_CACHE: Dict[str, float] = {}
_LOT_SIZE_CACHE_LOADED = False


def fetch_lot_sizes_from_binance() -> Dict[str, float]:
    """Fetch lot sizes (stepSize) for all symbols from Binance API."""
    global _LOT_SIZE_CACHE, _LOT_SIZE_CACHE_LOADED
    if _LOT_SIZE_CACHE_LOADED:
        return _LOT_SIZE_CACHE

    try:
        url = "https://api.binance.com/api/v3/exchangeInfo"
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            for symbol_info in data.get("symbols", []):
                symbol = symbol_info.get("symbol", "")
                for filter_info in symbol_info.get("filters", []):
                    if filter_info.get("filterType") == "LOT_SIZE":
                        step_size = float(filter_info.get("stepSize", 1))
                        _LOT_SIZE_CACHE[symbol] = step_size
                        break
            _LOT_SIZE_CACHE_LOADED = True
            print(f"[LotSize] Loaded {len(_LOT_SIZE_CACHE)} lot sizes from Binance API")
        else:
            print(f"[LotSize] Failed to fetch exchange info: HTTP {response.status_code}")
    except Exception as e:
        print(f"[LotSize] Error fetching lot sizes: {e}")

    return _LOT_SIZE_CACHE


def get_lot_size(symbol: str) -> float:
    """Get the lot size (stepSize) for a symbol. Returns 1e-8 as fallback."""
    # Normalize symbol (remove / and handle USDC->USDT)
    clean_symbol = symbol.replace("/", "").upper()

    # Ensure cache is loaded
    if not _LOT_SIZE_CACHE_LOADED:
        fetch_lot_sizes_from_binance()

    # Try exact match
    if clean_symbol in _LOT_SIZE_CACHE:
        return _LOT_SIZE_CACHE[clean_symbol]

    # Try USDC -> USDT variant
    usdt_symbol = clean_symbol.replace("USDC", "USDT")
    if usdt_symbol in _LOT_SIZE_CACHE:
        return _LOT_SIZE_CACHE[usdt_symbol]

    # Fallback to very small step size (effectively no rounding)
    return 1e-8


def round_to_lot_size(amount: float, symbol: str) -> float:
    """Round amount DOWN to the nearest valid lot size for the symbol."""
    lot_size = get_lot_size(symbol)
    if lot_size <= 0:
        return amount
    # Floor to nearest lot size
    import math
    return math.floor(amount / lot_size) * lot_size


def correct_historical_trades_pnl(json_path: str = None) -> int:
    """
    Correct PnL for all historical trades using the correct formula:
    - size_units = round_to_lot_size(stake / entry_price)
    - fees = (entry_price + exit_price) * size_units * fee_rate
    - pnl = size_units * (exit_price - entry_price) - fees

    Returns number of trades corrected.
    """
    # Pre-fetch lot sizes from Binance API
    fetch_lot_sizes_from_binance()

    if json_path is None:
        json_path = SIMULATION_LOG_JSON

    if not os.path.exists(json_path):
        return 0

    try:
        with open(json_path, 'r') as f:
            trades = json.load(f)

        if not isinstance(trades, list):
            return 0

        fee_rate = st.FEE_RATE
        corrected = 0

        for t in trades:
            entry_price = float(t.get('entry_price', 0) or 0)
            exit_price = float(t.get('exit_price', 0) or 0)
            stake = float(t.get('stake', 0) or 0)
            symbol = str(t.get('symbol', '')).replace('/', '')  # Convert "TNSR/USDT" to "TNSRUSDT"

            if entry_price > 0 and exit_price > 0 and stake > 0:
                raw_size = stake / entry_price
                size_units = round_to_lot_size(raw_size, symbol) if symbol else raw_size
                fees = (entry_price + exit_price) * size_units * fee_rate
                direction = str(t.get('direction', 'Long')).lower()

                if direction == 'long':
                    new_pnl = size_units * (exit_price - entry_price) - fees
                else:
                    new_pnl = size_units * (entry_price - exit_price) - fees

                # Only update if different (avoid unnecessary writes)
                old_pnl = float(t.get('pnl', 0) or 0)
                if abs(new_pnl - old_pnl) > 0.001:
                    t['pnl'] = round(new_pnl, 8)
                    t['fees'] = round(fees, 8)
                    t['size_units'] = size_units
                    corrected += 1

        if corrected > 0:
            with open(json_path, 'w') as f:
                json.dump(trades, f, indent=2, default=str)
            print(f"[PnL-Fix] Corrected {corrected} trades in {json_path}")

        return corrected
    except Exception as e:
        print(f"[PnL-Fix] Error correcting trades: {e}")
        return 0


CONFIG_FILE = "paper_trading_config.csv"
STATE_FILE = "paper_trading_state.json"
TRADE_LOG_FILE = "paper_trading_log.csv"
SIMULATION_LOG_FILE = "paper_trading_simulation_log.csv"
SIMULATION_LOG_JSON = "paper_trading_simulation_log.json"
SIMULATION_OPEN_POSITIONS_FILE = "paper_trading_actual_trades.csv"
SIMULATION_OPEN_POSITIONS_JSON = "paper_trading_actual_trades.json"
# Default paths - will be overridden by get_report_dir() for testnet
REPORT_DIR = "report_html"
SIMULATION_SUMMARY_HTML = os.path.join("report_html", "trading_summary.html")
SIMULATION_SUMMARY_JSON = os.path.join("report_html", "trading_summary.json")


def get_report_dir(use_testnet: bool = False) -> str:
    """Return the appropriate report directory based on testnet mode."""
    return "report_testnet" if use_testnet else "report_html"
BEST_PARAMS_CSV = st.OVERALL_PARAMS_CSV
START_TOTAL_CAPITAL = 16_500.0
MAX_OPEN_POSITIONS = 10
STAKE_DIVISOR = 10  # stake = current total_capital / STAKE_DIVISOR = 16500/10 = 1650
# Separate limits for Long (Spot) vs Short (Margin)
MAX_LONG_POSITIONS = 10
MAX_SHORT_POSITIONS = 0  # Shorts disabled for long-only mode
LONG_STAKE = 1650.0  # 16500 / 10 positions
SHORT_STAKE = 200.0  # 1000 / 5 positions
DEFAULT_DIRECTION_CAPITAL = 2_800.0
BASE_BAR_MINUTES = st.timeframe_to_minutes(st.TIMEFRAME)
DEFAULT_SYMBOL_ALLOWLIST = [sym.strip() for sym in st.SYMBOLS if sym and sym.strip()]
DEFAULT_FIXED_STAKE = 2000.0  # Fixed stake per trade
DEFAULT_ALLOWED_DIRECTIONS = ["long"]  # Long only mode
DEFAULT_USE_TESTNET = False  # Testnet should be opt-in with --testnet flag
USE_TIME_BASED_EXIT = True  # Enable time-based exits based on optimal hold times
DISABLE_TREND_FLIP_EXIT = False  # Enable trend flip exits (like original working version)
SIGNAL_DEBUG = False
USE_FUTURES_SIGNALS = False  # Use futures data for signal generation (Option 1 from futures lead analysis)
USE_TREND_STRENGTH_FILTER = False  # Only enter if price is far enough from HTF indicator
TREND_STRENGTH_MIN_PCT = 0.5  # Minimum distance from HTF indicator in % (0.5 = 0.5%)
_TESTNET_ACTIVE = False  # Track if testnet mode is active for dashboard updates
DEFAULT_SIGNAL_INTERVAL_MIN = 15
DEFAULT_SPIKE_INTERVAL_MIN = 5
DEFAULT_ATR_SPIKE_MULT = 2.5
DEFAULT_POLL_SECONDS = 30
TESTNET_DEFAULT_STAKE = 2000.0
TESTNET_POSITIONS_FILE = "crypto9_testnet_positions.json"
TESTNET_CLOSED_TRADES_FILE = "crypto9_testnet_closed_trades.json"


def save_testnet_position(position: Dict, use_testnet: bool = False) -> None:
    """Save a position to the Crypto9 testnet tracking file."""
    if not use_testnet:
        return
    try:
        positions = load_testnet_positions()
        # Add timestamp if not present
        if "tracked_at" not in position:
            position["tracked_at"] = datetime.now().isoformat()
        # Check if position already exists (by symbol + direction)
        key = f"{position.get('symbol', '')}_{position.get('direction', '')}"
        existing_idx = None
        for i, p in enumerate(positions):
            p_key = f"{p.get('symbol', '')}_{p.get('direction', '')}"
            if p_key == key:
                existing_idx = i
                break
        if existing_idx is not None:
            positions[existing_idx] = position
        else:
            positions.append(position)
        with open(TESTNET_POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2, default=str)
        print(f"[Testnet] Position gespeichert: {position.get('symbol')} {position.get('direction')}")
    except Exception as e:
        print(f"[Testnet] Fehler beim Speichern der Position: {e}")


def load_testnet_positions() -> List[Dict]:
    """Load Crypto9 testnet positions from local file."""
    try:
        if os.path.exists(TESTNET_POSITIONS_FILE):
            with open(TESTNET_POSITIONS_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[Testnet] Fehler beim Laden der Positionen: {e}")
    return []


def remove_testnet_position(symbol: str, direction: str) -> None:
    """Remove a closed position from testnet tracking."""
    try:
        positions = load_testnet_positions()
        key = f"{symbol}_{direction}"
        positions = [p for p in positions if f"{p.get('symbol', '')}_{p.get('direction', '')}" != key]
        with open(TESTNET_POSITIONS_FILE, "w") as f:
            json.dump(positions, f, indent=2, default=str)
    except Exception as e:
        print(f"[Testnet] Fehler beim Entfernen der Position: {e}")


def save_testnet_closed_trade(trade: Dict) -> None:
    """Save a closed trade to the Crypto9 testnet closed trades file (with duplicate check)."""
    try:
        trades = []
        if os.path.exists(TESTNET_CLOSED_TRADES_FILE):
            with open(TESTNET_CLOSED_TRADES_FILE, "r") as f:
                trades = json.load(f)

        # Check for duplicate before adding
        trade_key = (
            trade.get("symbol", ""),
            trade.get("entry_time", ""),
            trade.get("exit_time", ""),
            trade.get("indicator", ""),
        )
        for existing in trades:
            existing_key = (
                existing.get("symbol", ""),
                existing.get("entry_time", ""),
                existing.get("exit_time", ""),
                existing.get("indicator", ""),
            )
            if trade_key == existing_key:
                print(f"[Testnet] Trade already exists, skipping duplicate: {trade.get('symbol')}")
                return

        trade["closed_at"] = datetime.now().isoformat()
        trades.append(trade)
        with open(TESTNET_CLOSED_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    except Exception as e:
        print(f"[Testnet] Fehler beim Speichern des geschlossenen Trades: {e}")


def set_max_open_positions(value: int) -> None:
    global MAX_OPEN_POSITIONS
    try:
        val = int(value)
        if val <= 0:
            raise ValueError("max-open-positions must be positive")
        MAX_OPEN_POSITIONS = val
        print(f"[Config] Max open positions set to {MAX_OPEN_POSITIONS}")
    except Exception as exc:
        print(f"[Config] Invalid max-open-positions '{value}': {exc}")


def set_signal_debug(enabled: bool) -> None:
    global SIGNAL_DEBUG
    SIGNAL_DEBUG = bool(enabled)


def set_use_futures_signals(enabled: bool) -> None:
    """Enable/disable using futures data for signal generation."""
    global USE_FUTURES_SIGNALS
    USE_FUTURES_SIGNALS = bool(enabled)
    if USE_FUTURES_SIGNALS:
        print("[Config] Using FUTURES data for signal generation")


def set_testnet_active(enabled: bool) -> None:
    """Track if testnet mode is active for dashboard updates."""
    global _TESTNET_ACTIVE
    _TESTNET_ACTIVE = bool(enabled)


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.is_file():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and value:
                os.environ[key] = value
    except OSError as exc:
        print(f"[Env] Failed to load {env_path}: {exc}")


load_env_file()


def get_api_credentials(use_testnet: bool = False) -> Tuple[Optional[str], Optional[str]]:
    if use_testnet:
        return (
            os.environ.get("BINANCE_API_KEY_TEST"),
            os.environ.get("BINANCE_API_SECRET_TEST"),
        )
    return (
        os.environ.get("BINANCE_API_KEY"),
        os.environ.get("BINANCE_API_SECRET"),
    )


@dataclass
class ConfigEntry:
    symbol: str
    enable_long: bool
    enable_short: bool
    long_capital: float
    short_capital: float


@dataclass
class Position:
    key: str
    symbol: str
    direction: str
    indicator: str
    htf: str
    param_a: float
    param_b: float
    atr_mult: Optional[float]
    min_hold_bars: int
    entry_price: float
    entry_time: str
    entry_atr: float
    stake: float
    size_units: float


@dataclass
class StrategyContext:
    symbol: str
    direction: str
    indicator: str
    htf: str
    param_a: float
    param_b: float
    atr_mult: Optional[float]
    min_hold_bars: int

    @property
    def key(self) -> str:
        return position_key(self.symbol, self.indicator, self.htf, self.direction)

    @property
    def param_desc(self) -> str:
        atr_text = f"{self.atr_mult}" if self.atr_mult is not None else "None"
        return f"ParamA={self.param_a}, ParamB={self.param_b}, ATR={atr_text}"


@dataclass
class TradeResult:
    symbol: str
    direction: str
    indicator: str
    htf: str
    param_desc: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    stake: float
    fees: float
    pnl: float
    equity_after: float
    reason: str
    size_units: float


class OrderExecutor:
    """Interface for submitting real exchange orders."""

    def submit_entry(self, position: Dict) -> None:
        """Called right after an entry is opened."""

    def submit_exit(self, position: Dict, trade: TradeResult) -> None:
        """Called right after an exit trade is recorded."""


class BinanceOrderExecutor(OrderExecutor):
    def __init__(self, exchange):
        self.exchange = exchange

    def _submit(self, symbol: str, side: str, amount: float):
        if amount <= 0:
            print(f"[Order] Skipping {side} {symbol}: zero amount")
            return
        try:
            order = self.exchange.create_order(symbol, "market", side, amount)
            order_id = order.get("id") if isinstance(order, dict) else order
            print(f"[Order] {side.upper()} {amount:.6f} {symbol} -> {order_id}")
            # Fetch full order to get accurate fills/avg price
            try:
                detailed = self.exchange.fetch_order(order_id, symbol)
                return detailed or order
            except Exception as exc:
                print(f"[Order] Fetch order detail failed for {symbol}: {exc}")
                return order
        except Exception as exc:
            print(f"[Order] Failed to submit {side} {symbol}: {exc}")
            return None

    def submit_entry(self, position: Dict):
        direction = str(position.get("direction", "long")).lower()
        side = "buy" if direction == "long" else "sell"
        amount = float(position.get("size_units", 0.0) or 0.0)
        symbol = position.get("symbol")
        if symbol:
            return self._submit(symbol, side, amount)
        return None

    def submit_exit(self, position: Dict, trade: TradeResult):
        direction = str(position.get("direction", "long")).lower()
        side = "sell" if direction == "long" else "buy"
        amount = float(position.get("size_units", trade.size_units))
        symbol = position.get("symbol")
        if symbol:
            return self._submit(symbol, side, amount)
        return None


# Futures Testnet URL
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"


class BinanceFuturesOrderExecutor(OrderExecutor):
    """Order executor for Binance Futures Testnet (used for SHORT trades)."""

    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or os.getenv("BINANCE_API_KEY_TEST", "")
        self.api_secret = api_secret or os.getenv("BINANCE_API_SECRET_TEST", "")
        self.base_url = FUTURES_TESTNET_URL
        self.session = requests.Session()
        self._precision_cache: Dict[str, int] = {}

    def _sign(self, params: dict) -> str:
        """Create HMAC SHA256 signature."""
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        signature = hmac.new(
            self.api_secret.encode(),
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> Optional[dict]:
        """Execute API request."""
        if params is None:
            params = {}
        headers = {"X-MBX-APIKEY": self.api_key}
        url = f"{self.base_url}{endpoint}"

        if signed:
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)

        try:
            if method == "GET":
                response = self.session.get(url, params=params, headers=headers, timeout=30)
            elif method == "POST":
                response = self.session.post(url, params=params, headers=headers, timeout=30)
            else:
                return None

            if response.status_code == 200:
                return response.json()
            else:
                print(f"[Futures] API Error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"[Futures] Request Error: {e}")
            return None

    def _get_precision(self, symbol: str) -> int:
        """Get quantity precision for a symbol."""
        if symbol in self._precision_cache:
            return self._precision_cache[symbol]
        result = self._request("GET", "/fapi/v1/exchangeInfo")
        if result:
            for s in result.get("symbols", []):
                self._precision_cache[s["symbol"]] = s.get("quantityPrecision", 3)
        return self._precision_cache.get(symbol, 3)

    def _get_price(self, symbol: str) -> Optional[float]:
        """Get current futures price."""
        result = self._request("GET", "/fapi/v1/ticker/price", {"symbol": symbol})
        if result:
            return float(result.get("price", 0))
        return None

    def _ccxt_to_binance_symbol(self, symbol: str) -> str:
        """Convert CCXT symbol (BTC/USDT) to Binance format (BTCUSDT)."""
        return symbol.replace("/", "")

    def _submit_futures_order(self, symbol: str, side: str, amount: float) -> Optional[dict]:
        """Submit a futures market order."""
        binance_symbol = self._ccxt_to_binance_symbol(symbol)
        precision = self._get_precision(binance_symbol)
        quantity = round(amount, precision)

        if quantity <= 0:
            print(f"[Futures] Skipping {side} {symbol}: quantity too small")
            return None

        params = {
            "symbol": binance_symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": quantity,
        }

        result = self._request("POST", "/fapi/v1/order", params, signed=True)
        if result:
            order_id = result.get("orderId", "unknown")
            avg_price = float(result.get("avgPrice", 0)) or self._get_price(binance_symbol)
            print(f"[Futures] {side.upper()} {quantity:.6f} {symbol} @ ${avg_price:.2f} -> {order_id}")
            return result
        return None

    def submit_entry(self, position: Dict):
        """Submit SHORT entry (SELL to open)."""
        direction = str(position.get("direction", "long")).lower()
        if direction != "short":
            print(f"[Futures] Skipping non-short entry: {direction}")
            return None
        amount = float(position.get("size_units", 0.0) or 0.0)
        symbol = position.get("symbol")
        if symbol and amount > 0:
            return self._submit_futures_order(symbol, "SELL", amount)
        return None

    def submit_exit(self, position: Dict, trade: TradeResult):
        """Submit SHORT exit (BUY to close)."""
        direction = str(position.get("direction", "long")).lower()
        if direction != "short":
            print(f"[Futures] Skipping non-short exit: {direction}")
            return None
        amount = float(position.get("size_units", trade.size_units))
        symbol = position.get("symbol")
        if symbol and amount > 0:
            return self._submit_futures_order(symbol, "BUY", amount)
        return None


class HybridOrderExecutor(OrderExecutor):
    """Routes Long trades to Spot, Short trades to Futures."""

    def __init__(self, spot_executor: OrderExecutor, futures_executor: OrderExecutor):
        self.spot_executor = spot_executor
        self.futures_executor = futures_executor

    def submit_entry(self, position: Dict):
        direction = str(position.get("direction", "long")).lower()
        if direction == "short":
            return self.futures_executor.submit_entry(position)
        else:
            return self.spot_executor.submit_entry(position)

    def submit_exit(self, position: Dict, trade: TradeResult):
        direction = str(position.get("direction", "long")).lower()
        if direction == "short":
            return self.futures_executor.submit_exit(position, trade)
        else:
            return self.spot_executor.submit_exit(position, trade)


def create_order_executor(use_testnet: bool = False) -> OrderExecutor:
    """Create the appropriate order executor based on mode.

    For testnet:
      - Long trades → Spot Testnet (via ccxt)
      - Short trades → Futures Testnet (via direct API)
    For live:
      - All trades → Spot (via ccxt)
    """
    spot_executor = BinanceOrderExecutor(st.get_exchange())

    if use_testnet:
        futures_executor = BinanceFuturesOrderExecutor()
        print("[Config] Using HybridOrderExecutor: Long→Spot, Short→Futures")
        return HybridOrderExecutor(spot_executor, futures_executor)
    else:
        return spot_executor


def derive_fill_price(order: Dict) -> float:
    if not isinstance(order, dict):
        return 0.0
    price = float(order.get("average") or order.get("price") or 0.0)
    filled = float(order.get("filled") or order.get("executedQty") or 0.0)
    cost = float(order.get("cost") or order.get("cummulativeQuoteQty") or 0.0)
    if price == 0.0 and cost == 0.0:
        # try fills array if available
        fills = order.get("fills") or []
        total_cost = 0.0
        total_qty = 0.0
        for fill in fills:
            qty = float(fill.get("qty") or fill.get("quantity") or 0.0)
            price_fill = float(fill.get("price") or 0.0)
            total_qty += qty
            total_cost += qty * price_fill
        if total_qty > 0:
            price = total_cost / total_qty
            cost = total_cost
    if price == 0.0 and filled > 0:
        price = cost / filled if filled else 0.0
    return float(price or 0.0)


class TradeNotifier:
    """Interface for sending human-facing trade alerts."""

    def notify_entry(self, position: Dict) -> None:
        """Called right after a paper/live entry is opened."""

    def notify_exit(self, trade: TradeResult) -> None:
        """Called right after a trade exit is recorded."""


class SmsNotifier(TradeNotifier):
    def __init__(self, account_sid: str, auth_token: str, from_number: str, to_numbers: Sequence[str]):
        if TwilioClient is None:
            raise RuntimeError("twilio package not installed; run 'pip install twilio'")
        if not account_sid or not auth_token:
            raise ValueError("Twilio credentials are required for SMS alerts")
        if not from_number:
            raise ValueError("Twilio sender number is required")
        recipients = [number.strip() for number in to_numbers if number and number.strip()]
        if not recipients:
            raise ValueError("At least one SMS recipient is required")
        self.client = TwilioClient(account_sid, auth_token)
        self.from_number = from_number
        self.to_numbers = recipients

    def _send(self, body: str) -> None:
        for dest in self.to_numbers:
            try:
                self.client.messages.create(body=body, from_=self.from_number, to=dest)
                print(f"[Notify] SMS sent to {dest}")
            except Exception as exc:
                print(f"[Notify] SMS to {dest} failed: {exc}")

    def notify_entry(self, position: Dict) -> None:
        symbol = position.get("symbol", "?")
        direction = str(position.get("direction", "long")).upper()
        price = float(position.get("entry_price", 0.0) or 0.0)
        stake = float(position.get("stake", 0.0) or 0.0)
        ts = position.get("entry_time", "")
        body = (
            f"PaperTrader ENTRY {symbol} {direction} @ {price:.4f} | stake {stake:.2f} USDT"
            f" | {ts}"
        )
        self._send(body)

    def notify_exit(self, trade: TradeResult) -> None:
        direction = str(trade.direction).upper()
        body = (
            f"PaperTrader EXIT {trade.symbol} {direction} @ {trade.exit_price:.4f}"
            f" | PnL {trade.pnl:.2f} USDT | reason: {trade.reason}"
        )
        self._send(body)


def _parse_phone_list(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [token.strip() for token in value.split(",") if token.strip()]


def build_sms_notifier(enable: bool, override_to: Optional[str] = None) -> Optional[TradeNotifier]:
    if not enable:
        return None
    if TwilioClient is None:
        print("[Notify] Twilio package missing; install it with 'pip install twilio' to send SMS alerts.")
        return None
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_number = os.environ.get("TWILIO_FROM_NUMBER")
    to_numbers = _parse_phone_list(override_to) or _parse_phone_list(os.environ.get("TWILIO_TO_NUMBERS"))
    if not account_sid or not auth_token:
        print("[Notify] TWILIO_ACCOUNT_SID/AUTH_TOKEN missing; SMS alerts disabled.")
        return None
    if not from_number:
        print("[Notify] TWILIO_FROM_NUMBER missing; SMS alerts disabled.")
        return None
    if not to_numbers:
        print("[Notify] Provide recipients via --sms-to or TWILIO_TO_NUMBERS.")
        return None
    try:
        return SmsNotifier(account_sid, auth_token, from_number, to_numbers)
    except Exception as exc:
        print(f"[Notify] Failed to initialize SMS notifier: {exc}")
        return None


def normalize_symbol_list(symbols: Optional[List[str]]) -> List[str]:
    if not symbols:
        return []
    normalized = []
    for sym in symbols:
        if not sym:
            continue
        normalized.append(sym.strip().upper())
    return normalized


def parse_symbol_argument(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return normalize_symbol_list([token for token in arg.split(",")])


def parse_indicator_argument(arg: Optional[str]) -> List[str]:
    if not arg:
        return []
    return [token.strip().lower() for token in arg.split(",") if token.strip()]


def parse_force_entry_argument(arg: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if not arg:
        return None, None
    token = arg.strip()
    if not token:
        return None, None
    parts = token.split(":", 1)
    if len(parts) == 2:
        symbol = parts[0].strip().upper()
        direction = parts[1].strip().lower()
    else:
        symbol = token.strip().upper()
        direction = None
    if direction and direction not in {"long", "short"}:
        print(f"[Force] Unknown direction '{direction}' – defaulting to long")
        direction = None
    return symbol or None, direction


def prune_state_for_indicators(state: Dict, allowed_indicators: Optional[List[str]]) -> None:
    if not allowed_indicators:
        return
    allowed = {item.strip().lower() for item in allowed_indicators if item}
    if not allowed:
        return
    positions = state.get("positions", []) or []
    before_positions = len(positions)
    state["positions"] = [
        pos for pos in positions if str(pos.get("indicator", "")).strip().lower() in allowed
    ]
    removed_positions = before_positions - len(state["positions"])
    last_processed = state.get("last_processed_bar") or {}
    removed_markers = 0
    if isinstance(last_processed, dict):
        for key in list(last_processed.keys()):
            parts = key.split("|")
            indicator_key = parts[1].strip().lower() if len(parts) >= 2 else ""
            if indicator_key not in allowed:
                last_processed.pop(key, None)
                removed_markers += 1
    if removed_positions:
        print(f"[State] Removed {removed_positions} open position(s) blocked by indicator filter: {sorted(allowed)}")
    if removed_markers:
        print(f"[State] Cleared {removed_markers} last-processed markers outside indicator filter")


def _signal_log(message: str) -> None:
    if not SIGNAL_DEBUG:
        return
    ts = datetime.now(st.BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f"[Signal] {ts} {message}")


def filter_best_rows_by_symbol(df: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
    if df.empty or not symbols:
        return df
    mask = df["Symbol"].str.upper().isin(set(symbols))
    filtered = df[mask]
    if filtered.empty:
        print(f"[Filter] Keine Overall-Einträge für {', '.join(symbols)} gefunden.")
    return filtered


def remap_best_rows_for_testnet(df: pd.DataFrame) -> pd.DataFrame:
    """Remap USDC symbols to testnet equivalents for trading.

    Uses parameters trained on USDC pairs but trades USDT/USDC on testnet.
    """
    if df.empty or "Symbol" not in df.columns:
        return df
    # Reverse map: USDC -> testnet symbol
    usdc_to_testnet = {v: k for k, v in st.TESTNET_TO_USDC_MAP.items()}
    df = df.copy()
    df["Symbol"] = df["Symbol"].map(lambda s: usdc_to_testnet.get(s, s))
    return df


def get_usdc_equivalents_for_testnet(testnet_symbols: List[str]) -> List[str]:
    """Get USDC symbol equivalents for testnet symbols."""
    return [st.map_symbol_for_params(s) for s in testnet_symbols]


def filter_best_rows_by_direction(df: pd.DataFrame, allowed: Optional[List[str]] = None) -> pd.DataFrame:
    if df.empty or "Direction" not in df.columns:
        return df
    if not allowed:
        return df
    allowed_set = {item.strip().lower() for item in allowed if item}
    filtered = df[df["Direction"].str.lower().isin(allowed_set)]
    if filtered.empty:
        print(f"[Filter] Keine Einträge für Richtungen: {', '.join(sorted(allowed_set))}.")
    return filtered


def select_best_indicator_per_symbol(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Symbol" not in df.columns or "FinalEquity" not in df.columns:
        return df
    group_cols = ["Symbol"]
    if "Direction" in df.columns:
        group_cols.append("Direction")
    sorted_df = df.sort_values("FinalEquity", ascending=False)
    reduced_df = sorted_df.drop_duplicates(subset=group_cols, keep="first")
    return reduced_df.reset_index(drop=True)


def determine_position_size(symbol: str, state: Dict, fixed_stake: Optional[float], direction: str = "long") -> float:
    """
    Determine position size for a trade.

    Args:
        symbol: Trading symbol
        state: Current state dict with total_capital
        fixed_stake: Fixed stake amount (if provided), None for dynamic sizing
        direction: Trade direction ("long" or "short")

    Returns:
        Position size in USDT

    Notes:
        - Long (Spot): Dynamic sizing = total_capital / MAX_LONG_POSITIONS
        - Short (Margin): Static 200 USDT (limited by 999 USD margin)
    """
    # Use fixed stake if provided and > 0
    if fixed_stake is not None and fixed_stake > 0:
        return fixed_stake

    # Direction-specific stake sizes
    if direction.lower() == "short":
        # Short: Static stake (limited by margin)
        return SHORT_STAKE
    else:
        # Long: Dynamic sizing based on current capital
        total_capital = float(state.get("total_capital", START_TOTAL_CAPITAL))
        dynamic_stake = total_capital / MAX_LONG_POSITIONS
        return max(dynamic_stake, 100.0)  # Minimum 100 USDT


def record_symbol_trade(state: Dict, symbol: str) -> None:
    trade_counts = state.setdefault("symbol_trade_counts", {})
    trade_counts[symbol] = trade_counts.get(symbol, 0) + 1


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        print(f"[Cleanup] Failed to remove {path}: {exc}")


def reset_state_file() -> None:
    _safe_remove(STATE_FILE)


def clear_positions_in_state() -> None:
    """Clear open positions and last_processed_bar in the saved state."""
    state = load_state()
    state["positions"] = []
    state["last_processed_bar"] = {}
    save_state(state)
    print("[State] Cleared open positions and last-processed markers.")


def clear_output_artifacts(include_state: bool = False) -> None:
    targets = [
        TRADE_LOG_FILE,
        SIMULATION_LOG_FILE,
        SIMULATION_LOG_JSON,
        SIMULATION_OPEN_POSITIONS_FILE,
        SIMULATION_OPEN_POSITIONS_JSON,
        SIMULATION_SUMMARY_HTML,
        SIMULATION_SUMMARY_JSON,
    ]
    if include_state:
        targets.append(STATE_FILE)
    for path in targets:
        _safe_remove(path)


def ensure_config(symbols: List[str]) -> pd.DataFrame:
    normalized_symbols = normalize_symbol_list(symbols)
    if os.path.exists(CONFIG_FILE):
        df = pd.read_csv(CONFIG_FILE)
    else:
        df = pd.DataFrame(columns=[
            "Symbol",
            "EnableLong",
            "EnableShort",
            "LongInitialCapital",
            "ShortInitialCapital",
        ])
    existing = set(df.get("Symbol", pd.Series(dtype=str)).astype(str).str.strip().str.upper())
    missing = []
    for sym in normalized_symbols:
        if sym not in existing:
            missing.append(sym)
    if df.empty and not missing:
        missing = normalized_symbols
    if missing:
        rows = []
        for sym in missing:
            rows.append({
                "Symbol": sym,
                "EnableLong": True,
                "EnableShort": True,
                "LongInitialCapital": DEFAULT_DIRECTION_CAPITAL,
                "ShortInitialCapital": DEFAULT_DIRECTION_CAPITAL,
            })
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        df.to_csv(CONFIG_FILE, index=False)
        print(f"[Init] Ergänze Konfiguration für: {', '.join(missing)}")
    return df


def load_config_lookup(df: pd.DataFrame) -> Dict[str, ConfigEntry]:
    lookup: Dict[str, ConfigEntry] = {}
    for _, row in df.iterrows():
        raw_symbol = row.get("Symbol")
        if pd.isna(raw_symbol) or not isinstance(raw_symbol, str):
            continue  # Skip invalid rows
        symbol = raw_symbol.strip()
        if not symbol:
            continue
        lookup[symbol] = ConfigEntry(
            symbol=symbol,
            enable_long=bool(row.get("EnableLong", True)),
            enable_short=bool(row.get("EnableShort", True)),
            long_capital=float(row.get("LongInitialCapital", DEFAULT_DIRECTION_CAPITAL)),
            short_capital=float(row.get("ShortInitialCapital", DEFAULT_DIRECTION_CAPITAL)),
        )
    return lookup


def default_state() -> Dict:
    return {
        "total_capital": START_TOTAL_CAPITAL,
        "positions": [],
        "last_processed_bar": {},
        "symbol_trade_counts": {},
    }


def load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return default_state()


def save_state(state: Dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)


def sync_simulation_positions(state: Dict, use_testnet: bool = False) -> int:
    """Sync simulation open positions to paper trading state.

    If simulation shows a position that isn't in paper trading state, add it.
    This ensures the monitor catches up with positions from the last backtest.
    Returns the number of positions synced.
    """
    sim_file = SIMULATION_OPEN_POSITIONS_JSON
    if not os.path.exists(sim_file):
        return 0

    try:
        with open(sim_file, "r", encoding="utf-8") as fh:
            sim_positions = json.load(fh)
    except Exception as e:
        print(f"[Sync] Could not load simulation positions: {e}")
        return 0

    if not sim_positions:
        return 0

    # Get existing position keys
    existing_keys = set()
    for pos in state.get("positions", []):
        key = f"{pos.get('symbol', '')}|{pos.get('direction', '')}|{pos.get('indicator', '')}|{pos.get('htf', '')}"
        existing_keys.add(key)

    synced = 0
    for sim_pos in sim_positions:
        sim_key = f"{sim_pos.get('symbol', '')}|{sim_pos.get('direction', '')}|{sim_pos.get('indicator', '')}|{sim_pos.get('htf', '')}"
        if sim_key not in existing_keys:
            # Add simulation position to state
            state.setdefault("positions", []).append(sim_pos)
            existing_keys.add(sim_key)
            synced += 1
            print(f"[Sync] Added position from simulation: {sim_pos.get('symbol')} {sim_pos.get('direction')} {sim_pos.get('indicator')}/{sim_pos.get('htf')}")

    if synced > 0:
        save_state(state)
        print(f"[Sync] Synced {synced} positions from simulation to paper trading state")

    return synced


def clone_state(use_saved_state: bool = False) -> Dict:
    base = load_state() if use_saved_state else default_state()
    return json.loads(json.dumps(base))


def append_trade_log(trade: TradeResult) -> None:
    file_exists = os.path.exists(TRADE_LOG_FILE)
    with open(TRADE_LOG_FILE, "a", encoding="utf-8") as fh:
        if not file_exists:
            fh.write(
                "Timestamp,Symbol,Direction,Indicator,HTF,ParamDesc,EntryTime,EntryPrice,ExitTime,ExitPrice,"
                "Stake,Fees,PnL,EquityAfter,Reason\n"
            )
        # Quote ParamDesc and Reason to handle commas inside them
        param_desc_quoted = f'"{trade.param_desc}"' if "," in trade.param_desc else trade.param_desc
        reason_quoted = f'"{trade.reason}"' if "," in trade.reason else trade.reason
        fh.write(
            f"{datetime.utcnow().isoformat()},"
            f"{trade.symbol},{trade.direction},{trade.indicator},{trade.htf},{param_desc_quoted},"
            f"{trade.entry_time},{trade.entry_price:.8f},{trade.exit_time},{trade.exit_price:.8f},"
            f"{trade.stake:.2f},{trade.fees:.2f},{trade.pnl:.2f},{trade.equity_after:.2f},{reason_quoted}\n"
        )


def _load_trade_log_dataframe(log_path: str) -> pd.DataFrame:
    """Read the cumulative trade log CSV and normalize to the same columns as TradeResult."""
    if not os.path.exists(log_path):
        return pd.DataFrame()
    try:
        # Use quotechar to handle commas in ParamDesc and Reason
        df = pd.read_csv(log_path, quotechar='"')
    except Exception as exc:
        print(f"[Live] Failed to read trade log {log_path}: {exc}")
        return pd.DataFrame()

    if df.empty:
        return df

    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()].copy()

    print(f"[DEBUG] CSV columns: {list(df.columns)}")

    # Helper to check if column has actual data
    def has_data(col_name):
        if col_name not in df.columns:
            return False
        col = df[col_name]
        # Check if all values are NaN or empty strings
        return not (col.isna().all() or (col.astype(str).str.strip() == "").all())

    # Ensure all required lowercase columns exist and have data
    # Priority: keep existing lowercase data, fallback to Uppercase, else empty
    if not has_data("symbol"):
        df["symbol"] = df["Symbol"] if "Symbol" in df.columns else ""
    if not has_data("direction"):
        df["direction"] = df["Direction"].astype(str).str.strip().str.capitalize() if "Direction" in df.columns else ""
    if not has_data("indicator"):
        df["indicator"] = df["Indicator"] if "Indicator" in df.columns else ""
    if not has_data("htf"):
        df["htf"] = df["HTF"] if "HTF" in df.columns else ""
    if not has_data("param_desc"):
        df["param_desc"] = df["ParamDesc"] if "ParamDesc" in df.columns else ""
    if not has_data("entry_time"):
        df["entry_time"] = df["EntryTime"] if "EntryTime" in df.columns else ""
    if not has_data("entry_price"):
        df["entry_price"] = pd.to_numeric(df["EntryPrice"], errors="coerce") if "EntryPrice" in df.columns else 0.0
    if not has_data("exit_time"):
        df["exit_time"] = df["ExitTime"] if "ExitTime" in df.columns else ""
    if not has_data("exit_price"):
        df["exit_price"] = pd.to_numeric(df["ExitPrice"], errors="coerce") if "ExitPrice" in df.columns else 0.0
    if not has_data("stake"):
        df["stake"] = pd.to_numeric(df["Stake"], errors="coerce") if "Stake" in df.columns else 0.0
    if not has_data("fees"):
        df["fees"] = pd.to_numeric(df["Fees"], errors="coerce") if "Fees" in df.columns else 0.0
    if not has_data("pnl"):
        df["pnl"] = pd.to_numeric(df["PnL"], errors="coerce") if "PnL" in df.columns else 0.0
    if not has_data("equity_after"):
        df["equity_after"] = pd.to_numeric(df["EquityAfter"], errors="coerce") if "EquityAfter" in df.columns else 0.0
    if not has_data("reason"):
        df["reason"] = df["Reason"] if "Reason" in df.columns else ""

    # Drop the original Uppercase columns to avoid duplicates in saved CSV
    uppercase_cols = ["Symbol", "Direction", "Indicator", "HTF", "ParamDesc",
                     "EntryTime", "EntryPrice", "ExitTime", "ExitPrice",
                     "Stake", "Fees", "PnL", "EquityAfter", "Reason"]
    for col in uppercase_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Debug output
    print(f"[Live] Loaded {len(df)} trades from log.")
    if not df.empty and "symbol" in df.columns:
        print(f"[DEBUG] After mapping - First symbol='{df.iloc[0]['symbol']}', indicator='{df.iloc[0]['indicator']}'")
        unique_symbols = df["symbol"].dropna().unique()
        print(f"[Live] Symbols in log: {', '.join(str(s) for s in unique_symbols)}")

    return df


def parse_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() == "none":
            return None
        stripped = stripped.replace(",", ".")
        try:
            return float(stripped)
        except ValueError:
            return None
    if pd.isna(value):
        return None
    return float(value)


def normalize_params(row: pd.Series, indicator_key: str) -> tuple[float, float]:
    param_a = parse_float(row.get("ParamA"))
    param_b = parse_float(row.get("ParamB"))
    param_a = param_a if param_a is not None else parse_float(row.get(st.PARAM_A_LABEL))
    param_b = param_b if param_b is not None else parse_float(row.get(st.PARAM_B_LABEL))
    if param_a is None:
        param_a = st.DEFAULT_PARAM_A
    if param_b is None:
        param_b = st.DEFAULT_PARAM_B
    if indicator_key in {"supertrend", "jma", "kama"}:
        param_a = int(round(param_a))
    if indicator_key in {"jma", "kama"}:
        param_b = int(round(param_b))
    return float(param_a), float(param_b)


def build_strategy_context(row: pd.Series) -> StrategyContext:
    symbol = row["Symbol"].strip()
    direction = row["Direction"].strip().lower()
    indicator_key = row["Indicator"].strip()
    htf_value = str(row.get("HTF", st.HIGHER_TIMEFRAME) or st.HIGHER_TIMEFRAME).strip()
    param_a, param_b = normalize_params(row, indicator_key)
    atr_mult = parse_float(row.get("ATRStopMultValue", row.get("ATRStopMult")))
    # MinHoldBars from CSV - number of HTF bars before forced exit
    min_hold_bars = int(parse_float(row.get("MinHoldBars")) or 0)
    return StrategyContext(
        symbol=symbol,
        direction=direction,
        indicator=indicator_key,
        htf=htf_value,
        param_a=param_a,
        param_b=param_b,
        atr_mult=atr_mult,
        min_hold_bars=min_hold_bars,
    )


def build_dataframe_for_context(context: StrategyContext, use_all_data: bool = False) -> pd.DataFrame:
    return build_indicator_dataframe(context.symbol, context.indicator, context.htf, context.param_a, context.param_b, use_all_data=use_all_data)


def resolve_timestamp(value: Optional[str], default: Optional[pd.Timestamp] = None) -> pd.Timestamp:
    if value is None or value == "":
        if default is None:
            raise ValueError("Timestamp value is required")
        ts = default
    else:
        ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(st.BERLIN_TZ)
    else:
        ts = ts.tz_convert(st.BERLIN_TZ)
    return ts


def to_berlin_iso(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    if ts.tzinfo is None:
        ts = ts.tz_localize(st.BERLIN_TZ)
    else:
        ts = ts.tz_convert(st.BERLIN_TZ)
    return ts.isoformat()


def _to_numeric_series(obj: Any) -> pd.Series:
    if isinstance(obj, pd.DataFrame):
        obj = obj.iloc[:, 0]
    return pd.to_numeric(pd.Series(obj), errors="coerce")


def _infer_timestamp_from_df(df: pd.DataFrame, column: str, fallback: pd.Timestamp) -> pd.Timestamp:
    if column not in df.columns:
        return fallback
    series = pd.to_datetime(df[column], errors="coerce")
    series = series.dropna()
    if series.empty:
        return fallback
    ts = series.min()
    if ts.tzinfo is None:
        ts = ts.tz_localize(st.BERLIN_TZ)
    else:
        ts = ts.tz_convert(st.BERLIN_TZ)
    return ts


def load_best_rows(active_indicators: Optional[List[str]] = None) -> pd.DataFrame:
    if not os.path.exists(BEST_PARAMS_CSV):
        raise FileNotFoundError(
            f"Overall summary file {BEST_PARAMS_CSV} not found. Run a parameter sweep first."
        )
    # Try American format first (comma separator), fall back to European (semicolon)
    try:
        df = pd.read_csv(BEST_PARAMS_CSV)
        if "Symbol" not in df.columns:
            raise ValueError("Symbol column not found")
    except (ValueError, pd.errors.ParserError):
        df = pd.read_csv(BEST_PARAMS_CSV, sep=";", decimal=",")
    if df.empty:
        return df
    if active_indicators:
        allowed = {item.strip().lower() for item in active_indicators if item}
        df = df[df["Indicator"].str.lower().isin(allowed)]
    return df


def load_best_rows_from_detailed(path: str = os.path.join("report_html", "overall_best_detailed.html")) -> pd.DataFrame:
    """Load best rows from the detailed HTML report if available."""
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        tables = pd.read_html(path)
    except Exception as exc:
        print(f"[Force] Konnte {path} nicht lesen: {exc}")
        return pd.DataFrame()
    for table in tables:
        if "Symbol" in table.columns:
            return table
    return pd.DataFrame()


def load_replay_trades_csv(path: str) -> pd.DataFrame:
    """Load a precomputed trades CSV (e.g., last48_from_detailed_all.csv) for replay/summary purposes."""
    if not path:
        return pd.DataFrame()
    if not os.path.exists(path):
        print(f"[Replay] Trades CSV not found: {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[Replay] Failed to read trades CSV {path}: {exc}")
        return pd.DataFrame()
    if df.empty:
        print(f"[Replay] Trades CSV {path} is empty.")
        return df

    # Drop duplicate columns that break JSON export
    df = df.loc[:, ~df.columns.duplicated()].copy()

    rename_map = {
        "PnL (USD)": "pnl_usd",
        "PnL (recalc)": "pnl_recalc",
        "ExitReason": "reason",
        "CapitalAfter": "equity_after",
        "Equity": "equity_after",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Normalize required numeric fields
    for col in ("pnl_recalc", "pnl_usd", "Stake", "Fees", "StakeUsed", "Entry", "ExitPreis", "equity_after"):
        if col in df.columns:
            df[col] = _to_numeric_series(df[col])

    if "pnl" not in df.columns:
        for candidate in ("pnl_recalc", "pnl_usd"):
            if candidate in df.columns:
                df["pnl"] = pd.to_numeric(df[candidate], errors="coerce")
                break
    if "pnl" not in df.columns:
        df["pnl"] = 0.0

    if "equity_after" not in df.columns:
        df["equity_after"] = df["pnl"].cumsum() + START_TOTAL_CAPITAL

    if "reason" not in df.columns and "ExitReason" in df.columns:
        df["reason"] = df["ExitReason"]

    df["symbol"] = df.get("Symbol", "")
    df["direction"] = df.get("Direction", "").astype(str).str.lower()
    df["entry_price"] = df.get("Entry", np.nan)
    df["exit_price"] = df.get("ExitPreis", np.nan)
    df["stake"] = df.get("Stake", np.nan)
    df["fees"] = df.get("Fees", np.nan)

    if "Zeit" in df.columns:
        df["entry_time"] = df["Zeit"].apply(to_berlin_iso)
    if "ExitZeit" in df.columns:
        df["exit_time"] = df["ExitZeit"].apply(to_berlin_iso)

    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def build_indicator_dataframe(symbol: str, indicator_key: str, htf_value: str, param_a: float, param_b: float, use_all_data: bool = False) -> pd.DataFrame:
    st.apply_indicator_type(indicator_key)
    st.apply_higher_timeframe(htf_value)
    df_raw = st.prepare_symbol_dataframe(symbol, use_all_cached_data=use_all_data)

    # Return empty DataFrame if no data available for this symbol
    if df_raw.empty:
        return pd.DataFrame()

    # Option 1: Use futures data for signal generation (futures lead analysis)
    if USE_FUTURES_SIGNALS:
        # Fetch futures data for the same time period
        futures_df = st.fetch_futures_data(symbol, st.TIMEFRAME, len(df_raw) + 100)
        if not futures_df.empty:
            # Compute indicator on futures data
            futures_ind = st.compute_indicator(futures_df.copy(), param_a, param_b)

            # Compute indicator on spot data (keeps spot prices and bands)
            df_ind = st.compute_indicator(df_raw.copy(), param_a, param_b)

            # Align futures signals with spot data by timestamp
            common_idx = df_ind.index.intersection(futures_ind.index)
            if len(common_idx) > 100:
                # Only copy TREND DIRECTION signals from futures
                # DO NOT copy price-based bands (supertrend/upper_band/lower_band)
                # because futures is in USDT, spot is in EUR - different price scales!
                for col in ["trend_flag", "trend_flip"]:
                    if col in futures_ind.columns:
                        df_ind.loc[common_idx, col] = futures_ind.loc[common_idx, col]
                print(f"[Futures] Applied futures trend signals to {symbol} ({len(common_idx)} bars)")
            else:
                print(f"[Futures] Not enough common bars for {symbol} ({len(common_idx)}), using spot signals")
        else:
            print(f"[Futures] No futures data for {symbol}, falling back to spot signals")
            df_ind = st.compute_indicator(df_raw.copy(), param_a, param_b)
    else:
        df_ind = st.compute_indicator(df_raw.copy(), param_a, param_b)

    for col in ("htf_trend", "htf_indicator", "momentum"):
        if col in df_raw.columns:
            df_ind[col] = df_raw[col]

    # Add EMA slope filter columns for long trade filtering
    if USE_EMA_SLOPE_FILTER:
        # Primary EMA-20 for all symbols
        ema_period = 20
        df_ind[f'ema_{ema_period}'] = df_ind['close'].ewm(span=ema_period, adjust=False).mean()
        # Calculate slope as percentage change (lookback=1 bar)
        ema = df_ind[f'ema_{ema_period}']
        df_ind[f'ema_{ema_period}_slope'] = ((ema - ema.shift(1)) / ema.shift(1)) * 100

        # Add EMA-50 for dual EMA confirmation (Option 3)
        if USE_DUAL_EMA_FILTER:
            secondary_period = SECONDARY_EMA_PERIOD
            df_ind[f'ema_{secondary_period}'] = df_ind['close'].ewm(span=secondary_period, adjust=False).mean()
            ema_secondary = df_ind[f'ema_{secondary_period}']
            df_ind[f'ema_{secondary_period}_slope'] = ((ema_secondary - ema_secondary.shift(1)) / ema_secondary.shift(1)) * 100

        # Store symbol for later reference
        df_ind['symbol'] = symbol

    return df_ind.dropna(subset=["trend_flag"])


def direction_allowed(config: ConfigEntry, direction: str) -> bool:
    if direction == "long":
        return config.enable_long
    return config.enable_short


def position_key(symbol: str, indicator: str, htf: str, direction: str) -> str:
    return f"{symbol}|{indicator}|{htf}|{direction}"


def find_position(state: Dict, key: str) -> Optional[Dict]:
    for pos in state["positions"]:
        if pos["key"] == key:
            return pos
    return None


def remove_position(state: Dict, key: str) -> None:
    state["positions"] = [pos for pos in state["positions"] if pos["key"] != key]


def bars_in_position(entry_iso: str, latest_ts: pd.Timestamp, htf_timeframe: Optional[str] = None) -> int:
    """
    Calculate number of bars held in position.

    Args:
        entry_iso: ISO timestamp of entry
        latest_ts: Current timestamp
        htf_timeframe: Higher timeframe (e.g. '6h', '24h'). If None, uses BASE_BAR_MINUTES.

    Returns:
        Number of bars held in the position's timeframe
    """
    entry_ts = pd.Timestamp(entry_iso)
    delta_minutes = max(0.0, (latest_ts - entry_ts).total_seconds() / 60.0)

    # Determine bar size in minutes
    if htf_timeframe:
        try:
            bar_minutes = st.timeframe_to_minutes(htf_timeframe)
        except:
            bar_minutes = BASE_BAR_MINUTES
    else:
        bar_minutes = BASE_BAR_MINUTES

    if bar_minutes <= 0:
        return 0
    return int(delta_minutes // bar_minutes)


def filters_allow_entry(direction: str, df: pd.DataFrame) -> tuple[bool, str]:
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    htf_value = int(curr.get("htf_trend", 0))
    if st.USE_HIGHER_TIMEFRAME_FILTER:
        if direction == "long" and htf_value < 1:
            return False, f"HTF filter blocked (value={htf_value})"
        if direction == "short" and htf_value > -1:
            return False, f"HTF filter blocked (value={htf_value})"
    if st.USE_MOMENTUM_FILTER and "momentum" in df.columns:
        mom_value = curr.get("momentum")
        if np.isnan(mom_value):
            return False, "Momentum unavailable"
        if direction == "long" and mom_value < st.RSI_LONG_THRESHOLD:
            return False, f"Momentum below threshold ({mom_value:.2f})"
        if direction == "short" and mom_value > st.RSI_SHORT_THRESHOLD:
            return False, f"Momentum above threshold ({mom_value:.2f})"
    if st.USE_BREAKOUT_FILTER:
        atr_curr = curr.get("atr")
        if atr_curr is None or atr_curr <= 0:
            return False, "ATR unavailable"
        candle_range = float(curr["high"] - curr["low"])
        allows = candle_range >= st.BREAKOUT_ATR_MULT * float(atr_curr)
        if allows and st.BREAKOUT_REQUIRE_DIRECTION:
            prev_high = float(prev["high"])
            prev_low = float(prev["low"])
            close_curr = float(curr["close"])
            if direction == "long":
                allows = close_curr > prev_high
            else:
                allows = close_curr < prev_low
        if not allows:
            return False, f"Breakout filter blocked (range={candle_range:.4f})"

    # EMA Slope filter for LONG trades only
    if USE_EMA_SLOPE_FILTER and direction == "long":
        ema_slope = curr.get("ema_20_slope")
        if pd.isna(ema_slope):
            return False, "EMA slope unavailable"
        # Block entry if EMA is falling (slope < 0%)
        # Using slope >= 0% threshold: allows neutral or rising trends only
        if ema_slope < 0.0:
            return False, f"EMA-20 slope negative ({ema_slope:.3f}%) - downtrend blocked"

        # OPTION 1: Price must be above EMA-20
        if USE_PRICE_ABOVE_EMA_FILTER:
            current_price = float(curr["close"])
            ema_20_value = curr.get("ema_20")
            if pd.isna(ema_20_value):
                return False, "EMA-20 value unavailable"
            if current_price < ema_20_value:
                return False, f"Price below EMA-20 ({current_price:.2f} < {ema_20_value:.2f})"

        # OPTION 3: Dual EMA confirmation - EMA-50 must also be rising
        if USE_DUAL_EMA_FILTER:
            ema_50_slope = curr.get(f"ema_{SECONDARY_EMA_PERIOD}_slope")
            if pd.isna(ema_50_slope):
                return False, f"EMA-{SECONDARY_EMA_PERIOD} slope unavailable"
            if ema_50_slope < SECONDARY_SLOPE_THRESHOLD:
                return False, f"EMA-{SECONDARY_EMA_PERIOD} slope too low ({ema_50_slope:.3f}% < {SECONDARY_SLOPE_THRESHOLD}%)"

    return True, ""


def evaluate_entry(df: pd.DataFrame, direction: str) -> tuple[bool, str]:
    if len(df) < 2:
        return False, "Not enough bars"
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    trend_curr = int(curr["trend_flag"])
    trend_prev = int(prev["trend_flag"])
    if direction == "long":
        signal = trend_prev == -1 and trend_curr == 1
    else:
        signal = trend_prev == 1 and trend_curr == -1
    if not signal:
        return False, "Trend did not flip"

    # Trend strength filter: only enter if price is far enough from HTF indicator
    if USE_TREND_STRENGTH_FILTER and "htf_indicator" in curr.index:
        close_price = float(curr["close"])
        htf_ind = float(curr["htf_indicator"])
        if htf_ind > 0:
            distance_pct = ((close_price - htf_ind) / htf_ind) * 100
            if direction == "long" and distance_pct < TREND_STRENGTH_MIN_PCT:
                return False, f"Trend strength too weak: {distance_pct:.2f}% < {TREND_STRENGTH_MIN_PCT}%"
            if direction == "short" and distance_pct > -TREND_STRENGTH_MIN_PCT:
                return False, f"Trend strength too weak: {distance_pct:.2f}% > -{TREND_STRENGTH_MIN_PCT}%"

    allows, reason = filters_allow_entry(direction, df)
    if not allows:
        return False, reason
    return True, ""


def find_last_signal_bar(df: pd.DataFrame, direction: str, lookback_hours: float = 24.0) -> tuple[pd.Timestamp, float, bool]:
    """Locate the most recent trend-flip bar for the given direction within lookback_hours.
    Returns (timestamp, price, is_within_window).
    """
    now_ts = pd.Timestamp.now(tz=st.BERLIN_TZ)
    cutoff = now_ts - pd.Timedelta(hours=lookback_hours)
    if df is None or len(df) < 2:
        ts = df.index[-1] if df is not None and len(df) else now_ts
        price = float(df.iloc[-1]["close"]) if df is not None and len(df) else 0.0
        return ts, price, ts >= cutoff
    long_mode = direction == "long"
    for i in range(len(df) - 1, 0, -1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        trend_prev = int(prev["trend_flag"])
        trend_curr = int(curr["trend_flag"])
        if long_mode and trend_prev == -1 and trend_curr == 1:
            ts = df.index[i]
            return ts, float(curr["close"]), ts >= cutoff
        if (not long_mode) and trend_prev == 1 and trend_curr == -1:
            ts = df.index[i]
            return ts, float(curr["close"]), ts >= cutoff
    ts = df.index[-1]
    price = float(df.iloc[-1]["close"])
    return ts, price, ts >= cutoff


def evaluate_exit(position: Dict, df: pd.DataFrame, atr_mult: Optional[float], min_hold_bars: int) -> Optional[Dict]:
    if len(df) < 2:
        return None
    curr = df.iloc[-1]
    prev = df.iloc[-2]
    direction = str(position.get("direction", "long")).lower()
    long_mode = direction == "long"
    entry_price = float(position.get("entry_price", 0.0) or 0.0)
    entry_atr = float(position.get("entry_atr", 0.0) or 0.0)
    stake = float(position.get("stake", 0.0) or 0.0)
    entry_time = position.get("entry_time")
    htf = position.get("htf", None)  # Get HTF from position for correct bars calculation
    latest_ts = df.index[-1]
    bars_held = bars_in_position(entry_time, latest_ts, htf) if entry_time else 0

    exit_price = None
    reason = None
    if atr_mult is not None and entry_atr > 0:
        stop_price = entry_price - atr_mult * entry_atr if long_mode else entry_price + atr_mult * entry_atr
        hit_stop = (long_mode and float(curr["low"]) <= stop_price) or (
            (not long_mode) and float(curr["high"]) >= stop_price
        )
        if hit_stop:
            exit_price = stop_price
            reason = f"ATR stop x{atr_mult:.2f}"

    # Time-based exit: FORCED exit after optimal hold time (regardless of profit/loss)
    # Get symbol-specific optimal hold time from configuration
    symbol = position.get("symbol", "")
    direction_str = "long" if long_mode else "short"
    optimal_hold_bars = get_optimal_hold_bars(symbol, direction_str) if USE_TIME_BASED_EXIT else 0

    if USE_TIME_BASED_EXIT and exit_price is None:
        # FORCED exit after optimal_hold_bars - cuts losses early
        if bars_held >= optimal_hold_bars:
            current_price = float(curr["close"])
            exit_price = current_price
            reason = f"Time-based exit ({bars_held} bars, optimal={optimal_hold_bars})"

    # Trend flip exit - but only AFTER optimal hold time if time-based exits enabled
    trend_curr = int(curr["trend_flag"])
    trend_prev = int(prev["trend_flag"])
    flip_long = long_mode and trend_prev == 1 and trend_curr == -1
    flip_short = (not long_mode) and trend_prev == -1 and trend_curr == 1
    trend_flipped = flip_long or flip_short

    # Determine minimum bars before allowing trend flip exit
    min_bars_for_trend_flip = max(optimal_hold_bars, min_hold_bars) if USE_TIME_BASED_EXIT else max(0, min_hold_bars)

    if exit_price is None and trend_flipped and bars_held >= min_bars_for_trend_flip:
        exit_price = float(curr["close"])
        reason = "Trend flip"

    if exit_price is None:
        return None

    # Correct PnL calculation using amount (size_units)
    size_units = float(position.get("size_units", 0.0) or (stake / entry_price if entry_price else 0.0))
    # Fees based on total traded volume (entry + exit)
    fees = (entry_price + exit_price) * size_units * st.FEE_RATE
    # PnL = amount * price_diff - fees
    if long_mode:
        pnl = size_units * (exit_price - entry_price) - fees
    else:
        pnl = size_units * (entry_price - exit_price) - fees
    return {
        "exit_price": exit_price,
        "reason": reason,
        "fees": fees,
        "pnl": pnl,
    }


def process_snapshot(
    context: StrategyContext,
    df_slice: pd.DataFrame,
    config: ConfigEntry,
    state: Dict,
    emit_entry_log: bool = True,
    fixed_stake: Optional[float] = None,
    use_testnet: bool = DEFAULT_USE_TESTNET,
    order_executor: Optional[OrderExecutor] = None,
    trade_notifier: Optional[TradeNotifier] = None,
) -> List[TradeResult]:
    trades: List[TradeResult] = []

    # Debug: Track why BTC/ETH never reach evaluate_entry
    if context.symbol in ("BTC/EUR", "ETH/EUR"):
        debug_key = f"_snapshot_{context.symbol}_{context.direction}"
        count = getattr(process_snapshot, debug_key, 0) + 1
        setattr(process_snapshot, debug_key, count)
        if count == 1:
            print(f"[DEBUG] process_snapshot called for {context.symbol} {context.direction}")
        if count == 100:
            existing = find_position(state, context.key)
            print(f"[DEBUG] {context.symbol} {context.direction} after 100 calls: existing={existing is not None}, positions={len(state.get('positions', []))}")

    if not direction_allowed(config, context.direction):
        return trades
    if df_slice is None or len(df_slice) < 2:
        return trades

    latest_ts = df_slice.index[-1]
    latest_iso = latest_ts.isoformat()
    last_processed = state.setdefault("last_processed_bar", {}).get(context.key)
    if last_processed == latest_iso:
        return trades
    state["last_processed_bar"][context.key] = latest_iso

    existing = find_position(state, context.key)
    if existing:
        prior_total = state["total_capital"]
        exit_info = evaluate_exit(existing, df_slice, context.atr_mult, context.min_hold_bars)
        if exit_info:
            size_units = float(existing.get("size_units", 0.0))
            stake_val = float(existing.get("stake"))
            entry_price_val = float(existing.get("entry_price"))
            trade = TradeResult(
                symbol=context.symbol,
                direction=context.direction.capitalize(),
                indicator=context.indicator,
                htf=context.htf,
                param_desc=context.param_desc,
                entry_time=existing["entry_time"],
                entry_price=entry_price_val,
                exit_time=latest_iso,
                exit_price=float(exit_info["exit_price"]),
                stake=stake_val,
                fees=float(exit_info["fees"]),
                pnl=float(exit_info["pnl"]),
                equity_after=prior_total + float(exit_info["pnl"]),
                reason=exit_info["reason"],
                size_units=size_units,
            )
            # Submit live exit and adjust fill price/pnl if available
            if order_executor is not None:
                try:
                    order = order_executor.submit_exit(existing, trade)
                    fill_price = derive_fill_price(order) if order else 0.0
                    if fill_price > 0:
                        trade.exit_price = fill_price
                        # Correct PnL calculation using amount (size_units)
                        fees = (entry_price_val + fill_price) * size_units * st.FEE_RATE
                        trade.fees = fees
                        if context.direction == "long":
                            trade.pnl = size_units * (fill_price - entry_price_val) - fees
                        else:
                            trade.pnl = size_units * (entry_price_val - fill_price) - fees
                        trade.equity_after = prior_total + trade.pnl
                except Exception as exc:
                    print(f"[Order] Exit submission failed for {context.symbol}: {exc}")
            # Persist equity after potential adjustment
            state["total_capital"] = trade.equity_after
            if trade_notifier is not None:
                try:
                    trade_notifier.notify_exit(trade)
                except Exception as exc:
                    print(f"[Notify] Exit alert failed for {context.symbol}: {exc}")
            trades.append(trade)
            remove_position(state, context.key)
            save_state(state)  # Always save state after closing position
            # Save closed trade to testnet tracking file
            if use_testnet:
                closed_trade_data = {
                    "symbol": context.symbol,
                    "direction": context.direction,
                    "indicator": context.indicator,
                    "htf": context.htf,
                    "entry_time": existing_entry_time if 'existing_entry_time' in dir() else trade.entry_time,
                    "entry_price": trade.entry_price,
                    "exit_time": trade.exit_time,
                    "exit_price": trade.exit_price,
                    "stake": trade.stake,
                    "size_units": trade.size_units,
                    "fees": trade.fees,
                    "pnl": trade.pnl,
                    "pnl_pct": (trade.pnl / trade.stake * 100) if trade.stake else 0,
                    "reason": trade.reason,
                }
                save_testnet_closed_trade(closed_trade_data)
                remove_testnet_position(context.symbol, context.direction)
            existing = None

    if existing is None:
        open_positions = state.get("positions", [])
        # Count Long and Short positions separately
        long_count = sum(1 for p in open_positions if p.get("direction", "").lower() == "long")
        short_count = sum(1 for p in open_positions if p.get("direction", "").lower() == "short")

        # Check direction-specific limits
        if context.direction.lower() == "long" and long_count >= MAX_LONG_POSITIONS:
            if emit_entry_log:
                print(f"[Entry] Skip {context.symbol} {context.direction} – max {MAX_LONG_POSITIONS} Long-Positionen erreicht")
            _signal_log(f"{context.symbol} {context.direction} blocked: max long positions reached ({MAX_LONG_POSITIONS})")
            return trades
        if context.direction.lower() == "short" and short_count >= MAX_SHORT_POSITIONS:
            if emit_entry_log:
                print(f"[Entry] Skip {context.symbol} {context.direction} – max {MAX_SHORT_POSITIONS} Short-Positionen erreicht")
            _signal_log(f"{context.symbol} {context.direction} blocked: max short positions reached ({MAX_SHORT_POSITIONS})")
            return trades
        # Also check total limit
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            if emit_entry_log:
                print(f"[Entry] Skip {context.symbol} {context.direction} – max {MAX_OPEN_POSITIONS} Positionen erreicht")
            _signal_log(f"{context.symbol} {context.direction} blocked: max positions reached ({MAX_OPEN_POSITIONS})")
            return trades

    if existing is not None:
        return trades

    entry_allowed, entry_reason = evaluate_entry(df_slice, context.direction)

    # Debug: Count entry attempts for BTC/ETH
    if context.symbol in ("BTC/EUR", "ETH/EUR"):
        debug_key = f"_entry_count_{context.symbol}_{context.direction}"
        count = getattr(evaluate_entry, debug_key, 0) + 1
        setattr(evaluate_entry, debug_key, count)
        if count <= 3:  # Show first 3 attempts
            print(f"[DEBUG] {context.symbol} {context.direction}: entry_allowed={entry_allowed}, reason={entry_reason}")

    if not entry_allowed:
        if entry_reason:
            _signal_log(f"{context.symbol} {context.direction} entry skipped: {entry_reason}")
            # Debug: show first blocked entry for each symbol
            if context.symbol in ("BTC/EUR", "ETH/EUR") and not hasattr(evaluate_entry, f"_debug_{context.symbol}_{context.direction}"):
                setattr(evaluate_entry, f"_debug_{context.symbol}_{context.direction}", True)
                print(f"[DEBUG] {context.symbol} {context.direction} {context.indicator} {context.htf}: {entry_reason}")
        return trades

    entry_price = float(df_slice.iloc[-1]["close"])
    # Dynamic stake: total_capital / 10 (or fixed if explicitly provided)
    stake = determine_position_size(context.symbol, state, fixed_stake, context.direction)
    # Round to valid lot size for the symbol
    size_units = round_to_lot_size(stake / entry_price, context.symbol) if entry_price else 0.0
    entry = Position(
        key=context.key,
        symbol=context.symbol,
        direction=context.direction,
        indicator=context.indicator,
        htf=context.htf,
        param_a=context.param_a,
        param_b=context.param_b,
        atr_mult=context.atr_mult,
        min_hold_bars=context.min_hold_bars,
        entry_price=entry_price,
        entry_time=latest_iso,
        entry_atr=float(df_slice.iloc[-1].get("atr", 0.0)),
        stake=stake,
        size_units=size_units,
    )
    entry_record = dict(entry.__dict__)
    state.setdefault("positions", []).append(entry_record)
    record_symbol_trade(state, context.symbol)
    save_state(state)  # Always save state after opening position
    _signal_log(f"{context.symbol} {context.direction} entry triggered at {entry_price:.4f} (stake={stake:.2f})")
    if order_executor is not None:
        try:
            order = order_executor.submit_entry(entry_record)
            fill_price = derive_fill_price(order) if order else 0.0
            if fill_price > 0:
                entry_record["entry_price_live"] = fill_price
                entry_record["entry_price"] = fill_price
                entry_record["size_units"] = round_to_lot_size(entry_record["stake"] / fill_price, context.symbol) if fill_price else entry_record["size_units"]
                save_state(state)
        except Exception as exc:
            print(f"[Order] Entry submission failed for {context.symbol}: {exc}")
            # If live order failed, remove the paper position to keep portfolio clean
            try:
                remove_position(state, context.key)
                save_state(state)
                print(f"[Order] Removed paper position {context.key} due to failed live order.")
            except Exception as exc2:
                print(f"[Order] Failed to remove position after order error: {exc2}")
    if emit_entry_log:
        print(f"[Entry] {context.symbol} {context.direction} at {entry_record['entry_price']:.8f} stake {stake:.2f}")
    if trade_notifier is not None:
        try:
            trade_notifier.notify_entry(entry_record)
        except Exception as exc:
            print(f"[Notify] Entry alert failed for {context.symbol}: {exc}")
    return trades


def process_strategy_row(
    row: pd.Series,
    cfg_lookup: Dict[str, ConfigEntry],
    state: Dict,
    fixed_stake: Optional[float] = None,
    use_testnet: bool = DEFAULT_USE_TESTNET,
    order_executor: Optional[OrderExecutor] = None,
    trade_notifier: Optional[TradeNotifier] = None,
) -> List[TradeResult]:
    context = build_strategy_context(row)
    config = cfg_lookup.get(context.symbol)
    if not config or not direction_allowed(config, context.direction):
        return []
    df = build_dataframe_for_context(context)
    if df.empty:
        return []
    return process_snapshot(
        context,
        df,
        config,
        state,
        fixed_stake=fixed_stake,
        use_testnet=use_testnet,
        order_executor=order_executor,
        trade_notifier=trade_notifier,
    )


def trades_to_dataframe(trades: List[TradeResult]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame([trade.__dict__ for trade in trades])

    if "entry_time" in df.columns:
        df["Zeit"] = df["entry_time"].apply(to_berlin_iso)
    if "exit_time" in df.columns:
        df["ExitZeit"] = df["exit_time"].apply(to_berlin_iso)
    if "entry_price" in df.columns:
        df["Entry"] = df["entry_price"].astype(float)
    if "exit_price" in df.columns:
        df["ExitPreis"] = df["exit_price"].astype(float)
    return df


def _parse_param_desc(param_desc: str) -> tuple[float, float, Optional[float]]:
    param_a = 0.0
    param_b = 0.0
    atr_mult = None
    try:
        parts = [p.strip() for p in param_desc.split(",")]
        for part in parts:
            if part.lower().startswith("parama="):
                param_a = float(part.split("=", 1)[1])
            elif part.lower().startswith("paramb="):
                param_b = float(part.split("=", 1)[1])
            elif part.lower().startswith("atr="):
                val = part.split("=", 1)[1]
                if val and val.lower() != "none":
                    atr_mult = float(val)
    except Exception:
        pass
    return param_a, param_b, atr_mult


def _context_from_position(pos: Dict) -> StrategyContext:
    indicator_key = str(pos.get("indicator", st.INDICATOR_TYPE)).strip() or st.INDICATOR_TYPE
    htf_value = str(pos.get("htf", st.HIGHER_TIMEFRAME)).strip() or st.HIGHER_TIMEFRAME
    param_a_val = parse_float(pos.get("param_a"))
    param_b_val = parse_float(pos.get("param_b"))
    if param_a_val is None:
        param_a_val = float(st.DEFAULT_PARAM_A)
    if param_b_val is None:
        param_b_val = float(st.DEFAULT_PARAM_B)
    min_hold_bars = int(pos.get("min_hold_bars", 0) or 0)
    return StrategyContext(
        symbol=str(pos.get("symbol", "")).strip(),
        direction=str(pos.get("direction", "long")).strip().lower() or "long",
        indicator=indicator_key,
        htf=htf_value,
        param_a=float(param_a_val),
        param_b=float(param_b_val),
        atr_mult=parse_float(pos.get("atr_mult")),
        min_hold_bars=min_hold_bars,
    )


def enrich_open_positions(positions: List[Dict]) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame()
    enriched = []
    df_cache: Dict[Tuple[str, str, str, float, float], pd.DataFrame] = {}
    ticker_cache: Dict[str, float] = {}
    now_ts = pd.Timestamp.now(tz=st.BERLIN_TZ)
    for pos in positions:
        context = _context_from_position(pos)
        cache_key = (context.symbol, context.indicator, context.htf, context.param_a, context.param_b)
        if cache_key not in df_cache:
            try:
                df_cache[cache_key] = build_indicator_dataframe(
                    context.symbol,
                    context.indicator,
                    context.htf,
                    context.param_a,
                    context.param_b,
                )
            except Exception as exc:
                print(f"[OpenPositions] Datenabruf für {cache_key} fehlgeschlagen: {exc}")
                df_cache[cache_key] = pd.DataFrame()
        df = df_cache[cache_key]
        latest_price = None
        latest_ts = now_ts
        if not df.empty:
            latest_bar = df.iloc[-1]
            latest_price = float(latest_bar.get("close")) if not pd.isna(latest_bar.get("close")) else None
            latest_ts = df.index[-1]
        # If dataset is stale, try live ticker for a fresher price
        age_minutes = (now_ts - latest_ts).total_seconds() / 60.0 if latest_ts else None
        if (latest_price is None or (age_minutes is not None and age_minutes > 10)):
            try:
                if context.symbol not in ticker_cache:
                    tkr = st.get_exchange().fetch_ticker(context.symbol)
                    ticker_cache[context.symbol] = float(tkr.get("last") or tkr.get("close") or 0.0)
                live_price = ticker_cache.get(context.symbol)
                if live_price and live_price > 0:
                    latest_price = live_price
                    latest_ts = now_ts
            except Exception as exc:
                print(f"[OpenPositions] Live ticker failed for {context.symbol}: {exc}")
        entry_price = float(pos.get("entry_price", 0.0) or 0.0)
        stake = float(pos.get("stake", 0.0) or 0.0)
        long_mode = context.direction == "long"
        unrealized_pct = 0.0
        unrealized_pnl = 0.0
        if latest_price is not None and entry_price:
            diff = latest_price - entry_price if long_mode else entry_price - latest_price
            unrealized_pct = diff / entry_price if entry_price else 0.0
            unrealized_pnl = unrealized_pct * stake
        status = "Gewinn" if unrealized_pnl > 0 else "Verlust" if unrealized_pnl < 0 else "Flat"
        entry_time = pos.get("entry_time")
        bars_held = bars_in_position(entry_time, latest_ts) if entry_time else 0
        enriched.append({
            "symbol": context.symbol,
            "direction": context.direction,
            "indicator": context.indicator,
            "htf": context.htf,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "stake": stake,
            "param_a": context.param_a,
            "param_b": context.param_b,
            "atr_mult": context.atr_mult,
            "min_hold_bars": context.min_hold_bars,
            "last_price": latest_price,
            "bars_held": bars_held,
            "unrealized_pct": unrealized_pct * 100.0,
            "unrealized_pnl": unrealized_pnl,
            "status": status,
        })
    return pd.DataFrame(enriched)


def open_positions_to_dataframe(positions: List[Dict]) -> pd.DataFrame:
    return enrich_open_positions(positions)


def _write_dataframe_outputs(df: pd.DataFrame, csv_path: Optional[str], json_path: Optional[str], label: str) -> None:
    row_count = len(df)
    if df.empty:
        print(f"[Simulation] No {label} written (no rows) – keeping previous files if any.")
        return
    if csv_path:
        df.to_csv(csv_path, index=False)
    if json_path:
        df.to_json(json_path, orient="records", indent=2)
    targets = [p for p in (csv_path, json_path) if p]
    target_text = " & ".join(targets) if targets else "memory"
    print(f"[Simulation] Wrote {row_count} {label} rows to {target_text}")


def write_closed_trades_report(trades_df: pd.DataFrame, csv_path: str, json_path: str) -> None:
    _write_dataframe_outputs(trades_df, csv_path, json_path, label="closed trades")


def write_open_positions_report(positions: List[Dict], csv_path: str, json_path: Optional[str]) -> None:
    df = open_positions_to_dataframe(positions)
    _write_dataframe_outputs(df, csv_path, json_path, label="open positions")


def calc_symbol_stats(trades_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Calculate per-symbol statistics including trades, PnL, win rate, drawdown, etc."""
    if trades_df.empty or "symbol" not in trades_df.columns:
        return []

    symbol_stats = []
    # Filter out NaN symbols and sort
    valid_symbols = [s for s in trades_df["symbol"].unique() if pd.notna(s) and isinstance(s, str)]
    for symbol in sorted(valid_symbols):
        sym_df = trades_df[trades_df["symbol"] == symbol].copy()
        if sym_df.empty:
            continue

        # Sort by exit_time for proper drawdown calculation
        if "exit_time" in sym_df.columns:
            sym_df = sym_df.sort_values("exit_time")

        total_trades = len(sym_df)
        pnl_series = sym_df["pnl"] if "pnl" in sym_df.columns else pd.Series([0.0])
        total_pnl = float(pnl_series.sum())
        avg_pnl = float(pnl_series.mean())

        winners = len(sym_df[sym_df["pnl"] > 0]) if "pnl" in sym_df.columns else 0
        losers = len(sym_df[sym_df["pnl"] < 0]) if "pnl" in sym_df.columns else 0
        win_rate = (winners / total_trades * 100.0) if total_trades else 0.0

        best_trade = float(pnl_series.max()) if not pnl_series.empty else 0.0
        worst_trade = float(pnl_series.min()) if not pnl_series.empty else 0.0

        # Calculate max drawdown from cumulative PnL
        cumulative_pnl = pnl_series.cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = running_max - cumulative_pnl
        max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0

        # Long/Short breakdown
        long_trades = len(sym_df[sym_df["direction"].str.lower() == "long"]) if "direction" in sym_df.columns else 0
        short_trades = len(sym_df[sym_df["direction"].str.lower() == "short"]) if "direction" in sym_df.columns else 0

        long_pnl = float(sym_df[sym_df["direction"].str.lower() == "long"]["pnl"].sum()) if "direction" in sym_df.columns and "pnl" in sym_df.columns else 0.0
        short_pnl = float(sym_df[sym_df["direction"].str.lower() == "short"]["pnl"].sum()) if "direction" in sym_df.columns and "pnl" in sym_df.columns else 0.0

        # Profit factor: gross profit / gross loss (avoid division by zero)
        gross_profit = float(pnl_series[pnl_series > 0].sum()) if not pnl_series.empty else 0.0
        gross_loss = abs(float(pnl_series[pnl_series < 0].sum())) if not pnl_series.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        symbol_stats.append({
            "symbol": symbol,
            "trades": total_trades,
            "winners": winners,
            "losers": losers,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "best_trade": round(best_trade, 2),
            "worst_trade": round(worst_trade, 2),
            "max_drawdown": round(max_drawdown, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "long_trades": long_trades,
            "short_trades": short_trades,
            "long_pnl": round(long_pnl, 2),
            "short_pnl": round(short_pnl, 2),
        })

    # Sort by total PnL descending (best symbols first)
    symbol_stats.sort(key=lambda x: x["total_pnl"], reverse=True)

    return symbol_stats


def build_summary_payload(
    trades_df: pd.DataFrame,
    open_positions_df: pd.DataFrame,
    final_state: Dict,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> Dict[str, Any]:
    # Overall statistics
    total_trades = len(trades_df)
    winners = len(trades_df[trades_df["pnl"] > 0]) if not trades_df.empty else 0
    losers = len(trades_df[trades_df["pnl"] < 0]) if not trades_df.empty else 0
    pnl_sum = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    avg_pnl = float(trades_df["pnl"].mean()) if not trades_df.empty else 0.0
    win_rate = (winners / total_trades * 100.0) if total_trades else 0.0
    open_count = len(open_positions_df)
    open_equity = compute_net_open_equity(open_positions_df)

    # Calculate lifetime capital: START_EQUITY + all historical closed PnL + unrealized open equity
    # This gives accurate total capital across all historical trades
    lifetime_capital = st.START_EQUITY + pnl_sum + open_equity

    # Separate statistics for Long and Short
    def calc_direction_stats(df, direction_name):
        if "direction" not in df.columns or df.empty:
            return {}
        dir_df = df[df["direction"].str.lower() == direction_name.lower()]
        if dir_df.empty:
            return {
                f"{direction_name}_trades": 0,
                f"{direction_name}_pnl": 0.0,
                f"{direction_name}_avg_pnl": 0.0,
                f"{direction_name}_win_rate": 0.0,
                f"{direction_name}_winners": 0,
                f"{direction_name}_losers": 0,
            }
        count = len(dir_df)
        wins = len(dir_df[dir_df["pnl"] > 0])
        losses = len(dir_df[dir_df["pnl"] < 0])
        pnl = float(dir_df["pnl"].sum())
        avg = float(dir_df["pnl"].mean())
        wr = (wins / count * 100.0) if count else 0.0
        return {
            f"{direction_name}_trades": int(count),
            f"{direction_name}_pnl": round(pnl, 6),
            f"{direction_name}_avg_pnl": round(avg, 6),
            f"{direction_name}_win_rate": round(wr, 4),
            f"{direction_name}_winners": int(wins),
            f"{direction_name}_losers": int(losses),
        }

    long_stats = calc_direction_stats(trades_df, "long")
    short_stats = calc_direction_stats(trades_df, "short")

    # Open position stats by direction
    def calc_open_direction_stats(df, direction_name):
        if "direction" not in df.columns or df.empty:
            return {f"{direction_name}_open": 0, f"{direction_name}_open_equity": 0.0}
        dir_df = df[df["direction"].str.lower() == direction_name.lower()]
        count = len(dir_df)
        equity = compute_net_open_equity(dir_df)
        return {
            f"{direction_name}_open": int(count),
            f"{direction_name}_open_equity": round(equity, 6),
        }

    long_open_stats = calc_open_direction_stats(open_positions_df, "long")
    short_open_stats = calc_open_direction_stats(open_positions_df, "short")

    # Per-symbol statistics
    symbol_stats = calc_symbol_stats(trades_df)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "closed_trades": int(total_trades),
        "open_positions": int(open_count),
        "closed_pnl": round(pnl_sum, 6),
        "avg_trade_pnl": round(avg_pnl, 6),
        "win_rate_pct": round(win_rate, 4),
        "winners": int(winners),
        "losers": int(losers),
        "open_equity": round(open_equity, 6),
        "final_capital": round(lifetime_capital, 6),
        **long_stats,
        **short_stats,
        **long_open_stats,
        **short_open_stats,
        "symbol_stats": symbol_stats,
    }


def generate_summary_html(
    summary: Dict[str, Any],
    trades_df: pd.DataFrame,
    open_positions_df: pd.DataFrame,
    path: str,
) -> None:
    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Paper Trading Simulation Summary</title>",
        "<style>body{font-family:Arial,sans-serif;margin:20px;}table{border-collapse:collapse;margin-top:12px;width:auto;}th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;}th{text-align:center;background:#f0f0f0;font-weight:bold;}td:first-child{text-align:left;}h1{margin-bottom:10px;}h2{margin-top:30px;margin-bottom:10px;}.stats-container{display:flex;gap:20px;flex-wrap:wrap;}</style>",
        "</head><body>",
        f"<h1>Simulation Summary {summary['start']} → {summary['end']}</h1>",

        # Combined Statistics Table (Overall + Long + Short)
        "<h2>Statistics</h2>",
        "<table>",
        "<tr><th>Metric</th><th>Overall</th><th>Long</th><th>Short</th></tr>",
        f"<tr><td>Closed trades</td><td>{summary['closed_trades']}</td><td>{summary.get('long_trades', 0)}</td><td>{summary.get('short_trades', 0)}</td></tr>",
        f"<tr><td>Open positions</td><td>{summary['open_positions']}</td><td>{summary.get('long_open', 0)}</td><td>{summary.get('short_open', 0)}</td></tr>",
        f"<tr><td>PnL (USDT)</td><td>{summary['closed_pnl']:.2f}</td><td>{summary.get('long_pnl', 0):.2f}</td><td>{summary.get('short_pnl', 0):.2f}</td></tr>",
        f"<tr><td>Avg PnL (USDT)</td><td>{summary['avg_trade_pnl']:.2f}</td><td>{summary.get('long_avg_pnl', 0):.2f}</td><td>{summary.get('short_avg_pnl', 0):.2f}</td></tr>",
        f"<tr><td>Win rate (%)</td><td>{summary['win_rate_pct']:.2f}</td><td>{summary.get('long_win_rate', 0):.2f}</td><td>{summary.get('short_win_rate', 0):.2f}</td></tr>",
        f"<tr><td>Winners</td><td>{summary['winners']}</td><td>{summary.get('long_winners', 0)}</td><td>{summary.get('short_winners', 0)}</td></tr>",
        f"<tr><td>Losers</td><td>{summary['losers']}</td><td>{summary.get('long_losers', 0)}</td><td>{summary.get('short_losers', 0)}</td></tr>",
        f"<tr><td>Open equity (USDT)</td><td>{summary['open_equity']:.2f}</td><td>{summary.get('long_open_equity', 0):.2f}</td><td>{summary.get('short_open_equity', 0):.2f}</td></tr>",
        f"<tr style='font-weight:bold;'><td>Final capital (USDT)</td><td>{summary['final_capital']:.2f}</td><td>-</td><td>-</td></tr>",
        "</table>",
    ]

    # Per-Symbol Statistics Table
    symbol_stats = summary.get("symbol_stats", [])
    if symbol_stats:
        html_parts.append("<h2>Statistics by Symbol</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Symbol</th><th>Trades</th><th>Win</th><th>Loss</th><th>Win%</th><th>Total PnL</th><th>Avg PnL</th><th>Best</th><th>Worst</th><th>Max DD</th><th>PF</th><th>Long</th><th>Short</th><th>Long PnL</th><th>Short PnL</th></tr>")
        for ss in symbol_stats:
            pnl_color = "green" if ss["total_pnl"] >= 0 else "red"
            html_parts.append(
                f"<tr>"
                f"<td>{ss['symbol']}</td>"
                f"<td>{ss['trades']}</td>"
                f"<td>{ss['winners']}</td>"
                f"<td>{ss['losers']}</td>"
                f"<td>{ss['win_rate']:.1f}%</td>"
                f"<td style='color:{pnl_color}'>{ss['total_pnl']:.2f}</td>"
                f"<td>{ss['avg_pnl']:.2f}</td>"
                f"<td style='color:green'>{ss['best_trade']:.2f}</td>"
                f"<td style='color:red'>{ss['worst_trade']:.2f}</td>"
                f"<td style='color:orange'>{ss['max_drawdown']:.2f}</td>"
                f"<td>{ss['profit_factor']}</td>"
                f"<td>{ss['long_trades']}</td>"
                f"<td>{ss['short_trades']}</td>"
                f"<td>{ss['long_pnl']:.2f}</td>"
                f"<td>{ss['short_pnl']:.2f}</td>"
                f"</tr>"
            )
        html_parts.append("</table>")

    if not trades_df.empty:
        full_cols = [c for c in [
            "symbol","direction","indicator","htf","entry_time","entry_price","exit_time","exit_price","stake","pnl","reason"
        ] if c in trades_df.columns]

        # Prepare display DataFrame - ensure numeric columns are actually numeric
        trades_display = trades_df[full_cols].copy()

        # Remove rows where essential columns are NaN (filter out empty rows)
        if "symbol" in trades_display.columns:
            trades_display = trades_display[trades_display["symbol"].notna()]

        for col in ["entry_price", "exit_price", "stake", "pnl"]:
            if col in trades_display.columns:
                trades_display[col] = pd.to_numeric(trades_display[col], errors="coerce")

        # Use formatters parameter to format specific columns during HTML generation
        # Note: Must use default parameter to avoid closure bug with lambda in loop
        def make_formatter(precision):
            return lambda x: f"{x:.{precision}f}" if pd.notna(x) else ""

        formatters = {}
        for col in ["entry_price", "exit_price", "stake", "pnl"]:
            if col in trades_display.columns:
                formatters[col] = make_formatter(8)

        # Separate Long and Short trades
        if "direction" in trades_display.columns:
            long_trades = trades_display[trades_display["direction"].str.lower() == "long"].copy()
            short_trades = trades_display[trades_display["direction"].str.lower() == "short"].copy()

            # Display Long Trades
            if not long_trades.empty:
                long_pnl = long_trades["pnl"].sum() if "pnl" in long_trades.columns else 0
                html_parts.append(f"<h2>Long Trades ({len(long_trades)} trades, PnL: {long_pnl:.2f} USDT)</h2>")
                html_parts.append(long_trades.to_html(index=False, escape=False, formatters=formatters))

            # Display Short Trades
            if not short_trades.empty:
                short_pnl = short_trades["pnl"].sum() if "pnl" in short_trades.columns else 0
                html_parts.append(f"<h2>Short Trades ({len(short_trades)} trades, PnL: {short_pnl:.2f} USDT)</h2>")
                html_parts.append(short_trades.to_html(index=False, escape=False, formatters=formatters))
        else:
            # Fallback if no direction column
            html_parts.append("<h2>Complete Closed Trades (with Entry and Exit)</h2>")
            html_parts.append(trades_display.to_html(index=False, escape=False, formatters=formatters))

    if not open_positions_df.empty:
        # Prepare display DataFrame - ensure numeric columns are actually numeric
        open_display = open_positions_df.copy()

        # Remove rows where essential columns are NaN (filter out empty rows)
        if "symbol" in open_display.columns:
            open_display = open_display[open_display["symbol"].notna()]

        # Convert all numeric columns to proper types
        for col in ["entry_price", "stake", "last_price", "unrealized_pnl", "unrealized_pct"]:
            if col in open_display.columns:
                open_display[col] = pd.to_numeric(open_display[col], errors="coerce")

        for col in ["param_a", "param_b", "atr_mult"]:
            if col in open_display.columns:
                open_display[col] = pd.to_numeric(open_display[col], errors="coerce")

        for col in ["min_hold_bars", "bars_held"]:
            if col in open_display.columns:
                open_display[col] = pd.to_numeric(open_display[col], errors="coerce")

        # Use formatters parameter to format specific columns during HTML generation
        # Note: Must use factory function to avoid closure bug with lambda in loop
        def make_float_formatter(precision):
            return lambda x: f"{x:.{precision}f}" if pd.notna(x) else ""

        def make_int_formatter():
            return lambda x: f"{int(x)}" if pd.notna(x) else "0"

        formatters = {}

        # 8 decimal places for prices and amounts
        for col in ["entry_price", "stake", "last_price", "unrealized_pnl", "unrealized_pct"]:
            if col in open_display.columns:
                formatters[col] = make_float_formatter(8)

        # 2 decimal places for float params
        for col in ["param_a", "param_b", "atr_mult"]:
            if col in open_display.columns:
                formatters[col] = make_float_formatter(2)

        # Integers for counts
        for col in ["min_hold_bars", "bars_held"]:
            if col in open_display.columns:
                formatters[col] = make_int_formatter()

        # Separate Long and Short open positions
        if "direction" in open_display.columns:
            long_open = open_display[open_display["direction"].str.lower() == "long"].copy()
            short_open = open_display[open_display["direction"].str.lower() == "short"].copy()

            # Display Long Open Positions
            if not long_open.empty:
                long_equity = compute_net_open_equity(long_open)
                html_parts.append(f"<h2>Long Open Positions ({len(long_open)} positions, Equity: {long_equity:.2f} USDT)</h2>")
                html_parts.append(long_open.to_html(index=False, escape=False, formatters=formatters))

            # Display Short Open Positions
            if not short_open.empty:
                short_equity = compute_net_open_equity(short_open)
                html_parts.append(f"<h2>Short Open Positions ({len(short_open)} positions, Equity: {short_equity:.2f} USDT)</h2>")
                html_parts.append(short_open.to_html(index=False, escape=False, formatters=formatters))
        else:
            # Fallback if no direction column
            html_parts.append("<h2>Open positions</h2>")
            html_parts.append(open_display.to_html(index=False, escape=False, formatters=formatters))

    html_parts.append("</body></html>")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(html_parts))
    print(f"[Simulation] Summary HTML saved to {path}")


def write_summary_json(summary: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"[Simulation] Summary JSON saved to {path}")


def generate_trade_charts(trades_df: pd.DataFrame, open_positions_df: pd.DataFrame = None, output_dir: str = os.path.join("report_html", "charts")) -> None:
    if (trades_df is None or trades_df.empty) and (open_positions_df is None or open_positions_df.empty):
        print("[Chart] No trades or open positions available for chart generation.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    # Combine symbols from both closed trades and open positions
    symbols_to_chart = set()
    if trades_df is not None and not trades_df.empty and "symbol" in trades_df.columns:
        symbols_to_chart.update(trades_df["symbol"].dropna().unique())
    if open_positions_df is not None and not open_positions_df.empty and "symbol" in open_positions_df.columns:
        symbols_to_chart.update(open_positions_df["symbol"].dropna().unique())
    
    for symbol in symbols_to_chart:
        try:
            # Get trades and open positions for this symbol
            symbol_trades = trades_df[trades_df["symbol"] == symbol] if trades_df is not None and not trades_df.empty and "symbol" in trades_df.columns else pd.DataFrame()
            symbol_open = open_positions_df[open_positions_df["symbol"] == symbol] if open_positions_df is not None and not open_positions_df.empty and "symbol" in open_positions_df.columns else pd.DataFrame()
            
            # Get indicator/htf from trades or open positions
            row = None
            if not symbol_trades.empty:
                row = symbol_trades.iloc[0]
            elif not symbol_open.empty:
                row = symbol_open.iloc[0]
            else:
                continue
                
            indicator = str(row.get("indicator") or st.INDICATOR_TYPE or "supertrend")
            htf_value = str(row.get("htf") or st.HIGHER_TIMEFRAME)
            param_desc = str(row.get("param_desc") or "")
            param_a, param_b, _ = _parse_param_desc(param_desc)

            df = pd.DataFrame()
            try:
                df = build_indicator_dataframe(symbol, indicator, htf_value, param_a, param_b, use_all_data=True)
            except Exception as exc:
                print(f"[Chart] Indicator data failed for {symbol}: {exc}")

            if df is None or df.empty:
                try:
                    # Fetch 4 weeks of data for charts (approx 672 5-min bars per day * 28 days)
                    df_raw = st.fetch_data(symbol, st.TIMEFRAME, min(8000, 672 * 28))
                except Exception as exc:
                    print(f"[Chart] Raw data fetch failed for {symbol}: {exc}")
                    df_raw = pd.DataFrame()
                if df_raw is not None and not df_raw.empty:
                    df = df_raw.copy()
                    print(f"[Chart] Using raw price data for {symbol} (indicator unavailable).")

            if df is None or df.empty:
                print(f"[Chart] Skipping {symbol}: no data.")
                continue

            # Normalize index to timezone-aware timestamps for consistent plotting
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df.loc[~df.index.isna()].copy()
            if df.empty:
                print(f"[Chart] Skipping {symbol}: all timestamps invalid.")
                continue
            try:
                if df.index.tzinfo is None:
                    df.index = df.index.tz_localize(st.BERLIN_TZ)
                else:
                    df.index = df.index.tz_convert(st.BERLIN_TZ)
            except Exception:
                pass

            # Ensure OHLC columns exist; if missing, synthesize from close so candlesticks still render
            df_candles = df.copy()
            if "open" not in df_candles.columns:
                df_candles["open"] = df_candles["close"]
            if "high" not in df_candles.columns:
                df_candles["high"] = df_candles["close"]
            if "low" not in df_candles.columns:
                df_candles["low"] = df_candles["close"]

            fig = go.Figure()
            fig.add_candlestick(
                x=df_candles.index,
                open=df_candles["open"],
                high=df_candles["high"],
                low=df_candles["low"],
                close=df_candles["close"],
                name="Candles",
            )

            if "indicator_line" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["indicator_line"], mode="lines", name=f"{indicator} line", line=dict(color="blue")))
            if "htf_indicator" in df.columns:
                fig.add_trace(go.Scatter(x=df.index, y=df["htf_indicator"], mode="lines", name=f"HTF {indicator}", line=dict(color="orange", dash="dash")))

            # Calculate ATR if available for offset
            atr_series = None
            if "atr" in df.columns:
                atr_series = df["atr"]
            elif all(c in df.columns for c in ["high", "low", "close"]):
                try:
                    from ta.volatility import AverageTrueRange
                    atr_series = AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
                except Exception:
                    pass

            entries_x = []
            entries_y = []
            exits_x = []
            exits_y = []
            open_entries_x = []
            open_entries_y = []
            
            # Add closed trades
            if not symbol_trades.empty:
                for _, trade in symbol_trades.iterrows():
                    try:
                        if "entry_time" in trade:
                            entry_ts = pd.to_datetime(trade["entry_time"])
                            entry_price = float(trade.get("entry_price", np.nan))
                            entries_x.append(entry_ts)
                            # Offset by -2.5 * ATR
                            offset = 0
                            if atr_series is not None and entry_ts in df.index:
                                atr_val = float(atr_series.loc[entry_ts]) if not pd.isna(atr_series.loc[entry_ts]) else 0
                                offset = 2.5 * atr_val
                            entries_y.append(entry_price - offset if entry_price else entry_price)
                        if "exit_time" in trade and pd.notna(trade["exit_time"]):
                            exit_ts = pd.to_datetime(trade["exit_time"])
                            exit_price = float(trade.get("exit_price", np.nan))
                            exits_x.append(exit_ts)
                            # Offset by +2.5 * ATR
                            offset = 0
                            if atr_series is not None and exit_ts in df.index:
                                atr_val = float(atr_series.loc[exit_ts]) if not pd.isna(atr_series.loc[exit_ts]) else 0
                                offset = 2.5 * atr_val
                            exits_y.append(exit_price + offset if exit_price else exit_price)
                    except Exception:
                        continue
            
            # Add open positions
            if not symbol_open.empty:
                for _, pos in symbol_open.iterrows():
                    try:
                        if "entry_time" in pos:
                            entry_ts = pd.to_datetime(pos["entry_time"])
                            entry_price = float(pos.get("entry_price", np.nan))
                            open_entries_x.append(entry_ts)
                            # Offset by -2.5 * ATR
                            offset = 0
                            if atr_series is not None and entry_ts in df.index:
                                atr_val = float(atr_series.loc[entry_ts]) if not pd.isna(atr_series.loc[entry_ts]) else 0
                                offset = 2.5 * atr_val
                            open_entries_y.append(entry_price - offset if entry_price else entry_price)
                    except Exception:
                        continue

            if entries_x:
                fig.add_trace(go.Scatter(x=entries_x, y=entries_y, mode="markers", name="Buy (closed)", marker=dict(symbol="triangle-up", color="green", size=12)))
            if exits_x:
                fig.add_trace(go.Scatter(x=exits_x, y=exits_y, mode="markers", name="Sell", marker=dict(symbol="triangle-down", color="red", size=12)))
            if open_entries_x:
                fig.add_trace(go.Scatter(x=open_entries_x, y=open_entries_y, mode="markers", name="Buy (open)", marker=dict(symbol="triangle-up", color="blue", size=14)))

            fig.update_layout(
                title=f"{symbol} – {indicator} / HTF {htf_value}",
                xaxis_title="Time",
                yaxis_title="Price",
                xaxis_rangeslider=dict(visible=True, thickness=0.08),
                template="plotly_white",
            )
            out_path = os.path.join(output_dir, f"{symbol.replace('/', '_')}.html")
            fig.write_html(out_path, include_plotlyjs="cdn")
            print(f"[Chart] Saved {out_path}")
        except Exception as exc:
            print(f"[Chart] Failed for {symbol}: {exc}")


def generate_equity_curve(
    trades_df: pd.DataFrame,
    start_capital: float = None,
    output_dir: str = os.path.join("report_html", "charts"),
) -> None:
    """Generate equity curve and drawdown charts from closed trades."""
    if trades_df is None or trades_df.empty:
        print("[Equity] No trades available for equity curve.")
        return

    if start_capital is None:
        start_capital = START_TOTAL_CAPITAL

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data - sort by exit time
    df = trades_df.copy()
    if "exit_time" not in df.columns or "pnl" not in df.columns:
        print("[Equity] Missing required columns (exit_time, pnl).")
        return

    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df = df.dropna(subset=["exit_time", "pnl"])
    df = df.sort_values("exit_time").reset_index(drop=True)

    if df.empty:
        print("[Equity] No valid trades for equity curve.")
        return

    # Calculate cumulative PnL and equity
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["equity"] = start_capital + df["cumulative_pnl"]

    # Calculate drawdown
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["peak"] - df["equity"]
    df["drawdown_pct"] = (df["drawdown"] / df["peak"]) * 100

    max_drawdown = df["drawdown"].max()
    max_drawdown_pct = df["drawdown_pct"].max()
    max_dd_idx = df["drawdown"].idxmax()
    max_dd_date = df.loc[max_dd_idx, "exit_time"] if max_dd_idx in df.index else None

    # Create figure with secondary y-axis
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Equity Curve", "Drawdown (USDT)", "Cumulative PnL by Symbol")
    )

    # Equity curve
    fig.add_trace(
        go.Scatter(
            x=df["exit_time"],
            y=df["equity"],
            mode="lines",
            name="Equity",
            line=dict(color="blue", width=2),
            fill="tozeroy",
            fillcolor="rgba(0, 100, 255, 0.1)"
        ),
        row=1, col=1
    )

    # Peak equity line
    fig.add_trace(
        go.Scatter(
            x=df["exit_time"],
            y=df["peak"],
            mode="lines",
            name="Peak Equity",
            line=dict(color="green", width=1, dash="dot")
        ),
        row=1, col=1
    )

    # Starting capital reference line
    fig.add_hline(
        y=start_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Start: {start_capital:,.0f}",
        row=1, col=1
    )

    # Drawdown area
    fig.add_trace(
        go.Scatter(
            x=df["exit_time"],
            y=-df["drawdown"],  # Negative to show below zero
            mode="lines",
            name="Drawdown",
            line=dict(color="red", width=1),
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.3)"
        ),
        row=2, col=1
    )

    # Max drawdown annotation
    if max_dd_date is not None:
        fig.add_annotation(
            x=max_dd_date,
            y=-max_drawdown,
            text=f"Max DD: {max_drawdown:,.0f} ({max_drawdown_pct:.1f}%)",
            showarrow=True,
            arrowhead=2,
            arrowcolor="red",
            font=dict(color="red"),
            row=2, col=1
        )

    # Per-symbol cumulative PnL
    if "symbol" in df.columns:
        symbols = df["symbol"].unique()
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        for i, symbol in enumerate(sorted(symbols)):
            sym_df = df[df["symbol"] == symbol].copy()
            sym_df["sym_cumulative_pnl"] = sym_df["pnl"].cumsum()
            color = colors[i % len(colors)]

            fig.add_trace(
                go.Scatter(
                    x=sym_df["exit_time"],
                    y=sym_df["sym_cumulative_pnl"],
                    mode="lines",
                    name=symbol,
                    line=dict(color=color, width=1.5)
                ),
                row=3, col=1
            )

    # Final statistics
    final_equity = df["equity"].iloc[-1]
    total_return = ((final_equity - start_capital) / start_capital) * 100
    total_trades = len(df)

    fig.update_layout(
        title=dict(
            text=f"Equity Curve | Start: {start_capital:,.0f} → Final: {final_equity:,.0f} USDT | Return: {total_return:.1f}% | Trades: {total_trades} | Max DD: {max_drawdown:,.0f} ({max_drawdown_pct:.1f}%)",
            font=dict(size=14)
        ),
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        hovermode="x unified"
    )

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Equity (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (USDT)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative PnL", row=3, col=1)

    # Save chart
    out_path = os.path.join(output_dir, "equity_curve.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[Equity] Saved equity curve to {out_path}")

    # Also create a simple monthly/weekly returns heatmap if enough data
    _generate_returns_summary(df, start_capital, output_dir)


def _generate_returns_summary(df: pd.DataFrame, start_capital: float, output_dir: str) -> None:
    """Generate monthly returns summary."""
    if df.empty or "exit_time" not in df.columns:
        return

    try:
        df = df.copy()
        # Ensure exit_time is datetime
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
        df = df.dropna(subset=["exit_time"])
        if df.empty:
            print("[Equity] No valid dates for monthly returns.")
            return
        df["month"] = df["exit_time"].dt.to_period("M")

        monthly = df.groupby("month").agg({
            "pnl": ["sum", "count", lambda x: (x > 0).sum(), lambda x: (x < 0).sum()]
        }).reset_index()
        monthly.columns = ["month", "pnl", "trades", "winners", "losers"]
        monthly["win_rate"] = (monthly["winners"] / monthly["trades"] * 100).round(1)
        monthly["month_str"] = monthly["month"].astype(str)

        # Create bar chart for monthly returns
        colors = ["green" if x >= 0 else "red" for x in monthly["pnl"]]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["month_str"],
            y=monthly["pnl"],
            marker_color=colors,
            text=[f"{p:,.0f}<br>{t} trades<br>{wr:.0f}% WR" for p, t, wr in zip(monthly["pnl"], monthly["trades"], monthly["win_rate"])],
            textposition="outside",
            name="Monthly PnL"
        ))

        total_pnl = monthly["pnl"].sum()
        avg_monthly = monthly["pnl"].mean()
        best_month = monthly.loc[monthly["pnl"].idxmax()]
        worst_month = monthly.loc[monthly["pnl"].idxmin()]

        fig.update_layout(
            title=f"Monthly Returns | Total: {total_pnl:,.0f} USDT | Avg: {avg_monthly:,.0f}/month | Best: {best_month['month_str']} ({best_month['pnl']:,.0f}) | Worst: {worst_month['month_str']} ({worst_month['pnl']:,.0f})",
            xaxis_title="Month",
            yaxis_title="PnL (USDT)",
            template="plotly_white",
            height=500
        )

        out_path = os.path.join(output_dir, "monthly_returns.html")
        fig.write_html(out_path, include_plotlyjs="cdn")
        print(f"[Equity] Saved monthly returns to {out_path}")
    except Exception as exc:
        print(f"[Equity] Monthly returns generation failed: {exc}")


def compute_net_open_equity(open_positions_df: pd.DataFrame) -> float:
    if open_positions_df.empty:
        return 0.0
    pnl_sum = float(open_positions_df.get("unrealized_pnl", pd.Series(dtype=float)).sum())
    if "stake" in open_positions_df.columns:
        fee_reserve = float((open_positions_df["stake"].fillna(0.0) * st.FEE_RATE).sum())
    else:
        fee_reserve = 0.0
    return pnl_sum - fee_reserve


def print_daily_closed_trades(log_path: str = TRADE_LOG_FILE) -> None:
    if not os.path.exists(log_path):
        print("[Live] No trade log available yet.")
        return
    try:
        df = pd.read_csv(log_path)
    except Exception as exc:
        print(f"[Live] Unable to read trade log {log_path}: {exc}")
        return
    if df.empty or "Timestamp" not in df.columns:
        print("[Live] No closed trades recorded yet.")
        return
    try:
        ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce", format="%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        ts = pd.to_datetime(df["Timestamp"], utc=True, errors="coerce", format="%Y-%m-%dT%H:%M:%S")
    except Exception as exc:
        print(f"[Live] Failed to parse trade timestamps: {exc}")
        return
    df = df.assign(Timestamp=ts).dropna(subset=["Timestamp"])
    if df.empty:
        print("[Live] No closed trades recorded yet.")
        return
    berlin_now = datetime.now(st.BERLIN_TZ)
    start_of_day = berlin_now.replace(hour=0, minute=0, second=0, microsecond=0)
    df["Timestamp"] = df["Timestamp"].dt.tz_convert(st.BERLIN_TZ)
    todays = df[df["Timestamp"] >= start_of_day]
    if todays.empty:
        print(f"[Live] No closed trades yet today ({start_of_day.date()}).")
        return
    print(f"[Live] Closed trades today ({len(todays)}):")
    preview_cols = [
        "Timestamp",
        "Symbol",
        "Direction",
        "Indicator",
        "ExitPrice",
        "PnL",
        "Reason",
    ]
    for _, row in todays.sort_values("Timestamp").iterrows():
        values = []
        for col in preview_cols:
            if col == "Timestamp":
                ts_val = row[col]
                if pd.isna(ts_val):
                    values.append("Timestamp=?")
                else:
                    values.append(ts_val.strftime("%H:%M"))
            elif col in row:
                if col in {"ExitPrice", "PnL"} and not pd.isna(row[col]):
                    values.append(f"{col}={float(row[col]):.2f}")
                elif isinstance(row[col], str):
                    values.append(f"{col}={row[col]}")
                else:
                    values.append(f"{col}={row[col]}")
        print("  - " + ", ".join(values))


def _derive_summary_window(trades_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Derive time window for summary metrics - uses full history from earliest to now."""
    end_ts = pd.Timestamp.now(tz=st.BERLIN_TZ)
    if trades_df.empty or "exit_time" not in trades_df.columns:
        # Default to 7 days back if no trades
        return end_ts - pd.Timedelta(days=7), end_ts
    exit_times = pd.to_datetime(trades_df["exit_time"], errors="coerce")
    exit_times = exit_times.dropna()
    if exit_times.empty:
        return end_ts - pd.Timedelta(days=7), end_ts
    # Use earliest trade as start (forever history)
    start_ts = exit_times.min()
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(st.BERLIN_TZ)
    else:
        start_ts = start_ts.tz_convert(st.BERLIN_TZ)
    return start_ts, end_ts


def write_live_reports(final_state: Dict, closed_trades: List[TradeResult]) -> None:
    # Append new trades to cumulative log FIRST
    for trade in closed_trades:
        append_trade_log(trade)
    
    # Also write to snapshot files
    current_trades_df = trades_to_dataframe(closed_trades)
    write_closed_trades_report(current_trades_df, SIMULATION_LOG_FILE, SIMULATION_LOG_JSON)
    
    # NOW load ALL historical trades from cumulative log for display
    all_trades_df = _load_trade_log_dataframe(TRADE_LOG_FILE)
    if all_trades_df.empty:
        all_trades_df = current_trades_df
    
    # Use ALL historical trades for summary and charts
    open_positions = final_state.get("positions", [])
    write_open_positions_report(open_positions, SIMULATION_OPEN_POSITIONS_FILE, SIMULATION_OPEN_POSITIONS_JSON)
    open_df = open_positions_to_dataframe(open_positions)
    start_ts, end_ts = _derive_summary_window(all_trades_df)
    summary = build_summary_payload(all_trades_df, open_df, final_state, start_ts, end_ts)
    generate_summary_html(summary, all_trades_df, open_df, SIMULATION_SUMMARY_HTML)
    write_summary_json(summary, SIMULATION_SUMMARY_JSON)
    
    # Generate charts with ALL historical trades + open positions
    if not all_trades_df.empty or not open_df.empty:
        chart_df = all_trades_df.copy() if not all_trades_df.empty else pd.DataFrame()
        if not chart_df.empty:
            chart_df = chart_df.loc[:, ~chart_df.columns.duplicated()].copy()
            if {"symbol", "entry_time", "exit_time", "exit_price"}.issubset(chart_df.columns):
                chart_df = chart_df.drop_duplicates(
                    subset=["symbol", "entry_time", "exit_time", "exit_price"], keep="last"
                )
        generate_trade_charts(chart_df, open_df, output_dir=os.path.join(REPORT_DIR, "charts"))
    
    if current_trades_df.empty:
        print(f"[Live] Snapshot updated with no new trades this cycle. Total history: {len(all_trades_df)} trades.")
    else:
        print(f"[Live] Snapshot includes {len(current_trades_df)} new trade(s). Total history: {len(all_trades_df)} trades.")

    # Update testnet dashboard if in testnet mode
    if _TESTNET_ACTIVE:
        try:
            from TestnetDashboard import generate_dashboard
            generate_dashboard()
            print("[Dashboard] Testnet dashboard updated")
        except Exception as e:
            print(f"[Dashboard] Failed to update: {e}")

    return float(summary.get("final_capital", final_state.get("total_capital", 0.0)))


def _now_str() -> str:
    return (
        datetime.now(st.BERLIN_TZ)
        .strftime("%Y-%m-%d %H:%M:%S %Z")
    )


def run_signal_cycle(
    symbols: Sequence[str],
    indicators: Sequence[str],
    stake: Optional[float],
    use_testnet: bool,
    order_executor: Optional[OrderExecutor],
    trade_notifier: Optional[TradeNotifier],
) -> None:
    print(f"[{_now_str()}] Running scheduled trading cycle (symbols={symbols or 'ALL'})")

    # First, run a mini-simulation to get current simulated positions
    # This updates paper_trading_actual_trades.json with current state
    now = pd.Timestamp.now(tz=st.BERLIN_TZ)
    sim_start = now - pd.Timedelta(days=30)  # 30-day lookback for simulation
    print(f"[{_now_str()}] Running simulation from {sim_start.strftime('%Y-%m-%d')} to update positions...")

    # Temporarily enable synthetic bars for the simulation
    # This ensures we get current price data even if the current hour isn't complete
    original_skip_synthetic = st.SKIP_SYNTHETIC_BARS
    st.SKIP_SYNTHETIC_BARS = False
    # Clear data cache to ensure fresh synthetic bars are fetched
    st.DATA_CACHE.clear()

    try:
        trades, sim_state = run_simulation(
            sim_start,
            now,
            use_saved_state=False,
            emit_entry_log=False,
            allowed_symbols=list(symbols) if symbols else None,
            allowed_indicators=list(indicators) if indicators else None,
            fixed_stake=stake,
            use_testnet=use_testnet,
            refresh_params=False,
            reset_state=False,
            clear_outputs=False,
        )
        # Write simulated open positions to file
        sim_positions = sim_state.get("positions", [])
        write_open_positions_report(sim_positions, SIMULATION_OPEN_POSITIONS_FILE, SIMULATION_OPEN_POSITIONS_JSON)
        print(f"[{_now_str()}] Simulation complete: {len(trades)} trades, {len(sim_positions)} open positions")
    except Exception as e:
        print(f"[{_now_str()}] Simulation failed: {e}")
    finally:
        # Restore original setting
        st.SKIP_SYNTHETIC_BARS = original_skip_synthetic

    # Now run the regular signal check (paper trading)
    main(
        allowed_symbols=list(symbols) if symbols else None,
        allowed_indicators=list(indicators) if indicators else None,
        fixed_stake=stake,
        use_testnet=use_testnet,
        order_executor=order_executor,
        trade_notifier=trade_notifier,
        configure_exchange=False,
    )
    print_daily_closed_trades()


def detect_atr_spikes(symbols: Sequence[str], atr_mult: float, use_testnet: bool = False) -> List[str]:
    triggered: List[str] = []
    target_symbols = list(symbols) if symbols else st.get_symbols(use_testnet)
    for symbol in target_symbols:
        try:
            df = st.fetch_data(symbol, st.TIMEFRAME, max(st.ATR_WINDOW + 5, 50))
        except Exception as exc:
            print(f"[Spike] Failed to fetch {symbol}: {exc}")
            continue
        if df is None or len(df) <= st.ATR_WINDOW:
            continue
        atr_series = AverageTrueRange(df["high"], df["low"], df["close"], window=st.ATR_WINDOW).average_true_range()
        candle = df.iloc[-1]
        candle_range = float(candle["high"] - candle["low"])
        atr_value = float(atr_series.iloc[-2] if len(atr_series) >= 2 else atr_series.iloc[-1])
        if atr_value <= 0:
            continue
        if candle_range >= atr_mult * atr_value:
            print(
                f"[Spike] {symbol} candle range {candle_range:.4f} >= {atr_mult:.2f} * ATR {atr_value:.4f}"
            )
            triggered.append(symbol)
    return triggered


def monitor_loop(
    symbols: Sequence[str],
    indicators: Sequence[str],
    stake: Optional[float],
    use_testnet: bool,
    signal_interval_min: float,
    spike_interval_min: float,
    atr_mult: float,
    poll_seconds: float,
    order_executor: Optional[OrderExecutor],
    trade_notifier: Optional[TradeNotifier],
) -> None:
    st.configure_exchange(use_testnet=use_testnet)
    next_signal = 0.0
    next_spike = 0.0
    try:
        while True:
            now = time.time()
            if now >= next_signal:
                run_signal_cycle(symbols, indicators, stake, use_testnet, order_executor, trade_notifier)
                next_signal = now + signal_interval_min * 60.0
            if now >= next_spike:
                spike_symbols = detect_atr_spikes(symbols, atr_mult, use_testnet)
                if spike_symbols:
                    print(f"[{_now_str()}] Spike trigger for {', '.join(spike_symbols)}")
                    run_signal_cycle(spike_symbols, indicators, stake, use_testnet, order_executor, trade_notifier)
                next_spike = now + spike_interval_min * 60.0
            time.sleep(max(1.0, poll_seconds))
    except KeyboardInterrupt:
        print("[Monitor] Stopped by user")


def force_entry_position(
    symbol: Optional[str],
    direction: Optional[str],
    allowed_symbols: Optional[List[str]],
    allowed_indicators: Optional[List[str]],
    fixed_stake: Optional[float],
    lookback_hours: float,
    use_testnet: bool,
    order_executor: Optional[OrderExecutor],
    trade_notifier: Optional[TradeNotifier],
) -> bool:
    """Force-open a position for quick manual testing."""
    direction_value = (direction or "long").strip().lower()
    if direction_value not in {"long", "short"}:
        print(f"[Force] Direction '{direction_value}' unbekannt – verwende long.")
        direction_value = "long"
    requested_symbol = symbol.strip().upper() if symbol else None
    symbol_pool = normalize_symbol_list(allowed_symbols) or list(DEFAULT_SYMBOL_ALLOWLIST)
    if requested_symbol:
        if requested_symbol not in symbol_pool:
            symbol_pool.append(requested_symbol)
    elif symbol_pool:
        requested_symbol = symbol_pool[0]
    if not requested_symbol:
        print("[Force] Kein Symbol gefunden, bitte --force-entry SYMBOL[:direction] angeben.")
        return False
    config_df = ensure_config(symbol_pool or [requested_symbol])
    cfg_lookup = load_config_lookup(config_df)
    config = cfg_lookup.get(requested_symbol)
    if not config:
        print(f"[Force] Keine Konfiguration für {requested_symbol} gefunden.")
        return False
    if not direction_allowed(config, direction_value):
        print(f"[Force] {requested_symbol} {direction_value} ist in der Konfiguration deaktiviert.")
        return False
    best_df = load_best_rows_from_detailed()
    if best_df.empty:
        best_df = load_best_rows(active_indicators=allowed_indicators)
    if best_df.empty:
        print("[Force] Keine Strategiedaten gefunden (weder detailed HTML noch overall CSV).")
        return False
    # For testnet: map USDT symbol to USDC equivalent for parameter lookup
    param_symbol = st.map_symbol_for_params(requested_symbol) if use_testnet else requested_symbol
    print(f"[Force] Suche Parameter für {param_symbol} (trade as {requested_symbol})")
    filtered = filter_best_rows_by_symbol(best_df, [param_symbol])
    filtered = filter_best_rows_by_direction(filtered, [direction_value])
    filtered = select_best_indicator_per_symbol(filtered)
    if filtered.empty:
        print(f"[Force] Keine Strategiezeile für {param_symbol} {direction_value} vorhanden.")
        return False
    row = filtered.iloc[0]
    context = build_strategy_context(row)
    # For testnet: override context symbol to use USDT pair for trading
    if use_testnet and context.symbol != requested_symbol:
        context.symbol = requested_symbol
        print(f"[Force] Symbol für Trade auf {requested_symbol} gesetzt (key={context.key})")
    df = build_dataframe_for_context(context)
    if df.empty:
        print(f"[Force] Keine Marktdaten für {requested_symbol} verfügbar.")
        return False
    signal_ts, entry_price, within_window = find_last_signal_bar(df, direction_value, lookback_hours=lookback_hours)
    if not within_window:
        if use_testnet:
            # Testnet has limited data - use current price for forced entry
            print(f"[Force] Kein Signal gefunden - verwende aktuellen Preis (Testnet-Modus)")
            signal_ts = df.index[-1]
            entry_price = float(df.iloc[-1]["close"])
        else:
            print(
                f"[Force] Kein Signal in den letzten {lookback_hours:g}h für {requested_symbol} {direction_value} gefunden."
            )
            return False
    if entry_price is None or pd.isna(entry_price):
        print(f"[Force] Schlusskurs für {requested_symbol} nicht verfügbar.")
        return False
    entry_price = float(entry_price)
    if entry_price <= 0:
        print(f"[Force] Ungültiger Kurs {entry_price} für {requested_symbol}.")
        return False
    state = load_state()
    positions = state.setdefault("positions", [])
    if len(positions) >= MAX_OPEN_POSITIONS:
        print(f"[Force] Maximale Anzahl ({MAX_OPEN_POSITIONS}) offener Trades erreicht.")
        return False
    if find_position(state, context.key):
        print(f"[Force] Position {context.key} ist bereits offen.")
        return False
    # Use direction-specific stake sizes (SHORT_STAKE for shorts, dynamic for longs)
    stake_value = determine_position_size(context.symbol, state, fixed_stake, context.direction)
    if stake_value <= 0:
        print("[Force] Stake-Betrag ist 0 – prüfe Kapital oder --stake.")
        return False
    entry_iso = signal_ts.isoformat()
    try:
        latest_bar = df.loc[signal_ts]
    except Exception:
        latest_bar = df.iloc[-1]
    entry_atr_val = float(latest_bar.get("atr", 0.0) or 0.0)

    entry = Position(
        key=context.key,
        symbol=context.symbol,
        direction=context.direction,
        indicator=context.indicator,
        htf=context.htf,
        param_a=context.param_a,
        param_b=context.param_b,
        atr_mult=context.atr_mult,
        min_hold_bars=context.min_hold_bars,
        entry_price=entry_price,
        entry_time=entry_iso,
        entry_atr=entry_atr_val,
        stake=stake_value,
        size_units=round_to_lot_size(stake_value / entry_price, context.symbol),
    )
    entry_record = dict(entry.__dict__)
    positions.append(entry_record)
    state.setdefault("last_processed_bar", {})[context.key] = entry_iso
    record_symbol_trade(state, context.symbol)
    save_state(state)
    side_text = "testnet" if use_testnet else "live"
    print(
        f"[Force] Öffne {context.symbol} {context.direction} @ {entry_price:.8f} "
        f"(Stake {stake_value:.2f}, {side_text})."
    )
    if order_executor is not None:
        try:
            order = order_executor.submit_entry(entry_record)
            fill_price = derive_fill_price(order) if order else 0.0
            if fill_price > 0:
                entry_record["entry_price_live"] = fill_price
                entry_record["entry_price"] = fill_price
                entry_record["size_units"] = round_to_lot_size(entry_record["stake"] / fill_price, context.symbol) if fill_price else entry_record["size_units"]
                save_state(state)
            # Save to Crypto9 testnet tracking file
            if use_testnet:
                save_testnet_position(entry_record, use_testnet=True)
        except Exception as exc:
            print(f"[Force] Orderausführung fehlgeschlagen: {exc}")
            # Remove the paper position if live order failed
            try:
                remove_position(state, context.key)
                save_state(state)
                print(f"[Force] Entferne Papier-Position {context.key} nach fehlgeschlagener Order.")
            except Exception as exc2:
                print(f"[Force] Entfernen der Position fehlgeschlagen: {exc2}")
    if trade_notifier is not None:
        try:
            trade_notifier.notify_entry(entry_record)
        except Exception as exc:
            print(f"[Force] SMS-Benachrichtigung fehlgeschlagen: {exc}")
    return True

def run_simulation(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    use_saved_state: bool = False,
    emit_entry_log: bool = False,
    allowed_symbols: Optional[List[str]] = None,
    allowed_indicators: Optional[List[str]] = None,
    fixed_stake: Optional[float] = None,
    use_testnet: bool = DEFAULT_USE_TESTNET,
    refresh_params: bool = False,
    reset_state: bool = False,
    clear_outputs: bool = False,
) -> Tuple[List[TradeResult], Dict]:
    if end_ts <= start_ts:
        raise ValueError("End timestamp must be greater than start timestamp")
    if clear_outputs:
        clear_output_artifacts(include_state=reset_state)
    elif reset_state:
        reset_state_file()
    if refresh_params:
        st.run_overall_best_params()
    st.configure_exchange(use_testnet=use_testnet)
    # Use testnet symbols (USDT pairs) when in testnet mode
    default_symbols = st.get_symbols(use_testnet)
    raw_symbols = allowed_symbols if allowed_symbols else default_symbols
    config_df = ensure_config(raw_symbols or default_symbols)
    cfg_lookup = load_config_lookup(config_df)
    best_df = load_best_rows(active_indicators=allowed_indicators)
    print(f"[DEBUG] After load_best_rows: {len(best_df)} rows, symbols: {best_df['Symbol'].unique().tolist() if 'Symbol' in best_df.columns else 'N/A'}")
    # For testnet: filter by USDC equivalents (what's in CSV), then remap to USDT
    if use_testnet:
        usdc_equivalents = get_usdc_equivalents_for_testnet(raw_symbols)
        symbol_filter = normalize_symbol_list(usdc_equivalents)
        print(f"[DEBUG] Testnet mode: using USDC equivalents for params: {symbol_filter}")
    else:
        symbol_filter = normalize_symbol_list(raw_symbols)
    print(f"[DEBUG] symbol_filter: {symbol_filter}")
    best_df = filter_best_rows_by_symbol(best_df, symbol_filter)
    print(f"[DEBUG] After filter_by_symbol: {len(best_df)} rows, symbols: {best_df['Symbol'].unique().tolist() if 'Symbol' in best_df.columns else 'N/A'}")
    # Remap EUR -> USDT for testnet trading
    if use_testnet:
        best_df = remap_best_rows_for_testnet(best_df)
        print(f"[DEBUG] Remapped to testnet symbols: {best_df['Symbol'].unique().tolist() if 'Symbol' in best_df.columns else 'N/A'}")
    best_df = filter_best_rows_by_direction(best_df, DEFAULT_ALLOWED_DIRECTIONS)
    print(f"[DEBUG] After filter_by_direction: {len(best_df)} rows")
    best_df = select_best_indicator_per_symbol(best_df)
    print(f"[DEBUG] After select_best: {len(best_df)} rows, symbols: {best_df['Symbol'].unique().tolist() if 'Symbol' in best_df.columns else 'N/A'}")
    print(f"[Simulation] Loaded {len(best_df)} strategy rows")
    if best_df.empty:
        print("[Simulation] best_params_overall.csv enthält keine Daten.")
        return [], clone_state(use_saved_state)
    sim_state = clone_state(use_saved_state)
    prune_state_for_indicators(sim_state, allowed_indicators)

    # Pre-download historical data if needed
    print(f"[Simulation] Checking historical data availability...")
    unique_symbols = best_df['Symbol'].unique() if 'Symbol' in best_df.columns else []

    # Only download Binance-supported timeframes
    # Non-standard timeframes (3h, 9h, 15h, 18h, 21h, etc.) will be synthesized from 1h at runtime
    BINANCE_SUPPORTED_TIMEFRAMES = {"1h", "2h", "4h", "6h", "8h", "12h", "1d"}
    unique_timeframes = ["1h"]  # Always need 1h for synthesis
    if 'HTF' in best_df.columns:
        for tf in best_df['HTF'].unique():
            if tf in BINANCE_SUPPORTED_TIMEFRAMES and tf not in unique_timeframes:
                unique_timeframes.append(tf)

    # Add buffer for indicator warmup (30 days before start for safety)
    download_start = start_ts - pd.Timedelta(days=30)

    # Always update to latest available data (now)
    now = pd.Timestamp.now(tz=st.BERLIN_TZ)
    download_end = max(end_ts, now)

    for symbol in unique_symbols:
        for timeframe in unique_timeframes:
            try:
                # Check persistent cache directly (not limit-constrained)
                cached_df = st.load_ohlcv_from_cache(symbol, timeframe)

                needs_download = False
                if cached_df.empty:
                    print(f"[Simulation] No cached data for {symbol} {timeframe} - downloading...")
                    needs_download = True
                else:
                    earliest = cached_df.index.min()
                    latest = cached_df.index.max()

                    # Check if we need historical data
                    if earliest > download_start:
                        print(f"[Simulation] {symbol} {timeframe}: Cache starts {earliest.strftime('%Y-%m-%d')}, need {download_start.strftime('%Y-%m-%d')} - downloading...")
                        needs_download = True

                    # Check if cache is outdated (older than 2 hours)
                    elif latest < now - pd.Timedelta(hours=2):
                        print(f"[Simulation] {symbol} {timeframe}: Cache outdated (last: {latest.strftime('%Y-%m-%d %H:%M')}), updating to now...")
                        needs_download = True
                    else:
                        print(f"[Simulation] {symbol} {timeframe}: {len(cached_df)} bars, {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d %H:%M')} ✓")

                if needs_download:
                    st.download_historical_ohlcv(symbol, timeframe, download_start, download_end)

                    # Clear memory cache to force reload from persistent cache
                    for key in list(st.DATA_CACHE.keys()):
                        if key[0] == symbol and key[1] == timeframe:
                            del st.DATA_CACHE[key]

            except Exception as exc:
                print(f"[Simulation] Warning: Could not check/download {symbol} {timeframe}: {exc}")

    print(f"[Simulation] Historical data check complete")

    buffer = pd.Timedelta(minutes=BASE_BAR_MINUTES * 5)
    all_trades: List[TradeResult] = []
    # Pass stake through: None = dynamic sizing, value = fixed stake
    stake_value = fixed_stake
    print(f"[Simulation] Processing {len(best_df)} strategy rows...")
    for _, row in best_df.iterrows():
        context = build_strategy_context(row)
        config = cfg_lookup.get(context.symbol)
        if not config or not direction_allowed(config, context.direction):
            print(f"[Simulation] SKIP {context.symbol} {context.direction}: config missing or direction not allowed")
            continue
        # Use ALL cached historical data for simulations (not just LOOKBACK limit)
        df_full = build_dataframe_for_context(context, use_all_data=True)
        if df_full.empty:
            print(f"[Simulation] SKIP {context.symbol} {context.direction} {context.indicator} {context.htf}: NO DATA")
            continue
        mask = (df_full.index >= (start_ts - buffer)) & (df_full.index <= end_ts)
        df_range = df_full.loc[mask]
        if len(df_range) < 2:
            print(f"[Simulation] SKIP {context.symbol} {context.direction} {context.indicator} {context.htf}: only {len(df_range)} bars in range (need 2+)")
            continue
        # Debug: Count trend flips for BTC/ETH
        if context.symbol in ("BTC/EUR", "ETH/EUR"):
            trend_flags = df_range["trend_flag"].values
            flips = sum(1 for i in range(1, len(trend_flags)) if trend_flags[i] != trend_flags[i-1])
            long_entries = sum(1 for i in range(1, len(trend_flags)) if trend_flags[i-1] == -1 and trend_flags[i] == 1)
            short_entries = sum(1 for i in range(1, len(trend_flags)) if trend_flags[i-1] == 1 and trend_flags[i] == -1)
            print(f"[DEBUG] {context.symbol} {context.direction} {context.indicator} {context.htf}: {len(df_range)} bars, {flips} flips, {long_entries} long signals, {short_entries} short signals")
        else:
            print(f"[Simulation] Processing {context.symbol} {context.direction} {context.indicator} {context.htf}: {len(df_range)} bars")
        for idx in range(1, len(df_range)):
            curr_ts = df_range.index[idx]
            if curr_ts < start_ts:
                sim_state.setdefault("last_processed_bar", {})[context.key] = curr_ts.isoformat()
                continue
            snapshot = df_range.iloc[: idx + 1]
            trades = process_snapshot(
                context,
                snapshot,
                config,
                sim_state,
                emit_entry_log,
                fixed_stake=stake_value,
                use_testnet=use_testnet,
                order_executor=None,
            )
            if trades:
                all_trades.extend(trades)
    # Show first and last trade dates to identify data gaps
    if all_trades:
        first_trade = min(all_trades, key=lambda t: t.entry_time)
        last_trade = max(all_trades, key=lambda t: t.entry_time)
        print(f"[Simulation] First trade: {first_trade.entry_time} ({first_trade.symbol} {first_trade.direction})")
        print(f"[Simulation] Last trade: {last_trade.entry_time} ({last_trade.symbol} {last_trade.direction})")
    return all_trades, sim_state


def main(
    allowed_symbols: Optional[List[str]] = None,
    allowed_indicators: Optional[List[str]] = None,
    fixed_stake: Optional[float] = None,
    use_testnet: bool = DEFAULT_USE_TESTNET,
    order_executor: Optional[OrderExecutor] = None,
    trade_notifier: Optional[TradeNotifier] = None,
    refresh_params: bool = False,
    reset_state: bool = False,
    clear_outputs: bool = False,
    configure_exchange: bool = True,
) -> None:
    # Auto-correct any historical trades with wrong PnL formula
    correct_historical_trades_pnl()

    if refresh_params:
        st.run_overall_best_params()
    if clear_outputs:
        clear_output_artifacts(include_state=reset_state)
    if reset_state and not clear_outputs:
        reset_state_file()
    if configure_exchange:
        st.configure_exchange(use_testnet=use_testnet)
    # Clear data cache to get fresh synthetic bars for current period
    st.clear_data_cache()
    # Use testnet symbols (USDT pairs) when in testnet mode
    default_symbols = st.get_symbols(use_testnet)
    raw_symbols = allowed_symbols if allowed_symbols else default_symbols
    print(f"[DEBUG] use_testnet={use_testnet}, raw_symbols={raw_symbols}")
    config_df = ensure_config(raw_symbols or default_symbols)
    cfg_lookup = load_config_lookup(config_df)
    best_df = load_best_rows(active_indicators=allowed_indicators)
    print(f"[DEBUG] Loaded {len(best_df)} rows from best_params_overall.csv")
    # For testnet: filter by USDC equivalents (what's in CSV), then remap to USDT
    if use_testnet:
        usdc_equivalents = get_usdc_equivalents_for_testnet(raw_symbols)
        symbol_filter = normalize_symbol_list(usdc_equivalents)
        print(f"[DEBUG] Testnet: USDC equivalents = {usdc_equivalents}")
    else:
        symbol_filter = normalize_symbol_list(raw_symbols)
    print(f"[DEBUG] symbol_filter = {symbol_filter}")
    best_df = filter_best_rows_by_symbol(best_df, symbol_filter)
    print(f"[DEBUG] After filter: {len(best_df)} rows")
    # Remap EUR -> USDT for testnet trading
    if use_testnet:
        best_df = remap_best_rows_for_testnet(best_df)
        print(f"[DEBUG] After remap: symbols = {best_df['Symbol'].unique().tolist() if not best_df.empty else []}")
    best_df = select_best_indicator_per_symbol(best_df)
    print(f"[DEBUG] After select_best: {len(best_df)} rows")
    if best_df.empty:
        print("[Skip] best_params_overall.csv enthält keine Daten.")
        return
    state = load_state()
    # Sync positions from simulation (catch up with backtest)
    sync_simulation_positions(state, use_testnet=use_testnet)
    prune_state_for_indicators(state, allowed_indicators)
    # Pass stake through: None = dynamic sizing, value = fixed stake
    stake_value = fixed_stake
    closed_trades: List[TradeResult] = []
    for _, row in best_df.iterrows():
        trades = process_strategy_row(
            row,
            cfg_lookup,
            state,
            fixed_stake=stake_value,
            use_testnet=use_testnet,
            order_executor=order_executor,
            trade_notifier=trade_notifier,
        )
        if not trades:
            continue
        closed_trades.extend(trades)
    save_state(state)
    marked_capital = write_live_reports(state, closed_trades)
    print(f"[Done] Total capital now {marked_capital:.2f} USDT (incl. open PnL - exit fees)")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper trading runner for overall-best strategies")
    parser.add_argument("--simulate", action="store_true", help="Run a historical simulation instead of a single live tick")
    parser.add_argument("--start", type=str, default=None, help="Simulation start timestamp (ISO, default: 24h before end)")
    parser.add_argument("--end", type=str, default=None, help="Simulation end timestamp (ISO, default: now)")
    parser.add_argument("--use-saved-state", action="store_true", help="Seed simulations with the saved JSON state instead of a fresh account")
    parser.add_argument("--sim-log", type=str, default=SIMULATION_LOG_FILE, help="CSV path for simulated trades")
    parser.add_argument("--sim-json", type=str, default=SIMULATION_LOG_JSON, help="JSON path for simulated trades")
    parser.add_argument("--open-log", type=str, default=SIMULATION_OPEN_POSITIONS_FILE, help="CSV path for simulated open positions")
    parser.add_argument("--open-json", type=str, default=SIMULATION_OPEN_POSITIONS_JSON, help="JSON path for simulated open positions")
    parser.add_argument("--summary-html", type=str, default=SIMULATION_SUMMARY_HTML, help="HTML summary output path")
    parser.add_argument("--summary-json", type=str, default=SIMULATION_SUMMARY_JSON, help="JSON summary output path")
    parser.add_argument("--replay-trades-csv", type=str, default=None, help="Use a precomputed trades CSV (e.g. last48_from_detailed_all.csv) instead of simulating")
    parser.add_argument("--verbose-sim-entries", action="store_true", help="Print entry messages while simulating")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated symbol allowlist (default: st.SYMBOLS)")
    parser.add_argument(
        "--indicators",
        type=str,
        default=None,
        help="Comma-separated indicator allowlist (e.g. jma only)",
    )
    parser.add_argument(
        "--stake",
        type=float,
        default=None,
        help="Fixed stake size per trade (default: dynamic sizing with total_capital/7)",
    )
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet credentials and endpoints")
    parser.add_argument("--debug-signals", action="store_true", help="Verbose logging for entry filter decisions")
    parser.add_argument("--refresh-params", action="store_true", help="Re-run overall-best parameter export before trading")
    parser.add_argument("--reset-state", action="store_true", help="Delete the saved state before running")
    parser.add_argument(
        "--clear-outputs",
        action="store_true",
        help="Delete generated CSV/JSON/HTML outputs before writing new ones",
    )
    parser.add_argument(
        "--close-at-end",
        action="store_true",
        help="Force close all open positions at simulation end (marks them as 'simulation_end' exit)",
    )
    parser.add_argument("--monitor", action="store_true", help="Run continuous monitor loop (scheduled cycle + optional ATR spikes)")
    parser.add_argument("--signal-interval", type=float, default=DEFAULT_SIGNAL_INTERVAL_MIN, help="Minutes between scheduled monitor cycles (default 15)")
    parser.add_argument("--spike-interval", type=float, default=DEFAULT_SPIKE_INTERVAL_MIN, help="Minutes between ATR spike scans (default 5)")
    parser.add_argument("--atr-mult", type=float, default=DEFAULT_ATR_SPIKE_MULT, help="ATR multiple to trigger spike reruns (default 2.5)")
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS, help="Sleep duration within the monitor loop (default 30)")
    parser.add_argument("--place-orders", action="store_true", help="Submit real Binance market orders via CCXT (default: paper only)")
    parser.add_argument("--notify-sms", action="store_true", help="Send SMS alerts for entries/exits via Twilio (requires env vars)")
    parser.add_argument("--sms-to", type=str, default=None, help="Comma-separated phone numbers overriding TWILIO_TO_NUMBERS")
    parser.add_argument(
        "--force-entry",
        type=str,
        default=None,
        help="Force a single entry SYMBOL[:direction] and exit immediately afterwards",
    )
    parser.add_argument(
        "--force-lookback-hours",
        type=float,
        default=24.0,
        help="Lookback window in hours to accept the last trend-flip signal for --force-entry (default 24)",
    )
    parser.add_argument(
        "--clear-positions",
        action="store_true",
        help="Clear all open positions and last-processed markers in the saved state",
    )
    parser.add_argument(
        "--clear-all",
        action="store_true",
        help="Clear both saved state (positions, markers) and all generated CSV/JSON/HTML outputs",
    )
    parser.add_argument(
        "--max-open-positions",
        type=int,
        default=None,
        help="Override the maximum number of concurrent open positions (default 5)",
    )
    parser.add_argument(
        "--use-futures-signals",
        action="store_true",
        help="Use futures data for signal generation (entries/exits from futures, trades on spot prices)",
    )
    return parser.parse_args(argv)


def run_cli(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    allowed_symbols = parse_symbol_argument(args.symbols)
    allowed_indicators = parse_indicator_argument(args.indicators)
    force_symbol, force_direction = parse_force_entry_argument(args.force_entry)

    use_testnet = bool(args.testnet or DEFAULT_USE_TESTNET)
    set_testnet_active(use_testnet)

    # Handle stake sizing: dynamic by default, fixed if --stake provided
    if args.stake is not None:
        stake_value = args.stake
        print(f"[Config] Using fixed stake: {stake_value} USDT")
    else:
        stake_value = None  # None triggers dynamic sizing
        print(f"[Config] Using dynamic position sizing: stake = total_capital / {STAKE_DIVISOR}")

    # Update output paths based on testnet mode
    global SIMULATION_SUMMARY_HTML, SIMULATION_SUMMARY_JSON, REPORT_DIR
    REPORT_DIR = get_report_dir(use_testnet)
    SIMULATION_SUMMARY_HTML = os.path.join(REPORT_DIR, "trading_summary.html")
    SIMULATION_SUMMARY_JSON = os.path.join(REPORT_DIR, "trading_summary.json")
    if use_testnet:
        print(f"[Config] Using testnet report directory: {REPORT_DIR}")

    set_signal_debug(args.debug_signals)
    set_use_futures_signals(args.use_futures_signals)
    api_key, api_secret = get_api_credentials(use_testnet=use_testnet)
    if use_testnet and (not api_key or not api_secret):
        print("[Warn] Testnet API credentials are missing in .env; requests may fail.")
    if (not use_testnet) and (not api_key or not api_secret):
        print("[Info] Live Binance API credentials missing; proceeding without authenticated requests.")
    sms_notifier = build_sms_notifier(args.notify_sms, args.sms_to)
    if args.max_open_positions is not None:
        set_max_open_positions(args.max_open_positions)
    if args.clear_all:
        clear_positions_in_state()
        clear_output_artifacts(include_state=True)
        print("[Cleanup] Cleared state and all generated outputs.")
        # Exit early if no other actions requested
        if not (args.force_entry or args.monitor or args.simulate or args.refresh_params or args.place_orders or args.symbols or args.indicators):
            return
    if args.clear_positions:
        clear_positions_in_state()
        # If only clearing positions was requested (no other actions), exit early
        if not (args.force_entry or args.monitor or args.simulate or args.refresh_params or args.clear_outputs or args.reset_state or args.place_orders or args.symbols or args.indicators):
            return
    if args.force_entry:
        if args.refresh_params:
            st.run_overall_best_params()
        if args.clear_outputs:
            clear_output_artifacts(include_state=args.reset_state)
        elif args.reset_state:
            reset_state_file()
        st.configure_exchange(use_testnet=use_testnet)
        order_executor = None
        if args.place_orders:
            order_executor = create_order_executor(use_testnet)
        success = force_entry_position(
            force_symbol,
            force_direction,
            allowed_symbols,
            allowed_indicators,
            stake_value,
            args.force_lookback_hours,
            use_testnet,
            order_executor=order_executor,
            trade_notifier=sms_notifier,
        )
        if not success:
            print("[Force] Manuelle Order konnte nicht erstellt werden.")
        return
    if args.monitor:
        if args.refresh_params:
            st.run_overall_best_params()
        if args.clear_outputs:
            clear_output_artifacts(include_state=args.reset_state)
        elif args.reset_state:
            reset_state_file()
        st.configure_exchange(use_testnet=use_testnet)
        order_executor = None
        if args.place_orders:
            order_executor = create_order_executor(use_testnet)
        monitor_loop(
            allowed_symbols,
            allowed_indicators,
            stake_value,
            use_testnet,
            signal_interval_min=args.signal_interval,
            spike_interval_min=args.spike_interval,
            atr_mult=args.atr_mult,
            poll_seconds=args.poll_seconds,
            order_executor=order_executor,
            trade_notifier=sms_notifier,
        )
        return

    order_executor = None
    configure_exchange_flag = True
    if args.place_orders:
        st.configure_exchange(use_testnet=use_testnet)
        order_executor = create_order_executor(use_testnet)
        configure_exchange_flag = False

    if args.simulate:
        trades: List[TradeResult] = []
        open_positions: List[Dict[str, Any]] = []
        if args.replay_trades_csv:
            if args.clear_outputs:
                clear_output_artifacts(include_state=args.reset_state)
            elif args.reset_state:
                reset_state_file()
            trades_df = load_replay_trades_csv(args.replay_trades_csv)
            if trades_df.empty:
                print(f"[Replay] No trades loaded from {args.replay_trades_csv}")
                return
            start_default = _infer_timestamp_from_df(
                trades_df,
                "entry_time",
                _infer_timestamp_from_df(
                    trades_df,
                    "Zeit",
                    pd.Timestamp.now(tz=st.BERLIN_TZ) - pd.Timedelta(days=2),
                ),
            )
            end_default = _infer_timestamp_from_df(
                trades_df,
                "exit_time",
                _infer_timestamp_from_df(trades_df, "ExitZeit", pd.Timestamp.now(tz=st.BERLIN_TZ)),
            )
            end_ts = resolve_timestamp(args.end, end_default)
            start_ts = resolve_timestamp(args.start, start_default)
            pnl_series = _to_numeric_series(trades_df.get("pnl", pd.Series(dtype=float))).fillna(0.0)
            # For replay, anchor capital to START_TOTAL_CAPITAL and add closed PnL only (ignore embedded equity_after from source files)
            final_capital = float(START_TOTAL_CAPITAL + pnl_series.sum())
            final_state = {
                "total_capital": final_capital,
                "positions": [],
                "last_processed_bar": {},
                "symbol_trade_counts": {
                    sym: int(count) for sym, count in trades_df.get("symbol", pd.Series(dtype=str)).value_counts().items()
                }
                if "symbol" in trades_df.columns
                else {},
            }
        else:
            end_ts = resolve_timestamp(args.end, pd.Timestamp.now(tz=st.BERLIN_TZ))
            default_start = end_ts - pd.Timedelta(days=1)
            start_ts = resolve_timestamp(args.start, default_start)
            print(f"[Simulation] Period: {start_ts.strftime('%Y-%m-%d %H:%M')} to {end_ts.strftime('%Y-%m-%d %H:%M')}")
            trades, final_state = run_simulation(
                start_ts,
                end_ts,
                use_saved_state=args.use_saved_state,
                emit_entry_log=args.verbose_sim_entries,
                allowed_symbols=allowed_symbols,
                allowed_indicators=allowed_indicators,
                fixed_stake=stake_value,
                use_testnet=use_testnet,
                refresh_params=args.refresh_params,
                reset_state=args.reset_state,
                clear_outputs=args.clear_outputs,
            )
            trades_df = trades_to_dataframe(trades)
            open_positions = final_state.get("positions", [])

            # Force close all open positions at simulation end if requested
            if args.close_at_end and open_positions:
                print(f"[Simulation] Force closing {len(open_positions)} open positions at simulation end...")
                for pos in open_positions[:]:  # Copy list to allow modification
                    # Get last price from cache or use entry price as fallback
                    symbol = pos.get("symbol", "")
                    try:
                        df_1h = st.load_ohlcv_from_cache(symbol, "1h")
                        if df_1h is not None and not df_1h.empty:
                            last_price = float(df_1h['close'].iloc[-1])
                        else:
                            last_price = float(pos.get("entry_price", 0))
                    except Exception:
                        last_price = float(pos.get("entry_price", 0))

                    entry_price = float(pos.get("entry_price", 0))
                    stake = float(pos.get("stake", 0))
                    direction = pos.get("direction", "long").lower()

                    # Calculate PnL
                    if direction == "long":
                        pnl_pct = (last_price - entry_price) / entry_price if entry_price else 0
                    else:
                        pnl_pct = (entry_price - last_price) / entry_price if entry_price else 0
                    pnl_usd = stake * pnl_pct

                    # Create trade result
                    trade = TradeResult(
                        symbol=symbol,
                        direction=pos.get("direction", "Long"),
                        indicator=pos.get("indicator", ""),
                        htf=pos.get("htf", ""),
                        entry_time=pos.get("entry_time", ""),
                        exit_time=end_ts.isoformat(),
                        entry_price=entry_price,
                        exit_price=last_price,
                        stake=stake,
                        pnl_pct=pnl_pct * 100,
                        pnl_usd=pnl_usd,
                        exit_reason="simulation_end",
                        equity_after=final_state["total_capital"] + pnl_usd,
                    )
                    trades.append(trade)
                    final_state["total_capital"] += pnl_usd
                    print(f"  Closed {symbol} {direction}: PnL {pnl_usd:+.2f} USDT ({pnl_pct*100:+.1f}%)")

                # Clear positions after force close
                final_state["positions"] = []
                open_positions = []
                trades_df = trades_to_dataframe(trades)
                print(f"[Simulation] All positions closed. New total: {len(trades)} trades")

            print(f"[Simulation] Generated {len(trades)} trades during simulation")
        log_path = args.sim_log or SIMULATION_LOG_FILE
        log_json_path = args.sim_json or SIMULATION_LOG_JSON
        write_closed_trades_report(trades_df, log_path, log_json_path)
        print(f"[Simulation] Final capital: {final_state['total_capital']:.2f} USDT")
        open_path = args.open_log or SIMULATION_OPEN_POSITIONS_FILE
        open_json_path = args.open_json or SIMULATION_OPEN_POSITIONS_JSON
        write_open_positions_report(open_positions, open_path, open_json_path)

        # Auto-sync simulation positions to paper trading state
        paper_state = load_state()
        synced = sync_simulation_positions(paper_state, use_testnet=use_testnet)
        if synced > 0:
            save_state(paper_state)
            print(f"[Sync] Added {synced} simulation positions to paper trading state")
        else:
            print(f"[Sync] Paper trading state already up to date ({len(paper_state.get('positions', []))} positions)")

        open_df = open_positions_to_dataframe(open_positions)
        summary_data = build_summary_payload(trades_df, open_df, final_state, start_ts, end_ts)
        summary_html_path = args.summary_html or SIMULATION_SUMMARY_HTML
        generate_summary_html(summary_data, trades_df, open_df, summary_html_path)
        summary_json_path = args.summary_json or SIMULATION_SUMMARY_JSON
        write_summary_json(summary_data, summary_json_path)

        # Print per-symbol statistics to console
        symbol_stats = summary_data.get("symbol_stats", [])
        if symbol_stats:
            print("\n" + "=" * 120)
            print("STATISTICS BY SYMBOL")
            print("=" * 120)
            print(f"{'Symbol':<12} {'Trades':>7} {'Win':>5} {'Loss':>5} {'Win%':>6} {'Total PnL':>12} {'Avg PnL':>10} {'Best':>10} {'Worst':>10} {'Max DD':>10} {'PF':>6}")
            print("-" * 120)
            for ss in symbol_stats:
                pf_str = f"{ss['profit_factor']}" if ss['profit_factor'] != "∞" else "inf"
                print(f"{ss['symbol']:<12} {ss['trades']:>7} {ss['winners']:>5} {ss['losers']:>5} {ss['win_rate']:>5.1f}% {ss['total_pnl']:>12.2f} {ss['avg_pnl']:>10.2f} {ss['best_trade']:>10.2f} {ss['worst_trade']:>10.2f} {ss['max_drawdown']:>10.2f} {pf_str:>6}")
            print("=" * 120 + "\n")

        generate_trade_charts(trades_df, output_dir=os.path.join(REPORT_DIR, "charts"))
        generate_equity_curve(trades_df, start_capital=START_TOTAL_CAPITAL, output_dir=os.path.join(REPORT_DIR, "charts"))
        if open_positions:
            print(f"[Simulation] Open positions remaining: {len(open_positions)}")
        else:
            print("[Simulation] No open positions remaining.")
        if not trades_df.empty:
            if args.replay_trades_csv:
                last_row = trades_df.tail(1).iloc[0]
                sym_val = last_row.get("symbol") or last_row.get("Symbol") or "?"
                dir_val = last_row.get("direction") or last_row.get("Direction") or "?"
                exit_val = last_row.get("exit_time") or last_row.get("ExitZeit") or "?"
                print(f"[Replay] Last trade: {sym_val} {dir_val} exited {exit_val}")
            else:
                last_trade = trades[-1]
                print(f"[Simulation] Last trade: {last_trade.symbol} {last_trade.direction} exited {last_trade.exit_time}")
    else:
        main(
            allowed_symbols=allowed_symbols,
            allowed_indicators=allowed_indicators,
            fixed_stake=stake_value,
            use_testnet=use_testnet,
            order_executor=order_executor,
            trade_notifier=sms_notifier,
            refresh_params=args.refresh_params,
            reset_state=args.reset_state,
            clear_outputs=args.clear_outputs,
            configure_exchange=configure_exchange_flag,
        )


if __name__ == "__main__":
    run_cli()
