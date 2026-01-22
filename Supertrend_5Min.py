import os
import math
import json
import shutil
from pathlib import Path
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from ta.volatility import AverageTrueRange
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import requests


def _load_env_file(path: str = ".env") -> None:
	env_path = Path(path)
	if not env_path.is_absolute():
		env_path = Path(__file__).resolve().parent / env_path
	if not env_path.exists():
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


_load_env_file()

BERLIN_TZ = ZoneInfo("Europe/Berlin")



def _truthy(value) -> bool:
	return str(value).strip().lower() in {"1", "true", "yes", "on"}


USE_TESTNET = _truthy(os.getenv("BINANCE_USE_TESTNET", "false"))

# ========== LOT SIZE / STEP SIZE HANDLING ==========
_LOT_SIZE_CACHE = {}
_LOT_SIZE_CACHE_LOADED = False


def fetch_lot_sizes_from_binance():
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
    except Exception as e:
        print(f"[LotSize] Error fetching lot sizes: {e}")
    return _LOT_SIZE_CACHE


def get_lot_size(symbol: str) -> float:
    """Get the lot size (stepSize) for a symbol."""
    clean_symbol = symbol.replace("/", "").upper()
    if not _LOT_SIZE_CACHE_LOADED:
        fetch_lot_sizes_from_binance()
    if clean_symbol in _LOT_SIZE_CACHE:
        return _LOT_SIZE_CACHE[clean_symbol]
    usdt_symbol = clean_symbol.replace("USDC", "USDT")
    if usdt_symbol in _LOT_SIZE_CACHE:
        return _LOT_SIZE_CACHE[usdt_symbol]
    return 1e-8


def round_to_lot_size(amount: float, symbol: str) -> float:
    """Round amount DOWN to the nearest valid lot size for the symbol."""
    lot_size = get_lot_size(symbol)
    if lot_size <= 0:
        return amount
    return math.floor(amount / lot_size) * lot_size


def timeframe_to_minutes(tf_str: str) -> int:
	unit = tf_str[-1].lower()
	value = int(tf_str[:-1])
	if unit == "m":
		return value
	if unit == "h":
		return value * 60
	if unit == "d":
		return value * 1440
	raise ValueError(f"Unsupported timeframe unit in {tf_str}")


EXCHANGE_ID = "binance"
TIMEFRAME = "1h"
LOOKBACK = 8760  # ~1 year of hourly bars (365 days × 24 hours)
OHLCV_CACHE_DIR = "ohlcv_cache"  # Directory for persistent OHLCV data storage
# USDx symbols for optimization sweep
SYMBOLS = [
	"BTC/USDC",
	"ETH/USDC",
	"SOL/USDC",
	"XRP/USDC",
	"LINK/USDC",
	"SUI/USDC",
	"ZEC/USDC",
	"TNSR/USDC",
	"ADA/USDC",
	"ICP/USDC",
	"BNB/USDC",
	"LUNC/USDT",  # nur USDT verfügbar
	"TAO/USDC",
]

# Testnet symbols - use USDC where available, USDT otherwise
# Note: Limited to symbols actually available on Binance Spot Testnet
TESTNET_SYMBOLS = [
	"BTC/USDT",
	"ETH/USDT",
	"SOL/USDT",
	"XRP/USDT",
	"LINK/USDT",
	"SUI/USDC",
	"ZEC/USDC",
	"TNSR/USDC",
	"BNB/USDT",
	"TAO/USDC",
	# ICP/USDT, ADA/USDT, LUNC/USDT not available on Spot testnet
]

# Map testnet symbols to production USDC equivalents for parameter lookup
TESTNET_TO_USDC_MAP = {
	"BTC/USDT": "BTC/USDC",
	"ETH/USDT": "ETH/USDC",
	"SOL/USDT": "SOL/USDC",
	"XRP/USDT": "XRP/USDC",
	"LINK/USDT": "LINK/USDC",
	"SUI/USDC": "SUI/USDC",
	"ZEC/USDC": "ZEC/USDC",
	"TNSR/USDC": "TNSR/USDC",
	"BNB/USDT": "BNB/USDC",
	"TAO/USDC": "TAO/USDC",
}


def get_symbols(use_testnet: bool = None) -> list:
	"""Return TESTNET_SYMBOLS if testnet mode, otherwise SYMBOLS."""
	if use_testnet is None:
		use_testnet = USE_TESTNET
	return TESTNET_SYMBOLS if use_testnet else SYMBOLS


def map_symbol_for_params(symbol: str) -> str:
	"""Map testnet symbol to production USDC equivalent for parameter lookup."""
	return TESTNET_TO_USDC_MAP.get(symbol, symbol)


RUN_PARAMETER_SWEEP = False  # ← Deaktiviert, Parameter bereits berechnet
RUN_SAVED_PARAMS = False
RUN_OVERALL_BEST = True  # ← AKTIVIERT für Portfolio-Simulation
ENABLE_LONGS = True
ENABLE_SHORTS = True  # Enabled for both long and short trading

# === PERFORMANCE OPTIMIZATIONS ===
SKIP_SYNTHETIC_BARS = True  # Skip synthetic bar creation for backtesting (big speedup!)
PARALLEL_DATA_FETCH = True  # Fetch multiple symbols in parallel
MAX_PARALLEL_WORKERS = 8    # Number of parallel workers for data fetching

USE_MIN_HOLD_FILTER = True
DEFAULT_MIN_HOLD_BARS = 0
# Min hold bar values - examples for 1h timeframe: [0, 12, 24, 48] = [0h, 12h, 1d, 2d]
MIN_HOLD_BAR_VALUES = [0, 12, 24]

USE_HIGHER_TIMEFRAME_FILTER = True
HIGHER_TIMEFRAME = "6h"
HTF_LOOKBACK = 1000  # Increased for longer backtests
HTF_LENGTH = 20
HTF_FACTOR = 3.0
HTF_PSAR_STEP = 0.02
HTF_PSAR_MAX_STEP = 0.2
HTF_JMA_LENGTH = 30
HTF_JMA_PHASE = 0
HTF_KAMA_LENGTH = 20
HTF_KAMA_SLOW_LENGTH = 40
HTF_MAMA_FAST_LIMIT = 0.5
HTF_MAMA_SLOW_LIMIT = 0.05

USE_MOMENTUM_FILTER = False
MOMENTUM_TYPE = "RSI"
MOMENTUM_WINDOW = 14
RSI_LONG_THRESHOLD = 55
RSI_SHORT_THRESHOLD = 45

USE_JMA_TREND_FILTER = False  # Disabled to match overall_best_detailed.html backtest settings
JMA_TREND_LENGTH = 20  # Length for JMA
JMA_TREND_PHASE = 0  # Phase for JMA
JMA_TREND_THRESH_UP = 0.0001  # Positive threshold for uptrend
JMA_TREND_THRESH_DOWN = -0.0001  # Negative threshold for downtrend

USE_BREAKOUT_FILTER = False
BREAKOUT_ATR_MULT = 1.5
BREAKOUT_REQUIRE_DIRECTION = True

# Bull/Bear Trap Filters
USE_MA_SLOPE_FILTER = True  # Require MA to be rising/falling (not just price above/below)
MA_SLOPE_PERIOD = 5  # Number of bars to check for MA slope direction
MA_SLOPE_MIN_CHANGE = 0.0001  # Minimum percentage change required for valid slope

USE_CANDLESTICK_PATTERN_FILTER = True  # Filter out reversal candlestick patterns
PATTERN_FILTER_SENSITIVITY = "medium"  # "low", "medium", "high" - how strict pattern detection is

USE_DIVERGENCE_FILTER = True  # Detect price/RSI divergence (bull traps)
DIVERGENCE_LOOKBACK = 10  # How many bars to look back for divergence
DIVERGENCE_RSI_PERIOD = 14  # RSI period for divergence detection

START_EQUITY = 14000.0
RISK_FRACTION = 1
STAKE_DIVISOR = 8  # Kapital / 8 pro Trade
FEE_RATE = 0.00075  # VIP Level 1
ATR_WINDOW = 14
ATR_STOP_MULTS = [None, 1.0, 1.5, 2.0]

# Advanced Exit Strategies - Based on Peak Profit Analysis
USE_TRAILING_STOP = True  # Enable trailing stop after peak
TRAILING_STOP_PCT = 0.05  # 5% drawdown from peak triggers exit
TRAILING_STOP_ACTIVATION_PCT = 0.02  # Activate after 2% profit

USE_PARTIAL_EXIT = True  # Take partial profits at targets
PARTIAL_EXIT_LEVELS = [
    {"profit_pct": 0.03, "exit_pct": 0.30},  # At +3%, sell 30%
    {"profit_pct": 0.05, "exit_pct": 0.30},  # At +5%, sell another 30%
]

USE_PROFIT_TARGET = True  # Full exit at profit target
PROFIT_TARGET_PCT = 0.10  # 10% profit = full exit

BASE_OUT_DIR = "report_html"
BARS_PER_DAY = max(1, int(1440 / timeframe_to_minutes(TIMEFRAME)))
CLEAR_BASE_OUTPUT_ON_SWEEP = True

# Output file paths
OVERALL_SUMMARY_HTML = os.path.join(BASE_OUT_DIR, "overall_best_results.html")
OVERALL_PARAMS_CSV = os.path.join(BASE_OUT_DIR, "best_params_overall.csv")
OVERALL_DETAILED_HTML = os.path.join(BASE_OUT_DIR, "overall_best_detailed.html")
OVERALL_FLAT_CSV = os.path.join(BASE_OUT_DIR, "overall_best_flat_trades.csv")
OVERALL_FLAT_JSON = os.path.join(BASE_OUT_DIR, "overall_best_flat_trades.json")
GLOBAL_BEST_RESULTS = {}

INDICATOR_PRESETS = {
	"supertrend": {
		"display_name": "Supertrend",
		"slug": "supertrend",
		"param_a_label": "Length",
		"param_b_label": "Factor",
		"param_a_values": [7, 10, 14],
		"param_b_values": [2.0, 3.0, 4.0],
		"default_a": 10,
		"default_b": 3.0,
	},
	"htf_crossover": {
		"display_name": "HTF Crossover",
		"slug": "htf_crossover",
		"param_a_label": "Length",
		"param_b_label": "Factor",
		"param_a_values": [7, 10, 14],
		"param_b_values": [2.0, 3.0, 4.0],
		"default_a": 10,
		"default_b": 3.0,
	},
	"psar": {
		"display_name": "Parabolic SAR",
		"slug": "psar",
		"param_a_label": "Step",
		"param_b_label": "MaxStep",
		"param_a_values": [0.01, 0.02, 0.03],
		"param_b_values": [0.1, 0.2, 0.3],
		"default_a": 0.02,
		"default_b": 0.2,
	},
	"jma": {
		"display_name": "Jurik Moving Average",
		"slug": "jma",
		"param_a_label": "Length",
		"param_b_label": "Phase",
		"param_a_values": [20, 30, 50],
		"param_b_values": [-50, 0, 50],
		"default_a": 30,
		"default_b": 0,
	},
	"kama": {
		"display_name": "Kaufman AMA",
		"slug": "kama",
		"param_a_label": "Length",
		"param_b_label": "SlowLength",
		"param_a_values": [10, 20, 30],
		"param_b_values": [30, 40, 50],
		"default_a": 20,
		"default_b": 40,
	},
	"mama": {
		"display_name": "Mesa Adaptive MA",
		"slug": "mama",
		"param_a_label": "FastLimit",
		"param_b_label": "SlowLimit",
		"param_a_values": [0.5, 0.4, 0.3],
		"param_b_values": [0.05, 0.03, 0.01],
		"default_a": 0.5,
		"default_b": 0.05,
	},
}

ACTIVE_INDICATORS = ["htf_crossover", "jma", "kama", "supertrend"]

INDICATOR_TYPE = ""
INDICATOR_DISPLAY_NAME = ""
INDICATOR_SLUG = ""
PARAM_A_LABEL = ""
PARAM_B_LABEL = ""
PARAM_A_VALUES: list = []
PARAM_B_VALUES: list = []
DEFAULT_PARAM_A = 0
DEFAULT_PARAM_B = 0

OUT_DIR = BASE_OUT_DIR
REPORT_FILE = "supertrend_report.html"
BEST_PARAMS_FILE = "best_params.csv"

_exchange = None
_data_exchange = None
DATA_CACHE = {}


def clear_data_cache():
	"""Clear the data cache to force fresh data fetch including updated synthetic bars."""
	global DATA_CACHE, FUTURES_DATA_CACHE
	DATA_CACHE = {}
	FUTURES_DATA_CACHE = {}


# Global futures exchange (unauthenticated, for public OHLCV data)
_futures_exchange = None
FUTURES_DATA_CACHE = {}


def get_futures_exchange():
	"""Get or create unauthenticated futures exchange for public data."""
	global _futures_exchange
	if _futures_exchange is None:
		_futures_exchange = ccxt.binance({
			'options': {'defaultType': 'future'},
			'enableRateLimit': True,
		})
	return _futures_exchange


def fetch_futures_data(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
	"""
	Fetch futures OHLCV data for a symbol.
	Converts spot symbol (e.g., BTC/EUR) to futures symbol (e.g., BTC/USDT:USDT).
	"""
	# Convert spot symbol to futures symbol
	base = symbol.split("/")[0]
	futures_symbol = f"{base}/USDT:USDT"

	cache_key = (futures_symbol, timeframe, limit)
	if cache_key in FUTURES_DATA_CACHE:
		return FUTURES_DATA_CACHE[cache_key].copy()

	try:
		exchange = get_futures_exchange()
		ohlcv = exchange.fetch_ohlcv(futures_symbol, timeframe, limit=limit)
		df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
		df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
		df.set_index("timestamp", inplace=True)
		df.index = df.index.tz_convert(BERLIN_TZ)
		FUTURES_DATA_CACHE[cache_key] = df
		print(f"[Futures] Loaded {len(df)} bars for {futures_symbol} {timeframe}")
		return df.copy()
	except Exception as e:
		print(f"[Futures] Failed to fetch {futures_symbol}: {e}")
		return pd.DataFrame()


def clear_futures_cache():
	"""Clear futures data cache."""
	global FUTURES_DATA_CACHE
	FUTURES_DATA_CACHE = {}


# ============================================================================
# LOCAL CONFIGURATION OVERRIDE
# ============================================================================
# Import local config overrides if config_local.py exists
# This allows you to override any config variable without git conflicts
try:
	from config_local import *
	print("[Config] Loaded local configuration overrides from config_local.py")
except ImportError:
	pass  # No local config file - use defaults above


def configure_exchange(use_testnet=None) -> None:
	global USE_TESTNET, _exchange, _data_exchange
	if use_testnet is None or use_testnet == USE_TESTNET:
		return
	USE_TESTNET = use_testnet
	os.environ["BINANCE_USE_TESTNET"] = "1" if use_testnet else "0"
	# Reset trading exchange (uses testnet for orders when enabled)
	_exchange = None
	# Data exchange stays on production - no reset needed
	# _data_exchange = None  # Keep production endpoint for data


def _build_exchange(include_keys: bool):
	cls = getattr(ccxt, EXCHANGE_ID)
	args = {"enableRateLimit": True}
	if include_keys:
		api_key, api_secret = _current_api_credentials()
		if api_key and api_secret:
			args.update({"apiKey": api_key, "secret": api_secret})
	exchange = cls(args)
	options = dict(getattr(exchange, "options", {}))
	options["warnOnFetchCurrenciesWithoutPermission"] = False
	exchange.options = options
	if hasattr(exchange, "has") and isinstance(exchange.has, dict):
		exchange.has["fetchCurrencies"] = False
	if USE_TESTNET and hasattr(exchange, "set_sandbox_mode"):
		try:
			exchange.set_sandbox_mode(True)
		except Exception as exc:
			print(f"[Exchange] Failed to enable Binance sandbox mode: {exc}")
	return exchange


def _current_api_credentials():
	if USE_TESTNET:
		return (
			os.getenv("BINANCE_API_KEY_TEST"),
			os.getenv("BINANCE_API_SECRET_TEST"),
		)
	return (
		os.getenv("BINANCE_API_KEY"),
		os.getenv("BINANCE_API_SECRET"),
	)


def clear_directory(path: str) -> None:
	if not os.path.isdir(path):
		return
	for entry in os.listdir(path):
		full_path = os.path.join(path, entry)
		if os.path.isdir(full_path):
			shutil.rmtree(full_path, ignore_errors=True)
		else:
			try:
				os.remove(full_path)
			except OSError:
				pass


def clear_sweep_targets(indicator_names, htf_values):
	if not indicator_names or not htf_values:
		return
	for indicator_name in indicator_names:
		preset = INDICATOR_PRESETS.get(indicator_name)
		if not preset:
			continue
		slug = preset.get("slug", indicator_name)
		for htf_value in htf_values:
			htf_clean = str(htf_value).replace("/", "")
			folder = os.path.join(BASE_OUT_DIR, f"{slug}_{htf_clean}")
			clear_directory(folder)


def get_exchange():
	global _exchange
	if _exchange is None:
		_exchange = _build_exchange(include_keys=True)
	return _exchange


def get_data_exchange():
	"""Get exchange for data fetching - always uses production Binance.

	Note: We always use production endpoint for data because:
	1. Simulations need real historical data
	2. Testnet has limited/different price data
	3. Only order execution should use testnet
	"""
	global _data_exchange
	if _data_exchange is None:
		# Always use production Binance for data (NOT testnet)
		cls = getattr(ccxt, EXCHANGE_ID)
		args = {"enableRateLimit": True}
		_data_exchange = cls(args)
		# Do NOT enable sandbox mode - we want production data
		options = dict(getattr(_data_exchange, "options", {}))
		options["warnOnFetchCurrenciesWithoutPermission"] = False
		_data_exchange.options = options
		if hasattr(_data_exchange, "has") and isinstance(_data_exchange.has, dict):
			_data_exchange.has["fetchCurrencies"] = False
	return _data_exchange


def _fetch_futures_testnet_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
	"""Fetch klines directly from Binance Futures testnet API (bypasses ccxt issues).

	Handles both USDT and USDC pairs by converting USDC to USDT for the API call.
	Futures testnet has more symbols available than Spot testnet.
	"""
	import requests
	try:
		# Convert symbol format: BTC/USDT -> BTCUSDT, BTC/USDC -> BTCUSDT
		api_symbol = symbol.replace("/", "").replace("USDC", "USDT")
		# Binance has a max limit of 1500 per request
		api_limit = min(limit, 1500)
		url = "https://testnet.binancefuture.com/fapi/v1/klines"
		params = {"symbol": api_symbol, "interval": interval, "limit": api_limit}
		response = requests.get(url, params=params, timeout=30)
		if response.status_code != 200:
			print(f"[API-Direct] Error {response.status_code} for {symbol} (as {api_symbol}): {response.text[:100]}")
			return pd.DataFrame()
		data = response.json()
		if not data:
			return pd.DataFrame()
		# Binance kline format: [open_time, open, high, low, close, volume, ...]
		cols = ["timestamp", "open", "high", "low", "close", "volume"]
		rows = [[d[0], float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])] for d in data]
		df = pd.DataFrame(rows, columns=cols)
		df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(BERLIN_TZ)
		result = df.set_index("timestamp")
		print(f"[API-Direct] Got {len(result)} bars for {symbol} {interval} (via Futures testnet as {api_symbol})")
		return result
	except Exception as exc:
		print(f"[API-Direct] ERROR fetching {symbol} {interval}: {exc}")
		return pd.DataFrame()


def _fetch_direct_ohlcv(symbol, timeframe, limit):
	"""Fetch OHLCV data directly from exchange API with proper error handling."""
	try:
		exchange = get_data_exchange()
		buffer = max(50, limit // 5)
		fetch_limit = limit + buffer
		print(f"[API] Fetching {symbol} {timeframe} (limit={fetch_limit})...")
		ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=fetch_limit)
		if not ohlcv:
			print(f"[API] Warning: No data returned for {symbol} {timeframe}")
			return pd.DataFrame()
		cols = ["timestamp", "open", "high", "low", "close", "volume"]
		df = pd.DataFrame(ohlcv, columns=cols)
		df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(BERLIN_TZ)
		result = df.set_index("timestamp").tail(limit)
		print(f"[API] Got {len(result)} bars for {symbol} {timeframe}")
		return result
	except Exception as exc:
		print(f"[API] ccxt failed for {symbol} {timeframe}: {exc}")
		# Fallback to direct Futures testnet API for USDT and USDC pairs
		# Futures testnet works from EU and has more symbols
		if "USDT" in symbol or "USDC" in symbol:
			print(f"[API] Trying direct Futures testnet API for {symbol}...")
			return _fetch_futures_testnet_klines(symbol, timeframe, limit)
		return pd.DataFrame()


def _maybe_append_synthetic_bar(df, symbol, timeframe):
	"""
	Append a synthetic bar for the current incomplete period using 1m bars or ticker data.

	This ensures simulations always include the very latest data, even if the current
	hour/period hasn't completed yet.
	"""
	# Skip synthetic bars for backtesting (big performance improvement!)
	if SKIP_SYNTHETIC_BARS:
		return df

	try:
		tf_minutes = timeframe_to_minutes(timeframe)
	except ValueError:
		return df
	if tf_minutes <= 1:
		return df
	if df is None:
		df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)

	now = pd.Timestamp.now(BERLIN_TZ)
	bucket = pd.Timedelta(minutes=tf_minutes)
	current_end = now.floor(f"{tf_minutes}min") + bucket

	# Check if we already have the current bar
	if not df.empty:
		last_idx = df.index.max()
		if last_idx > current_end:
			return df
		if last_idx == current_end:
			# Remove existing incomplete bar - we'll recreate it
			df = df.drop(index=current_end)

	current_start = current_end - bucket
	minutes_needed = max(2, int(np.ceil((now - current_start).total_seconds() / 60.0)) + 2)

	# Try to fetch 1-minute bars first (most accurate)
	try:
		minute_df = _fetch_direct_ohlcv(symbol, "1m", limit=minutes_needed)
		slice_df = minute_df[(minute_df.index > current_start) & (minute_df.index <= now)]

		if not slice_df.empty:
			synthetic = pd.DataFrame({
				"open": float(slice_df["open"].iloc[0]),
				"high": float(slice_df["high"].max()),
				"low": float(slice_df["low"].min()),
				"close": float(slice_df["close"].iloc[-1]),
				"volume": float(slice_df["volume"].sum()),
			}, index=[current_start])

			combined = pd.concat([df, synthetic])
			combined = combined[~combined.index.duplicated(keep="last")]
			combined = combined.sort_index()

			print(f"[Synthetic] Created current bar for {symbol} {timeframe} using {len(slice_df)} 1m bars (starts {current_start.strftime('%Y-%m-%d %H:%M')})")
			return combined
	except Exception as exc:
		print(f"[Synthetic] Failed to fetch 1m data for {symbol}: {exc}")

	# Fallback: try to use ticker data for current price
	try:
		exchange = get_data_exchange()
		ticker = exchange.fetch_ticker(symbol)

		if ticker and 'last' in ticker and ticker['last']:
			# For ticker-based bar, we use last price for all OHLC values
			# This is less accurate but better than no data
			last_price = float(ticker['last'])

			# Try to get previous close as open price
			if not df.empty:
				prev_close = float(df['close'].iloc[-1])
				synthetic_open = prev_close
			else:
				synthetic_open = last_price

			synthetic = pd.DataFrame({
				"open": synthetic_open,
				"high": last_price,  # Conservative: use current price
				"low": last_price,   # Conservative: use current price
				"close": last_price,
				"volume": 0.0,  # Ticker doesn't provide volume for incomplete bar
			}, index=[current_start])

			combined = pd.concat([df, synthetic])
			combined = combined[~combined.index.duplicated(keep="last")]
			combined = combined.sort_index()

			print(f"[Synthetic] Created current bar for {symbol} {timeframe} using ticker data @ {last_price:.2f} (starts {current_start.strftime('%Y-%m-%d %H:%M')})")
			return combined
	except Exception as exc:
		print(f"[Synthetic] Failed to fetch ticker for {symbol}: {exc}")

	# If both methods fail, return original dataframe
	print(f"[Synthetic] Could not create synthetic bar for {symbol} {timeframe} - no data available")
	return df


def fetch_data(symbol, timeframe, limit):
	key = (symbol, timeframe, limit)
	if key in DATA_CACHE:
		base_df = DATA_CACHE[key]
	else:
		# Try loading from persistent cache first
		persistent_df = load_ohlcv_from_cache(symbol, timeframe)

		# If limit is None or 0, return ALL cached data (for simulations)
		if limit is None or limit == 0:
			if not persistent_df.empty:
				cache_df = persistent_df
				print(f"[Cache] Loaded {len(cache_df)} bars for {symbol} {timeframe} from {cache_df.index[0].strftime('%Y-%m-%d')} to {cache_df.index[-1].strftime('%Y-%m-%d')}")
			else:
				# Fall back to API with large limit
				cache_df = _fetch_direct_ohlcv(symbol, timeframe, 10000)
				if not cache_df.empty:
					print(f"[API] Fetched {len(cache_df)} bars for {symbol} {timeframe} from {cache_df.index[0].strftime('%Y-%m-%d')} to {cache_df.index[-1].strftime('%Y-%m-%d')}")
					save_ohlcv_to_cache(symbol, timeframe, cache_df)
		# If we have data in persistent cache, use it (prefer cache over API)
		elif not persistent_df.empty:
			# Use cached data - even if less than requested (API can't give more anyway)
			cache_df = persistent_df.tail(limit) if len(persistent_df) >= limit else persistent_df
			print(f"[Cache] Loaded {len(cache_df)} bars for {symbol} {timeframe}")
		else:
			# No cached data - fall back to API
			cache_df = None
			exchange = get_data_exchange()
			supported_timeframes = getattr(exchange, "timeframes", {}) or {}
			if timeframe in supported_timeframes:
				cache_df = _fetch_direct_ohlcv(symbol, timeframe, limit)
				if cache_df is not None and not cache_df.empty:
					save_ohlcv_to_cache(symbol, timeframe, cache_df)
			else:
				target_minutes = timeframe_to_minutes(timeframe)
				base_minutes = timeframe_to_minutes(TIMEFRAME)
				if target_minutes < base_minutes or target_minutes % base_minutes != 0:
					raise ValueError(f"Cannot synthesize timeframe {timeframe} from base {TIMEFRAME}")
				factor = target_minutes // base_minutes
				base_limit = limit * factor + 10
				base_df_source = fetch_data(symbol, TIMEFRAME, base_limit)
				if base_df_source.empty:
					cache_df = base_df_source
				else:
					agg_rule = f"{target_minutes}min"
					synth = base_df_source.resample(agg_rule, label="right", closed="right").agg({
						"open": "first",
						"high": "max",
						"low": "min",
						"close": "last",
						"volume": "sum",
					})
					synth = synth.dropna(subset=["open", "high", "low", "close"])
					cache_df = synth.tail(limit)
		DATA_CACHE[key] = cache_df
		base_df = cache_df
	df_copy = base_df.copy() if base_df is not None else pd.DataFrame()
	df_with_live = _maybe_append_synthetic_bar(df_copy, symbol, timeframe)
	return df_with_live


def download_historical_ohlcv(symbol, timeframe, start_date, end_date=None):
	"""
	Download historical OHLCV data from Binance for a specific date range.

	Args:
		symbol: Trading pair (e.g., "BTC/EUR")
		timeframe: Timeframe string (e.g., "1h", "4h", "1d")
		start_date: Start date (pd.Timestamp or datetime)
		end_date: End date (pd.Timestamp or datetime), defaults to now

	Returns:
		DataFrame with OHLCV data for the requested period
	"""
	import time

	# Ensure timestamps
	if isinstance(start_date, str):
		start_date = pd.Timestamp(start_date, tz=BERLIN_TZ)
	elif not hasattr(start_date, 'tz_localize'):
		start_date = pd.Timestamp(start_date)
	if start_date.tzinfo is None:
		start_date = start_date.tz_localize(BERLIN_TZ)
	else:
		start_date = start_date.tz_convert(BERLIN_TZ)

	if end_date is None:
		end_date = pd.Timestamp.now(tz=BERLIN_TZ)
	elif isinstance(end_date, str):
		end_date = pd.Timestamp(end_date, tz=BERLIN_TZ)
	elif not hasattr(end_date, 'tz_localize'):
		end_date = pd.Timestamp(end_date)
	if end_date.tzinfo is None:
		end_date = end_date.tz_localize(BERLIN_TZ)
	else:
		end_date = end_date.tz_convert(BERLIN_TZ)

	# Convert to UTC milliseconds for Binance API
	start_ms = int(start_date.tz_convert('UTC').timestamp() * 1000)
	end_ms = int(end_date.tz_convert('UTC').timestamp() * 1000)

	exchange = get_data_exchange()
	tf_minutes = timeframe_to_minutes(timeframe)
	tf_ms = tf_minutes * 60 * 1000

	# Binance limit is 1000 bars per request
	max_bars_per_request = 1000
	all_data = []
	current_start = start_ms

	print(f"[Download] Fetching {symbol} {timeframe} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

	batch_count = 0
	while current_start < end_ms:
		try:
			# Fetch batch
			ohlcv = exchange.fetch_ohlcv(
				symbol,
				timeframe=timeframe,
				since=current_start,
				limit=max_bars_per_request
			)

			if not ohlcv:
				break

			# Add to results
			all_data.extend(ohlcv)
			batch_count += 1

			# Move to next batch (last timestamp + 1 interval)
			last_timestamp = ohlcv[-1][0]
			current_start = last_timestamp + tf_ms

			# Progress indicator
			current_date = pd.Timestamp(last_timestamp, unit='ms', tz='UTC').tz_convert(BERLIN_TZ)
			print(f"[Download] Batch {batch_count}: Got {len(ohlcv)} bars, up to {current_date.strftime('%Y-%m-%d %H:%M')}")

			# Rate limiting - sleep between requests
			if current_start < end_ms:
				time.sleep(0.5)  # 500ms between requests to avoid rate limits

		except Exception as exc:
			print(f"[Download] Error fetching data: {exc}")
			break

	if not all_data:
		print(f"[Download] No data retrieved for {symbol} {timeframe}")
		return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

	# Convert to DataFrame
	cols = ["timestamp", "open", "high", "low", "close", "volume"]
	df = pd.DataFrame(all_data, columns=cols)
	df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert(BERLIN_TZ)
	df = df.set_index("timestamp")

	# Remove duplicates and sort
	df = df[~df.index.duplicated(keep='last')]
	df = df.sort_index()

	# Filter to requested range
	df = df[(df.index >= start_date) & (df.index <= end_date)]

	print(f"[Download] Complete: {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

	# Save to persistent cache
	save_ohlcv_to_cache(symbol, timeframe, df)

	return df


def get_cache_filename(symbol, timeframe):
	"""Generate cache filename for a symbol/timeframe pair."""
	# Replace / with _ for filesystem compatibility
	safe_symbol = symbol.replace('/', '_')
	return os.path.join(OHLCV_CACHE_DIR, f"{safe_symbol}_{timeframe}.csv")


def load_ohlcv_from_cache(symbol, timeframe):
	"""Load OHLCV data from persistent cache."""
	cache_file = get_cache_filename(symbol, timeframe)

	if not os.path.exists(cache_file):
		return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

	try:
		df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

		# Ensure index is DatetimeIndex with proper timezone handling
		if not isinstance(df.index, pd.DatetimeIndex):
			# Use utc=True to handle timezone-aware datetime strings in CSV
			df.index = pd.to_datetime(df.index, utc=True)

		# Ensure timezone is BERLIN_TZ
		if df.index.tz is None:
			df.index = df.index.tz_localize('UTC').tz_convert(BERLIN_TZ)
		else:
			df.index = df.index.tz_convert(BERLIN_TZ)

		return df
	except Exception as exc:
		print(f"[Cache] Error loading {cache_file}: {exc}")
		return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def save_ohlcv_to_cache(symbol, timeframe, df):
	"""Save OHLCV data to persistent cache."""
	if df.empty:
		return

	# Create cache directory if needed
	os.makedirs(OHLCV_CACHE_DIR, exist_ok=True)

	cache_file = get_cache_filename(symbol, timeframe)

	try:
		# Load existing data if any
		existing_df = load_ohlcv_from_cache(symbol, timeframe)

		# Merge with new data
		if not existing_df.empty:
			combined = pd.concat([existing_df, df])
			combined = combined[~combined.index.duplicated(keep='last')]
			combined = combined.sort_index()
		else:
			combined = df

		# Save to CSV
		combined.to_csv(cache_file)
		print(f"[Cache] Saved {len(combined)} bars to {cache_file}")

	except Exception as exc:
		print(f"[Cache] Error saving {cache_file}: {exc}")


def update_output_targets():
	global OUT_DIR, REPORT_FILE, BEST_PARAMS_FILE
	slug = INDICATOR_SLUG or "supertrend"
	htf = HIGHER_TIMEFRAME.replace("/", "")
	folder = f"{slug}_{htf}"
	OUT_DIR = os.path.join(BASE_OUT_DIR, folder)
	REPORT_FILE = f"long_strategy_report_{folder}.html"
	BEST_PARAMS_FILE = f"best_params_{folder}.csv"


def apply_indicator_type(name: str):
	global INDICATOR_TYPE, INDICATOR_DISPLAY_NAME, INDICATOR_SLUG
	global PARAM_A_LABEL, PARAM_B_LABEL, PARAM_A_VALUES, PARAM_B_VALUES
	global DEFAULT_PARAM_A, DEFAULT_PARAM_B
	preset = INDICATOR_PRESETS[name]
	INDICATOR_TYPE = name
	INDICATOR_DISPLAY_NAME = preset["display_name"]
	INDICATOR_SLUG = preset["slug"]
	PARAM_A_LABEL = preset["param_a_label"]
	PARAM_B_LABEL = preset["param_b_label"]
	PARAM_A_VALUES = preset["param_a_values"]
	PARAM_B_VALUES = preset["param_b_values"]
	DEFAULT_PARAM_A = preset["default_a"]
	DEFAULT_PARAM_B = preset["default_b"]
	update_output_targets()


def apply_higher_timeframe(htf_value: str):
	global HIGHER_TIMEFRAME
	HIGHER_TIMEFRAME = htf_value
	update_output_targets()


def get_indicator_candidates():
	if ACTIVE_INDICATORS:
		return [name for name in ACTIVE_INDICATORS if name in INDICATOR_PRESETS]
	return list(INDICATOR_PRESETS.keys())


def get_enabled_directions():
	directions = []
	if ENABLE_LONGS:
		directions.append("long")
	if ENABLE_SHORTS:
		directions.append("short")
	if not directions:
		directions.append("long")
	return directions


def get_highertimeframe_candidates():
	# Include shorter timeframes (1h, 2h) plus standard range (3h-24h)
	return ["1h", "2h"] + [f"{hours}h" for hours in range(3, 25)]


def compute_supertrend(df, length=10, factor=3.0):
	df = df.copy()
	length = max(1, int(length))
	factor = float(factor)

	if df.empty:
		df["supertrend"] = np.nan
		df["st_trend"] = 0
		return df

	for col in ["high", "low", "close"]:
		if col not in df.columns:
			df["supertrend"] = np.nan
			df["st_trend"] = 0
			return df

	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=length).average_true_range()

	# Use numpy arrays for speed
	high = df["high"].values
	low = df["low"].values
	close = df["close"].values
	atr_vals = atr.values
	n = len(df)

	hl2 = (high + low) / 2.0
	basic_ub = hl2 + factor * atr_vals
	basic_lb = hl2 - factor * atr_vals

	# Find first valid ATR index (after warmup period)
	first_valid = 0
	for i in range(n):
		if not np.isnan(atr_vals[i]):
			first_valid = i
			break

	# Initialize arrays with NaN
	final_ub = np.full(n, np.nan)
	final_lb = np.full(n, np.nan)
	supertrend = np.full(n, np.nan)
	trend = np.zeros(n, dtype=np.int32)

	# Start from first valid ATR value
	if first_valid < n:
		final_ub[first_valid] = basic_ub[first_valid]
		final_lb[first_valid] = basic_lb[first_valid]
		trend[first_valid] = 1 if close[first_valid] >= final_lb[first_valid] else -1
		supertrend[first_valid] = final_lb[first_valid] if trend[first_valid] == 1 else final_ub[first_valid]

		# Calculate for remaining bars
		for i in range(first_valid + 1, n):
			prev_close = close[i - 1]

			# Upper band
			if basic_ub[i] < final_ub[i - 1] or prev_close > final_ub[i - 1]:
				final_ub[i] = basic_ub[i]
			else:
				final_ub[i] = final_ub[i - 1]

			# Lower band
			if basic_lb[i] > final_lb[i - 1] or prev_close < final_lb[i - 1]:
				final_lb[i] = basic_lb[i]
			else:
				final_lb[i] = final_lb[i - 1]

			# Trend and supertrend
			if trend[i - 1] == 1:
				if close[i] <= final_lb[i]:
					trend[i] = -1
					supertrend[i] = final_ub[i]
				else:
					trend[i] = 1
					supertrend[i] = final_lb[i]
			else:
				if close[i] >= final_ub[i]:
					trend[i] = 1
					supertrend[i] = final_lb[i]
				else:
					trend[i] = -1
					supertrend[i] = final_ub[i]

	df["supertrend"] = supertrend
	df["st_trend"] = trend
	df["atr"] = atr
	df["indicator_line"] = df["supertrend"]
	df["trend_flag"] = df["st_trend"]
	return df


def compute_psar(df, step=0.02, max_step=0.2):
	df = df.copy()
	step = float(step)
	max_step = float(max_step)
	high = df["high"].values
	low = df["low"].values
	psar_vals = np.zeros(len(df))
	trend_flags = np.ones(len(df))
	af = step
	ep = high[0]
	psar_vals[0] = low[0]
	bullish = True

	for i in range(1, len(df)):
		prior_psar = psar_vals[i - 1]
		if bullish:
			psar_candidate = prior_psar + af * (ep - prior_psar)
			if i >= 2:
				psar_candidate = min(psar_candidate, low[i - 1], low[i - 2])
			else:
				psar_candidate = min(psar_candidate, low[i - 1])
			if low[i] < psar_candidate:
				bullish = False
				psar_vals[i] = ep
				ep = low[i]
				af = step
			else:
				psar_vals[i] = psar_candidate
				if high[i] > ep:
					ep = high[i]
					af = min(af + step, max_step)
		else:
			psar_candidate = prior_psar + af * (ep - prior_psar)
			if i >= 2:
				psar_candidate = max(psar_candidate, high[i - 1], high[i - 2])
			else:
				psar_candidate = max(psar_candidate, high[i - 1])
			if high[i] > psar_candidate:
				bullish = True
				psar_vals[i] = ep
				ep = high[i]
				af = step
			else:
				psar_vals[i] = psar_candidate
				if low[i] < ep:
					ep = low[i]
					af = min(af + step, max_step)
		trend_flags[i] = 1 if bullish else -1

	df["psar"] = pd.Series(psar_vals, index=df.index)
	df["psar_trend"] = pd.Series(trend_flags, index=df.index)
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
	df["atr"] = atr
	df["indicator_line"] = df["psar"]
	df["trend_flag"] = df["psar_trend"]
	return df


def jurik_moving_average(series: pd.Series, length: int, phase: int) -> pd.Series:
	if series.empty:
		return pd.Series(index=series.index, dtype=float)
	length = max(1, int(length))
	phase = int(np.clip(phase, -100, 100))
	beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2) if length > 1 else 0.0
	alpha = beta ** 2
	phase_ratio = (phase + 100) / 200
	jma_values = pd.Series(index=series.index, dtype=float)
	e0 = float(series.iloc[0])
	e1 = 0.0
	e2 = 0.0
	for idx, (_, price) in enumerate(series.items()):
		e0 = (1 - alpha) * price + alpha * e0
		e1 = price - e0
		e2 = (1 - beta) * e1 + beta * e2
		jma = e0 + phase_ratio * e2
		jma_values.iloc[idx] = jma
	return jma_values


def compute_jma(df, length=20, phase=0):
	df = df.copy()
	jma = jurik_moving_average(df["close"], length=length, phase=phase)
	trend = np.where(df["close"] >= jma, 1, -1)
	df["jma"] = jma
	df["jma_trend"] = trend
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
	df["atr"] = atr
	df["indicator_line"] = df["jma"]
	df["trend_flag"] = df["jma_trend"]
	return df


def compute_kama(df, length=10, slow_length=30, fast_length=2):
	df = df.copy()
	close = df["close"].astype(float)
	length = max(1, int(length))
	slow_length = max(length + 1, int(slow_length))
	fast_length = max(1, int(fast_length))
	fast_sc = 2.0 / (fast_length + 1)
	slow_sc = 2.0 / (slow_length + 1)
	direction = close.diff(length).abs()
	volatility = close.diff().abs().rolling(length).sum()
	er = (direction / volatility).fillna(0.0).clip(lower=0.0, upper=1.0)
	sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
	kama = close.copy()
	for i in range(1, len(close)):
		prev = kama.iloc[i - 1]
		kama.iloc[i] = prev + sc.iloc[i] * (close.iloc[i] - prev)
	trend = np.where(close >= kama, 1, -1)
	df["kama"] = kama
	df["kama_trend"] = trend
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
	df["atr"] = atr
	df["indicator_line"] = df["kama"]
	df["trend_flag"] = df["kama_trend"]
	return df


def calculate_jma_trend_filter(df, length=20, phase=0, thresh_up=0.0001, thresh_down=-0.0001):
	"""
	Calculate JMA trend filter using double-smoothed JMA and slope.

	Algorithm:
	1. Calculate JMA of close prices
	2. Calculate JMA of the JMA (double smoothing for minimal lag)
	3. Calculate slope = first difference of double-smoothed JMA
	4. Classify trend:
	   - UP: slope > thresh_up
	   - DOWN: slope < thresh_down
	   - FLAT: between thresholds

	Returns dataframe with 'jma_trend_slope' and 'jma_trend_direction' columns
	"""
	df = df.copy()

	# Step 1: Calculate JMA of close
	jma_1 = jurik_moving_average(df["close"], length=length, phase=phase)

	# Step 2: Calculate JMA of JMA (double smoothing)
	jma_2 = jurik_moving_average(jma_1, length=length, phase=phase)

	# Step 3: Calculate slope (first difference)
	slope = jma_2.diff()

	# Step 4: Classify trend
	trend_direction = pd.Series("FLAT", index=df.index)
	trend_direction[slope > thresh_up] = "UP"
	trend_direction[slope < thresh_down] = "DOWN"

	df["jma_trend_slope"] = slope
	df["jma_trend_direction"] = trend_direction

	return df


def mesa_adaptive_moving_average(series: pd.Series, fast_limit: float = 0.5, slow_limit: float = 0.05):
	if series.empty:
		return pd.Series(dtype=float), pd.Series(dtype=float)
	values = series.astype(float)
	n = len(values)
	mama = np.zeros(n)
	fama = np.zeros(n)
	period = np.ones(n) * 10.0
	smooth = values.copy().to_numpy()
	detrender = np.zeros(n)
	I1 = np.zeros(n)
	Q1 = np.zeros(n)
	jI = np.zeros(n)
	jQ = np.zeros(n)
	I2 = np.zeros(n)
	Q2 = np.zeros(n)
	Re = np.zeros(n)
	Im = np.zeros(n)
	phase = np.zeros(n)
	fast_limit = float(max(0.01, fast_limit))
	slow_limit = float(max(0.001, min(fast_limit, slow_limit)))
	for i in range(n):
		price = values.iloc[i]
		if i >= 3:
			smooth[i] = (4 * price + 3 * values.iloc[i - 1] + 2 * values.iloc[i - 2] + values.iloc[i - 3]) / 10.0
		else:
			smooth[i] = price
		if i >= 6:
			detrender[i] = (0.0962 * smooth[i] + 0.5769 * smooth[i - 2] - 0.5769 * smooth[i - 4] - 0.0962 * smooth[i - 6]) * (0.075 * period[i - 1] + 0.54)
			Q1[i] = (0.0962 * detrender[i] + 0.5769 * detrender[i - 2] - 0.5769 * detrender[i - 4] - 0.0962 * detrender[i - 6]) * (0.075 * period[i - 1] + 0.54)
			I1[i] = detrender[i - 3]
			jI[i] = (0.0962 * I1[i] + 0.5769 * I1[i - 2] - 0.5769 * I1[i - 4] - 0.0962 * I1[i - 6]) * (0.075 * period[i - 1] + 0.54)
			jQ[i] = (0.0962 * Q1[i] + 0.5769 * Q1[i - 2] - 0.5769 * Q1[i - 4] - 0.0962 * Q1[i - 6]) * (0.075 * period[i - 1] + 0.54)
		else:
			detrender[i] = 0.0
			Q1[i] = 0.0
			I1[i] = 0.0
			jI[i] = 0.0
			jQ[i] = 0.0
		I2[i] = I1[i] - jQ[i]
		Q2[i] = Q1[i] + jI[i]
		if i > 0:
			Re[i] = I2[i] * I2[i - 1] + Q2[i] * Q2[i - 1]
			Im[i] = I2[i] * Q2[i - 1] - Q2[i] * I2[i - 1]
			angle = np.arctan2(Im[i], Re[i])
			if angle != 0.0:
				period[i] = abs(2 * np.pi / angle)
			period[i] = np.clip(period[i], 6.0, 50.0)
			period[i] = 0.2 * period[i] + 0.8 * period[i - 1]
		else:
			period[i] = period[i - 1] if i > 0 else 10.0
		if I1[i] != 0.0:
			phase[i] = np.degrees(np.arctan2(Q1[i], I1[i]))
		else:
			phase[i] = 0.0
		delta_phase = phase[i - 1] - phase[i] if i > 0 else 0.0
		if delta_phase < 1.0:
			delta_phase = 1.0
		alpha = fast_limit / delta_phase
		alpha = np.clip(alpha, slow_limit, fast_limit)
		mama[i] = alpha * price + (1 - alpha) * (mama[i - 1] if i > 0 else price)
		fama[i] = 0.5 * alpha * mama[i] + (1 - 0.5 * alpha) * (fama[i - 1] if i > 0 else price)
	return pd.Series(mama, index=series.index), pd.Series(fama, index=series.index)


def compute_mama(df, fast_limit=0.5, slow_limit=0.05):
	df = df.copy()
	mama, fama = mesa_adaptive_moving_average(df["close"], fast_limit=fast_limit, slow_limit=slow_limit)
	trend = np.where(df["close"] >= mama, 1, -1)
	df["mama"] = mama
	df["fama"] = fama
	df["mama_trend"] = trend
	atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
	df["atr"] = atr
	df["indicator_line"] = df["mama"]
	df["trend_flag"] = df["mama_trend"]
	return df


def compute_indicator(df, param_a, param_b):
	if INDICATOR_TYPE == "supertrend" or INDICATOR_TYPE == "htf_crossover":
		return compute_supertrend(df, length=int(param_a), factor=float(param_b))
	if INDICATOR_TYPE == "psar":
		return compute_psar(df, step=float(param_a), max_step=float(param_b))
	if INDICATOR_TYPE == "jma":
		return compute_jma(df, length=int(param_a), phase=int(param_b))
	if INDICATOR_TYPE == "kama":
		return compute_kama(df, length=int(param_a), slow_length=int(param_b))
	if INDICATOR_TYPE == "mama":
		return compute_mama(df, fast_limit=float(param_a), slow_limit=float(param_b))
	raise ValueError(f"Unsupported INDICATOR_TYPE: {INDICATOR_TYPE}")


def attach_higher_timeframe_trend(df_low, symbol):
	if not USE_HIGHER_TIMEFRAME_FILTER:
		df_low = df_low.copy()
		df_low["htf_trend"] = 0
		df_low["htf_indicator"] = np.nan
		return df_low

	# Calculate how many HTF bars we need to cover the low TF date range
	low_tf_minutes = timeframe_to_minutes(TIMEFRAME)
	htf_minutes = timeframe_to_minutes(HIGHER_TIMEFRAME)
	factor = max(1, htf_minutes // low_tf_minutes)
	# Need enough HTF bars to cover low TF range, plus buffer for indicator warmup
	htf_bars_needed = (len(df_low) // factor) + 100

	df_high = fetch_data(symbol, HIGHER_TIMEFRAME, htf_bars_needed)
	if df_high.empty:
		print(f"[HTF] WARNING: No HTF data for {symbol} {HIGHER_TIMEFRAME}")
		df_low = df_low.copy()
		df_low["htf_trend"] = 0
		df_low["htf_indicator"] = np.nan
		return df_low

	# Debug: Check time ranges
	print(f"[HTF] {symbol} {HIGHER_TIMEFRAME}: {len(df_high)} bars (needed ~{htf_bars_needed})")
	print(f"[HTF] HTF range: {df_high.index[0]} to {df_high.index[-1]}")
	print(f"[HTF] LowTF range: {df_low.index[0]} to {df_low.index[-1]}")

	# Debug: Check df_high content
	print(f"[HTF] df_high columns: {list(df_high.columns)}")
	print(f"[HTF] df_high dtypes: {df_high.dtypes.to_dict()}")
	nan_check = {col: df_high[col].isna().sum() for col in ['open', 'high', 'low', 'close', 'volume'] if col in df_high.columns}
	print(f"[HTF] NaN counts: {nan_check}")

	if INDICATOR_TYPE == "supertrend" or INDICATOR_TYPE == "htf_crossover":
		print(f"[HTF] Calling compute_supertrend with length={HTF_LENGTH}, factor={HTF_FACTOR}")
		df_high_ind = compute_supertrend(df_high, length=HTF_LENGTH, factor=HTF_FACTOR)
		print(f"[HTF] compute_supertrend returned, supertrend col exists: {'supertrend' in df_high_ind.columns}")
		indicator_col = "supertrend"
		trend_col = "st_trend"
	elif INDICATOR_TYPE == "psar":
		df_high_ind = compute_psar(df_high, step=HTF_PSAR_STEP, max_step=HTF_PSAR_MAX_STEP)
		indicator_col = "psar"
		trend_col = "psar_trend"
	elif INDICATOR_TYPE == "jma":
		df_high_ind = compute_jma(df_high, length=HTF_JMA_LENGTH, phase=HTF_JMA_PHASE)
		indicator_col = "jma"
		trend_col = "jma_trend"
	elif INDICATOR_TYPE == "kama":
		df_high_ind = compute_kama(df_high, length=HTF_KAMA_LENGTH, slow_length=HTF_KAMA_SLOW_LENGTH)
		indicator_col = "kama"
		trend_col = "kama_trend"
	elif INDICATOR_TYPE == "mama":
		df_high_ind = compute_mama(df_high, fast_limit=HTF_MAMA_FAST_LIMIT, slow_limit=HTF_MAMA_SLOW_LIMIT)
		indicator_col = "mama"
		trend_col = "mama_trend"
	else:
		raise ValueError(f"Unsupported HTF indicator type: {INDICATOR_TYPE}")

	htf = df_high_ind[[indicator_col, trend_col]].rename(columns={
		indicator_col: "htf_indicator",
		trend_col: "htf_trend"
	})

	# Debug: Check indicator values before reindex
	htf_valid = htf["htf_indicator"].notna().sum()
	print(f"[HTF] Indicator computed: {htf_valid}/{len(htf)} valid values")

	aligned = htf.reindex(df_low.index, method="ffill")

	# Debug: Check after reindex
	aligned_valid = aligned["htf_indicator"].notna().sum()
	print(f"[HTF] After reindex: {aligned_valid}/{len(aligned)} valid values")

	df_low = df_low.copy()
	df_low["htf_trend"] = aligned["htf_trend"].fillna(0).astype(int)
	df_low["htf_indicator"] = aligned["htf_indicator"]
	return df_low


def attach_momentum_filter(df):
	df = df.copy()
	if not USE_MOMENTUM_FILTER:
		df["momentum"] = np.nan
		return df

	if MOMENTUM_TYPE.lower() == "rsi":
		delta = df["close"].diff()
		gain = np.where(delta > 0, delta, 0.0)
		loss = np.where(delta < 0, -delta, 0.0)
		roll_gain = pd.Series(gain, index=df.index).rolling(MOMENTUM_WINDOW).mean()
		roll_loss = pd.Series(loss, index=df.index).rolling(MOMENTUM_WINDOW).mean()
		rs = roll_gain / roll_loss.replace(0, np.nan)
		rsi = 100 - (100 / (1 + rs))
		df["momentum"] = rsi
	else:
		df["momentum"] = np.nan
	return df


def attach_jma_trend_filter(df):
	"""
	Attach JMA trend filter to dataframe.

	Calculates double-smoothed JMA and slope to classify trend as UP/DOWN/FLAT.
	Only allows LONG entries when trend is UP, SHORT entries when trend is DOWN.
	"""
	df = df.copy()
	if not USE_JMA_TREND_FILTER:
		df["jma_trend_slope"] = np.nan
		df["jma_trend_direction"] = "FLAT"
		return df

	df = calculate_jma_trend_filter(
		df,
		length=JMA_TREND_LENGTH,
		phase=JMA_TREND_PHASE,
		thresh_up=JMA_TREND_THRESH_UP,
		thresh_down=JMA_TREND_THRESH_DOWN
	)
	return df


def attach_ma_slope_filter(df, indicator_col="close"):
	"""
	Calculate moving average slope to detect if MA is actually rising/falling.

	Prevents entries when price crosses MA but MA is still falling (bull trap) or vice versa.
	"""
	df = df.copy()
	if not USE_MA_SLOPE_FILTER or indicator_col not in df.columns:
		df["ma_slope_direction"] = "NEUTRAL"
		df["ma_slope_change_pct"] = 0.0
		return df

	# Calculate MA slope over the specified period
	ma_values = df[indicator_col].rolling(MA_SLOPE_PERIOD).mean()

	# Calculate percentage change in MA over the period
	ma_change = ma_values.pct_change(MA_SLOPE_PERIOD)
	df["ma_slope_change_pct"] = ma_change

	# Classify slope direction
	df["ma_slope_direction"] = np.where(
		ma_change > MA_SLOPE_MIN_CHANGE, "UP",
		np.where(ma_change < -MA_SLOPE_MIN_CHANGE, "DOWN", "NEUTRAL")
	)

	return df


def detect_candlestick_reversal_pattern(df):
	"""
	Detect bearish reversal patterns that signal bull traps:
	- Shooting Star (long upper wick after uptrend)
	- Bearish Engulfing (large red candle engulfing previous green)
	- Evening Star (3-candle reversal)
	- Dark Cloud Cover (bearish reversal)
	"""
	df = df.copy()
	if not USE_CANDLESTICK_PATTERN_FILTER:
		df["bearish_reversal"] = False
		df["bullish_reversal"] = False
		return df

	# Sensitivity thresholds
	sensitivity_map = {
		"low": {"body_ratio": 0.7, "wick_ratio": 2.5},
		"medium": {"body_ratio": 0.6, "wick_ratio": 2.0},
		"high": {"body_ratio": 0.5, "wick_ratio": 1.5},
	}
	thresholds = sensitivity_map.get(PATTERN_FILTER_SENSITIVITY, sensitivity_map["medium"])

	o = df["open"]
	h = df["high"]
	l = df["low"]
	c = df["close"]

	# Calculate body and wick sizes
	body = abs(c - o)
	range_size = h - l
	upper_wick = h - np.maximum(o, c)
	lower_wick = np.minimum(o, c) - l

	# Shooting Star: Small body at bottom, long upper wick
	shooting_star = (
		(upper_wick > body * thresholds["wick_ratio"]) &
		(body > 0) &
		(lower_wick < body * 0.3)
	)

	# Bearish Engulfing: Current red candle engulfs previous green candle
	prev_close = c.shift(1)
	prev_open = o.shift(1)
	bearish_engulfing = (
		(prev_close > prev_open) &  # Previous was green
		(c < o) &  # Current is red
		(c < prev_open) &  # Current close below previous open
		(o > prev_close) &  # Current open above previous close
		(body > body.shift(1) * 1.2)  # Current body larger
	)

	# Combine patterns
	df["bearish_reversal"] = shooting_star | bearish_engulfing

	# Bullish reversal patterns (for short entries)
	hammer = (
		(lower_wick > body * thresholds["wick_ratio"]) &
		(body > 0) &
		(upper_wick < body * 0.3)
	)

	bullish_engulfing = (
		(prev_close < prev_open) &  # Previous was red
		(c > o) &  # Current is green
		(c > prev_open) &  # Current close above previous open
		(o < prev_close) &  # Current open below previous close
		(body > body.shift(1) * 1.2)
	)

	df["bullish_reversal"] = hammer | bullish_engulfing

	return df


def detect_divergence(df):
	"""
	Detect price/RSI divergence to identify potential bull/bear traps.

	Bearish divergence: Price makes higher high, RSI makes lower high (bull trap)
	Bullish divergence: Price makes lower low, RSI makes higher low (bear trap)
	"""
	df = df.copy()
	if not USE_DIVERGENCE_FILTER:
		df["bearish_divergence"] = False
		df["bullish_divergence"] = False
		return df

	# Calculate RSI if not already present
	if "rsi" not in df.columns:
		delta = df["close"].diff()
		gain = np.where(delta > 0, delta, 0.0)
		loss = np.where(delta < 0, -delta, 0.0)
		roll_gain = pd.Series(gain, index=df.index).rolling(DIVERGENCE_RSI_PERIOD).mean()
		roll_loss = pd.Series(loss, index=df.index).rolling(DIVERGENCE_RSI_PERIOD).mean()
		rs = roll_gain / roll_loss.replace(0, np.nan)
		df["rsi"] = 100 - (100 / (1 + rs))

	# Find local highs and lows in price and RSI
	close = df["close"]
	rsi = df["rsi"]

	# Rolling max/min over lookback period
	price_high = close.rolling(DIVERGENCE_LOOKBACK, center=True).max()
	price_low = close.rolling(DIVERGENCE_LOOKBACK, center=True).min()
	rsi_high = rsi.rolling(DIVERGENCE_LOOKBACK, center=True).max()
	rsi_low = rsi.rolling(DIVERGENCE_LOOKBACK, center=True).min()

	# Detect peaks/troughs
	is_price_peak = (close == price_high)
	is_price_trough = (close == price_low)

	# Bearish divergence: Price making higher highs, RSI making lower highs
	prev_price_peak = close.where(is_price_peak).ffill()
	prev_rsi_at_price_peak = rsi.where(is_price_peak).ffill()

	bearish_div = (
		is_price_peak &
		(close > prev_price_peak.shift(1)) &
		(rsi < prev_rsi_at_price_peak.shift(1)) &
		(rsi > 50)  # Only in overbought territory
	)

	# Bullish divergence: Price making lower lows, RSI making higher lows
	prev_price_trough = close.where(is_price_trough).ffill()
	prev_rsi_at_price_trough = rsi.where(is_price_trough).ffill()

	bullish_div = (
		is_price_trough &
		(close < prev_price_trough.shift(1)) &
		(rsi > prev_rsi_at_price_trough.shift(1)) &
		(rsi < 50)  # Only in oversold territory
	)

	df["bearish_divergence"] = bearish_div.fillna(False)
	df["bullish_divergence"] = bullish_div.fillna(False)

	return df


def prepare_symbol_dataframe(symbol, use_all_cached_data=False):
	"""
	Prepare symbol dataframe with indicators.

	Args:
		symbol: Trading pair
		use_all_cached_data: If True, load ALL cached historical data (for simulations)
	                         If False, use LOOKBACK limit (for live trading)
	"""
	limit = None if use_all_cached_data else LOOKBACK
	df = fetch_data(symbol, TIMEFRAME, limit)
	# Check for empty or invalid dataframe before processing
	if df.empty or "open" not in df.columns:
		print(f"[WARN] No data available for {symbol}, skipping...")
		return pd.DataFrame()
	df = attach_higher_timeframe_trend(df, symbol)
	# Debug: Check htf_indicator
	htf_valid = df["htf_indicator"].notna().sum() if "htf_indicator" in df.columns else 0
	print(f"[DEBUG] {symbol}: htf_indicator has {htf_valid}/{len(df)} valid values")
	df = attach_momentum_filter(df)
	df = attach_jma_trend_filter(df)
	df = attach_ma_slope_filter(df)
	df = detect_candlestick_reversal_pattern(df)
	df = detect_divergence(df)
	return df


def calculate_dynamic_min_min_hold_bars(
	symbol: str,
	recent_trades_df: pd.DataFrame = None,
	lookback_days: int = 30,
	min_days: int = 0,
	max_days: int = 7
) -> int:
	"""
	Calculate optimal minimum hold days dynamically based on recent performance and volatility.

	Strategy:
	- High volatility -> shorter hold times (let profits run, cut losses quickly)
	- Low volatility -> longer hold times (avoid whipsaws)
	- Recent losses -> increase hold time slightly (avoid overtrading)
	- Recent wins -> maintain or reduce hold time

	Args:
		symbol: Trading symbol
		recent_trades_df: DataFrame of recent trades (optional)
		lookback_days: How many days to look back for volatility
		min_days: Minimum hold days allowed
		max_days: Maximum hold days allowed

	Returns:
		Optimal minimum hold days (integer)
	"""
	try:
		# Fetch recent data for volatility calculation
		lookback_bars = max(100, lookback_days * BARS_PER_DAY)
		df = fetch_data(symbol, TIMEFRAME, lookback_bars)

		if df.empty or len(df) < 20:
			return min_days

		# Calculate ATR-based volatility score
		atr_series = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
		recent_atr = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
		avg_atr = float(atr_series.mean()) if not atr_series.empty else 0.0
		avg_price = float(df["close"].mean())

		if avg_price == 0 or avg_atr == 0:
			return min_days

		# Volatility ratio: higher means more volatile
		volatility_ratio = recent_atr / avg_atr if avg_atr > 0 else 1.0
		atr_pct = (avg_atr / avg_price) * 100  # ATR as % of price

		# Base calculation: Higher volatility -> lower hold days
		# ATR% ranges typically: 1-3% low vol, 3-7% medium vol, 7%+ high vol
		if atr_pct < 2.0:  # Very low volatility
			base_hold = max_days
		elif atr_pct < 4.0:  # Low-medium volatility
			base_hold = (min_days + max_days) // 2 + 1
		elif atr_pct < 7.0:  # Medium-high volatility
			base_hold = (min_days + max_days) // 2
		else:  # High volatility
			base_hold = min_days

		# Adjust based on recent volatility spike
		if volatility_ratio > 1.5:  # Volatility spike
			base_hold = max(min_days, base_hold - 1)
		elif volatility_ratio < 0.7:  # Volatility drop
			base_hold = min(max_days, base_hold + 1)

		# Adjust based on recent trade performance (if provided)
		if recent_trades_df is not None and not recent_trades_df.empty:
			# Look at last 10 trades
			recent = recent_trades_df.tail(10)
			if len(recent) >= 3:
				win_rate = len(recent[recent["pnl"] > 0]) / len(recent)
				avg_pnl = recent["pnl"].mean()

				# If losing streak, increase hold time to avoid overtrading
				if win_rate < 0.3:
					base_hold = min(max_days, base_hold + 1)
				# If winning streak with good profits, maintain or reduce
				elif win_rate > 0.6 and avg_pnl > 0:
					base_hold = max(min_days, base_hold)

		return int(np.clip(base_hold, min_days, max_days))

	except Exception as exc:
		print(f"[DynamicHold] Error calculating for {symbol}: {exc}")
		return min_days


def backtest_supertrend(df, atr_stop_mult=None, direction="long", min_hold_bars=0, symbol=None):
	direction = direction.lower()
	if direction not in {"long", "short"}:
		raise ValueError("direction must be 'long' or 'short'")
	min_hold_bars = 0 if min_hold_bars is None else max(0, int(min_hold_bars))
	# Symbol for lot size rounding (optional)
	_symbol = symbol or "BTCUSDT"  # Fallback

	long_mode = direction == "long"
	equity = START_EQUITY
	trades = []
	in_position = False
	entry_price = None
	entry_ts = None
	entry_capital = None
	entry_atr = None
	bars_in_position = 0

	for i in range(1, len(df)):
		ts = df.index[i]
		trend = int(df["trend_flag"].iloc[i])
		prev_trend = int(df["trend_flag"].iloc[i - 1])

		enter_long = prev_trend == -1 and trend == 1
		enter_short = prev_trend == 1 and trend == -1

		if not in_position:
			htf_value = int(df["htf_trend"].iloc[i]) if "htf_trend" in df.columns else 0
			htf_allows = True
			if USE_HIGHER_TIMEFRAME_FILTER:
				htf_allows = htf_value >= 1 if long_mode else htf_value <= -1

			momentum_allows = True
			if USE_MOMENTUM_FILTER and "momentum" in df.columns:
				mom_value = df["momentum"].iloc[i]
				if pd.isna(mom_value):
					momentum_allows = False
				else:
					momentum_allows = mom_value >= RSI_LONG_THRESHOLD if long_mode else mom_value <= RSI_SHORT_THRESHOLD

			breakout_allows = True
			if USE_BREAKOUT_FILTER:
				atr_curr = df["atr"].iloc[i]
				if atr_curr is None or np.isnan(atr_curr) or atr_curr <= 0:
					breakout_allows = False
				else:
					candle_range = float(df["high"].iloc[i] - df["low"].iloc[i])
					breakout_allows = candle_range >= BREAKOUT_ATR_MULT * float(atr_curr)
					if breakout_allows and BREAKOUT_REQUIRE_DIRECTION:
						prev_high = float(df["high"].iloc[i - 1]) if i > 0 else float(df["high"].iloc[i])
						prev_low = float(df["low"].iloc[i - 1]) if i > 0 else float(df["low"].iloc[i])
						close_curr = float(df["close"].iloc[i])
						breakout_allows = close_curr > prev_high if long_mode else close_curr < prev_low

			jma_trend_allows = True
			if USE_JMA_TREND_FILTER and "jma_trend_direction" in df.columns:
				trend_direction = df["jma_trend_direction"].iloc[i]
				if long_mode:
					jma_trend_allows = trend_direction == "UP"
				else:
					jma_trend_allows = trend_direction == "DOWN"

			# MA Slope Filter: Require MA to be trending in entry direction
			ma_slope_allows = True
			if USE_MA_SLOPE_FILTER and "ma_slope_direction" in df.columns:
				slope_dir = df["ma_slope_direction"].iloc[i]
				if long_mode:
					ma_slope_allows = slope_dir == "UP"
				else:
					ma_slope_allows = slope_dir == "DOWN"

			# Candlestick Pattern Filter: Avoid reversal patterns
			pattern_allows = True
			if USE_CANDLESTICK_PATTERN_FILTER:
				if long_mode and "bearish_reversal" in df.columns:
					# Don't enter long if bearish reversal detected
					pattern_allows = not df["bearish_reversal"].iloc[i]
				elif not long_mode and "bullish_reversal" in df.columns:
					# Don't enter short if bullish reversal detected
					pattern_allows = not df["bullish_reversal"].iloc[i]

			# Divergence Filter: Avoid entries during divergence
			divergence_allows = True
			if USE_DIVERGENCE_FILTER:
				if long_mode and "bearish_divergence" in df.columns:
					# Don't enter long if bearish divergence (bull trap signal)
					divergence_allows = not df["bearish_divergence"].iloc[i]
				elif not long_mode and "bullish_divergence" in df.columns:
					# Don't enter short if bullish divergence (bear trap signal)
					divergence_allows = not df["bullish_divergence"].iloc[i]

			if long_mode and enter_long and htf_allows and momentum_allows and breakout_allows and jma_trend_allows and ma_slope_allows and pattern_allows and divergence_allows:
				in_position = True
			elif (not long_mode) and enter_short and htf_allows and momentum_allows and breakout_allows and jma_trend_allows and ma_slope_allows and pattern_allows and divergence_allows:
				in_position = True

			if in_position:
				entry_price = float(df["close"].iloc[i])
				entry_ts = ts
				entry_capital = equity / STAKE_DIVISOR
				atr_val = df["atr"].iloc[i]
				entry_atr = float(atr_val) if not np.isnan(atr_val) else 0.0
				bars_in_position = 0
				# Initialize exit strategy tracking
				highest_price = entry_price if long_mode else entry_price
				lowest_price = entry_price if not long_mode else entry_price
				remaining_position = 1.0  # 100% of position
				partial_exits_taken = []  # Track which partial exit levels hit
			continue

		bars_in_position += 1
		stake = entry_capital if entry_capital is not None else equity / STAKE_DIVISOR
		current_stake = stake * remaining_position
		atr_buffer = entry_atr if entry_atr is not None else 0.0
		stop_price = None
		if atr_stop_mult is not None and atr_buffer and atr_buffer > 0:
			stop_price = entry_price - atr_stop_mult * atr_buffer if long_mode else entry_price + atr_stop_mult * atr_buffer

		# Track peak price for trailing stop
		current_price = float(df["close"].iloc[i])
		current_high = float(df["high"].iloc[i])
		current_low = float(df["low"].iloc[i])

		if long_mode:
			highest_price = max(highest_price, current_high)
		else:
			lowest_price = min(lowest_price, current_low)

		exit_price = None
		exit_reason = None
		partial_exit_amount = 0.0

		# Advanced Exit Strategies (checked BEFORE traditional exits)

		# 1. PROFIT TARGET - Full exit at target
		if USE_PROFIT_TARGET and exit_price is None:
			profit_pct = (current_price - entry_price) / entry_price if long_mode else (entry_price - current_price) / entry_price
			if profit_pct >= PROFIT_TARGET_PCT:
				exit_price = current_price
				exit_reason = f"Profit target {PROFIT_TARGET_PCT*100:.1f}%"

		# 2. PARTIAL EXITS at profit levels
		if USE_PARTIAL_EXIT and remaining_position > 0.4 and exit_price is None:
			profit_pct = (current_price - entry_price) / entry_price if long_mode else (entry_price - current_price) / entry_price
			for level_idx, level in enumerate(PARTIAL_EXIT_LEVELS):
				if level_idx not in partial_exits_taken and profit_pct >= level["profit_pct"]:
					# Take partial exit
					exit_amount = level["exit_pct"]
					partial_exit_amount = exit_amount
					remaining_position -= exit_amount
					partial_exits_taken.append(level_idx)
					# Record partial exit as separate trade
					price_diff = current_price - entry_price if long_mode else entry_price - current_price
					partial_stake = stake * exit_amount
					size_units = round_to_lot_size(partial_stake / entry_price, _symbol)
					fees = (entry_price + current_price) * size_units * FEE_RATE
					pnl_usd = size_units * price_diff - fees
					equity += pnl_usd
					trades.append({
						"Zeit": entry_ts,
						"Entry": entry_price,
						"ExitZeit": ts,
						"ExitPreis": current_price,
						"Stake": partial_stake,
						"Fees": fees,
						"ExitReason": f"Partial exit {level['profit_pct']*100:.0f}% ({exit_amount*100:.0f}%)",
						"PnL (USD)": pnl_usd,
						"Equity": equity,
						"Direction": direction.capitalize(),
						"MinHoldBars": min_hold_bars
					})
					if remaining_position <= 0.01:  # Essentially fully exited
						in_position = False
						entry_capital = None
						entry_atr = None
						bars_in_position = 0
						continue
					break

		# 3. TRAILING STOP - Exit on drawdown from peak
		if USE_TRAILING_STOP and exit_price is None:
			profit_pct = (current_price - entry_price) / entry_price if long_mode else (entry_price - current_price) / entry_price

			# Only activate trailing stop after minimum profit reached
			if profit_pct >= TRAILING_STOP_ACTIVATION_PCT:
				if long_mode:
					# Check drawdown from highest price
					drawdown_from_peak = (highest_price - current_low) / highest_price
					if drawdown_from_peak >= TRAILING_STOP_PCT:
						exit_price = current_price
						exit_reason = f"Trailing stop {TRAILING_STOP_PCT*100:.1f}%"
				else:
					# Short: check rise from lowest price
					rise_from_trough = (current_high - lowest_price) / lowest_price
					if rise_from_trough >= TRAILING_STOP_PCT:
						exit_price = current_price
						exit_reason = f"Trailing stop {TRAILING_STOP_PCT*100:.1f}%"

		# Traditional exits (ATR stop, Trend flip) - only if no advanced exit triggered
		if stop_price is not None and exit_price is None:
			if long_mode and float(df["low"].iloc[i]) <= stop_price:
				exit_price = stop_price
				exit_reason = "ATR stop"
			elif (not long_mode) and float(df["high"].iloc[i]) >= stop_price:
				exit_price = stop_price
				exit_reason = "ATR stop"

		if exit_price is None:
			if long_mode and prev_trend == 1 and trend == -1 and bars_in_position >= min_hold_bars:
				exit_price = float(df["close"].iloc[i])
				exit_reason = "Trend flip"
			elif (not long_mode) and prev_trend == -1 and trend == 1 and bars_in_position >= min_hold_bars:
				exit_price = float(df["close"].iloc[i])
				exit_reason = "Trend flip"

		if exit_price is None:
			continue

		# Calculate PnL for remaining position
		price_diff = exit_price - entry_price if long_mode else entry_price - exit_price
		size_units = round_to_lot_size(current_stake / entry_price, _symbol)
		fees = (entry_price + exit_price) * size_units * FEE_RATE
		pnl_usd = size_units * price_diff - fees
		equity += pnl_usd
		trades.append({
			"Zeit": entry_ts,
			"Entry": entry_price,
			"ExitZeit": ts,
			"ExitPreis": exit_price,
			"Stake": current_stake,
			"Fees": fees,
			"ExitReason": exit_reason,
			"PnL (USD)": pnl_usd,
			"Equity": equity,
			"Direction": direction.capitalize(),
			"MinHoldBars": min_hold_bars
		})
		in_position = False
		entry_capital = None
		entry_atr = None
		bars_in_position = 0

	if in_position:
		last = df.iloc[-1]
		exit_ts = last.name
		exit_price = float(last["close"])
		stake = entry_capital if entry_capital is not None else equity / STAKE_DIVISOR
		price_diff = exit_price - entry_price if long_mode else entry_price - exit_price
		size_units = round_to_lot_size(stake / entry_price, _symbol)
		fees = (entry_price + exit_price) * size_units * FEE_RATE
		pnl_usd = size_units * price_diff - fees
		equity += pnl_usd
		trades.append({
			"Zeit": entry_ts,
			"Entry": entry_price,
			"ExitZeit": exit_ts,
			"ExitPreis": exit_price,
			"Stake": stake,
			"Fees": fees,
			"ExitReason": "Final bar",
			"PnL (USD)": pnl_usd,
			"Equity": equity,
			"Direction": direction.capitalize(),
			"MinHoldBars": min_hold_bars
		})

	return pd.DataFrame(trades)


def backtest_htf_crossover(df, atr_stop_mult=None, direction="long", min_hold_bars=0, symbol=""):
	"""
	Backtest HTF Crossover Strategy
	Entry: Close crosses HTF indicator (upward for long, downward for short)
	Exit: Close crosses HTF indicator (opposite direction) or ATR stop
	"""
	direction = direction.lower()
	if direction not in {"long", "short"}:
		raise ValueError("direction must be 'long' or 'short'")
	min_hold_bars = 0 if min_hold_bars is None else max(0, int(min_hold_bars))
	_symbol = symbol or "BTCUSDT"  # Fallback for lot size lookup

	long_mode = direction == "long"
	equity = START_EQUITY
	trades = []
	in_position = False
	entry_price = None
	entry_ts = None
	entry_capital = None
	entry_atr = None
	bars_in_position = 0

	# Check if HTF indicator column exists
	if "indicator_line" not in df.columns:
		print("[Warning] HTF indicator column 'indicator_line' not found")
		return pd.DataFrame(trades)

	for i in range(1, len(df)):
		ts = df.index[i]
		curr = df.iloc[i]
		prev = df.iloc[i - 1]

		close_curr = float(curr["close"])
		close_prev = float(prev["close"])
		# Use HTF indicator for crossover (not main TF indicator_line)
		htf_col = "htf_indicator" if "htf_indicator" in df.columns else "indicator_line"

		# Skip if HTF indicator is NaN
		if pd.isna(curr[htf_col]) or pd.isna(prev[htf_col]):
			continue

		htf_curr = float(curr[htf_col])
		htf_prev = float(prev[htf_col])

		# Entry logic: Close crosses HTF indicator
		if not in_position:
			crossover_entry = False

			if long_mode:
				# Long entry: Close crosses above HTF
				if close_prev < htf_prev and close_curr > htf_curr:
					crossover_entry = True
			else:
				# Short entry: Close crosses below HTF
				if close_prev > htf_prev and close_curr < htf_curr:
					crossover_entry = True

			# Apply filters (HTF, momentum, breakout) like in trend_flip strategy
			if crossover_entry:
				htf_value = int(curr["htf_trend"]) if "htf_trend" in df.columns else 0
				htf_allows = True
				if USE_HIGHER_TIMEFRAME_FILTER:
					htf_allows = htf_value >= 1 if long_mode else htf_value <= -1

				momentum_allows = True
				if USE_MOMENTUM_FILTER and "momentum" in df.columns:
					mom_value = curr["momentum"]
					if pd.isna(mom_value):
						momentum_allows = False
					else:
						momentum_allows = mom_value >= RSI_LONG_THRESHOLD if long_mode else mom_value <= RSI_SHORT_THRESHOLD

				breakout_allows = True
				if USE_BREAKOUT_FILTER:
					atr_curr = curr["atr"]
					if atr_curr is None or np.isnan(atr_curr) or atr_curr <= 0:
						breakout_allows = False
					else:
						candle_range = float(curr["high"] - curr["low"])
						breakout_allows = candle_range >= BREAKOUT_ATR_MULT * float(atr_curr)
						if breakout_allows and BREAKOUT_REQUIRE_DIRECTION:
							prev_high = float(prev["high"])
							prev_low = float(prev["low"])
							breakout_allows = close_curr > prev_high if long_mode else close_curr < prev_low

				jma_trend_allows = True
				if USE_JMA_TREND_FILTER and "jma_trend_direction" in df.columns:
					trend_direction = curr["jma_trend_direction"]
					if long_mode:
						jma_trend_allows = trend_direction == "UP"
					else:
						jma_trend_allows = trend_direction == "DOWN"

				# MA Slope Filter
				ma_slope_allows = True
				if USE_MA_SLOPE_FILTER and "ma_slope_direction" in df.columns:
					slope_dir = curr["ma_slope_direction"]
					if long_mode:
						ma_slope_allows = slope_dir == "UP"
					else:
						ma_slope_allows = slope_dir == "DOWN"

				# Candlestick Pattern Filter
				pattern_allows = True
				if USE_CANDLESTICK_PATTERN_FILTER:
					if long_mode and "bearish_reversal" in df.columns:
						pattern_allows = not curr["bearish_reversal"]
					elif not long_mode and "bullish_reversal" in df.columns:
						pattern_allows = not curr["bullish_reversal"]

				# Divergence Filter
				divergence_allows = True
				if USE_DIVERGENCE_FILTER:
					if long_mode and "bearish_divergence" in df.columns:
						divergence_allows = not curr["bearish_divergence"]
					elif not long_mode and "bullish_divergence" in df.columns:
						divergence_allows = not curr["bullish_divergence"]

				if htf_allows and momentum_allows and breakout_allows and jma_trend_allows and ma_slope_allows and pattern_allows and divergence_allows:
					in_position = True
					entry_price = close_curr
					entry_ts = ts
					entry_capital = equity / STAKE_DIVISOR
					atr_val = curr["atr"]
					entry_atr = float(atr_val) if not np.isnan(atr_val) else 0.0
					bars_in_position = 0
					# Initialize exit strategy tracking
					highest_price = entry_price if long_mode else entry_price
					lowest_price = entry_price if not long_mode else entry_price
					remaining_position = 1.0
					partial_exits_taken = []
			continue

		bars_in_position += 1
		stake = entry_capital if entry_capital is not None else equity / STAKE_DIVISOR
		current_stake = stake * remaining_position
		atr_buffer = entry_atr if entry_atr is not None else 0.0
		stop_price = None
		if atr_stop_mult is not None and atr_buffer and atr_buffer > 0:
			stop_price = entry_price - atr_stop_mult * atr_buffer if long_mode else entry_price + atr_stop_mult * atr_buffer

		# Track peak for trailing stop
		if long_mode:
			highest_price = max(highest_price, float(curr["high"]))
		else:
			lowest_price = min(lowest_price, float(curr["low"]))

		exit_price = None
		exit_reason = None
		partial_exit_amount = 0.0

		# Advanced Exit Strategies
		# 1. PROFIT TARGET
		if USE_PROFIT_TARGET and exit_price is None:
			profit_pct = (close_curr - entry_price) / entry_price if long_mode else (entry_price - close_curr) / entry_price
			if profit_pct >= PROFIT_TARGET_PCT:
				exit_price = close_curr
				exit_reason = f"Profit target {PROFIT_TARGET_PCT*100:.1f}%"

		# 2. PARTIAL EXITS
		if USE_PARTIAL_EXIT and remaining_position > 0.4 and exit_price is None:
			profit_pct = (close_curr - entry_price) / entry_price if long_mode else (entry_price - close_curr) / entry_price
			for level_idx, level in enumerate(PARTIAL_EXIT_LEVELS):
				if level_idx not in partial_exits_taken and profit_pct >= level["profit_pct"]:
					exit_amount = level["exit_pct"]
					partial_exit_amount = exit_amount
					remaining_position -= exit_amount
					partial_exits_taken.append(level_idx)
					price_diff = close_curr - entry_price if long_mode else entry_price - close_curr
					partial_stake = stake * exit_amount
					size_units = round_to_lot_size(partial_stake / entry_price, _symbol)
					fees = (entry_price + close_curr) * size_units * FEE_RATE
					pnl_usd = size_units * price_diff - fees
					equity += pnl_usd
					trades.append({
						"Zeit": entry_ts,
						"Entry": entry_price,
						"ExitZeit": ts,
						"ExitPreis": close_curr,
						"Stake": partial_stake,
						"Fees": fees,
						"ExitReason": f"Partial exit {level['profit_pct']*100:.0f}% ({exit_amount*100:.0f}%)",
						"PnL (USD)": pnl_usd,
						"Equity": equity,
						"Direction": direction.capitalize(),
						"MinHoldBars": min_hold_bars
					})
					if remaining_position <= 0.01:
						in_position = False
						entry_capital = None
						entry_atr = None
						bars_in_position = 0
						continue
					break

		# 3. TRAILING STOP
		if USE_TRAILING_STOP and exit_price is None:
			profit_pct = (close_curr - entry_price) / entry_price if long_mode else (entry_price - close_curr) / entry_price
			if profit_pct >= TRAILING_STOP_ACTIVATION_PCT:
				if long_mode:
					drawdown_from_peak = (highest_price - float(curr["low"])) / highest_price
					if drawdown_from_peak >= TRAILING_STOP_PCT:
						exit_price = close_curr
						exit_reason = f"Trailing stop {TRAILING_STOP_PCT*100:.1f}%"
				else:
					rise_from_trough = (float(curr["high"]) - lowest_price) / lowest_price
					if rise_from_trough >= TRAILING_STOP_PCT:
						exit_price = close_curr
						exit_reason = f"Trailing stop {TRAILING_STOP_PCT*100:.1f}%"

		# Traditional exits - only if no advanced exit triggered
		# ATR stop has priority
		if stop_price is not None and exit_price is None:
			if long_mode and float(curr["low"]) <= stop_price:
				exit_price = stop_price
				exit_reason = "ATR stop"
			elif (not long_mode) and float(curr["high"]) >= stop_price:
				exit_price = stop_price
				exit_reason = "ATR stop"

		# HTF crossover exit (only if min_hold_bars satisfied)
		if exit_price is None and bars_in_position >= min_hold_bars:
			if long_mode:
				# Long exit: Close crosses below HTF
				if close_prev > htf_prev and close_curr < htf_curr:
					exit_price = close_curr
					exit_reason = "HTF crossover exit"
			else:
				# Short exit: Close crosses above HTF
				if close_prev < htf_prev and close_curr > htf_curr:
					exit_price = close_curr
					exit_reason = "HTF crossover exit"

		if exit_price is None:
			continue

		# Calculate PnL for remaining position
		price_diff = exit_price - entry_price if long_mode else entry_price - exit_price
		size_units = round_to_lot_size(current_stake / entry_price, _symbol)
		fees = (entry_price + exit_price) * size_units * FEE_RATE
		pnl_usd = size_units * price_diff - fees
		equity += pnl_usd
		trades.append({
			"Zeit": entry_ts,
			"Entry": entry_price,
			"ExitZeit": ts,
			"ExitPreis": exit_price,
			"Stake": current_stake,
			"Fees": fees,
			"ExitReason": exit_reason,
			"PnL (USD)": pnl_usd,
			"Equity": equity,
			"Direction": direction.capitalize(),
			"MinHoldBars": min_hold_bars
		})
		in_position = False
		entry_capital = None
		entry_atr = None
		bars_in_position = 0

	# Close open position at final bar
	if in_position:
		last = df.iloc[-1]
		exit_ts = last.name
		exit_price = float(last["close"])
		stake = entry_capital if entry_capital is not None else equity / STAKE_DIVISOR
		price_diff = exit_price - entry_price if long_mode else entry_price - exit_price
		size_units = round_to_lot_size(stake / entry_price, _symbol)
		fees = (entry_price + exit_price) * size_units * FEE_RATE
		pnl_usd = size_units * price_diff - fees
		equity += pnl_usd
		trades.append({
			"Zeit": entry_ts,
			"Entry": entry_price,
			"ExitZeit": exit_ts,
			"ExitPreis": exit_price,
			"Stake": stake,
			"Fees": fees,
			"ExitReason": "Final bar",
			"PnL (USD)": pnl_usd,
			"Equity": equity,
			"Direction": direction.capitalize(),
			"MinHoldBars": min_hold_bars
		})

	return pd.DataFrame(trades)


def performance_report(trades_df, symbol, param_a, param_b, direction, min_hold_bars):
	base = {
		"Symbol": symbol,
		"ParamA": param_a,
		"ParamB": param_b,
		PARAM_A_LABEL: param_a,
		PARAM_B_LABEL: param_b,
	}
	if INDICATOR_TYPE == "supertrend":
		base["Length"] = param_a
		base["Factor"] = param_b

	if trades_df.empty:
		return {
			**base,
			"Trades": 0,
			"WinRate": 0.0,
			"AvgPnL": 0.0,
			"ProfitFactor": 0.0,
			"MaxDrawdown": 0.0,
			"FinalEquity": START_EQUITY,
			"Direction": direction,
			"MinHoldBars": min_hold_bars,
		}

	wins = trades_df[trades_df["PnL (USD)"] > 0]
	losses = trades_df[trades_df["PnL (USD)"] < 0]
	win_rate = len(wins) / len(trades_df)
	avg_pnl = trades_df["PnL (USD)"].mean()
	total_win = wins["PnL (USD)"].sum()
	total_loss = abs(losses["PnL (USD)"].sum())
	profit_factor = (total_win / total_loss) if total_loss > 0 else np.inf
	equity_curve = trades_df["Equity"]
	max_drawdown = (equity_curve.cummax() - equity_curve).max() if not equity_curve.empty else 0.0
	final_eq = float(equity_curve.iloc[-1]) if not equity_curve.empty else START_EQUITY
	return {
		**base,
		"Trades": len(trades_df),
		"WinRate": win_rate,
		"AvgPnL": avg_pnl,
		"ProfitFactor": profit_factor,
		"MaxDrawdown": max_drawdown,
		"FinalEquity": final_eq,
		"Direction": direction,
		"MinHoldBars": min_hold_bars,
	}


def build_equity_series(df, trades_df, direction):
	if df.empty:
		return pd.Series(dtype=float)
	equity_series = pd.Series(index=df.index, dtype=float)
	equity_series.iloc[0] = START_EQUITY
	if trades_df.empty:
		return equity_series.ffill()
	current_equity = START_EQUITY
	direction_key = str(direction).lower()
	for trade in trades_df.to_dict("records"):
		entry_ts = trade.get("Zeit")
		exit_ts = trade.get("ExitZeit")
		stake = float(trade.get("Stake", START_EQUITY / STAKE_DIVISOR))
		entry_price = float(trade.get("Entry", 0))
		if not entry_ts or not exit_ts or entry_price == 0:
			continue
		entry_idx = df.index.get_indexer([entry_ts])
		exit_idx = df.index.get_indexer([exit_ts])
		if entry_idx.size == 0 or exit_idx.size == 0:
			continue
		entry_pos = entry_idx[0]
		exit_pos = exit_idx[0]
		if entry_pos == -1 or exit_pos == -1 or exit_pos < entry_pos:
			continue
		price_slice = df.iloc[entry_pos:exit_pos + 1]["close"]
		baseline_equity = current_equity
		for ts, close_price in price_slice.items():
			if ts == exit_ts:
				equity_at_exit = float(trade.get("Equity", baseline_equity))
				equity_series.loc[ts] = equity_at_exit
				current_equity = equity_at_exit
				break
			close_val = float(close_price)
			if direction_key == "short":
				unrealized = (entry_price - close_val) / entry_price * stake
			else:
				unrealized = (close_val - entry_price) / entry_price * stake
			equity_series.loc[ts] = baseline_equity + unrealized
	return equity_series.ffill().fillna(START_EQUITY)


def build_two_panel_figure(symbol, df, trades_df, param_a, param_b, direction, min_hold_bars=None):
	direction_title = direction.capitalize()
	hold_text = f", Hold≥{min_hold_bars} bars" if min_hold_bars else ""
	indicator_desc = f"{INDICATOR_DISPLAY_NAME} {PARAM_A_LABEL}={param_a}, {PARAM_B_LABEL}={param_b}"
	line_name = "Supertrend" if INDICATOR_TYPE == "supertrend" else INDICATOR_DISPLAY_NAME
	fig = make_subplots(
		rows=3,
		cols=1,
		shared_xaxes=True,
		vertical_spacing=0.06,
		row_heights=[0.55, 0.3, 0.15],
		subplot_titles=(
			f"{symbol} {direction_title} {indicator_desc}{hold_text}",
			"Equity",
			"Momentum",
		),
	)

	fig.add_trace(
		go.Candlestick(
			x=df.index,
			open=df["open"],
			high=df["high"],
			low=df["low"],
			close=df["close"],
			name="Price",
		),
		row=1,
		col=1,
	)
	fig.add_trace(
		go.Scatter(
			x=df.index,
			y=df["indicator_line"],
			mode="lines",
			name=line_name,
			line=dict(color="orange"),
		),
		row=1,
		col=1,
	)
	if USE_HIGHER_TIMEFRAME_FILTER and "htf_indicator" in df.columns:
		fig.add_trace(
			go.Scatter(
				x=df.index,
				y=df["htf_indicator"],
				mode="lines",
				name=f"HTF {INDICATOR_DISPLAY_NAME} ({HIGHER_TIMEFRAME})",
				line=dict(color="purple", dash="dot"),
			),
			row=1,
			col=1,
		)

	if not trades_df.empty:
		entry_color = "green" if direction_title == "Long" else "red"
		exit_color = "red" if direction_title == "Long" else "green"
		entry_symbol = "triangle-up" if direction_title == "Long" else "triangle-down"
		exit_symbol = "triangle-down" if direction_title == "Long" else "triangle-up"
		fig.add_trace(
			go.Scatter(
				x=trades_df["Zeit"],
				y=trades_df["Entry"],
				mode="markers",
				marker=dict(color=entry_color, symbol=entry_symbol, size=10),
				name=f"{direction_title} Entry",
			),
			row=1,
			col=1,
		)
		fig.add_trace(
			go.Scatter(
				x=trades_df["ExitZeit"],
				y=trades_df["ExitPreis"],
				mode="markers",
				marker=dict(color=exit_color, symbol=exit_symbol, size=10),
				name=f"{direction_title} Exit",
			),
			row=1,
			col=1,
		)

		equity_series = build_equity_series(df, trades_df, direction_title)
		fig.add_trace(
			go.Scatter(x=equity_series.index, y=equity_series, mode="lines", name="Equity"),
			row=2,
			col=1,
		)
	else:
		equity_series = pd.Series(index=df.index, data=START_EQUITY, dtype=float) if len(df.index) else None
		if equity_series is not None:
			fig.add_trace(
				go.Scatter(x=equity_series.index, y=equity_series, mode="lines", name="Equity"),
				row=2,
				col=1,
			)

	if "momentum" in df.columns:
		fig.add_trace(
			go.Scatter(x=df.index, y=df["momentum"], mode="lines", name="Momentum", line=dict(color="teal")),
			row=3,
			col=1,
		)
		fig.add_hrect(
			y0=RSI_SHORT_THRESHOLD,
			y1=RSI_LONG_THRESHOLD,
			line_width=0,
			fillcolor="gray",
			opacity=0.15,
			row=3,
			col=1,
		)

	fig.update_layout(
		height=900,
		showlegend=True,
		xaxis=dict(rangeslider=dict(visible=False)),
		xaxis2=dict(rangeslider=dict(visible=False)),
		xaxis3=dict(rangeslider=dict(visible=True, thickness=0.03), type="date"),
	)
	fig.update_xaxes(title_text="Zeit", row=3, col=1)
	fig.update_yaxes(title_text="Preis", row=1, col=1)
	fig.update_yaxes(title_text="Equity (USD)", row=2, col=1)
	fig.update_yaxes(title_text="RSI", row=3, col=1)
	return fig


def df_to_html_table(df, title=None):
	html = ""
	if title:
		html += f"<h3>{title}</h3>\n"
	if df.empty:
		html += "<p>Keine Daten</p>"
	else:
		html += df.to_html(index=False, justify="left", border=0)
	return html


def _format_result_cell(entry):
	if not entry:
		return "-"
	win_pct = float(entry.get("WinRate", 0.0)) * 100.0
	max_dd = float(entry.get("MaxDrawdown", 0.0))
	return (
		f"HTF {entry.get('HTF', '-')}, Eq {entry.get('FinalEquity', START_EQUITY):.0f}, "
		f"Trades {entry.get('Trades', 0)}, Win {win_pct:.1f}%, DD {max_dd:.0f}"
	)


def record_global_best(indicator_key, summary_rows):
	if not summary_rows:
		return
	indicator_store = GLOBAL_BEST_RESULTS.setdefault(indicator_key, {})
	for row in summary_rows:
		symbol = row.get("Symbol")
		direction = str(row.get("Direction", "Long")).lower()
		if direction not in {"long", "short"} or not symbol:
			continue
		symbol_store = indicator_store.setdefault(symbol, {})
		existing = symbol_store.get(direction)
		candidate_equity = float(row.get("FinalEquity", START_EQUITY))
		existing_equity = float(existing.get("FinalEquity", START_EQUITY)) if existing else None
		if existing is None or candidate_equity > existing_equity:
			symbol_store[direction] = dict(row)


def write_overall_result_tables():
	if not GLOBAL_BEST_RESULTS:
		return
	indicator_order = list(INDICATOR_PRESETS.keys())
	indicator_labels = {key: INDICATOR_PRESETS[key]["display_name"] for key in indicator_order}
	long_rows = []
	short_rows = []
	for symbol in SYMBOLS:
		long_row = {"Symbol": symbol}
		short_row = {"Symbol": symbol}
		for key in indicator_order:
			entry_long = GLOBAL_BEST_RESULTS.get(key, {}).get(symbol, {}).get("long")
			entry_short = GLOBAL_BEST_RESULTS.get(key, {}).get(symbol, {}).get("short")
			col_name = indicator_labels[key]
			long_row[col_name] = _format_result_cell(entry_long)
			short_row[col_name] = _format_result_cell(entry_short)
		long_rows.append(long_row) 
		short_rows.append(short_row)
	long_df = pd.DataFrame(long_rows)
	short_df = pd.DataFrame(short_rows)
	os.makedirs(BASE_OUT_DIR, exist_ok=True)
	now = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
	html_parts = [
		"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Overall Indicator Results</title></head><body>",
		f"<h1>Overall Indicator Results</h1><p>Stand: {now}</p>",
		"<h2>Long Ergebnisse</h2>",
		long_df.to_html(index=False, justify="left", border=0) if not long_df.empty else "<p>Keine Daten</p>",
		"<h2>Short Ergebnisse</h2>",
		short_df.to_html(index=False, justify="left", border=0) if not short_df.empty else "<p>Keine Daten</p>",
		"</body></html>",
	]
	with open(OVERALL_SUMMARY_HTML, "w", encoding="utf-8") as f:
		f.write("\n".join(html_parts))
	csv_rows = []
	for key in indicator_order:
		indicator_store = GLOBAL_BEST_RESULTS.get(key, {})
		for symbol, dir_dict in indicator_store.items():
			for direction, entry in dir_dict.items():
				row = dict(entry)
				row["Indicator"] = key
				row["IndicatorDisplay"] = indicator_labels[key]
				row["Direction"] = direction.capitalize()
				csv_rows.append(row)
	if csv_rows:
		pd.DataFrame(csv_rows).to_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",", index=False, encoding="utf-8")


def write_combined_overall_best_report(sections):
	if not sections:
		return
	best_per_symbol = {}
	for item in sections:
		symbol = item.get("symbol")
		if not symbol:
			continue
		value = item.get("final_equity")
		try:
			value = float(value)
		except (TypeError, ValueError):
			value = float("-inf")
		current = best_per_symbol.get(symbol)
		if current is None or value > current[0]:
			best_per_symbol[symbol] = (value, item)
	sections = [entry for (_, entry) in best_per_symbol.values()]
	sections.sort(key=lambda item: item.get("symbol", ""))
	long_entries = [s for s in sections if s.get("direction", "").lower() == "long"]
	short_entries = [s for s in sections if s.get("direction", "").lower() == "short"]
	now = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
	html = [
		"<!DOCTYPE html>",
		"<html><head><meta charset='utf-8'><title>Overall-Best Detailreport</title>",
		"<style>body{font-family:Arial, sans-serif;margin:20px;}h2{margin-top:40px;}section{margin-bottom:50px;}table{margin-top:10px;}hr{margin:40px 0;}</style>",
		"</head><body>",
		f"<h1>Overall-Best Detailreport</h1><p>Stand: {now}</p>",
	]

	def render_entry(entry):
		meta = (
			f"<h3>{entry['symbol']} – {entry['direction']} – {entry['indicator']} ({entry['htf']})<br>"
			f"{entry['param_desc']}, ATRStop={entry['atr_label']}, MinHold={entry['min_min_hold_bars']}d</h3>"
		)
		html.append("<section>")
		html.append(meta)
		html.append(entry["fig_html"])
		html.append(entry["trade_table_html"])
		html.append("</section>")

	if long_entries:
		html.append("<h2>Long Trades</h2>")
		for item in long_entries:
			render_entry(item)
	if short_entries:
		html.append("<h2>Short Trades</h2>")
		for item in short_entries:
			render_entry(item)

	html.append("</body></html>")
	with open(OVERALL_DETAILED_HTML, "w", encoding="utf-8") as f:
		f.write("\n".join(html))


def write_flat_trade_list(rows):
	os.makedirs(BASE_OUT_DIR, exist_ok=True)
	flat_columns = [
		"Indicator",
		"IndicatorDisplay",
		"HTF",
		"Symbol",
		"Direction",
		"ParamA",
		"ParamB",
		PARAM_A_LABEL,
		PARAM_B_LABEL,
		"ATRStopMultValue",
		"ATRStopMult",
		"MinHoldBars",
		"FinalEquity",
		"Trades",
		"WinRate",
		"MaxDrawdown",
		"TradesCSV",
	]
	if not rows:
		flat_df = pd.DataFrame(columns=flat_columns)
	else:
		flat_df = pd.DataFrame(rows)
		for col in flat_columns:
			if col not in flat_df.columns:
				flat_df[col] = ""
		flat_df = flat_df.sort_values(["Indicator", "HTF", "Symbol", "Direction"]).reset_index(drop=True)
	flat_df = flat_df.reindex(columns=flat_columns)
	csv_path = OVERALL_FLAT_CSV
	json_path = OVERALL_FLAT_JSON
	try:
		flat_df.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
	except PermissionError as exc:
		timestamp = datetime.now(BERLIN_TZ).strftime("%Y%m%d_%H%M%S")
		csv_path = os.path.join(BASE_OUT_DIR, f"overall_best_flat_trades_{timestamp}.csv")
		flat_df.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
		print(f"[Flat] {exc}. Wrote fallback CSV {csv_path}")
	try:
		records = json.loads(flat_df.to_json(orient="records"))
		with open(json_path, "w", encoding="utf-8") as fh:
			json.dump(records, fh, ensure_ascii=False, indent=2)
	except PermissionError as exc:
		timestamp = datetime.now(BERLIN_TZ).strftime("%Y%m%d_%H%M%S")
		json_path = os.path.join(BASE_OUT_DIR, f"overall_best_flat_trades_{timestamp}.json")
		with open(json_path, "w", encoding="utf-8") as fh:
			json.dump(records, fh, ensure_ascii=False, indent=2)
		print(f"[Flat] {exc}. Wrote fallback JSON {json_path}")
	print(f"[Flat] Saved {len(flat_df)} trade definitions to {csv_path}")


def build_full_report(figs_html_blocks, sections_html, ranking_tables_html):
	html = []
	page_title = f"{INDICATOR_DISPLAY_NAME} Parameter Report ({HIGHER_TIMEFRAME})"
	html.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{page_title}</title></head><body>")
	html.append(f"<h1>{page_title}</h1>")
	now = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
	html.append(f"<p>Generiert: {now}</p>")

	for blk in figs_html_blocks:
		html.append(blk)
	for sec in sections_html:
		html.append(sec)

	html.append("<hr>")
	html.append("<h2>Parameter-Ranking je Symbol</h2>")
	for rtbl in ranking_tables_html:
		html.append(rtbl)

	html.append("</body></html>")
	return "\n".join(html)


def _normalize_param_values(param_a, param_b):
	if pd.isna(param_a):
		param_a = DEFAULT_PARAM_A
	if pd.isna(param_b):
		param_b = DEFAULT_PARAM_B
	if INDICATOR_TYPE in {"supertrend", "jma", "kama"}:
		param_a = int(round(float(param_a)))
	else:
		param_a = float(param_a)
	if INDICATOR_TYPE in {}:
		param_b = int(round(float(param_b)))
	else:
		param_b = float(param_b)
	return param_a, param_b


def _normalize_atr_value(raw_value):
	if isinstance(raw_value, str):
		raw_str = raw_value.strip().lower()
		if not raw_str or raw_str == "none":
			return None
		return float(raw_str)
	if pd.isna(raw_value):
		return None
	return float(raw_value)


def _run_saved_rows(rows_df, table_title, save_path=None, aggregate_sections=None):
	if rows_df is None or rows_df.empty:
		print("[Skip] No saved parameter rows to execute.")
		return []
	rows_df = rows_df.copy()
	print(f"[Run] {table_title} – {len(rows_df)} gespeicherte Kombinationen")
	figs_blocks = []
	sections_blocks = []
	ranking_tables = []
	data_cache = {}
	st_cache = {}
	updated_rows = []
	for _, row in rows_df.iterrows():
		row_dict = row.to_dict()
		symbol = row.get("Symbol")
		if not symbol:
			continue
		direction = str(row.get("Direction", "Long")).lower()
		if direction not in {"long", "short"}:
			continue
		if direction == "long" and not ENABLE_LONGS:
			continue
		if direction == "short" and not ENABLE_SHORTS:
			continue
		param_a = row.get("ParamA", DEFAULT_PARAM_A)
		param_b = row.get("ParamB", DEFAULT_PARAM_B)
		param_a, param_b = _normalize_param_values(param_a, param_b)
		atr_mult = row.get("ATRStopMultValue", row.get("ATRStopMult"))
		atr_mult = _normalize_atr_value(atr_mult)
		min_hold_bars = row.get("MinHoldBars", DEFAULT_MIN_HOLD_BARS)
		if pd.isna(min_hold_bars):
			min_hold_bars = DEFAULT_MIN_HOLD_BARS
		else:
			min_hold_bars = int(min_hold_bars)
		if symbol not in data_cache:
			data_cache[symbol] = prepare_symbol_dataframe(symbol)
		df_raw = data_cache[symbol]
		st_key = (symbol, param_a, param_b)
		if st_key not in st_cache:
			df_tmp = compute_indicator(df_raw, param_a, param_b)
			for col in ("htf_trend", "htf_indicator", "momentum"):
				if col in df_raw.columns:
					df_tmp[col] = df_raw[col]
			st_cache[st_key] = df_tmp
		df_st = st_cache[st_key]

		# Select backtest function based on indicator type
		if INDICATOR_TYPE == "htf_crossover":
			trades = backtest_htf_crossover(
				df_st,
				atr_stop_mult=atr_mult,
				direction=direction,
				min_hold_bars=min_hold_bars,
				symbol=symbol,
			)
		else:  # Default: trend_flip for all other indicators
			trades = backtest_supertrend(
				df_st,
				atr_stop_mult=atr_mult,
				direction=direction,
				min_hold_bars=min_hold_bars,
			)
		direction_title = direction.capitalize()
		atr_label = "None" if atr_mult is None else atr_mult
		param_desc = f"{PARAM_A_LABEL}={param_a}, {PARAM_B_LABEL}={param_b}"
		print(f"  · {symbol} {direction_title} ({param_desc}, ATR={atr_label}, MinHold={min_hold_bars} bars)")
		stats = performance_report(
			trades,
			symbol,
			param_a,
			param_b,
			direction_title,
			min_hold_bars,
		)
		updated_row = dict(row_dict)
		updated_row.update({
			"ParamA": param_a,
			"ParamB": param_b,
			PARAM_A_LABEL: param_a,
			PARAM_B_LABEL: param_b,
			"MinHoldBars": min_hold_bars,
			"ATRStopMult": atr_label,
			"ATRStopMultValue": atr_mult,
			"HTF": HIGHER_TIMEFRAME,
			"Trades": stats["Trades"],
			"WinRate": stats["WinRate"],
			"AvgPnL": stats["AvgPnL"],
			"ProfitFactor": stats["ProfitFactor"],
			"MaxDrawdown": stats["MaxDrawdown"],
			"FinalEquity": stats["FinalEquity"],
		})
		updated_rows.append(updated_row)
		fig = build_two_panel_figure(
			symbol,
			df_st,
			trades,
			param_a,
			param_b,
			direction_title,
			min_hold_bars=min_hold_bars,
		)
		fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
		figs_blocks.append(
			f"<h2>{symbol} – {direction_title} gespeicherte Parameter: {param_desc}, ATRStop={atr_label}, MinHold={min_hold_bars} bars</h2>\n"
			+ fig_html
		)
		trade_table_html = df_to_html_table(
			trades,
			title=f"Trade-Liste {symbol} ({direction_title} gespeicherte Parameter, MinHold={min_hold_bars} bars)",
		)
		sections_blocks.append(trade_table_html)
		csv_suffix = "" if direction == "long" else "_short"
		csv_path = os.path.join(OUT_DIR, f"trades_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_best{csv_suffix}.csv")
		os.makedirs(OUT_DIR, exist_ok=True)
		trades.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
		updated_row["TradesCSV"] = csv_path
		if aggregate_sections is not None:
			aggregate_sections.append({
				"indicator": INDICATOR_DISPLAY_NAME,
				"indicator_slug": INDICATOR_SLUG,
				"htf": HIGHER_TIMEFRAME,
				"symbol": symbol,
				"direction": direction_title,
				"param_desc": param_desc,
				"atr_label": atr_label,
				"min_min_hold_bars": min_hold_bars,
				"fig_html": fig_html,
				"trade_table_html": trade_table_html,
				"final_equity": stats.get("FinalEquity", START_EQUITY),
			})
	updated_df = pd.DataFrame(updated_rows) if updated_rows else rows_df
	if updated_df is not None:
		ranking_tables = [df_to_html_table(updated_df, title=table_title)]
	if figs_blocks or sections_blocks:
		report_html = build_full_report(figs_blocks, sections_blocks, ranking_tables)
		report_path = os.path.join(OUT_DIR, REPORT_FILE)
		with open(report_path, "w", encoding="utf-8") as f:
			f.write(report_html)
	if save_path and updated_df is not None and not updated_df.empty:
		dir_name = os.path.dirname(save_path) or "."
		os.makedirs(dir_name, exist_ok=True)
		updated_df.to_csv(save_path, sep=";", decimal=",", index=False, encoding="utf-8")
	return updated_df.to_dict("records") if updated_df is not None else []


def ensure_cache_populated(symbols, timeframe, min_bars):
	"""Ensure OHLCV cache has enough data for all symbols before running sweep."""
	import time

	# Fixed start date: 2024-05-01 for historical sweep data
	start_date = pd.Timestamp("2024-05-01", tz=BERLIN_TZ)

	# Only download Binance-supported timeframes
	# Unsupported timeframes (3h, 5h, 7h, etc.) will be synthesized from 1h data by fetch_data()
	binance_supported_tf = ["1h", "2h", "4h", "6h", "8h", "12h", "1d"]

	print(f"\n[Cache Init] Checking cache for {len(symbols)} symbols")
	print(f"[Cache Init] Main TF: {timeframe} needs {min_bars} bars")
	print(f"[Cache Init] Downloading Binance-supported TFs: {binance_supported_tf}")
	print(f"[Cache Init] (Other TFs like 3h, 5h, etc. will be synthesized from 1h data)")
	print(f"[Cache Init] Will download from {start_date.strftime('%Y-%m-%d')} if needed\n")

	for symbol in symbols:
		# Check and populate main timeframe cache
		cached_df = load_ohlcv_from_cache(symbol, timeframe)
		cached_bars = len(cached_df) if cached_df is not None else 0

		if cached_bars >= min_bars:
			print(f"[Cache Init] {symbol} {timeframe}: OK ({cached_bars} bars)")
		else:
			print(f"[Cache Init] {symbol} {timeframe}: Need data (only {cached_bars} bars, need {min_bars})")
			df = download_historical_ohlcv(symbol, timeframe, start_date)
			if not df.empty:
				save_ohlcv_to_cache(symbol, timeframe, df)
				print(f"[Cache Init] {symbol} {timeframe}: Downloaded {len(df)} bars")
			else:
				print(f"[Cache Init] {symbol} {timeframe}: WARNING - No data!")
			time.sleep(0.5)

		# Also cache Binance-supported HTF data for this symbol
		for htf in binance_supported_tf:
			if htf == timeframe:
				continue  # Already handled above

			htf_cached = load_ohlcv_from_cache(symbol, htf)
			htf_bars = len(htf_cached) if htf_cached is not None else 0

			if htf_bars >= HTF_LOOKBACK:
				continue  # HTF cache OK, skip silently

			print(f"[Cache Init] {symbol} {htf}: Downloading HTF data...")
			df = download_historical_ohlcv(symbol, htf, start_date)
			if not df.empty:
				save_ohlcv_to_cache(symbol, htf, df)
				print(f"[Cache Init] {symbol} {htf}: Downloaded {len(df)} bars")
			time.sleep(0.3)

	print("\n[Cache Init] Cache check complete\n")


def run_parameter_sweep():
	figs_blocks = []
	sections_blocks = []
	ranking_tables = []
	best_params_summary = []
	clear_directory(OUT_DIR)

	# Ensure cache is populated before running sweep
	ensure_cache_populated(SYMBOLS, TIMEFRAME, LOOKBACK)

	directions = get_enabled_directions()
	hold_bar_candidates = MIN_HOLD_BAR_VALUES if USE_MIN_HOLD_FILTER else [DEFAULT_MIN_HOLD_BARS]

	for symbol in SYMBOLS:
		df_raw = prepare_symbol_dataframe(symbol)
		results = {d: [] for d in directions}
		trades_per_combo = {d: {} for d in directions}
		df_cache = {}

		for param_a in PARAM_A_VALUES:
			for param_b in PARAM_B_VALUES:
				cache_key = (param_a, param_b)
				if cache_key not in df_cache:
					df_tmp = compute_indicator(df_raw, param_a, param_b)
					for col in ("htf_trend", "htf_indicator", "momentum"):
						if col in df_raw.columns:
							df_tmp[col] = df_raw[col]
					df_cache[cache_key] = df_tmp
				df_st = df_cache[cache_key]
				for atr_mult in ATR_STOP_MULTS:
					for min_hold_bars in hold_bar_candidates:
						for direction in directions:
							df_st_with_htf = df_st.copy()
							for col in ("htf_trend", "htf_indicator", "momentum"):
								if col in df_raw.columns:
									df_st_with_htf[col] = df_raw[col]
							# Select backtest function based on indicator type
							if INDICATOR_TYPE == "htf_crossover":
								trades = backtest_htf_crossover(
									df_st_with_htf,
									atr_stop_mult=atr_mult,
									direction=direction,
									min_hold_bars=min_hold_bars,
									symbol=symbol,
								)
							else:  # Default: trend_flip for all other indicators
								trades = backtest_supertrend(
									df_st_with_htf,
									atr_stop_mult=atr_mult,
									direction=direction,
									min_hold_bars=min_hold_bars,
								)
							stats = performance_report(
								trades,
								symbol,
								param_a,
								param_b,
								direction.capitalize(),
								min_hold_bars,
							)
							stats["ATRStopMult"] = atr_mult if atr_mult is not None else "None"
							stats["MinHoldBars"] = min_hold_bars
							results[direction].append(stats)
							trades_per_combo[direction][(param_a, param_b, atr_mult, min_hold_bars)] = trades

		for direction in directions:
			dir_results = results[direction]
			ranking_df = pd.DataFrame(dir_results)
			if not ranking_df.empty:
				ranking_df = ranking_df.sort_values("FinalEquity", ascending=False).reset_index(drop=True)
			ranking_tables.append(
				df_to_html_table(
					ranking_df,
					title=f"Ranking: {symbol} {INDICATOR_DISPLAY_NAME} ({direction.capitalize()} nach FinalEquity)",
				)
			)

			best_param_a, best_param_b = DEFAULT_PARAM_A, DEFAULT_PARAM_B
			best_atr = None
			best_hold_bars = DEFAULT_MIN_HOLD_BARS
			final_equity = START_EQUITY
			trades_count = 0
			win_rate = 0.0
			max_dd = 0.0
			if not ranking_df.empty:
				best_row = ranking_df.iloc[0]
				best_param_a = best_row.get("ParamA", best_row.get(PARAM_A_LABEL, DEFAULT_PARAM_A))
				best_param_b = best_row.get("ParamB", best_row.get(PARAM_B_LABEL, DEFAULT_PARAM_B))
				best_param_a = best_param_a if not pd.isna(best_param_a) else DEFAULT_PARAM_A
				best_param_b = best_param_b if not pd.isna(best_param_b) else DEFAULT_PARAM_B
				best_atr_raw = best_row.get("ATRStopMult", "None")
				best_atr = best_atr_raw if best_atr_raw != "None" else None
				best_hold_bars = int(best_row.get("MinHoldBars", DEFAULT_MIN_HOLD_BARS))
				final_equity = float(best_row.get("FinalEquity", START_EQUITY))
				trades_count = int(best_row.get("Trades", 0))
				win_rate = float(best_row.get("WinRate", 0.0))
				max_dd = float(best_row.get("MaxDrawdown", 0.0))
				best_df = df_cache[(best_param_a, best_param_b)]
				best_trades = trades_per_combo[direction][(best_param_a, best_param_b, best_atr, best_hold_bars)]
			else:
				best_df = compute_indicator(df_raw, best_param_a, best_param_b)
				for col in ("htf_trend", "htf_indicator", "momentum"):
					if col in df_raw.columns:
						best_df[col] = df_raw[col]
				best_trades = pd.DataFrame()

			atr_label = best_atr if best_atr is not None else "None"
			best_params_summary.append({
				"Symbol": symbol,
				"Direction": direction.capitalize(),
				"Indicator": INDICATOR_TYPE,
				"IndicatorDisplay": INDICATOR_DISPLAY_NAME,
				"ParamA": best_param_a,
				"ParamB": best_param_b,
				PARAM_A_LABEL: best_param_a,
				PARAM_B_LABEL: best_param_b,
				"Length": best_param_a if INDICATOR_TYPE == "supertrend" else None,
				"Factor": best_param_b if INDICATOR_TYPE == "supertrend" else None,
				"ATRStopMult": atr_label,
				"ATRStopMultValue": best_atr,
				"MinHoldBars": best_hold_bars,
				"HTF": HIGHER_TIMEFRAME,
				"FinalEquity": final_equity,
				"Trades": trades_count,
				"WinRate": win_rate,
				"MaxDrawdown": max_dd,
			})

			fig = build_two_panel_figure(
				symbol,
				best_df,
				best_trades,
				best_param_a,
				best_param_b,
				direction.capitalize(),
				min_hold_bars=best_hold_bars,
			)
			fig_html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
			figs_blocks.append(
				f"<h2>{symbol} – {direction.capitalize()} beste Parameter: {PARAM_A_LABEL}={best_param_a}, {PARAM_B_LABEL}={best_param_b}, ATRStop={atr_label}, MinHold={best_hold_bars} bars</h2>\n"
				+ fig_html
			)

			sections_blocks.append(
				df_to_html_table(
					best_trades,
					title=f"Trade-Liste {symbol} ({direction.capitalize()} beste Parameter, MinHold={best_hold_bars} bars)",
				)
			)

			csv_suffix = "" if direction == "long" else "_short"
			csv_path = os.path.join(OUT_DIR, f"trades_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_best{csv_suffix}.csv")
			best_trades.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
			csv_rank_suffix = "" if direction == "long" else "_short"
			csv_rank_path = os.path.join(OUT_DIR, f"ranking_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_params{csv_rank_suffix}.csv")
			ranking_df.to_csv(csv_rank_path, sep=";", decimal=",", index=False, encoding="utf-8")

	if best_params_summary:
		summary_df = pd.DataFrame(best_params_summary)
		summary_path = os.path.join(OUT_DIR, BEST_PARAMS_FILE)
		summary_df.to_csv(summary_path, sep=";", decimal=",", index=False, encoding="utf-8")

	report_html = build_full_report(figs_blocks, sections_blocks, ranking_tables)
	report_path = os.path.join(OUT_DIR, REPORT_FILE)
	with open(report_path, "w", encoding="utf-8") as f:
		f.write(report_html)
	return best_params_summary


def run_saved_params(rows_df=None):
	summary_df = rows_df
	summary_path = os.path.join(OUT_DIR, BEST_PARAMS_FILE)
	if summary_df is None:
		if os.path.exists(summary_path):
			summary_df = pd.read_csv(summary_path, sep=";", decimal=",")
		else:
			summary_path = None
	if summary_df is None or summary_df.empty:
		print("[Skip] No saved parameters available. Run the sweep to generate them.")
		return []
	return _run_saved_rows(
		summary_df,
		table_title="Gespeicherte Parameter (ohne Sweep)",
		save_path=summary_path,
	)


def run_overall_best_params():
	os.makedirs(OUT_DIR, exist_ok=True)
	os.makedirs(os.path.join(BASE_OUT_DIR, "charts"), exist_ok=True)
	if not os.path.exists(OVERALL_PARAMS_CSV):
		print("[Skip] Overall summary file missing. Run parameter sweeps first.")
		return []
	overall_df = pd.read_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",")
	if overall_df.empty:
		print("[Skip] Overall summary file is empty.")
		return []
	if ACTIVE_INDICATORS:
		overall_df = overall_df[overall_df["Indicator"].isin(ACTIVE_INDICATORS)]
		if overall_df.empty:
			print("[Skip] No overall rows match ACTIVE_INDICATORS.")
			return []
	allowed_dirs = []
	if ENABLE_LONGS:
		allowed_dirs.append("Long")
	if ENABLE_SHORTS:
		allowed_dirs.append("Short")
	if not allowed_dirs:
		allowed_dirs = ["Long"]
	overall_df = overall_df[overall_df["Direction"].isin(allowed_dirs)]
	if overall_df.empty:
		print("[Skip] No overall rows match the enabled trade directions.")
		return []
	if SYMBOLS:
		allowed_symbols = [sym.strip() for sym in SYMBOLS if sym and sym.strip()]
		if allowed_symbols:
			overall_df = overall_df[overall_df["Symbol"].isin(allowed_symbols)]
			if overall_df.empty:
				print(f"[Skip] Overall summary enthält keine Einträge für {', '.join(allowed_symbols)}.")
				return []
	group_cols = ["Indicator", "HTF"] if "HTF" in overall_df.columns else ["Indicator"]
	updated_all_rows = []
	aggregate_sections = []
	for group_key, rows in overall_df.groupby(group_cols):
		if isinstance(group_key, tuple):
			indicator_key = group_key[0]
			htf_value = group_key[1] if len(group_key) > 1 else HIGHER_TIMEFRAME
		else:
			indicator_key = group_key
			htf_value = HIGHER_TIMEFRAME
		if indicator_key not in INDICATOR_PRESETS:
			print(f"[Skip] Unknown indicator in overall file: {indicator_key}")
			continue
		apply_indicator_type(indicator_key)
		htf_str = str(htf_value) if not (isinstance(htf_value, float) and math.isnan(htf_value)) else HIGHER_TIMEFRAME
		htf_str = htf_str.strip()
		if not htf_str:
			htf_str = HIGHER_TIMEFRAME
		apply_higher_timeframe(htf_str)
		title = f"Overall-Beste Parameter ({INDICATOR_DISPLAY_NAME} {htf_str})"
		print(f"[Run] Indicator={INDICATOR_DISPLAY_NAME}, HTF={htf_str}, Kombinationen={len(rows)})")
		refreshed_rows = _run_saved_rows(rows, table_title=title, aggregate_sections=aggregate_sections)
		if refreshed_rows:
			updated_all_rows.extend(refreshed_rows)
	if updated_all_rows:
		updated_df = pd.DataFrame(updated_all_rows)
		os.makedirs(BASE_OUT_DIR, exist_ok=True)
		updated_df.to_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",", index=False, encoding="utf-8")
	write_flat_trade_list(updated_all_rows)
	write_combined_overall_best_report(aggregate_sections)
	return updated_all_rows


def run_current_configuration():
	os.makedirs(OUT_DIR, exist_ok=True)
	if RUN_PARAMETER_SWEEP:
		return run_parameter_sweep()
	elif RUN_SAVED_PARAMS:
		run_saved_params()
	elif RUN_OVERALL_BEST:
		run_overall_best_params()
	else:
		print("[Skip] Backtesting disabled. Enable RUN_PARAMETER_SWEEP or RUN_SAVED_PARAMS.")
	return []


apply_indicator_type("supertrend")
apply_higher_timeframe(HIGHER_TIMEFRAME)


if __name__ == "__main__":
	if RUN_OVERALL_BEST:
		run_overall_best_params()
	else:
		indicator_candidates = get_indicator_candidates()
		htf_candidates = get_highertimeframe_candidates()
		if RUN_PARAMETER_SWEEP and CLEAR_BASE_OUTPUT_ON_SWEEP:
			clear_sweep_targets(indicator_candidates, htf_candidates)
		for indicator_name in indicator_candidates:
			apply_indicator_type(indicator_name)
			for htf_value in htf_candidates:
				apply_higher_timeframe(htf_value)
				print(f"[Run] Indicator={INDICATOR_DISPLAY_NAME}, HTF={HIGHER_TIMEFRAME}")
				summary_rows = run_current_configuration()
				record_global_best(indicator_name, summary_rows)
		write_overall_result_tables()

