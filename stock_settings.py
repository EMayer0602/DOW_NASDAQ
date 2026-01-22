"""
Stock Trading Settings for DOW/NASDAQ
Replaces crypto-specific settings from Supertrend_5Min.py
"""

# ============================================
# TRADING MODE
# ============================================
TRADING_MODE = "STOCKS"  # "CRYPTO" or "STOCKS"
USE_IB = True  # Use Interactive Brokers (vs simulation only)
IB_PAPER_TRADING = True  # Paper trading mode

# ============================================
# TIMEFRAME SETTINGS
# ============================================
TIMEFRAME = "1h"  # Primary timeframe (1h works well for stocks)
HTF_TIMEFRAME = "1d"  # Higher timeframe for filters

# IB-compatible timeframe mapping
TIMEFRAME_MAP = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "4h": "4 hours",
    "1d": "1 day"
}

# Duration for historical data
DURATION_MAP = {
    "1m": "2 D",
    "5m": "1 W",
    "15m": "2 W",
    "30m": "1 M",
    "1h": "2 M",
    "4h": "6 M",
    "1d": "1 Y"
}

# ============================================
# SYMBOL CONFIGURATION
# ============================================
from stock_symbols import DEFAULT_TRADING_SYMBOLS, DOW_30, NASDAQ_100_TOP

# Active trading symbols
SYMBOLS = DEFAULT_TRADING_SYMBOLS

# All available symbols
ALL_SYMBOLS = list(set(DOW_30 + NASDAQ_100_TOP))

# ============================================
# POSITION SIZING
# ============================================
START_TOTAL_CAPITAL = 100_000.0  # Starting capital for stocks
MAX_OPEN_POSITIONS = 10
STAKE_DIVISOR = 10  # stake = capital / divisor
MAX_LONG_POSITIONS = 10
MAX_SHORT_POSITIONS = 0  # Long only (stocks harder to short)
POSITION_SIZE_USD = 10_000.0  # Fixed position size per trade

# ============================================
# TRADING FEES
# ============================================
FEE_RATE = 0.0  # IB has per-share fees, not percentage
IB_COMMISSION_PER_SHARE = 0.005  # $0.005 per share (IB tiered)
IB_MIN_COMMISSION = 1.0  # Minimum $1 per trade

# ============================================
# SUPERTREND PARAMETERS
# ============================================
DEFAULT_ATR_PERIOD = 10
DEFAULT_ATR_MULTIPLIER = 3.0

# ============================================
# EXIT SETTINGS
# ============================================
USE_TIME_BASED_EXIT = True
DISABLE_TREND_FLIP_EXIT = False
DEFAULT_HOLD_BARS = 5  # Default bars to hold

# ============================================
# MARKET HOURS (US Eastern)
# ============================================
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

# Only trade during market hours
RESPECT_MARKET_HOURS = True

# ============================================
# DATA SOURCES
# ============================================
# Primary: Interactive Brokers
# Fallback: Yahoo Finance (yfinance)
USE_YFINANCE_FALLBACK = True

# ============================================
# REPORT PATHS
# ============================================
REPORT_DIR = "report_stocks"
BEST_PARAMS_CSV = "report_stocks/best_params_overall.csv"
OVERALL_PARAMS_CSV = "report_stocks/best_params_overall.csv"

# ============================================
# HELPER FUNCTIONS
# ============================================
def timeframe_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes."""
    tf = tf.lower()
    if tf.endswith('m'):
        return int(tf[:-1])
    elif tf.endswith('h'):
        return int(tf[:-1]) * 60
    elif tf.endswith('d'):
        return int(tf[:-1]) * 60 * 24
    return 60  # Default 1 hour

def get_ib_timeframe(tf: str) -> str:
    """Convert timeframe to IB format."""
    return TIMEFRAME_MAP.get(tf.lower(), "1 hour")

def get_ib_duration(tf: str) -> str:
    """Get appropriate duration for timeframe."""
    return DURATION_MAP.get(tf.lower(), "2 M")

def is_market_open() -> bool:
    """Check if US stock market is currently open."""
    from datetime import datetime
    import pytz

    eastern = pytz.timezone('US/Eastern')
    now = datetime.now(eastern)

    # Weekend check
    if now.weekday() >= 5:
        return False

    # Time check
    market_open = now.replace(hour=MARKET_OPEN_HOUR, minute=MARKET_OPEN_MINUTE, second=0)
    market_close = now.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0)

    return market_open <= now <= market_close

# ============================================
# BARS PER DAY (for hold time calculations)
# ============================================
BARS_PER_DAY = {
    "1m": 390,    # 6.5 hours * 60
    "5m": 78,     # 6.5 hours * 12
    "15m": 26,    # 6.5 hours * 4
    "30m": 13,    # 6.5 hours * 2
    "1h": 7,      # ~7 bars per day (6.5h rounded)
    "4h": 2,      # ~2 bars per day
    "1d": 1
}

def get_bars_per_day(tf: str = None) -> int:
    """Get bars per trading day for timeframe."""
    if tf is None:
        tf = TIMEFRAME
    return BARS_PER_DAY.get(tf.lower(), 7)
