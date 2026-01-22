"""
Optimal Hold Times for DOW/NASDAQ Stocks
Based on typical stock market behavior patterns.

For stocks using 1h bars, we generally hold longer than crypto due to:
- Lower volatility
- Trading only during market hours (6.5h/day)
- No 24/7 trading

1 day = ~7 bars (1h timeframe, 6.5h market hours)
"""

# Optimal hold times for stocks
# Format: (symbol, direction) -> bars (1h bars)
OPTIMAL_HOLD_BARS = {
    # === High-volatility tech stocks ===
    # Hold shorter due to quick moves
    ("NVDA", "long"): 5,
    ("TSLA", "long"): 4,
    ("AMD", "long"): 5,
    ("META", "long"): 5,

    # === Large-cap stable stocks ===
    # Hold longer for trends to develop
    ("AAPL", "long"): 6,
    ("MSFT", "long"): 6,
    ("GOOGL", "long"): 6,
    ("GOOG", "long"): 6,
    ("AMZN", "long"): 6,
    ("V", "long"): 7,
    ("JPM", "long"): 7,

    # === DOW Industrial stocks ===
    # Generally more stable, hold longer
    ("HD", "long"): 7,
    ("CAT", "long"): 6,
    ("BA", "long"): 5,
    ("HON", "long"): 6,
    ("GS", "long"): 5,
    ("IBM", "long"): 7,
    ("JNJ", "long"): 8,
    ("KO", "long"): 8,
    ("MCD", "long"): 7,
    ("WMT", "long"): 7,
    ("PG", "long"): 8,
    ("UNH", "long"): 6,
    ("DIS", "long"): 6,
    ("NKE", "long"): 6,
    ("INTC", "long"): 5,
    ("VZ", "long"): 7,
    ("CVX", "long"): 6,
    ("MMM", "long"): 7,
    ("MRK", "long"): 7,
    ("AXP", "long"): 6,
    ("DOW", "long"): 6,
    ("TRV", "long"): 7,
    ("WBA", "long"): 5,
    ("CSCO", "long"): 6,
    ("AMGN", "long"): 7,

    # === NASDAQ Growth stocks ===
    ("NFLX", "long"): 5,
    ("ADBE", "long"): 6,
    ("CRM", "long"): 5,
    ("PYPL", "long"): 5,
    ("QCOM", "long"): 5,
    ("AVGO", "long"): 6,
    ("COST", "long"): 7,
    ("PEP", "long"): 7,
    ("CMCSA", "long"): 6,
    ("TMUS", "long"): 6,
    ("TXN", "long"): 6,
    ("INTU", "long"): 5,
    ("AMAT", "long"): 5,
    ("ISRG", "long"): 6,
    ("BKNG", "long"): 5,
    ("SBUX", "long"): 6,
    ("VRTX", "long"): 6,
    ("GILD", "long"): 7,
    ("ADI", "long"): 6,
    ("MDLZ", "long"): 7,
    ("ADP", "long"): 7,
    ("REGN", "long"): 6,
    ("LRCX", "long"): 5,
    ("MU", "long"): 5,
    ("PANW", "long"): 5,
    ("SNPS", "long"): 6,
    ("KLAC", "long"): 5,
    ("CDNS", "long"): 6,
    ("MELI", "long"): 5,
    ("ORLY", "long"): 7,
    ("MAR", "long"): 6,
    ("ABNB", "long"): 5,
    ("FTNT", "long"): 5,
    ("CTAS", "long"): 7,
    ("MNST", "long"): 6,
    ("WDAY", "long"): 5,
    ("KDP", "long"): 7,
    ("DXCM", "long"): 5,
}

# Default if symbol not in dictionary
DEFAULT_HOLD_BARS_LONG = 6
DEFAULT_HOLD_BARS_SHORT = 5


def get_optimal_hold_bars(symbol: str, direction: str) -> int:
    """
    Get optimal hold bars for a symbol/direction pair.

    Args:
        symbol: Stock symbol (e.g., "AAPL")
        direction: "long" or "short"

    Returns:
        Optimal number of bars to hold position
    """
    # Normalize inputs
    symbol = symbol.upper().strip()
    direction = direction.lower().strip()

    # Look up in dictionary
    key = (symbol, direction)
    if key in OPTIMAL_HOLD_BARS:
        return OPTIMAL_HOLD_BARS[key]

    # Default values
    if direction == "long":
        return DEFAULT_HOLD_BARS_LONG
    return DEFAULT_HOLD_BARS_SHORT


def get_optimal_hold_hours(symbol: str, direction: str, bars_per_hour: int = 1) -> float:
    """Get optimal hold time in hours."""
    bars = get_optimal_hold_bars(symbol, direction)
    return bars / bars_per_hour


def get_optimal_hold_days(symbol: str, direction: str, bars_per_day: int = 7) -> float:
    """Get optimal hold time in trading days (assuming 1h bars)."""
    bars = get_optimal_hold_bars(symbol, direction)
    return bars / bars_per_day
