"""
Custom Technical Indicators: JMA, KAMA, and HTF Supertrend
"""
import numpy as np
import pandas as pd
from typing import Tuple


def calculate_jma(
    series: pd.Series,
    period: int = 7,
    phase: int = 0,
    power: int = 2
) -> pd.Series:
    """
    Calculate Jurik Moving Average (JMA).

    JMA is a smoothed moving average that reduces noise while maintaining
    responsiveness to price changes.

    Args:
        series: Price series (typically close)
        period: Lookback period (default 7)
        phase: Phase shift (-100 to +100, default 0)
        power: Power/smoothing factor (default 2)

    Returns:
        JMA values as pandas Series
    """
    data = series.values.astype(float)
    n = len(data)

    # Initialize arrays
    jma = np.zeros(n)
    e0 = np.zeros(n)
    e1 = np.zeros(n)
    e2 = np.zeros(n)

    # Calculate beta and alpha
    beta = 0.45 * (period - 1) / (0.45 * (period - 1) + 2)

    # Phase adjustment
    phase_ratio = phase / 100.0 + 1.5 if phase < -100 else (
        phase / 100.0 + 0.5 if phase > 100 else phase / 100.0 + 1.0
    )

    alpha = beta ** power

    # Calculate JMA
    jma[0] = data[0]
    e0[0] = data[0]
    e1[0] = 0
    e2[0] = 0

    for i in range(1, n):
        # Update e0 (price deviation)
        e0[i] = (1 - alpha) * data[i] + alpha * e0[i-1]

        # Update e1 (trend)
        e1[i] = (data[i] - e0[i]) * (1 - beta) + beta * e1[i-1]

        # Update e2 (second smoothing)
        e2[i] = (e0[i] + phase_ratio * e1[i] - jma[i-1]) * (1 - alpha) ** 2 + (alpha ** 2) * e2[i-1]

        # Calculate JMA
        jma[i] = jma[i-1] + e2[i]

    result = pd.Series(jma, index=series.index)
    result[:period] = np.nan  # Mark warmup period as NaN

    return result


def calculate_kama(
    series: pd.Series,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30
) -> pd.Series:
    """
    Calculate Kaufman Adaptive Moving Average (KAMA).

    KAMA adapts its smoothing constant based on market efficiency,
    becoming faster in trending markets and slower in ranging markets.

    Args:
        series: Price series (typically close)
        er_period: Efficiency Ratio period (default 10)
        fast_period: Fast EMA period (default 2)
        slow_period: Slow EMA period (default 30)

    Returns:
        KAMA values as pandas Series
    """
    data = series.values.astype(float)
    n = len(data)

    # Smoothing constants
    fast_sc = 2.0 / (fast_period + 1)
    slow_sc = 2.0 / (slow_period + 1)

    # Calculate change and volatility
    change = np.zeros(n)
    volatility = np.zeros(n)

    for i in range(er_period, n):
        change[i] = abs(data[i] - data[i - er_period])
        volatility[i] = sum(abs(data[j] - data[j-1]) for j in range(i - er_period + 1, i + 1))

    # Efficiency Ratio
    er = np.where(volatility != 0, change / volatility, 0)

    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Calculate KAMA
    kama = np.zeros(n)
    kama[er_period - 1] = data[er_period - 1]  # Start value

    for i in range(er_period, n):
        kama[i] = kama[i-1] + sc[i] * (data[i] - kama[i-1])

    result = pd.Series(kama, index=series.index)
    result[:er_period] = np.nan  # Mark warmup period as NaN

    return result


def calculate_atr(
    df: pd.DataFrame,
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    return atr


def calculate_supertrend(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculate Supertrend indicator.

    Returns DataFrame with added columns: supertrend, trend
    """
    df = df.copy()

    high = df['high']
    low = df['low']
    close = df['close']

    atr = calculate_atr(df, period)

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
            supertrend.iloc[i] = final_upper.iloc[i] if close.iloc[i] <= final_upper.iloc[i] else final_lower.iloc[i]
        elif supertrend.iloc[i-1] == final_lower.iloc[i-1]:
            supertrend.iloc[i] = final_lower.iloc[i] if close.iloc[i] >= final_lower.iloc[i] else final_upper.iloc[i]
        else:
            supertrend.iloc[i] = final_lower.iloc[i]

    df['supertrend'] = supertrend
    df['trend'] = np.where(close > supertrend, 1, -1)

    return df


def calculate_supertrend_htf(
    df: pd.DataFrame,
    htf_df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.DataFrame:
    """
    Calculate Supertrend with Higher Timeframe filter.

    Trades are only taken when the lower timeframe signal aligns with
    the higher timeframe trend direction.

    Args:
        df: Lower timeframe OHLCV data
        htf_df: Higher timeframe OHLCV data
        period: ATR period
        multiplier: ATR multiplier

    Returns:
        DataFrame with supertrend, htf_trend, and combined signal
    """
    # Calculate LTF supertrend
    df = calculate_supertrend(df, period, multiplier)

    # Calculate HTF supertrend
    htf_df = calculate_supertrend(htf_df, period, multiplier)

    # Map HTF trend to LTF timeframe using forward fill
    htf_trend = htf_df['trend'].copy()
    htf_trend.index = pd.to_datetime(htf_trend.index)
    df.index = pd.to_datetime(df.index)

    # Reindex HTF to LTF, forward filling the values
    df['htf_trend'] = htf_trend.reindex(df.index, method='ffill')
    df['htf_trend'] = df['htf_trend'].fillna(0)

    # Combined signal: LTF signal must align with HTF trend
    # 1 = bullish, -1 = bearish, 0 = no trade (misaligned)
    df['aligned_trend'] = np.where(
        df['trend'] == df['htf_trend'],
        df['trend'],
        0
    )

    return df


def calculate_jma_crossover(
    df: pd.DataFrame,
    fast_period: int = 7,
    slow_period: int = 21,
    phase: int = 0
) -> pd.DataFrame:
    """
    Calculate JMA crossover signals.

    Buy signal when fast JMA crosses above slow JMA.
    Sell signal when fast JMA crosses below slow JMA.
    """
    df = df.copy()

    df['jma_fast'] = calculate_jma(df['close'], fast_period, phase)
    df['jma_slow'] = calculate_jma(df['close'], slow_period, phase)

    # Trend direction: 1 = bullish (fast > slow), -1 = bearish
    df['jma_trend'] = np.where(df['jma_fast'] > df['jma_slow'], 1, -1)

    return df


def calculate_kama_crossover(
    df: pd.DataFrame,
    fast_er: int = 5,
    slow_er: int = 20,
    fast_ema: int = 2,
    slow_ema: int = 30
) -> pd.DataFrame:
    """
    Calculate KAMA crossover signals.

    Uses two KAMA lines with different efficiency ratio periods.
    """
    df = df.copy()

    df['kama_fast'] = calculate_kama(df['close'], fast_er, fast_ema, slow_ema)
    df['kama_slow'] = calculate_kama(df['close'], slow_er, fast_ema, slow_ema)

    # Trend direction
    df['kama_trend'] = np.where(df['kama_fast'] > df['kama_slow'], 1, -1)

    return df


def calculate_kama_price_cross(
    df: pd.DataFrame,
    er_period: int = 10,
    fast_period: int = 2,
    slow_period: int = 30
) -> pd.DataFrame:
    """
    Calculate KAMA price crossover signals.

    Buy when price crosses above KAMA.
    Sell when price crosses below KAMA.
    """
    df = df.copy()

    df['kama'] = calculate_kama(df['close'], er_period, fast_period, slow_period)

    # Trend: 1 = price above KAMA, -1 = price below
    df['kama_trend'] = np.where(df['close'] > df['kama'], 1, -1)

    return df
