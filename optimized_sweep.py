"""
Optimized Sweep Runner for Crypto9
==================================
Wraps the existing Supertrend_5Min.py with:
- Parallel data fetching
- Skipped synthetic bars for backtesting
- Cached HTF computations
- Better performance (~3x faster)
"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import pandas as pd

# Import original module
import Supertrend_5Min as st

# Global flags for optimization
SKIP_SYNTHETIC_BARS = True  # Disable for backtesting (big speedup!)
PARALLEL_FETCH = True
MAX_WORKERS = 8

# HTF Cache - store computed HTF indicators per symbol
_htf_cache = {}


def prefetch_all_data_parallel(symbols, timeframe, limit):
    """
    Fetch all symbol data in parallel before processing.
    This is much faster than fetching one-by-one.
    """
    print(f"\n[PARALLEL] Prefetching {len(symbols)} symbols...")
    start_time = time.time()

    results = {}
    errors = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all fetch jobs
        future_to_symbol = {
            executor.submit(_fetch_symbol_data, sym, timeframe, limit): sym
            for sym in symbols
        }

        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results[symbol] = df
                    print(f"  ✓ {symbol}: {len(df)} bars")
                else:
                    errors.append(symbol)
            except Exception as e:
                print(f"  ✗ {symbol}: {e}")
                errors.append(symbol)

    elapsed = time.time() - start_time
    print(f"[PARALLEL] Fetched {len(results)}/{len(symbols)} symbols in {elapsed:.1f}s")

    if errors:
        print(f"[PARALLEL] Failed: {errors}")

    return results


def _fetch_symbol_data(symbol, timeframe, limit):
    """Fetch data for a single symbol (used by parallel fetcher)."""
    try:
        # Use cache if available
        cached = st.load_ohlcv_from_cache(symbol, timeframe)
        if cached is not None and not cached.empty and len(cached) >= limit * 0.9:
            return cached.tail(limit)

        # Otherwise fetch from API
        df = st._fetch_direct_ohlcv(symbol, timeframe, limit)
        if df is not None and not df.empty:
            st.save_ohlcv_to_cache(symbol, timeframe, df)
        return df
    except Exception as e:
        print(f"[ERROR] Fetching {symbol}: {e}")
        return pd.DataFrame()


def prefetch_htf_data_parallel(symbols, htf_timeframe, htf_limit):
    """
    Prefetch HTF data for all symbols in parallel.
    """
    print(f"\n[PARALLEL] Prefetching HTF ({htf_timeframe}) for {len(symbols)} symbols...")
    start_time = time.time()

    results = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_symbol = {
            executor.submit(_fetch_symbol_data, sym, htf_timeframe, htf_limit): sym
            for sym in symbols
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                df = future.result()
                if df is not None and not df.empty:
                    results[symbol] = df
            except Exception as e:
                print(f"  ✗ HTF {symbol}: {e}")

    elapsed = time.time() - start_time
    print(f"[PARALLEL] HTF data fetched in {elapsed:.1f}s")

    return results


def compute_htf_indicator_cached(symbol, df_htf):
    """
    Compute HTF indicator with caching.
    Only compute once per symbol, not per parameter combo!
    """
    cache_key = f"{symbol}_{st.INDICATOR_TYPE}_{st.HTF_LENGTH}_{st.HTF_FACTOR}"

    if cache_key in _htf_cache:
        return _htf_cache[cache_key]

    # Compute indicator
    if st.INDICATOR_TYPE == "supertrend" or st.INDICATOR_TYPE == "htf_crossover":
        df_ind = st.compute_supertrend(df_htf, length=st.HTF_LENGTH, factor=st.HTF_FACTOR)
        indicator_col = "supertrend"
        trend_col = "st_trend"
    elif st.INDICATOR_TYPE == "psar":
        df_ind = st.compute_psar(df_htf, step=st.HTF_PSAR_STEP, max_step=st.HTF_PSAR_MAX_STEP)
        indicator_col = "psar"
        trend_col = "psar_trend"
    else:
        df_ind = st.compute_supertrend(df_htf, length=st.HTF_LENGTH, factor=st.HTF_FACTOR)
        indicator_col = "supertrend"
        trend_col = "st_trend"

    htf_result = df_ind[[indicator_col, trend_col]].rename(columns={
        indicator_col: "htf_indicator",
        trend_col: "htf_trend"
    })

    _htf_cache[cache_key] = htf_result
    return htf_result


def attach_htf_trend_optimized(df_low, symbol, htf_data_cache):
    """
    Optimized HTF attachment using pre-fetched and cached data.
    """
    if not st.USE_HIGHER_TIMEFRAME_FILTER:
        df_low = df_low.copy()
        df_low["htf_trend"] = 0
        df_low["htf_indicator"] = float('nan')
        return df_low

    # Get pre-fetched HTF data
    df_htf = htf_data_cache.get(symbol)
    if df_htf is None or df_htf.empty:
        df_low = df_low.copy()
        df_low["htf_trend"] = 0
        df_low["htf_indicator"] = float('nan')
        return df_low

    # Use cached HTF indicator computation
    htf = compute_htf_indicator_cached(symbol, df_htf)

    # Reindex to low timeframe
    aligned = htf.reindex(df_low.index, method="ffill")

    df_low = df_low.copy()
    df_low["htf_trend"] = aligned["htf_trend"].fillna(0).astype(int)
    df_low["htf_indicator"] = aligned["htf_indicator"]

    return df_low


def run_optimized_sweep():
    """
    Optimized parameter sweep with parallel fetching.
    """
    global SKIP_SYNTHETIC_BARS

    print("=" * 70)
    print("  OPTIMIZED PARAMETER SWEEP")
    print("=" * 70)
    print(f"  Parallel Workers: {MAX_WORKERS}")
    print(f"  Skip Synthetic Bars: {SKIP_SYNTHETIC_BARS}")
    print(f"  Symbols: {len(st.SYMBOLS)}")
    print("=" * 70)

    total_start = time.time()

    # Step 1: Prefetch ALL data in parallel
    print("\n[STEP 1] Prefetching all OHLCV data...")
    ohlcv_cache = prefetch_all_data_parallel(
        st.SYMBOLS,
        st.TIMEFRAME,
        st.LOOKBACK
    )

    # Step 2: Prefetch HTF data in parallel
    htf_limit = (st.LOOKBACK // (st.timeframe_to_minutes(st.HIGHER_TIMEFRAME) // st.timeframe_to_minutes(st.TIMEFRAME))) + 100
    print(f"\n[STEP 2] Prefetching HTF ({st.HIGHER_TIMEFRAME}) data...")
    htf_cache = prefetch_htf_data_parallel(
        st.SYMBOLS,
        st.HIGHER_TIMEFRAME,
        htf_limit
    )

    # Step 3: Run the sweep with pre-cached data
    print("\n[STEP 3] Running parameter sweep with cached data...")

    # Temporarily disable synthetic bar creation
    original_func = st._maybe_append_synthetic_bar
    if SKIP_SYNTHETIC_BARS:
        st._maybe_append_synthetic_bar = lambda df, sym, tf: df

    # Pre-populate the DATA_CACHE
    for symbol, df in ohlcv_cache.items():
        key = (symbol, st.TIMEFRAME, st.LOOKBACK)
        st.DATA_CACHE[key] = df

    # Monkey-patch attach_higher_timeframe_trend for optimization
    original_htf_func = st.attach_higher_timeframe_trend
    st.attach_higher_timeframe_trend = lambda df, sym: attach_htf_trend_optimized(df, sym, htf_cache)

    try:
        # Run the original sweep
        st.run_parameter_sweep()
    finally:
        # Restore original functions
        st._maybe_append_synthetic_bar = original_func
        st.attach_higher_timeframe_trend = original_htf_func

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"  TOTAL TIME: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


def estimate_speedup():
    """Estimate the speedup from optimizations."""
    print("\n[ESTIMATE] Speedup factors:")
    print(f"  - Parallel fetch ({MAX_WORKERS} workers): ~{MAX_WORKERS}x for I/O")
    print(f"  - Skip synthetic bars: ~1.5x")
    print(f"  - HTF caching: ~1.3x")
    print(f"  - Expected total speedup: ~3-4x")


if __name__ == "__main__":
    estimate_speedup()

    print("\nStart optimized sweep? (y/n): ", end="")
    if input().strip().lower() == 'y':
        run_optimized_sweep()
    else:
        print("Aborted.")
