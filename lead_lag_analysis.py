#!/usr/bin/env python3
"""
Lead-Lag Analysis: USDC vs EUR pairs
Determines if USDC prices move before EUR prices (or vice versa)
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

OHLCV_CACHE_DIR = "ohlcv_cache"
BERLIN_TZ = "Europe/Berlin"

# Pairs to compare
PAIRS_TO_COMPARE = [
    ("BTC/USDC", "BTC/EUR"),
    ("ETH/USDC", "ETH/EUR"),
    ("SOL/USDC", "SOL/EUR"),
    ("XRP/USDC", "XRP/EUR"),
    ("LINK/USDC", "LINK/EUR"),
    ("SUI/USDC", "SUI/EUR"),
]


def load_ohlcv(symbol, timeframe="1h"):
    """Load OHLCV data from cache."""
    safe_symbol = symbol.replace("/", "_")
    cache_file = os.path.join(OHLCV_CACHE_DIR, f"{safe_symbol}_{timeframe}.csv")

    if not os.path.exists(cache_file):
        return None

    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert(BERLIN_TZ)
    else:
        df.index = df.index.tz_convert(BERLIN_TZ)
    return df


def cross_correlation(series1, series2, max_lag=24):
    """
    Calculate cross-correlation between two series at different lags.

    Positive lag: series1 leads series2
    Negative lag: series2 leads series1

    Returns: dict with lag -> correlation
    """
    # Calculate returns (price changes)
    ret1 = series1.pct_change().dropna()
    ret2 = series2.pct_change().dropna()

    # Align the series
    common_idx = ret1.index.intersection(ret2.index)
    ret1 = ret1.loc[common_idx]
    ret2 = ret2.loc[common_idx]

    if len(ret1) < max_lag * 2:
        return None, None

    correlations = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            # series1 leads: shift series1 back (compare series1[t] with series2[t+lag])
            corr = ret1.iloc[:-lag].corr(ret2.iloc[lag:])
        elif lag < 0:
            # series2 leads: shift series2 back
            corr = ret1.iloc[-lag:].corr(ret2.iloc[:lag])
        else:
            corr = ret1.corr(ret2)
        correlations[lag] = corr

    # Find optimal lag (highest absolute correlation)
    best_lag = max(correlations, key=lambda x: abs(correlations[x]))
    best_corr = correlations[best_lag]

    return correlations, (best_lag, best_corr)


def analyze_pair(usdc_symbol, eur_symbol, timeframe="1h"):
    """Analyze lead-lag relationship between USDC and EUR pair."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {usdc_symbol} vs {eur_symbol}")
    print(f"{'='*60}")

    df_usdc = load_ohlcv(usdc_symbol, timeframe)
    df_eur = load_ohlcv(eur_symbol, timeframe)

    if df_usdc is None:
        print(f"  ERROR: No data for {usdc_symbol}")
        return None
    if df_eur is None:
        print(f"  ERROR: No data for {eur_symbol}")
        return None

    print(f"  {usdc_symbol}: {len(df_usdc)} bars ({df_usdc.index[0].date()} to {df_usdc.index[-1].date()})")
    print(f"  {eur_symbol}: {len(df_eur)} bars ({df_eur.index[0].date()} to {df_eur.index[-1].date()})")

    # Cross-correlation analysis
    correlations, best = cross_correlation(df_usdc["close"], df_eur["close"], max_lag=12)

    if correlations is None:
        print("  ERROR: Not enough data for analysis")
        return None

    best_lag, best_corr = best

    print(f"\n  Cross-Correlation Results:")
    print(f"  ---------------------------")

    # Show correlations around 0
    for lag in range(-6, 7):
        marker = " <-- BEST" if lag == best_lag else ""
        print(f"    Lag {lag:+2d}h: {correlations[lag]:+.4f}{marker}")

    print(f"\n  Interpretation:")
    if best_lag > 0:
        print(f"    -> {usdc_symbol} LEADS by {best_lag}h (corr: {best_corr:.4f})")
        print(f"    -> USDC bewegt sich {best_lag}h VOR EUR")
    elif best_lag < 0:
        print(f"    -> {eur_symbol} LEADS by {-best_lag}h (corr: {best_corr:.4f})")
        print(f"    -> EUR bewegt sich {-best_lag}h VOR USDC")
    else:
        print(f"    -> Synchron (keine signifikante Lead/Lag)")

    return {
        "usdc": usdc_symbol,
        "eur": eur_symbol,
        "best_lag": best_lag,
        "best_corr": best_corr,
        "correlations": correlations
    }


def main():
    print("\n" + "="*60)
    print("  LEAD-LAG ANALYSIS: USDC vs EUR")
    print("  Positive lag = USDC leads, Negative lag = EUR leads")
    print("="*60)

    results = []
    for usdc_sym, eur_sym in PAIRS_TO_COMPARE:
        result = analyze_pair(usdc_sym, eur_sym)
        if result:
            results.append(result)

    # Summary
    if results:
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)

        usdc_leads = sum(1 for r in results if r["best_lag"] > 0)
        eur_leads = sum(1 for r in results if r["best_lag"] < 0)
        sync = sum(1 for r in results if r["best_lag"] == 0)

        print(f"\n  USDC führt: {usdc_leads} Paare")
        print(f"  EUR führt:  {eur_leads} Paare")
        print(f"  Synchron:   {sync} Paare")

        avg_lag = np.mean([r["best_lag"] for r in results])
        print(f"\n  Durchschnittlicher Lag: {avg_lag:+.1f}h")

        if avg_lag > 0.5:
            print(f"  -> USDC Paare haben im Schnitt einen Lead von {avg_lag:.1f}h")
        elif avg_lag < -0.5:
            print(f"  -> EUR Paare haben im Schnitt einen Lead von {-avg_lag:.1f}h")
        else:
            print(f"  -> Kein klarer Lead zwischen USDC und EUR")


if __name__ == "__main__":
    main()
