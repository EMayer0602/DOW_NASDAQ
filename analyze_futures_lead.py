#!/usr/bin/env python3
"""
Analyze if futures prices lead spot prices.
If futures lead, we can use futures signals for spot trading.
"""

import argparse
import os
from datetime import datetime, timedelta
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import Supertrend_5Min as st


def download_spot_and_futures(
    symbol: str,
    timeframe: str = "5m",
    days: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Download spot and futures data for comparison."""

    # Spot symbol (e.g., BTC/EUR -> BTC/USDT for comparison)
    spot_symbol = symbol

    # Futures symbol - use perpetual futures
    # Convert spot to futures format
    base = symbol.split("/")[0]
    futures_symbol = f"{base}/USDT:USDT"  # Perpetual futures format for ccxt

    print(f"[Futures] Downloading spot: {spot_symbol}")
    print(f"[Futures] Downloading futures: {futures_symbol}")

    # Calculate bars needed
    minutes_per_bar = st.timeframe_to_minutes(timeframe)
    bars_needed = int((days * 24 * 60) / minutes_per_bar)

    # Download spot data
    try:
        spot_df = st.fetch_data(spot_symbol, timeframe, bars_needed)
        print(f"[Futures] Spot data: {len(spot_df)} bars")
    except Exception as e:
        print(f"[Futures] Failed to fetch spot {spot_symbol}: {e}")
        spot_df = pd.DataFrame()

    # Download futures data
    try:
        # Create a separate unauthenticated exchange for futures (OHLCV is public data)
        import ccxt
        futures_exchange = ccxt.binance({
            'options': {'defaultType': 'future'},
            'enableRateLimit': True,
        })

        futures_ohlcv = futures_exchange.fetch_ohlcv(futures_symbol, timeframe, limit=bars_needed)
        futures_df = pd.DataFrame(
            futures_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        futures_df["timestamp"] = pd.to_datetime(futures_df["timestamp"], unit="ms", utc=True)
        futures_df.set_index("timestamp", inplace=True)
        futures_df.index = futures_df.index.tz_convert(st.BERLIN_TZ)

        print(f"[Futures] Futures data: {len(futures_df)} bars")
    except Exception as e:
        print(f"[Futures] Failed to fetch futures {futures_symbol}: {e}")
        # Try alternative futures symbol format
        try:
            import ccxt
            alt_futures = f"{base}USDT"
            futures_exchange = ccxt.binance({
                'options': {'defaultType': 'future'},
                'enableRateLimit': True,
            })
            futures_ohlcv = futures_exchange.fetch_ohlcv(alt_futures, timeframe, limit=bars_needed)
            futures_df = pd.DataFrame(
                futures_ohlcv,
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            futures_df["timestamp"] = pd.to_datetime(futures_df["timestamp"], unit="ms", utc=True)
            futures_df.set_index("timestamp", inplace=True)
            futures_df.index = futures_df.index.tz_convert(st.BERLIN_TZ)
            print(f"[Futures] Futures data (alt): {len(futures_df)} bars")
        except Exception as e2:
            print(f"[Futures] Also failed with alt symbol: {e2}")
            futures_df = pd.DataFrame()

    return spot_df, futures_df


def calculate_cross_correlation(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    max_lag: int = 20,
) -> pd.DataFrame:
    """
    Calculate cross-correlation between spot and futures returns.
    Positive lag means futures leads spot.
    """
    # Align dataframes on common index
    common_idx = spot_df.index.intersection(futures_df.index)
    if len(common_idx) < 100:
        print(f"[Futures] Not enough common data points: {len(common_idx)}")
        return pd.DataFrame()

    spot_aligned = spot_df.loc[common_idx, "close"]
    futures_aligned = futures_df.loc[common_idx, "close"]

    # Calculate returns
    spot_returns = spot_aligned.pct_change().dropna()
    futures_returns = futures_aligned.pct_change().dropna()

    # Align returns
    common_ret_idx = spot_returns.index.intersection(futures_returns.index)
    spot_returns = spot_returns.loc[common_ret_idx]
    futures_returns = futures_returns.loc[common_ret_idx]

    print(f"[Futures] Analyzing {len(spot_returns)} return pairs")

    # Calculate cross-correlation for different lags
    correlations = []
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            # Spot leads futures
            corr = spot_returns.iloc[:lag].corr(futures_returns.iloc[-lag:])
        elif lag > 0:
            # Futures leads spot
            corr = futures_returns.iloc[:-lag].corr(spot_returns.iloc[lag:])
        else:
            corr = spot_returns.corr(futures_returns)

        correlations.append({
            "lag": lag,
            "correlation": corr,
            "meaning": "futures_leads" if lag > 0 else ("spot_leads" if lag < 0 else "concurrent")
        })

    return pd.DataFrame(correlations)


def calculate_lead_lag_signals(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    indicator: str = "supertrend",
    param_a: float = 10.0,
    param_b: float = 3.0,
) -> pd.DataFrame:
    """
    Calculate indicator signals on both spot and futures.
    Check if futures signals precede spot signals.
    """
    from Supertrend_5Min import compute_supertrend

    # Align data
    common_idx = spot_df.index.intersection(futures_df.index)
    spot_aligned = spot_df.loc[common_idx].copy()
    futures_aligned = futures_df.loc[common_idx].copy()

    # Calculate supertrend on both
    spot_st = compute_supertrend(spot_aligned, int(param_a), param_b)
    futures_st = compute_supertrend(futures_aligned, int(param_a), param_b)

    # Get trend flags
    spot_trend = spot_st["trend_flag"] if "trend_flag" in spot_st.columns else pd.Series(0, index=spot_st.index)
    futures_trend = futures_st["trend_flag"] if "trend_flag" in futures_st.columns else pd.Series(0, index=futures_st.index)

    # Find signal changes
    spot_signals = spot_trend.diff().fillna(0)
    futures_signals = futures_trend.diff().fillna(0)

    # Analyze lead/lag of signals
    results = []

    # Find where futures signal changes
    futures_changes = futures_signals[futures_signals != 0].index

    for ft in futures_changes:
        # Look for corresponding spot signal within next N bars
        for look_ahead in range(1, 11):
            future_idx = spot_signals.index.get_indexer([ft], method="nearest")[0]
            if future_idx + look_ahead < len(spot_signals):
                check_time = spot_signals.index[future_idx + look_ahead]
                if spot_signals.loc[check_time] != 0:
                    # Found matching signal
                    results.append({
                        "futures_signal_time": ft,
                        "spot_signal_time": check_time,
                        "lead_bars": look_ahead,
                        "signal_type": "buy" if futures_signals.loc[ft] > 0 else "sell"
                    })
                    break

    return pd.DataFrame(results)


def plot_futures_spot_comparison(
    spot_df: pd.DataFrame,
    futures_df: pd.DataFrame,
    correlations: pd.DataFrame,
    symbol: str,
    output_dir: str = "report_html/charts",
) -> None:
    """Create visualization comparing futures and spot."""

    os.makedirs(output_dir, exist_ok=True)

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3],
        subplot_titles=(
            f"{symbol} - Spot vs Futures Prices",
            "Cross-Correlation (positive lag = futures leads)",
            "Price Difference (Futures - Spot)"
        )
    )

    # Align for plotting
    common_idx = spot_df.index.intersection(futures_df.index)
    spot_aligned = spot_df.loc[common_idx]
    futures_aligned = futures_df.loc[common_idx]

    # Plot 1: Price comparison
    fig.add_trace(
        go.Scatter(x=spot_aligned.index, y=spot_aligned["close"], name="Spot", line=dict(color="blue")),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=futures_aligned.index, y=futures_aligned["close"], name="Futures", line=dict(color="orange")),
        row=1, col=1
    )

    # Plot 2: Cross-correlation
    if not correlations.empty:
        colors = ["green" if c > 0 else "red" for c in correlations["correlation"]]
        fig.add_trace(
            go.Bar(x=correlations["lag"], y=correlations["correlation"], marker_color=colors, name="Correlation"),
            row=2, col=1
        )

        # Find max correlation and its lag
        max_idx = correlations["correlation"].abs().idxmax()
        max_lag = correlations.loc[max_idx, "lag"]
        max_corr = correlations.loc[max_idx, "correlation"]

        fig.add_annotation(
            x=max_lag, y=max_corr,
            text=f"Max: lag={max_lag}, corr={max_corr:.3f}",
            showarrow=True, arrowhead=2,
            row=2, col=1
        )

    # Plot 3: Price difference
    price_diff = futures_aligned["close"] - spot_aligned["close"]
    diff_colors = ["green" if d > 0 else "red" for d in price_diff]
    fig.add_trace(
        go.Scatter(x=price_diff.index, y=price_diff, name="Futures Premium",
                   line=dict(color="purple"), fill="tozeroy"),
        row=3, col=1
    )

    # Summary statistics
    if not correlations.empty:
        positive_lags = correlations[correlations["lag"] > 0]
        if not positive_lags.empty:
            best_lead = positive_lags.loc[positive_lags["correlation"].idxmax()]
            lead_info = f"Best futures lead: {int(best_lead['lag'])} bars (corr: {best_lead['correlation']:.3f})"
        else:
            lead_info = "No significant lead detected"
    else:
        lead_info = "Correlation analysis failed"

    fig.update_layout(
        title=f"Futures vs Spot Analysis: {symbol}<br><sub>{lead_info}</sub>",
        height=900,
        showlegend=True,
        template="plotly_white"
    )

    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Correlation", row=2, col=1)
    fig.update_yaxes(title_text="Premium (USDT)", row=3, col=1)

    out_path = os.path.join(output_dir, f"futures_spot_{symbol.replace('/', '_')}.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[Futures] Saved analysis to {out_path}")

    return correlations


def analyze_futures_lead(
    symbols: list = None,
    timeframe: str = "5m",
    days: int = 30,
    output_dir: str = "report_html/charts",
) -> pd.DataFrame:
    """
    Analyze multiple symbols for futures leading spot.
    Returns summary of which symbols show futures leading behavior.
    """
    if symbols is None:
        symbols = ["BTC/USDT", "ETH/USDT"]

    results = []

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Analyzing {symbol}")
        print('='*60)

        try:
            spot_df, futures_df = download_spot_and_futures(symbol, timeframe, days)

            if spot_df.empty or futures_df.empty:
                print(f"[Futures] Skipping {symbol} - missing data")
                continue

            correlations = calculate_cross_correlation(spot_df, futures_df, max_lag=20)

            if correlations.empty:
                print(f"[Futures] Skipping {symbol} - correlation failed")
                continue

            # Find best positive lag (futures leads)
            positive_lags = correlations[correlations["lag"] > 0]
            if not positive_lags.empty:
                best = positive_lags.loc[positive_lags["correlation"].idxmax()]
                results.append({
                    "symbol": symbol,
                    "best_lead_bars": int(best["lag"]),
                    "correlation": best["correlation"],
                    "significant": best["correlation"] > 0.3,
                })

            # Create visualization
            plot_futures_spot_comparison(spot_df, futures_df, correlations, symbol, output_dir)

        except Exception as e:
            print(f"[Futures] Error analyzing {symbol}: {e}")
            continue

    summary = pd.DataFrame(results)
    if not summary.empty:
        summary = summary.sort_values("correlation", ascending=False)
        print("\n" + "="*60)
        print("FUTURES LEAD ANALYSIS SUMMARY")
        print("="*60)
        print(summary.to_string(index=False))
        print("\nSymbols where futures significantly lead spot (corr > 0.3):")
        sig = summary[summary["significant"]]
        if not sig.empty:
            for _, row in sig.iterrows():
                print(f"  {row['symbol']}: {row['best_lead_bars']} bars lead, corr={row['correlation']:.3f}")
        else:
            print("  None found")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze futures leading spot prices")
    parser.add_argument("--symbols", nargs="+", default=["BTC/USDT", "ETH/USDT"],
                        help="Symbols to analyze")
    parser.add_argument("--timeframe", default="5m", help="Timeframe for analysis")
    parser.add_argument("--days", type=int, default=30, help="Days of data to analyze")
    parser.add_argument("--output", default="report_html/charts", help="Output directory")

    args = parser.parse_args()

    st.configure_exchange(use_testnet=False)

    analyze_futures_lead(
        symbols=args.symbols,
        timeframe=args.timeframe,
        days=args.days,
        output_dir=args.output,
    )
