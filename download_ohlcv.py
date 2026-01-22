"""Download historical OHLCV data for all symbols and timeframes.

Usage:
    python download_ohlcv.py --start 2025-01-01
"""
import argparse
import Supertrend_5Min as st

# All timeframes needed for simulation
# Binance supports: 1h, 2h, 4h, 6h, 8h, 12h, 1d
# Other timeframes (3h, 9h, 15h, etc.) are synthesized from 1h at runtime
TIMEFRAMES_TO_DOWNLOAD = ["1h", "4h", "6h", "8h", "12h", "1d"]


def main():
    parser = argparse.ArgumentParser(description="Download historical OHLCV data")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD), defaults to now")
    parser.add_argument("--symbols", default=None, help="Comma-separated symbols (default: all)")
    args = parser.parse_args()

    # Get symbols
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(",")]
    else:
        symbols = st.SYMBOLS

    print(f"Downloading OHLCV data from {args.start} to {args.end or 'now'}")
    print(f"Symbols: {symbols}")
    print(f"Timeframes: {TIMEFRAMES_TO_DOWNLOAD}")
    print()

    total_downloaded = 0

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing {symbol}")
        print(f"{'='*60}")

        for timeframe in TIMEFRAMES_TO_DOWNLOAD:
            try:
                df = st.download_historical_ohlcv(symbol, timeframe, args.start, args.end)
                if not df.empty:
                    st.save_ohlcv_to_cache(symbol, timeframe, df)
                    print(f"[OK] {symbol} {timeframe}: {len(df)} bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
                    total_downloaded += len(df)
                else:
                    print(f"[WARN] {symbol} {timeframe}: No data")
            except Exception as e:
                print(f"[ERROR] {symbol} {timeframe}: {e}")

    print("\n" + "="*60)
    print(f"Download complete! Total: {total_downloaded} bars")
    print(f"Data saved to: {st.OHLCV_CACHE_DIR}/")
    print()
    print("Note: Timeframes like 3h, 9h, 15h, 18h, 21h etc. will be")
    print("synthesized from 1h data automatically during simulation.")


if __name__ == "__main__":
    main()
