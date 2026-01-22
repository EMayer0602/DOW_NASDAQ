#!/usr/bin/env python3
"""
Sweep parameters for a single symbol and merge into existing results.

Usage:
    Add symbol:    python sweep_single_symbol.py --add "DOGE/USDT"
    Remove symbol: python sweep_single_symbol.py --remove "DOGE/USDT"
    List symbols:  python sweep_single_symbol.py --list
"""

import argparse
import os
import sys
import pandas as pd
import shutil
from datetime import datetime

# Import from Supertrend_5Min
import Supertrend_5Min as st

BASE_OUT_DIR = st.BASE_OUT_DIR
OVERALL_PARAMS_CSV = st.OVERALL_PARAMS_CSV
OVERALL_DETAILED_HTML = st.OVERALL_DETAILED_HTML
OVERALL_FLAT_CSV = st.OVERALL_FLAT_CSV
OVERALL_FLAT_JSON = st.OVERALL_FLAT_JSON


def backup_file(filepath: str) -> str:
    """Create backup of file before modification."""
    if os.path.exists(filepath):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{filepath}.backup_{timestamp}"
        shutil.copy2(filepath, backup_path)
        print(f"[Backup] {filepath} -> {backup_path}")
        return backup_path
    return None


def list_symbols():
    """List all symbols currently in best_params_overall.csv."""
    if not os.path.exists(OVERALL_PARAMS_CSV):
        print(f"[Error] File not found: {OVERALL_PARAMS_CSV}")
        return []

    df = pd.read_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",")
    symbols = sorted(df["Symbol"].unique())
    print(f"\n=== Current Symbols ({len(symbols)}) ===")
    for sym in symbols:
        count = len(df[df["Symbol"] == sym])
        print(f"  {sym} ({count} entries)")
    return symbols


def remove_symbol(symbol: str):
    """Remove a symbol from all output files."""
    symbol = symbol.strip()
    print(f"\n=== Removing Symbol: {symbol} ===\n")

    # 1. Remove from best_params_overall.csv
    if os.path.exists(OVERALL_PARAMS_CSV):
        backup_file(OVERALL_PARAMS_CSV)
        df = pd.read_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",")
        before_count = len(df)
        df = df[df["Symbol"] != symbol]
        after_count = len(df)
        removed = before_count - after_count
        if removed > 0:
            df.to_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",", index=False, encoding="utf-8")
            print(f"[OK] Removed {removed} entries from {OVERALL_PARAMS_CSV}")
        else:
            print(f"[Skip] Symbol not found in {OVERALL_PARAMS_CSV}")

    # 2. Remove from flat trades CSV
    if os.path.exists(OVERALL_FLAT_CSV):
        backup_file(OVERALL_FLAT_CSV)
        df = pd.read_csv(OVERALL_FLAT_CSV, sep=";", decimal=",")
        before_count = len(df)
        df = df[df["Symbol"] != symbol]
        after_count = len(df)
        removed = before_count - after_count
        if removed > 0:
            df.to_csv(OVERALL_FLAT_CSV, sep=";", decimal=",", index=False, encoding="utf-8")
            print(f"[OK] Removed {removed} entries from {OVERALL_FLAT_CSV}")

    # 3. Remove from flat trades JSON
    if os.path.exists(OVERALL_FLAT_JSON):
        import json
        backup_file(OVERALL_FLAT_JSON)
        with open(OVERALL_FLAT_JSON, 'r') as f:
            trades = json.load(f)
        before_count = len(trades)
        trades = [t for t in trades if t.get("Symbol") != symbol]
        after_count = len(trades)
        removed = before_count - after_count
        if removed > 0:
            with open(OVERALL_FLAT_JSON, 'w') as f:
                json.dump(trades, f, indent=2)
            print(f"[OK] Removed {removed} entries from {OVERALL_FLAT_JSON}")

    # 4. Remove symbol-specific files
    safe_symbol = symbol.replace("/", "_")
    for root, dirs, files in os.walk(BASE_OUT_DIR):
        for file in files:
            if safe_symbol in file:
                filepath = os.path.join(root, file)
                os.remove(filepath)
                print(f"[OK] Removed {filepath}")

    print(f"\n[Done] Symbol {symbol} removed from all files.")
    print("[Note] Run 'python Supertrend_5Min.py' to regenerate overall_best_detailed.html")


def add_symbol(symbol: str):
    """Run parameter sweep for a single symbol and merge into existing results."""
    symbol = symbol.strip()
    print(f"\n=== Adding Symbol: {symbol} ===\n")

    # Backup existing files
    backup_file(OVERALL_PARAMS_CSV)

    # Load existing results
    existing_df = None
    if os.path.exists(OVERALL_PARAMS_CSV):
        existing_df = pd.read_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",")
        # Remove any existing entries for this symbol (will be replaced)
        existing_df = existing_df[existing_df["Symbol"] != symbol]
        print(f"[Info] Loaded {len(existing_df)} existing entries (excluding {symbol})")

    # Temporarily override SYMBOLS to just this one
    original_symbols = st.SYMBOLS.copy() if hasattr(st, 'SYMBOLS') and st.SYMBOLS else []
    st.SYMBOLS = [symbol]

    print(f"[Run] Starting parameter sweep for {symbol}...")
    print(f"[Info] This may take several minutes...\n")

    try:
        # Run the parameter sweep
        new_rows = st.run_parameter_sweep()

        if not new_rows:
            print(f"[Warning] No results generated for {symbol}")
            st.SYMBOLS = original_symbols
            return

        print(f"\n[OK] Generated {len(new_rows)} parameter combinations for {symbol}")

        # Merge with existing results
        new_df = pd.DataFrame(new_rows)
        if existing_df is not None and not existing_df.empty:
            merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            merged_df = new_df

        # Save merged results
        merged_df.to_csv(OVERALL_PARAMS_CSV, sep=";", decimal=",", index=False, encoding="utf-8")
        print(f"[OK] Saved {len(merged_df)} total entries to {OVERALL_PARAMS_CSV}")

        # Regenerate flat trades and detailed HTML
        print("\n[Run] Regenerating overall reports...")
        st.SYMBOLS = list(merged_df["Symbol"].unique())
        st.run_overall_best_params()

        print(f"\n[Done] Symbol {symbol} added successfully!")

    except Exception as e:
        print(f"[Error] Sweep failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Restore original symbols
        st.SYMBOLS = original_symbols


def main():
    parser = argparse.ArgumentParser(
        description="Add or remove a single symbol without full sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python sweep_single_symbol.py --add "DOGE/USDT"
    python sweep_single_symbol.py --remove "DOGE/USDT"
    python sweep_single_symbol.py --list
        """
    )
    parser.add_argument("--add", type=str, help="Symbol to add (e.g. 'DOGE/USDT')")
    parser.add_argument("--remove", type=str, help="Symbol to remove (e.g. 'DOGE/USDT')")
    parser.add_argument("--list", action="store_true", help="List all current symbols")

    args = parser.parse_args()

    if not any([args.add, args.remove, args.list]):
        parser.print_help()
        sys.exit(1)

    if args.list:
        list_symbols()
    elif args.remove:
        remove_symbol(args.remove)
    elif args.add:
        add_symbol(args.add)


if __name__ == "__main__":
    main()
