#!/usr/bin/env python3
"""
Script to correct PnL for all historical trades using the correct formula:
- size_units = round_to_lot_size(stake / entry_price)
- fees = (entry_price + exit_price) * size_units * fee_rate
- pnl = size_units * (exit_price - entry_price) - fees
"""

import json
import os
import math
import requests
from typing import Dict

# Fee rate (0.075%)
FEE_RATE = 0.00075

# Lot size cache
_LOT_SIZE_CACHE: Dict[str, float] = {}
_LOT_SIZE_CACHE_LOADED = False


def fetch_lot_sizes_from_binance() -> Dict[str, float]:
    """Fetch lot sizes (stepSize) for all symbols from Binance API."""
    global _LOT_SIZE_CACHE, _LOT_SIZE_CACHE_LOADED
    if _LOT_SIZE_CACHE_LOADED:
        return _LOT_SIZE_CACHE

    print("[LotSize] Fetching lot sizes from Binance API...")
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
            print(f"[LotSize] Loaded {len(_LOT_SIZE_CACHE)} symbols")
        else:
            print(f"[LotSize] Failed to fetch: HTTP {response.status_code}")
    except Exception as e:
        print(f"[LotSize] Error fetching lot sizes: {e}")
    return _LOT_SIZE_CACHE


def get_lot_size(symbol: str) -> float:
    """Get lot size for a symbol, fetching from API if needed."""
    if not _LOT_SIZE_CACHE_LOADED:
        fetch_lot_sizes_from_binance()
    return _LOT_SIZE_CACHE.get(symbol, 1.0)


def round_to_lot_size(amount: float, symbol: str) -> float:
    """Round amount DOWN to the nearest valid lot size for the symbol."""
    lot_size = get_lot_size(symbol)
    if lot_size <= 0:
        return amount
    return math.floor(amount / lot_size) * lot_size


def correct_trades(json_path: str) -> int:
    """Correct PnL for all trades in the JSON file."""
    if not os.path.exists(json_path):
        print(f"[Error] File not found: {json_path}")
        return 0

    # Pre-fetch lot sizes
    fetch_lot_sizes_from_binance()

    with open(json_path, 'r') as f:
        trades = json.load(f)

    if not isinstance(trades, list):
        print("[Error] JSON is not a list of trades")
        return 0

    print(f"[Info] Processing {len(trades)} trades...")
    corrected = 0

    for t in trades:
        entry_price = float(t.get('entry_price', 0) or 0)
        exit_price = float(t.get('exit_price', 0) or 0)
        stake = float(t.get('stake', 0) or 0)
        symbol = str(t.get('symbol', '')).replace('/', '')  # "TNSR/USDT" -> "TNSRUSDT"

        if entry_price > 0 and exit_price > 0 and stake > 0:
            raw_size = stake / entry_price
            size_units = round_to_lot_size(raw_size, symbol) if symbol else raw_size
            fees = (entry_price + exit_price) * size_units * FEE_RATE
            direction = str(t.get('direction', 'Long')).lower()

            if direction == 'long':
                new_pnl = size_units * (exit_price - entry_price) - fees
            else:
                new_pnl = size_units * (entry_price - exit_price) - fees

            old_pnl = float(t.get('pnl', 0) or 0)
            if abs(new_pnl - old_pnl) > 0.001:
                t['pnl'] = round(new_pnl, 8)
                t['fees'] = round(fees, 8)
                t['size_units'] = size_units
                corrected += 1

    if corrected > 0:
        with open(json_path, 'w') as f:
            json.dump(trades, f, indent=2, default=str)
        print(f"[Success] Corrected {corrected} trades in {json_path}")
    else:
        print("[Info] No trades needed correction")

    return corrected


if __name__ == "__main__":
    import sys

    print("=== PnL Correction Script ===")
    print()

    # Default files to correct
    files_to_correct = [
        "paper_trading_simulation_log.json",
        "crypto9_testnet_closed_trades.json",
    ]

    # Allow custom file as argument
    if len(sys.argv) > 1:
        files_to_correct = sys.argv[1:]

    total_corrected = 0
    for json_file in files_to_correct:
        if os.path.exists(json_file):
            print(f"Processing: {json_file}")
            corrected = correct_trades(json_file)
            total_corrected += corrected
            print()
        else:
            print(f"[Skip] File not found: {json_file}")
            print()

    print(f"=== Done! Total corrected: {total_corrected} trades ===")
