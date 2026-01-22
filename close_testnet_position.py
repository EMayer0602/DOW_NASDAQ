#!/usr/bin/env python3
"""Close a Crypto9 testnet position and update local tracking."""

import os
import sys
import json
import time
import hmac
import hashlib
import requests
from datetime import datetime
from pathlib import Path

# Load env manually
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

FUTURES_KEY = os.getenv("BINANCE_API_KEY_TEST_F")
FUTURES_SECRET = os.getenv("BINANCE_API_SECRET_TEST_F")
FUTURES_URL = "https://testnet.binancefuture.com"

POSITIONS_FILE = "crypto9_testnet_positions.json"
CLOSED_TRADES_FILE = "crypto9_testnet_closed_trades.json"


def sign(params, secret):
    qs = "&".join([f"{k}={v}" for k, v in params.items()])
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    return qs + "&signature=" + sig


def close_futures_position(symbol, side, quantity):
    """Close a futures position."""
    close_side = "BUY" if side.upper() == "SHORT" else "SELL"

    params = {
        "symbol": symbol.replace("/", ""),
        "side": close_side,
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": int(time.time() * 1000),
        "recvWindow": 5000,
    }

    query = sign(params, FUTURES_SECRET)
    url = f"{FUTURES_URL}/fapi/v1/order?{query}"
    headers = {"X-MBX-APIKEY": FUTURES_KEY}

    response = requests.post(url, headers=headers, timeout=10)
    print(f"Response: {response.status_code}")
    if response.status_code != 200:
        print(f"Error: {response.text}")
        return None
    return response.json()


def update_local_tracking(symbol, direction, exit_price):
    """Move position from open to closed trades."""
    positions = []
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            positions = json.load(f)

    closed_trade = None
    new_positions = []

    for p in positions:
        if p["symbol"] == symbol and p["direction"] == direction:
            entry_price = p["entry_price"]
            size_units = p.get("size_units", 0)

            if direction == "short":
                pnl = (entry_price - exit_price) * size_units
                pnl_pct = (entry_price - exit_price) / entry_price * 100
            else:
                pnl = (exit_price - entry_price) * size_units
                pnl_pct = (exit_price - entry_price) / entry_price * 100

            closed_trade = {
                **p,
                "exit_price": exit_price,
                "exit_time": datetime.now().isoformat(),
                "pnl": pnl,
                "pnl_pct": pnl_pct,
            }
            print(f"  Entry: ${entry_price:.2f}")
            print(f"  Exit:  ${exit_price:.2f}")
            print(f"  PnL:   ${pnl:.2f} ({pnl_pct:+.2f}%)")
        else:
            new_positions.append(p)

    # Save updated positions
    with open(POSITIONS_FILE, "w") as f:
        json.dump(new_positions, f, indent=2)

    # Save closed trade
    if closed_trade:
        closed_trades = []
        if os.path.exists(CLOSED_TRADES_FILE):
            with open(CLOSED_TRADES_FILE) as f:
                closed_trades = json.load(f)
        closed_trades.append(closed_trade)
        with open(CLOSED_TRADES_FILE, "w") as f:
            json.dump(closed_trades, f, indent=2)
        print("  Trade moved to closed trades!")

    return closed_trade


def main():
    if len(sys.argv) < 2:
        print("Usage: python close_testnet_position.py <symbol> [direction]")
        print("Example: python close_testnet_position.py ETH/USDT short")
        return

    symbol = sys.argv[1].upper()
    direction = sys.argv[2].lower() if len(sys.argv) > 2 else None

    # Find the position
    positions = []
    if os.path.exists(POSITIONS_FILE):
        with open(POSITIONS_FILE) as f:
            positions = json.load(f)

    target = None
    for p in positions:
        if symbol in p["symbol"].upper():
            if direction is None or p["direction"] == direction:
                target = p
                break

    if not target:
        print(f"Position not found: {symbol} {direction or ''}")
        print(f"Open positions: {[p['symbol'] + ' ' + p['direction'] for p in positions]}")
        return

    print(f"=== Closing {target['symbol']} {target['direction'].upper()} ===")
    print(f"  Size: {target.get('size_units', 0)}")

    # Close on exchange
    result = close_futures_position(
        target["symbol"],
        target["direction"],
        target.get("size_units", 0)
    )

    if result and result.get("orderId"):
        avg_price = float(result.get("avgPrice", 0))
        print(f"  Closed at: ${avg_price:.2f}")
        update_local_tracking(target["symbol"], target["direction"], avg_price)
    else:
        print("  Order failed!")


if __name__ == "__main__":
    main()
