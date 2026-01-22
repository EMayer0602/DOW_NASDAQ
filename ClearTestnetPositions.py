#!/usr/bin/env python3
"""Clear all testnet positions by selling all non-base currency balances."""

import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

# Load .env
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY_TEST")
API_SECRET = os.getenv("BINANCE_API_SECRET_TEST")
BASE_URL = "https://testnet.binance.vision"
RECV_WINDOW_MS = 5_000

# Base currencies to keep (don't sell these)
BASE_CURRENCIES = {"EUR", "USDT", "USDC", "BUSD", "BTC"}

# Symbol mapping for selling (asset -> trading pair)
SELL_PAIRS = {
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
    "LINK": "LINKUSDT",
    "SUI": "SUIUSDT",
    "BNB": "BNBUSDT",
    "ICP": "ICPUSDT",
    "ADA": "ADAUSDT",
    "TNSR": "TNSRUSDC",
    "ZEC": "ZECUSDC",
    "LUNC": "LUNCUSDT",
}


def sign_request(params: dict, secret: str) -> str:
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return query_string + "&signature=" + signature


def get_account_balances() -> list:
    """Get all account balances from testnet."""
    endpoint = "/api/v3/account"
    url = BASE_URL + endpoint
    params = {
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }
    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}

    try:
        response = requests.get(url + "?" + query, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("balances", [])
    except Exception as e:
        print(f"Error fetching balances: {e}")
        return []


def sell_market(symbol: str, quantity: float) -> dict:
    """Place a market sell order."""
    endpoint = "/api/v3/order"
    url = BASE_URL + endpoint
    params = {
        "symbol": symbol,
        "side": "SELL",
        "type": "MARKET",
        "quantity": quantity,
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }
    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}

    try:
        response = requests.post(url + "?" + query, headers=headers, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def buy_market_quote(symbol: str, quote_qty: float) -> dict:
    """Place a market buy order using quote currency amount (e.g., 15 USDT)."""
    endpoint = "/api/v3/order"
    url = BASE_URL + endpoint
    params = {
        "symbol": symbol,
        "side": "BUY",
        "type": "MARKET",
        "quoteOrderQty": quote_qty,
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }
    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}

    try:
        response = requests.post(url + "?" + query, headers=headers, timeout=10)
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def main():
    if not API_KEY or not API_SECRET:
        print("Error: BINANCE_API_KEY_TEST or BINANCE_API_SECRET_TEST not set in .env")
        return

    print("Fetching testnet balances...")
    balances = get_account_balances()

    if not balances:
        print("No balances found or error fetching.")
        return

    # Find non-zero balances that aren't base currencies
    to_sell = []
    for bal in balances:
        asset = bal["asset"]
        free = float(bal["free"])
        if free > 0 and asset not in BASE_CURRENCIES:
            to_sell.append((asset, free))

    if not to_sell:
        print("No positions to close. All balances are base currencies or zero.")
        return

    print(f"\nFound {len(to_sell)} position(s) to close:")
    for asset, qty in to_sell:
        print(f"  {asset}: {qty}")

    print("\nSelling positions...")
    for asset, qty in to_sell:
        symbol = SELL_PAIRS.get(asset)
        if not symbol:
            print(f"  {asset}: No trading pair configured, skipping")
            continue

        # Round quantity appropriately
        if qty < 1:
            qty_str = f"{qty:.8f}"
        elif qty < 100:
            qty_str = f"{qty:.4f}"
        else:
            qty_str = f"{qty:.2f}"

        print(f"  Selling {qty_str} {asset} via {symbol}...", end=" ")
        result = sell_market(symbol, float(qty_str))

        if "orderId" in result:
            filled_qty = result.get("executedQty", "?")
            print(f"OK (orderId: {result['orderId']}, filled: {filled_qty})")
        else:
            error = result.get("msg") or result.get("error") or str(result)
            # Check if NOTIONAL error (position too small)
            if "NOTIONAL" in str(error):
                print(f"TOO SMALL - topping up first...")
                # Buy 15 USDT worth to get above minimum
                buy_result = buy_market_quote(symbol, 15.0)
                if "orderId" in buy_result:
                    bought_qty = float(buy_result.get("executedQty", 0))
                    print(f"    Bought {bought_qty} {asset}, now selling all...")
                    # Get updated balance and sell
                    time.sleep(0.5)
                    new_balances = get_account_balances()
                    new_qty = 0
                    for bal in new_balances:
                        if bal["asset"] == asset:
                            new_qty = float(bal["free"])
                            break
                    if new_qty > 0:
                        if new_qty < 1:
                            new_qty_str = f"{new_qty:.8f}"
                        elif new_qty < 100:
                            new_qty_str = f"{new_qty:.4f}"
                        else:
                            new_qty_str = f"{new_qty:.2f}"
                        sell_result = sell_market(symbol, float(new_qty_str))
                        if "orderId" in sell_result:
                            print(f"    OK - sold {sell_result.get('executedQty', '?')} {asset}")
                        else:
                            print(f"    FAILED: {sell_result.get('msg') or sell_result}")
                else:
                    print(f"    Buy failed: {buy_result.get('msg') or buy_result}")
            else:
                print(f"FAILED: {error}")

    print("\nDone.")


if __name__ == "__main__":
    main()
