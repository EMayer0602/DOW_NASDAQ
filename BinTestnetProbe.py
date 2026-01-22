import json
import os

import ccxt

from paper_trader import load_env_file, get_api_credentials


def main():
    load_env_file()
    key, secret = get_api_credentials(use_testnet=True)
    if not key or not secret:
        print("[Probe] Missing BINANCE_API_KEY_TEST / BINANCE_API_SECRET_TEST in .env")
        return
    ex = ccxt.binance({
        "apiKey": key,
        "secret": secret,
        "enableRateLimit": True,
        "timeout": 15000,
    })
    # Ensure sandbox mode (spot testnet)
    ex.set_sandbox_mode(True)
    try:
        bal = ex.fetch_balance()
        free = bal.get("free", {}) or {}
        total = bal.get("total", {}) or {}
        used = bal.get("used", {}) or {}
        print("[Probe] Free:")
        print(json.dumps(free, indent=2))
        print("[Probe] Total:")
        print(json.dumps(total, indent=2))
        print("[Probe] Used:")
        print(json.dumps(used, indent=2))
    except Exception as exc:
        print(f"[Probe] Failed to fetch balance: {exc}")


if __name__ == "__main__":
    main()
