import os
import time
import hmac
import hashlib
import requests
from urllib.parse import urlencode

from paper_trader import load_env_file, get_api_credentials

BASE_URL = "https://testnet.binance.vision"
ENDPOINT = "/sapi/v1/capital/faucet/currency"


def sign_query(secret: str, params: dict) -> str:
    query = urlencode(params)
    signature = hmac.new(secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    return signature


def request_faucet(asset: str) -> None:
    load_env_file()
    key, secret = get_api_credentials(use_testnet=True)
    if not key or not secret:
        print("[Faucet] Missing BINANCE_API_KEY_TEST / BINANCE_API_SECRET_TEST in .env")
        return
    ts = int(time.time() * 1000)
    params = {"asset": asset.upper(), "timestamp": ts}
    sig = sign_query(secret, params)
    headers = {"X-MBX-APIKEY": key}
    url = BASE_URL + ENDPOINT
    try:
        resp = requests.post(url, params={**params, "signature": sig}, headers=headers, timeout=15)
        if resp.status_code == 200:
            print(f"[Faucet] Credited {asset.upper()} successfully.")
            print(resp.text)
        else:
            print(f"[Faucet] Failed ({resp.status_code}): {resp.text}")
    except Exception as exc:
        print(f"[Faucet] Request error: {exc}")


if __name__ == "__main__":
    # Change assets as needed, e.g., "EUR", "USDT", "ETH"
    request_faucet("EUR")
