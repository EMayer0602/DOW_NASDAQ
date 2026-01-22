import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv

# .env laden (aus aktuellem Verzeichnis)
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY_TEST")
API_SECRET = os.getenv("BINANCE_API_SECRET_TEST")

BASE_URL = "https://testnet.binance.vision"

def sign_request(params, secret):
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    return query_string + "&signature=" + signature

def place_order(symbol, side, order_type, quantity, price=None):
    endpoint = "/api/v3/order"
    url = BASE_URL + endpoint

    params = {
        "symbol": symbol,
        "side": side,              # "BUY" oder "SELL"
        "type": order_type,        # "LIMIT" oder "MARKET"
        "timeInForce": "GTC",      # nur für LIMIT nötig
        "quantity": quantity,
        "timestamp": int(time.time() * 1000)
    }

    if price:
        params["price"] = price

    query = sign_request(params, API_SECRET)

    headers = {
        "X-MBX-APIKEY": API_KEY
    }

    response = requests.post(url, headers=headers, data=query)
    return response.json()

# Beispiel: Kaufe 0.001 BTC gegen USDT zu 90.000 USDT
order = place_order("BTCUSDT", "BUY", "LIMIT", 0.001, 90000)
print(order)
