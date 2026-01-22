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

def place_limit_order(symbol, side, quantity, price):
    endpoint = "/api/v3/order"
    url = BASE_URL + endpoint

    params = {
        "symbol": symbol,
        "side": side,              # "BUY" oder "SELL"
        "type": "LIMIT",
        "timeInForce": "GTC",
        "quantity": quantity,
        "price": price,
        "timestamp": int(time.time() * 1000)
    }

    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
    response = requests.post(url, headers=headers, data=query)
    return response.json()

def get_order(symbol, order_id):
    endpoint = "/api/v3/order"
    url = BASE_URL + endpoint

    params = {
        "symbol": symbol,
        "orderId": order_id,
        "timestamp": int(time.time() * 1000)
    }

    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
    response = requests.get(url + "?" + query, headers=headers)
    return response.json()

def cancel_order(symbol, order_id):
    endpoint = "/api/v3/order"
    url = BASE_URL + endpoint

    params = {
        "symbol": symbol,
        "orderId": order_id,
        "timestamp": int(time.time() * 1000)
    }

    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
    response = requests.delete(url + "?" + query, headers=headers)
    return response.json()

# --- Workflow ---
print("1️⃣ Limit-Order platzieren...")
order = place_limit_order("BTCUSDT", "BUY", 0.001, 10000)  # absichtlich weit unter Marktpreis
print(order)

order_id = order.get("orderId")

print("\n2️⃣ Orderstatus abfragen...")
status = get_order("BTCUSDT", order_id)
print(status)

print("\n3️⃣ Order stornieren...")
cancel = cancel_order("BTCUSDT", order_id)
print(cancel)
