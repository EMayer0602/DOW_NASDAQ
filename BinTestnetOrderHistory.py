import os
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv
from datetime import datetime, timezone

# .env laden (aus aktuellem Verzeichnis)
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY_TEST")
API_SECRET = os.getenv("BINANCE_API_SECRET_TEST")

BASE_URL = "https://testnet.binance.vision"
RECV_WINDOW_MS = 5_000

def sign_request(params, secret):
    query_string = "&".join([f"{k}={v}" for k, v in params.items()])
    signature = hmac.new(secret.encode("utf-8"), query_string.encode("utf-8"), hashlib.sha256).hexdigest()
    return query_string + "&signature=" + signature

def get_all_orders(symbol):
    endpoint = "/api/v3/allOrders"
    url = BASE_URL + endpoint

    params = {
        "symbol": symbol,
        "timestamp": int(time.time() * 1000),
        "recvWindow": RECV_WINDOW_MS,
    }

    query = sign_request(params, API_SECRET)
    headers = {"X-MBX-APIKEY": API_KEY}
    try:
        response = requests.get(url + "?" + query, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        return {"error": str(exc), "status_code": getattr(exc, "response", None) and getattr(exc.response, "status_code", None)}

def format_orders(orders, symbol_hint=None):
    rows = []
    for o in orders:
        ts = o.get("time") or o.get("transactTime") or o.get("updateTime")
        readable_time = (
            datetime.fromtimestamp(ts / 1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
            if ts
            else "N/A"
        )
        # Binance market orders often return price=0; derive avg price from filled quote/base
        price_raw = float(o.get("price", 0) or 0)
        executed_qty = float(o.get("executedQty", 0) or 0)
        quote_qty = float(o.get("cummulativeQuoteQty", 0) or 0)
        derived_price = price_raw
        if derived_price == 0 and executed_qty > 0:
            derived_price = quote_qty / executed_qty if executed_qty else 0

        rows.append([
                o.get("symbol") or symbol_hint or "?",
                o.get("orderId"),
                o.get("side"),
                o.get("type"),
                o.get("status"),
                derived_price,
                executed_qty,
                quote_qty,
                readable_time
        ])
    return rows

# --- Mehrere Symbole abfragen ---
symbols = ["BTCEUR", "ETHEUR", "SUIEUR", "SOLEUR", "XRPEUR", "LINKEUR", "ICPEUR", "BNBEUR", "BTCUSDC", "ETHUSDC", "BNBUSDC","ZECUSDC", "LUNCUSDT","TNSRUSDC"]  # beliebig erweiterbar

for sym in symbols:
    print(f"\n📊 Order-History für {sym}:")
    orders = get_all_orders(sym)
    if isinstance(orders, dict) and orders.get("error"):
        print("❌ Fehler:", orders)
        continue
    if isinstance(orders, list) and len(orders) > 0:
        print(f"{'Symbol':<10} {'ID':<12} {'Side':<6} {'Typ':<10} {'Status':<12} {'AvgPreis':<15} {'ExecMenge':<12} {'QuoteSumme':<15} {'Zeit':<22}")
        print("-"*124)
        for row in format_orders(orders, symbol_hint=sym):
            print(f"{row[0]:<10} {row[1]:<12} {row[2]:<6} {row[3]:<10} {row[4]:<12} {row[5]:<15.8f} {row[6]:<12.6f} {row[7]:<15.8f} {row[8]:<22}")
    else:
        print("❌ Keine Orders oder leere Antwort:", orders)
