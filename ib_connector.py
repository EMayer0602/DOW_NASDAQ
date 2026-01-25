"""
Interactive Brokers Connection Module
Uses ib_insync library for paper/live trading

Setup:
1. Install TWS or IB Gateway
2. Enable API connections in TWS/Gateway settings
3. For paper trading: port 7497 (TWS) or 4002 (Gateway)
4. For live trading: port 7496 (TWS) or 4001 (Gateway)
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
import logging

try:
    from ib_insync import IB, Stock, MarketOrder, LimitOrder, util
    from ib_insync import Contract, Order, Trade, Position
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("WARNING: ib_insync not installed. Run: pip install ib_insync")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection settings
IB_HOST = "127.0.0.1"
IB_PAPER_PORT = 7497  # TWS Paper Trading
IB_GATEWAY_PAPER_PORT = 4002  # IB Gateway Paper Trading
IB_LIVE_PORT = 7496   # TWS Live Trading
IB_CLIENT_ID = 1

class IBConnector:
    """Interactive Brokers connection handler."""

    def __init__(self, paper_trading: bool = True, use_gateway: bool = False):
        """
        Initialize IB connection.

        Args:
            paper_trading: True for paper trading, False for live
            use_gateway: True to use IB Gateway, False for TWS
        """
        if not IB_AVAILABLE:
            raise ImportError("ib_insync not installed. Run: pip install ib_insync")

        self.ib = IB()
        self.paper_trading = paper_trading
        self.use_gateway = use_gateway
        self.connected = False

        # Set port based on settings
        if paper_trading:
            self.port = IB_GATEWAY_PAPER_PORT if use_gateway else IB_PAPER_PORT
        else:
            self.port = 4001 if use_gateway else IB_LIVE_PORT

    def connect(self, client_id: int = IB_CLIENT_ID, timeout: int = 30) -> bool:
        """Connect to IB TWS/Gateway."""
        try:
            self.ib.connect(IB_HOST, self.port, clientId=client_id, timeout=timeout)
            self.ib.sleep(2)  # Wait for initial sync
            self.connected = True
            account_type = "Paper" if self.paper_trading else "LIVE"
            logger.info(f"Connected to IB ({account_type}) on port {self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            logger.info("Make sure TWS/Gateway is running and API is enabled")
            return False

    def disconnect(self):
        """Disconnect from IB."""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB")

    def get_contract(self, symbol: str) -> Stock:
        """Create stock contract for symbol."""
        contract = Stock(symbol, 'SMART', 'USD')
        self.ib.qualifyContracts(contract)
        return contract

    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for symbol."""
        try:
            contract = self.get_contract(symbol)
            ticker = self.ib.reqMktData(contract, '', False, False)
            self.ib.sleep(1)  # Wait for data

            price = ticker.marketPrice()
            if price and price > 0:
                return float(price)

            # Fallback to last price
            if ticker.last and ticker.last > 0:
                return float(ticker.last)

            # Fallback to close price
            if ticker.close and ticker.close > 0:
                return float(ticker.close)

            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None

    def get_ohlcv(self, symbol: str, timeframe: str = "1 hour",
                  duration: str = "1 M") -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for symbol.

        Args:
            symbol: Stock symbol
            timeframe: Bar size (1 min, 5 mins, 15 mins, 1 hour, 1 day)
            duration: How far back (1 D, 1 W, 1 M, 1 Y)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            contract = self.get_contract(symbol)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=duration,
                barSizeSetting=timeframe,
                whatToShow='TRADES',
                useRTH=True,  # Regular trading hours only
                formatDate=1
            )

            if not bars:
                return None

            df = util.df(bars)
            df = df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.set_index('timestamp')

            return df[['open', 'high', 'low', 'close', 'volume']]

        except Exception as e:
            logger.error(f"Error getting OHLCV for {symbol}: {e}")
            return None

    def place_market_order(self, symbol: str, quantity: int,
                           action: str = "BUY") -> Optional[Trade]:
        """
        Place a market order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            action: "BUY" or "SELL"

        Returns:
            Trade object or None
        """
        try:
            contract = self.get_contract(symbol)
            order = MarketOrder(action, quantity)
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Placed {action} market order: {quantity} {symbol}")
            return trade
        except Exception as e:
            logger.error(f"Error placing market order: {e}")
            return None

    def place_limit_order(self, symbol: str, quantity: int,
                          limit_price: float, action: str = "BUY") -> Optional[Trade]:
        """
        Place a limit order.

        Args:
            symbol: Stock symbol
            quantity: Number of shares
            limit_price: Limit price
            action: "BUY" or "SELL"

        Returns:
            Trade object or None
        """
        try:
            contract = self.get_contract(symbol)
            order = LimitOrder(action, quantity, limit_price)
            trade = self.ib.placeOrder(contract, order)
            logger.info(f"Placed {action} limit order: {quantity} {symbol} @ ${limit_price}")
            return trade
        except Exception as e:
            logger.error(f"Error placing limit order: {e}")
            return None

    def get_positions(self) -> List[Position]:
        """Get all current positions."""
        return self.ib.positions()

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol."""
        positions = self.get_positions()
        for pos in positions:
            if pos.contract.symbol == symbol:
                return pos
        return None

    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary."""
        summary = {}
        account_values = self.ib.accountSummary()
        for av in account_values:
            summary[av.tag] = av.value
        return summary

    def get_buying_power(self) -> float:
        """Get available buying power."""
        summary = self.get_account_summary()
        return float(summary.get('BuyingPower', 0))

    def get_net_liquidation(self) -> float:
        """Get net liquidation value (total account value)."""
        summary = self.get_account_summary()
        return float(summary.get('NetLiquidation', 0))

    def cancel_order(self, trade: Trade):
        """Cancel an open order."""
        self.ib.cancelOrder(trade.order)
        logger.info(f"Cancelled order: {trade.order.orderId}")

    def cancel_all_orders(self):
        """Cancel all open orders."""
        self.ib.reqGlobalCancel()
        logger.info("Cancelled all open orders")

    def close_position(self, symbol: str) -> Optional[Trade]:
        """Close position for symbol."""
        position = self.get_position(symbol)
        if position is None:
            logger.warning(f"No position found for {symbol}")
            return None

        quantity = abs(position.position)
        action = "SELL" if position.position > 0 else "BUY"

        return self.place_market_order(symbol, int(quantity), action)

    def close_all_positions(self):
        """Close all open positions."""
        positions = self.get_positions()
        for pos in positions:
            if pos.position != 0:
                self.close_position(pos.contract.symbol)


class IBDataFeed:
    """Real-time data feed from IB."""

    def __init__(self, connector: IBConnector):
        self.connector = connector
        self.subscriptions = {}

    def subscribe(self, symbol: str, callback):
        """Subscribe to real-time data for symbol."""
        contract = self.connector.get_contract(symbol)
        ticker = self.connector.ib.reqMktData(contract, '', False, False)
        ticker.updateEvent += callback
        self.subscriptions[symbol] = ticker
        logger.info(f"Subscribed to {symbol} data feed")

    def unsubscribe(self, symbol: str):
        """Unsubscribe from symbol data."""
        if symbol in self.subscriptions:
            self.connector.ib.cancelMktData(self.subscriptions[symbol].contract)
            del self.subscriptions[symbol]
            logger.info(f"Unsubscribed from {symbol} data feed")

    def unsubscribe_all(self):
        """Unsubscribe from all data feeds."""
        for symbol in list(self.subscriptions.keys()):
            self.unsubscribe(symbol)


# Utility functions for compatibility with Crypto9 code
def fetch_ohlcv_ib(connector: IBConnector, symbol: str,
                   timeframe: str = "6h", limit: int = 500) -> List:
    """
    Fetch OHLCV data in CCXT-compatible format.

    Args:
        connector: IBConnector instance
        symbol: Stock symbol
        timeframe: Timeframe string (1m, 5m, 15m, 1h, 6h, 1d)
        limit: Number of bars

    Returns:
        List of [timestamp, open, high, low, close, volume]
    """
    # Map timeframe to IB format
    tf_map = {
        "1m": "1 min",
        "5m": "5 mins",
        "15m": "15 mins",
        "1h": "1 hour",
        "6h": "1 hour",  # IB doesn't have 6h, use 1h
        "1d": "1 day"
    }

    # Map timeframe to duration
    duration_map = {
        "1m": "1 D",
        "5m": "1 W",
        "15m": "1 M",
        "1h": "1 M",
        "6h": "6 M",
        "1d": "1 Y"
    }

    ib_tf = tf_map.get(timeframe, "1 hour")
    duration = duration_map.get(timeframe, "1 M")

    df = connector.get_ohlcv(symbol, ib_tf, duration)
    if df is None:
        return []

    # Convert to CCXT format
    result = []
    for idx, row in df.iterrows():
        result.append([
            int(idx.timestamp() * 1000),  # timestamp in ms
            row['open'],
            row['high'],
            row['low'],
            row['close'],
            row['volume']
        ])

    return result[-limit:]  # Return last N bars


# Test connection
if __name__ == "__main__":
    print("Testing IB Connection...")
    print("Make sure TWS Paper Trading is running with API enabled on port 7497")

    connector = IBConnector(paper_trading=True)
    if connector.connect():
        print("\n=== Account Info ===")
        print(f"Buying Power: ${connector.get_buying_power():,.2f}")
        print(f"Net Liquidation: ${connector.get_net_liquidation():,.2f}")

        print("\n=== Current Positions ===")
        positions = connector.get_positions()
        for pos in positions:
            print(f"  {pos.contract.symbol}: {pos.position} shares @ ${pos.avgCost:.2f}")

        print("\n=== Testing Price Fetch ===")
        price = connector.get_current_price("AAPL")
        if price:
            print(f"  AAPL: ${price:.2f}")

        print("\n=== Testing OHLCV Fetch ===")
        df = connector.get_ohlcv("AAPL", "1 hour", "1 W")
        if df is not None:
            print(f"  Got {len(df)} bars for AAPL")
            print(df.tail())

        connector.disconnect()
    else:
        print("Could not connect to IB")
