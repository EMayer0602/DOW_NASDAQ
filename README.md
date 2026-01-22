# DOW_NASDAQ - Supertrend Stock Trading Bot

## Overview
Paper trading system for DOW 30 and NASDAQ 100 stocks using Supertrend strategy.
Adapted from Crypto9 for use with Interactive Brokers (IB).

## Features
- Supertrend-based entry/exit signals
- Time-based exits with optimized hold periods per symbol
- Trend flip exits
- Long-only mode (stocks are harder to short)
- Interactive Brokers integration for paper/live trading
- Yahoo Finance fallback for data (simulation mode)

---

## Prerequisites

### Python Dependencies
```bash
pip install pandas numpy yfinance ib_insync pytz
```

### For IB Trading
1. Install **TWS (Trader Workstation)** or **IB Gateway**
2. Enable API in TWS settings:
   - File → Global Configuration → API → Settings
   - Enable "ActiveX and Socket Clients"
   - Port: 7497 (paper) or 7496 (live)
3. For paper trading, use a Paper Trading account in TWS

---

## Quick Start

### 1) Simulation Mode (No IB Required)
Test with Yahoo Finance data:
```bash
python stock_paper_trader.py --symbols AAPL MSFT NVDA AMZN GOOGL
```

Or use the batch file:
```bash
run_stock_trader.bat
```

### 2) IB Paper Trading
Connect to TWS Paper Trading:
```bash
python stock_paper_trader.py --ib --symbols AAPL MSFT NVDA
```

Run continuously (1 hour intervals):
```bash
python stock_paper_trader.py --ib --loop --interval 3600
```

Or use the batch file:
```bash
run_ib_paper.bat
```

### 3) IB Live Trading (CAREFUL!)
```bash
python stock_paper_trader.py --ib --live --symbols AAPL MSFT
```

---

## Configuration Files

### stock_symbols.py
Contains DOW 30 and NASDAQ 100 stock lists:
```python
DOW_30 = ["AAPL", "MSFT", "JNJ", ...]
NASDAQ_100_TOP = ["NVDA", "TSLA", "AMZN", ...]
DEFAULT_TRADING_SYMBOLS = ["AAPL", "MSFT", "NVDA", ...]
```

### stock_settings.py
Trading parameters:
```python
START_TOTAL_CAPITAL = 100_000.0
MAX_OPEN_POSITIONS = 10
POSITION_SIZE_USD = 10_000.0
TIMEFRAME = "1h"
```

### optimal_hold_times_defaults.py
Symbol-specific hold periods (1h bars):
```python
OPTIMAL_HOLD_BARS = {
    ("NVDA", "long"): 5,  # High volatility - shorter hold
    ("AAPL", "long"): 6,  # Large cap - medium hold
    ("KO", "long"): 8,    # Stable - longer hold
}
```

### report_stocks/best_params_overall.csv
Supertrend parameters per symbol:
- ATR Period (ParamA): 7-14
- ATR Multiplier (ParamB): 2.0-3.5
- Min Hold Bars (MinHoldBars): 4-8

---

## Trading Logic

### Entry Signal
- Price crosses **above** Supertrend → LONG entry
- Long-only mode (shorts disabled)

### Exit Signals
1. **Time-based Exit**: After `optimal_hold_bars` bars
2. **Trend Flip Exit**: Price crosses below Supertrend

### Position Sizing
- Fixed $10,000 per position
- Maximum 10 open positions
- Starting capital: $100,000

---

## File Structure
```
DOW_NASDAQ/
├── stock_paper_trader.py     # Main trading script
├── ib_connector.py           # Interactive Brokers connection
├── stock_symbols.py          # DOW/NASDAQ symbol lists
├── stock_settings.py         # Trading configuration
├── optimal_hold_times_defaults.py  # Hold periods per symbol
├── report_stocks/
│   └── best_params_overall.csv     # Supertrend parameters
├── run_stock_trader.bat      # Simulation mode launcher
├── run_ib_paper.bat          # IB paper trading launcher
└── stock_trading_state.json  # Portfolio state (auto-created)
```

---

## IB Connection Ports
| Mode | TWS Port | Gateway Port |
|------|----------|--------------|
| Paper | 7497 | 4002 |
| Live | 7496 | 4001 |

---

## Market Hours
- US Market: 9:30 AM - 4:00 PM Eastern
- Pre-market: 4:00 AM - 9:30 AM
- After-hours: 4:00 PM - 8:00 PM

The bot respects market hours by default (`RESPECT_MARKET_HOURS = True`).

---

## Original Crypto Version
The original Crypto9 trading bot files are still available:
- `paper_trader.py` - Crypto paper trader
- `Supertrend_5Min.py` - Crypto Supertrend logic
- `report_html/` - Crypto parameters

---

## Disclaimer
This software is for educational and paper trading purposes only.
Trading stocks involves risk. Always test thoroughly in paper trading mode
before using real money. The authors are not responsible for any losses.
