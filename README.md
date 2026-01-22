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

## Dashboard (TestnetDashboard.py)

Das Dashboard zeigt alle offenen Positionen, geschlossene Trades und Performance-Statistiken in einer HTML-Übersicht.

### Dashboard starten

**Einmalig generieren:**
```bash
python TestnetDashboard.py
```

**Kontinuierlich aktualisieren (alle 30 Sekunden):**
```bash
python TestnetDashboard.py --loop
```

**Mit benutzerdefiniertem Intervall (z.B. 60 Sekunden):**
```bash
python TestnetDashboard.py --loop --interval 60
```

### Dashboard Optionen

| Option | Beschreibung | Standard |
|--------|--------------|----------|
| `--loop` | Kontinuierlicher Modus - aktualisiert automatisch | Aus |
| `--interval N` | Aktualisierungsintervall in Sekunden | 30 |

### Dashboard Ausgabe

Das Dashboard generiert `report_testnet/dashboard.html` mit:
- **Open Positions**: Aktuelle offene Trades mit Echtzeit-Preisen
- **Closed Trades**: Alle geschlossenen Trades mit PnL
- **Performance Summary**: Win Rate, Total PnL, Equity Curve
- **Symbol Statistics**: Performance pro Symbol

### Beispiele

```bash
# Dashboard einmal generieren und im Browser öffnen
python TestnetDashboard.py
start report_testnet\dashboard.html

# Dashboard alle 60 Sekunden aktualisieren
python TestnetDashboard.py --loop --interval 60

# Dashboard alle 5 Minuten aktualisieren
python TestnetDashboard.py --loop --interval 300
```

---

## Stock Paper Trader - Alle Optionen

### Kommandozeilen-Optionen

| Option | Beschreibung | Standard |
|--------|--------------|----------|
| `--ib` | Interactive Brokers verwenden | Aus (yfinance) |
| `--live` | Live-Trading (statt Paper) | Paper |
| `--symbols SYM1 SYM2 ...` | Zu handelnde Symbole | DEFAULT_TRADING_SYMBOLS |
| `--loop` | Kontinuierlicher Modus | Aus |
| `--interval N` | Loop-Intervall in Sekunden | 3600 (1 Stunde) |
| `--state FILE` | State-Datei Pfad | stock_trading_state.json |

### Beispiele

```bash
# Simulation mit yfinance (kein IB nötig)
python stock_paper_trader.py --symbols AAPL MSFT NVDA

# IB Paper Trading - einmaliger Durchlauf
python stock_paper_trader.py --ib --symbols AAPL MSFT NVDA

# IB Paper Trading - kontinuierlich alle 30 Minuten
python stock_paper_trader.py --ib --loop --interval 1800

# IB Paper Trading - kontinuierlich jede Stunde
python stock_paper_trader.py --ib --loop --interval 3600

# IB Paper Trading - alle DOW 30 Aktien
python stock_paper_trader.py --ib --symbols AAPL MSFT JNJ JPM V HD CAT BA GS HON

# IB LIVE Trading (VORSICHT!)
python stock_paper_trader.py --ib --live --symbols AAPL MSFT

# Eigene State-Datei verwenden
python stock_paper_trader.py --ib --state my_portfolio.json
```

---

## Crypto Paper Trader (paper_trader.py) - Alle Optionen

Der originale Crypto-Trader für Binance:

| Option | Beschreibung | Standard |
|--------|--------------|----------|
| `--simulate` | Historische Simulation | Aus |
| `--start DATETIME` | Startzeit für Simulation | 24h zurück |
| `--end DATETIME` | Endzeit für Simulation | Jetzt |
| `--symbols "SYM1,SYM2"` | Zu handelnde Symbole | Aus Config |
| `--testnet` | Binance Testnet verwenden | Aus |
| `--max-positions N` | Max. offene Positionen | 10 |
| `--loop` | Kontinuierlicher Modus | Aus |
| `--interval N` | Loop-Intervall in Sekunden | 1800 |

### Crypto Beispiele

```bash
# Live-Tick (einmaliger Durchlauf)
python paper_trader.py

# Historische Simulation letzte 24h
python paper_trader.py --simulate

# Simulation ab bestimmtem Datum
python paper_trader.py --simulate --start 2025-01-01

# Binance Testnet mit Loop
python paper_trader.py --testnet --loop --interval 1800

# Bestimmte Symbole
python paper_trader.py --symbols "BTC/USDC,ETH/USDC,SOL/USDC"
```

---

## Batch-Dateien

| Datei | Beschreibung |
|-------|--------------|
| `run_stock_trader.bat` | Stock Simulation (yfinance) |
| `run_ib_paper.bat` | IB Paper Trading (Loop) |
| `run_paper_trader.bat` | Crypto Paper Trading |
| `run_live_trader.bat` | Crypto Live Trading |
| `run_full_refresh.bat` | Alle Reports neu generieren |

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
