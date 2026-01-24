# DOW_NASDAQ - Supertrend Stock Trading Bot

**Version 2.0.0** - Shared Capital Pool with Dynamic Position Sizing

## Overview
Paper trading system for DOW 30 and NASDAQ 100 stocks using Supertrend strategy.
Designed for Interactive Brokers (IB) paper and live trading.

## Key Features
- **Shared Capital Pool**: All symbols trade from one capital pool
- **Dynamic Position Sizing**: Stake = Current Cash / Max Positions (compounding effect)
- **Multi-Indicator Support**: Supertrend, JMA, KAMA, HTF Supertrend
- **Long & Short Trading**: Both directions enabled
- **Interactive HTML Reports**: Plotly charts with equity curve, P&L distribution
- **Time-based & Trend Flip Exits**: Configurable exit strategies
- **Interactive Brokers Integration**: Paper and live trading

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
- **Dynamic**: Current Cash / Max Positions (compounding)
- Gewinne → größere Stakes → größere Gewinne
- Verluste → kleinere Stakes → kleinere Verluste
- Maximum 10 open positions (default)
- Starting capital: $20,000 (default)
- Shared capital pool across all symbols

---

## Backtesting

### Basic Backtest
Run backtest on historical data:
```bash
# Backtest default symbols
python backtester.py

# Backtest specific symbols
python backtester.py --symbols AAPL MSFT NVDA

# Backtest all NASDAQ 100 stocks
python backtester.py --all-nasdaq

# Backtest all DOW 30 stocks
python backtester.py --all-dow

# Custom period and output
python backtester.py --all-nasdaq --period 1y --output results.csv --trades trades.csv
```

### Backtest Options
| Option | Description | Default |
|--------|-------------|---------|
| `--symbols SYM1 SYM2` | Symbols to backtest | DEFAULT_TRADING_SYMBOLS |
| `--all-nasdaq` | All NASDAQ 100 stocks | - |
| `--all-dow` | All DOW 30 stocks | - |
| `--period` | Historical period (1mo, 3mo, 6mo, 1y, 2y) | 1y |
| `--interval` | Bar interval (1h, 1d) | 1h |
| `--capital` | Initial capital | 20000 |
| `--max-positions` | Max open positions (stake = capital/max) | 10 |
| `--start` | Start date (YYYY-MM-DD) | - |
| `--end` | End date (YYYY-MM-DD) | - |
| `--no-optimized` | Use default params instead of optimized | - |

### Example: 2 Year Backtest
```bash
python backtester.py --all-nasdaq --period 2y
```
Output: `trades.csv` and `report.html` (auto-generated with Plotly charts)

---

## Parameter Sweep / Optimization

Find optimal parameters for each symbol using the multi-indicator parameter sweep.

### Supported Indicators
| Indicator | Description | Parameters |
|-----------|-------------|------------|
| `supertrend` | Basic Supertrend | ATR period, ATR multiplier |
| `jma` | Jurik Moving Average crossover | Fast period, Slow period, Phase |
| `kama` | Kaufman Adaptive MA | ER period, Fast EMA, Slow EMA |
| `supertrend_htf` | Supertrend with HTF filter | ATR period, ATR multiplier |

### Basic Sweep
```bash
# Supertrend sweep (default)
python parameter_sweep.py --all-nasdaq

# JMA sweep
python parameter_sweep.py --all-nasdaq --indicator jma

# KAMA sweep
python parameter_sweep.py --all-nasdaq --indicator kama

# Supertrend with HTF filter
python parameter_sweep.py --all-nasdaq --htf
python parameter_sweep.py --all-nasdaq --indicator supertrend_htf
```

### Sweep Modes
```bash
# Quick sweep (fewer combinations, faster)
python parameter_sweep.py --all-nasdaq --quick

# Standard sweep (balanced)
python parameter_sweep.py --all-nasdaq

# Thorough sweep (more combinations)
python parameter_sweep.py --all-nasdaq --thorough

# Full sweep (test all exit strategies)
python parameter_sweep.py --all-nasdaq --full
```

### Custom Parameters
```bash
# Custom ATR periods and multipliers
python parameter_sweep.py --all-nasdaq --param-a 7 10 14 20 --param-b 2.0 2.5 3.0 3.5

# Custom hold bars (max hold time)
python parameter_sweep.py --all-nasdaq --hold-bars 3 5 7 10 14 20

# JMA with custom periods
python parameter_sweep.py --all-nasdaq --indicator jma --param-a 5 7 10 --param-b 14 21 30

# KAMA with custom ER periods
python parameter_sweep.py --all-nasdaq --indicator kama --param-a 5 10 15 --param-c 20 30 40
```

### Sweep CLI Options
| Option | Description | Default |
|--------|-------------|---------|
| `--indicator, -i` | Indicator type (supertrend, jma, kama, supertrend_htf) | supertrend |
| `--htf` | Enable HTF filter | Off |
| `--htf-interval` | HTF interval | 1d |
| `--quick` | Quick sweep (fewer params) | - |
| `--thorough` | Thorough sweep (more params) | - |
| `--full` | Test all exit strategies | - |
| `--param-a` | ParamA values (ATR period / JMA fast / KAMA ER) | Auto |
| `--param-b` | ParamB values (ATR mult / JMA slow / KAMA fast) | Auto |
| `--param-c` | ParamC values (Phase / Slow EMA) | Auto |
| `--hold-bars` | Hold bars to test | 3 5 7 10 14 |
| `--output` | Output CSV file | report_stocks/best_params_overall.csv |

### Output Files
After sweep, results are saved to:
- `report_stocks/best_params_overall.csv` - Best params per symbol (for backtester)
- `report_stocks/best_params_overall_full.csv` - Full results with all metrics

---

## Performance Analysis

Generate HTML reports from backtest results:
```bash
python performance_analyzer.py --trades trades.csv --html report.html
```

Or use the backtester directly (auto-generates HTML):
```bash
python backtester.py --all-nasdaq --trades trades.csv
# Creates trades.csv and trades.html automatically
```

---

## File Structure
```
DOW_NASDAQ/
├── stock_paper_trader.py         # Main trading script
├── backtester.py                 # Backtesting engine
├── parameter_sweep.py            # Multi-indicator parameter optimization
├── performance_analyzer.py       # HTML report generator
├── ib_connector.py               # Interactive Brokers connection
├── stock_symbols.py              # DOW/NASDAQ symbol lists
├── stock_settings.py             # Trading configuration
├── optimal_hold_times_defaults.py # Hold periods per symbol
├── report_stocks/
│   ├── best_params_overall.csv   # Optimized parameters
│   └── best_params_overall_full.csv # Full sweep results
├── ta/
│   ├── __init__.py
│   ├── indicators.py             # JMA, KAMA, HTF Supertrend
│   └── volatility.py             # ATR calculation
├── run_stock_trader.bat          # Simulation mode launcher
├── run_ib_paper.bat              # IB paper trading launcher
└── stock_trading_state.json      # Portfolio state (auto-created)
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

## Batch Files

| Datei | Beschreibung |
|-------|--------------|
| `run_stock_trader.bat` | Stock Simulation (yfinance) |
| `run_ib_paper.bat` | IB Paper Trading (Loop) |

---

## Disclaimer
This software is for educational and paper trading purposes only.
Trading stocks involves risk. Always test thoroughly in paper trading mode
before using real money. The authors are not responsible for any losses.
