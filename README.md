# DOW/NASDAQ Stock Trading Bot

Supertrend-basierter Trading Bot für US-Aktien (DOW 30 & NASDAQ 100) mit Interactive Brokers Integration.

## Features

- **Supertrend-Strategie**: Trend-folgende Signale mit ATR-basierter Volatilitätsanpassung
- **Long & Short**: Unterstützt beide Richtungen (Short erfordert Margin-Konto)
- **Zeitbasierte Exits**: Optimierte Haltezeiten pro Symbol
- **IB Integration**: Paper Trading und Live Trading mit Interactive Brokers
- **Yahoo Finance Fallback**: Automatischer Fallback für Daten
- **Backtesting**: Vollständiges Backtesting-Framework
- **Parameter-Optimierung**: Grid-Search für optimale Parameter
- **Performance-Analyse**: Detaillierte Statistiken und Reports

## Installation

```bash
# Repository klonen
git clone https://github.com/EMayer0602/DOW_NASDAQ.git
cd DOW_NASDAQ

# Dependencies installieren
pip install pandas numpy yfinance pytz

# Für IB Trading
pip install ib_insync

# Für Charts
pip install matplotlib
```

## Schnellstart

### 1. Simulation mit Yahoo Finance (ohne IB)

```bash
python stock_paper_trader.py
```

### 2. IB Paper Trading

1. TWS oder IB Gateway starten
2. API-Verbindungen aktivieren (Port 7497 für Paper)
3. Bot starten:

```bash
python stock_paper_trader.py --ib
```

### 3. Kontinuierliches Trading

```bash
python stock_paper_trader.py --ib --loop --interval 3600
```

## Backtesting

### Einfaches Backtest

```bash
python backtester.py
```

### Mit spezifischen Symbolen

```bash
python backtester.py --symbols AAPL MSFT NVDA --period 1y
```

### Ergebnisse speichern

```bash
python backtester.py --output results.csv --trades trades.csv
```

## Parameter-Optimierung

### Standard-Sweep

```bash
python parameter_sweep.py
```

### Schneller Sweep

```bash
python parameter_sweep.py --quick
```

### Gründlicher Sweep

```bash
python parameter_sweep.py --thorough
```

## Performance-Analyse

```bash
python performance_analyzer.py --trades trades.csv --plot
```

## Konfiguration

### stock_settings.py

| Parameter | Default | Beschreibung |
|-----------|---------|--------------|
| `START_TOTAL_CAPITAL` | 100,000 | Startkapital in USD |
| `MAX_OPEN_POSITIONS` | 10 | Maximale gleichzeitige Positionen |
| `MAX_LONG_POSITIONS` | 10 | Maximale Long-Positionen |
| `MAX_SHORT_POSITIONS` | 5 | Maximale Short-Positionen |
| `POSITION_SIZE_USD` | 10,000 | Positionsgröße pro Trade |
| `TIMEFRAME` | "1h" | Trading-Timeframe |
| `DEFAULT_ATR_PERIOD` | 10 | ATR-Periode für Supertrend |
| `DEFAULT_ATR_MULTIPLIER` | 3.0 | ATR-Multiplikator |
| `USE_TIME_BASED_EXIT` | True | Zeitbasierte Exits aktivieren |
| `RESPECT_MARKET_HOURS` | True | Nur während Marktzeiten handeln |

### stock_symbols.py

Enthält DOW 30 und NASDAQ 100 Symbole. Default-Symbole:
- AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, V, HD

## Dateistruktur

```
DOW_NASDAQ/
├── stock_paper_trader.py      # Haupt-Trading-Script
├── ib_connector.py            # IB Verbindungsmodul
├── stock_settings.py          # Konfiguration
├── stock_symbols.py           # Symbollisten
├── optimal_hold_times_defaults.py  # Haltezeiten
├── backtester.py              # Backtesting-Engine
├── parameter_sweep.py         # Parameter-Optimierung
├── performance_analyzer.py    # Performance-Analyse
├── ta/                        # Technische Analyse Module
│   ├── __init__.py
│   └── volatility.py
└── report_stocks/             # Output-Verzeichnis
    └── best_params_overall.csv
```

## IB Setup

### TWS Einstellungen

1. TWS öffnen → Configure → API → Settings
2. "Enable ActiveX and Socket Clients" aktivieren
3. "Read-Only API" deaktivieren (für Trading)
4. Port: 7497 (Paper) oder 7496 (Live)
5. Trusted IP: 127.0.0.1

### Ports

| Modus | TWS Port | Gateway Port |
|-------|----------|--------------|
| Paper | 7497 | 4002 |
| Live | 7496 | 4001 |

## Trading-Logik

### Entry Signale
- **LONG**: Preis kreuzt über Supertrend
- **SHORT**: Preis kreuzt unter Supertrend

### Exit Signale
1. **Zeit-Exit**: Nach `optimal_hold_bars` Bars
2. **Trend-Flip**: Preis kreuzt Supertrend in Gegenrichtung

### Position Sizing
- Feste $10,000 pro Position
- Maximal 10 offene Positionen (10 Long + 5 Short)
- Startkapital: $100,000

## Beispiel-Workflow

```bash
# 1. Parameter optimieren
python parameter_sweep.py --thorough --output report_stocks/best_params_overall.csv

# 2. Backtest mit optimierten Parametern
python backtester.py --output report_stocks/results.csv --trades report_stocks/trades.csv

# 3. Performance analysieren
python performance_analyzer.py --trades report_stocks/trades.csv --save-plot report_stocks/equity.png

# 4. Paper Trading starten
python stock_paper_trader.py --ib --loop
```

## Marktzeiten

- US Market: 9:30 - 16:00 Eastern
- Pre-Market: 4:00 - 9:30 Eastern
- After-Hours: 16:00 - 20:00 Eastern

Der Bot respektiert Marktzeiten standardmäßig (`RESPECT_MARKET_HOURS = True`).

## Risiko-Hinweise

- **Paper Trading zuerst**: Immer erst im Paper-Modus testen
- **Margin für Shorts**: Short-Positionen erfordern ein Margin-Konto
- **Marktrisiko**: Vergangene Performance ist keine Garantie für zukünftige Ergebnisse
- **Live Trading**: `--live` Flag nur mit Bedacht verwenden

## Lizenz

MIT License
