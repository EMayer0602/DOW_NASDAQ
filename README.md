# DOW/NASDAQ Stock Trading Bot

Supertrend-basierter Trading Bot für US-Aktien (DOW 30 & NASDAQ 100) mit Interactive Brokers Integration.

## Features

- **Supertrend-Strategie**: Trend-folgende Signale mit ATR-basierter Volatilitätsanpassung
- **Long & Short**: Unterstützt beide Richtungen (Short erfordert Margin-Konto)
- **Zeitbasierte Exits**: Optimierte Haltezeiten pro Symbol
- **IB Integration**: Paper Trading und Live Trading mit Interactive Brokers
- **Yahoo Finance Fallback**: Automatischer Fallback für Daten
- **Backtesting**: Vollständiges Backtesting-Framework
- **Parameter-Optimierung**: Grid-Search für ATR, Hold Time, Exit-Strategien
- **Performance-Analyse**: Detaillierte Statistiken und Reports
- **Analyst Ratings**: Upgrades/Downgrades von Yahoo Finance

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

```bash
python stock_paper_trader.py --ib
```

### 3. Kontinuierliches Trading

```bash
python stock_paper_trader.py --ib --loop --interval 3600
```

---

## Parameter-Optimierung (Sweep)

Der Parameter-Sweep testet verschiedene Kombinationen und findet die optimalen Einstellungen pro Symbol.

### Getestete Parameter

| Parameter | Werte | Beschreibung |
|-----------|-------|--------------|
| ATR Period | 5-25 | Periode für ATR-Berechnung |
| ATR Multiplier | 1.5-5.0 | Multiplikator für Supertrend-Bänder |
| Hold Bars | 3-20 | Haltezeit in Bars (Zeit-Exit) |
| Time Exit | Ja/Nein | Zeitbasierter Exit aktiviert |
| Trend Flip | Ja/Nein | Trend-Flip Exit aktiviert |

### Sweep-Modi

```bash
# Quick Sweep (schnell, weniger Parameter)
python parameter_sweep.py --all-nasdaq --quick

# Standard Sweep
python parameter_sweep.py --all-nasdaq

# Full Sweep (testet alle Exit-Kombinationen)
python parameter_sweep.py --all-nasdaq --full

# Thorough Sweep (maximale Parameter, dauert lang)
python parameter_sweep.py --all-nasdaq --thorough
```

### Sweep für alle Symbole

```bash
# NASDAQ 100
python parameter_sweep.py --all-nasdaq

# DOW 30
python parameter_sweep.py --all-dow

# Spezifische Symbole
python parameter_sweep.py --symbols AAPL MSFT NVDA TSLA
```

### Output

Der Sweep erzeugt:
- `report_stocks/best_params_overall.csv` - Beste Parameter pro Symbol
- `report_stocks/best_params_overall_full.csv` - Detaillierte Ergebnisse

---

## Backtesting

### Mit optimierten Parametern (Standard)

```bash
# Verwendet automatisch best_params_overall.csv
python backtester.py --all-nasdaq
```

### Mit festen Parametern

```bash
python backtester.py --all-nasdaq --no-optimized --atr-period 14 --atr-mult 3.0
```

### Ergebnisse speichern

```bash
python backtester.py --all-nasdaq --output results.csv --trades trades.csv
```

---

## Performance-Analyse

```bash
# Analyse der Trades
python performance_analyzer.py --trades trades.csv

# Mit Equity-Kurve
python performance_analyzer.py --trades trades.csv --plot

# Als Bild speichern
python performance_analyzer.py --trades trades.csv --save-plot equity.png
```

---

## Analyst Ratings

Abrufen von Analysten-Empfehlungen und Upgrades/Downgrades.

### Recommendation Summary

```bash
# NASDAQ 100 Übersicht
python analyst_ratings.py --all-nasdaq --summary

# DOW 30 Übersicht
python analyst_ratings.py --all-dow --summary
```

### Upgrades/Downgrades

```bash
# Letzte 7 Tage
python analyst_ratings.py --all-nasdaq --recent 7

# Nur Upgrades
python analyst_ratings.py --all-nasdaq --upgrades-only

# Nur Downgrades
python analyst_ratings.py --all-dow --downgrades-only
```

### Als CSV speichern

```bash
python analyst_ratings.py --all-nasdaq --summary --output analyst_summary.csv
```

---

## Kompletter Workflow

```bash
# 1. Analyst Ratings prüfen
python analyst_ratings.py --all-nasdaq --summary

# 2. Parameter optimieren (dauert ~30-60 Min für NASDAQ 100)
python parameter_sweep.py --all-nasdaq --full

# 3. Backtest mit optimierten Parametern
python backtester.py --all-nasdaq --output results.csv --trades trades.csv

# 4. Performance analysieren
python performance_analyzer.py --trades trades.csv --plot

# 5. Paper Trading starten
python stock_paper_trader.py --ib --loop
```

---

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
| `DISABLE_TREND_FLIP_EXIT` | False | Trend-Flip Exit deaktivieren |

### stock_symbols.py

| Liste | Anzahl | Beschreibung |
|-------|--------|--------------|
| `DOW_30` | 30 | Alle DOW Jones Komponenten |
| `NASDAQ_100_TOP` | 50 | Top NASDAQ 100 Aktien |
| `DEFAULT_TRADING_SYMBOLS` | 10 | Standard-Symbole |

---

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
├── analyst_ratings.py         # Analyst Upgrades/Downgrades
├── ta/                        # Technische Analyse Module
│   ├── __init__.py
│   └── volatility.py
└── report_stocks/             # Output-Verzeichnis
    ├── best_params_overall.csv
    └── best_params_overall_full.csv
```

---

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

---

## Trading-Logik

### Entry Signale
- **LONG**: Preis kreuzt über Supertrend
- **SHORT**: Preis kreuzt unter Supertrend

### Exit Signale
1. **Zeit-Exit**: Nach `hold_bars` Bars (optimiert pro Symbol)
2. **Trend-Flip**: Preis kreuzt Supertrend in Gegenrichtung

### Position Sizing
- Feste $10,000 pro Position
- Maximal 15 offene Positionen (10 Long + 5 Short)
- Startkapital: $100,000

---

## CLI-Referenz

### parameter_sweep.py

| Option | Beschreibung |
|--------|--------------|
| `--all-nasdaq` | Alle NASDAQ 100 Symbole |
| `--all-dow` | Alle DOW 30 Symbole |
| `--symbols X Y Z` | Spezifische Symbole |
| `--quick` | Schneller Sweep (weniger Parameter) |
| `--full` | Alle Exit-Kombinationen testen |
| `--thorough` | Maximale Parameter |
| `--period 1y` | Historischer Zeitraum |
| `--output FILE` | Output-Datei |

### backtester.py

| Option | Beschreibung |
|--------|--------------|
| `--all-nasdaq` | Alle NASDAQ 100 Symbole |
| `--all-dow` | Alle DOW 30 Symbole |
| `--no-optimized` | Feste Parameter verwenden |
| `--params-file FILE` | Eigene Parameter-CSV |
| `--output FILE` | Ergebnisse speichern |
| `--trades FILE` | Trades speichern |

### analyst_ratings.py

| Option | Beschreibung |
|--------|--------------|
| `--all-nasdaq` | Alle NASDAQ 100 Symbole |
| `--all-dow` | Alle DOW 30 Symbole |
| `--summary` | Empfehlungs-Übersicht |
| `--upgrades-only` | Nur Upgrades |
| `--downgrades-only` | Nur Downgrades |
| `--recent N` | Letzte N Tage |
| `--output FILE` | Als CSV speichern |

---

## Risiko-Hinweise

- **Paper Trading zuerst**: Immer erst im Paper-Modus testen
- **Margin für Shorts**: Short-Positionen erfordern ein Margin-Konto
- **Marktrisiko**: Vergangene Performance ist keine Garantie für zukünftige Ergebnisse
- **Live Trading**: `--live` Flag nur mit Bedacht verwenden

## Lizenz

MIT License
