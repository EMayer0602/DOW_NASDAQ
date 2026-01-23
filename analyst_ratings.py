#!/usr/bin/env python3
"""
Analyst Ratings Module for DOW/NASDAQ Trading
Fetches analyst upgrades/downgrades from Yahoo Finance and IB.

Usage:
    python analyst_ratings.py                    # Show all ratings
    python analyst_ratings.py --symbols AAPL     # Specific symbols
    python analyst_ratings.py --upgrades-only    # Only show upgrades
    python analyst_ratings.py --recent 7         # Last 7 days
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import pandas as pd

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("WARNING: yfinance not installed. Run: pip install yfinance")

from stock_symbols import DEFAULT_TRADING_SYMBOLS, DOW_30, NASDAQ_100_TOP


# ============================================
# DATA CLASSES
# ============================================
@dataclass
class AnalystRating:
    """Single analyst rating/recommendation."""
    symbol: str
    firm: str
    to_grade: str
    from_grade: str
    action: str  # up, down, main, init, reit
    date: datetime

    @property
    def is_upgrade(self) -> bool:
        return self.action.lower() in ['up', 'upgrade']

    @property
    def is_downgrade(self) -> bool:
        return self.action.lower() in ['down', 'downgrade']

    @property
    def is_new(self) -> bool:
        return self.action.lower() in ['init', 'initiated']


@dataclass
class AnalystSummary:
    """Summary of analyst recommendations for a symbol."""
    symbol: str
    strong_buy: int = 0
    buy: int = 0
    hold: int = 0
    sell: int = 0
    strong_sell: int = 0
    target_mean: float = 0.0
    target_low: float = 0.0
    target_high: float = 0.0
    current_price: float = 0.0

    @property
    def total_analysts(self) -> int:
        return self.strong_buy + self.buy + self.hold + self.sell + self.strong_sell

    @property
    def bullish_pct(self) -> float:
        if self.total_analysts == 0:
            return 0.0
        return (self.strong_buy + self.buy) / self.total_analysts * 100

    @property
    def upside_pct(self) -> float:
        if self.current_price == 0:
            return 0.0
        return (self.target_mean - self.current_price) / self.current_price * 100

    @property
    def consensus(self) -> str:
        if self.total_analysts == 0:
            return "N/A"
        score = (self.strong_buy * 5 + self.buy * 4 + self.hold * 3 +
                 self.sell * 2 + self.strong_sell * 1) / self.total_analysts
        if score >= 4.5:
            return "Strong Buy"
        elif score >= 3.5:
            return "Buy"
        elif score >= 2.5:
            return "Hold"
        elif score >= 1.5:
            return "Sell"
        else:
            return "Strong Sell"


# ============================================
# YAHOO FINANCE RATINGS
# ============================================
def get_yf_upgrades_downgrades(symbol: str, days: int = 30) -> List[AnalystRating]:
    """Get recent analyst upgrades/downgrades from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        return []

    try:
        ticker = yf.Ticker(symbol)

        # Get upgrades/downgrades
        upgrades = ticker.upgrades_downgrades
        if upgrades is None or upgrades.empty:
            return []

        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)

        ratings = []
        for idx, row in upgrades.iterrows():
            try:
                # Handle timezone-aware timestamps
                if hasattr(idx, 'tz_localize'):
                    date = idx.to_pydatetime()
                else:
                    date = pd.to_datetime(idx).to_pydatetime()

                if date.tzinfo:
                    date = date.replace(tzinfo=None)

                if date < cutoff:
                    continue

                ratings.append(AnalystRating(
                    symbol=symbol,
                    firm=row.get('Firm', 'Unknown'),
                    to_grade=row.get('ToGrade', ''),
                    from_grade=row.get('FromGrade', ''),
                    action=row.get('Action', ''),
                    date=date
                ))
            except Exception:
                continue

        return sorted(ratings, key=lambda r: r.date, reverse=True)

    except Exception as e:
        print(f"Error getting upgrades for {symbol}: {e}")
        return []


def get_yf_recommendations(symbol: str) -> Optional[AnalystSummary]:
    """Get analyst recommendation summary from Yahoo Finance."""
    if not YFINANCE_AVAILABLE:
        return None

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get current price
        current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))

        # Get recommendations
        recs = ticker.recommendations
        if recs is not None and not recs.empty:
            # Get most recent recommendation counts
            latest = recs.iloc[-1] if len(recs) > 0 else None
            if latest is not None:
                summary = AnalystSummary(
                    symbol=symbol,
                    strong_buy=int(latest.get('strongBuy', 0)),
                    buy=int(latest.get('buy', 0)),
                    hold=int(latest.get('hold', 0)),
                    sell=int(latest.get('sell', 0)),
                    strong_sell=int(latest.get('strongSell', 0)),
                    target_mean=info.get('targetMeanPrice', 0) or 0,
                    target_low=info.get('targetLowPrice', 0) or 0,
                    target_high=info.get('targetHighPrice', 0) or 0,
                    current_price=current_price or 0
                )
                return summary

        # Fallback to info recommendations
        return AnalystSummary(
            symbol=symbol,
            strong_buy=info.get('recommendationMean', 0) or 0,
            target_mean=info.get('targetMeanPrice', 0) or 0,
            target_low=info.get('targetLowPrice', 0) or 0,
            target_high=info.get('targetHighPrice', 0) or 0,
            current_price=current_price or 0
        )

    except Exception as e:
        print(f"Error getting recommendations for {symbol}: {e}")
        return None


# ============================================
# IB RATINGS (requires connection)
# ============================================
def get_ib_ratings(symbol: str, connector) -> Optional[AnalystSummary]:
    """Get analyst ratings from Interactive Brokers."""
    if connector is None:
        return None

    try:
        # IB provides analyst ratings through fundamental data
        # This requires IB market data subscription
        from ib_insync import Stock
        contract = Stock(symbol, 'SMART', 'USD')

        # Request fundamental data (requires subscription)
        # Note: This is a placeholder - actual implementation depends on IB subscription
        return None

    except Exception as e:
        print(f"Error getting IB ratings for {symbol}: {e}")
        return None


# ============================================
# COMBINED FUNCTIONS
# ============================================
def get_all_ratings(
    symbols: List[str],
    days: int = 30,
    source: str = "yf"
) -> Dict[str, List[AnalystRating]]:
    """Get all ratings for multiple symbols."""
    all_ratings = {}

    for symbol in symbols:
        print(f"Fetching ratings for {symbol}...", end=" ", flush=True)

        if source == "yf":
            ratings = get_yf_upgrades_downgrades(symbol, days)
        else:
            ratings = []

        all_ratings[symbol] = ratings
        print(f"{len(ratings)} ratings")

    return all_ratings


def get_all_summaries(
    symbols: List[str],
    source: str = "yf"
) -> Dict[str, AnalystSummary]:
    """Get recommendation summaries for multiple symbols."""
    summaries = {}

    for symbol in symbols:
        print(f"Fetching summary for {symbol}...", end=" ", flush=True)

        if source == "yf":
            summary = get_yf_recommendations(symbol)
        else:
            summary = None

        if summary:
            summaries[symbol] = summary
            print(f"{summary.consensus} ({summary.total_analysts} analysts)")
        else:
            print("N/A")

    return summaries


def filter_by_rating(
    symbols: List[str],
    min_bullish_pct: float = 50.0,
    min_analysts: int = 5,
    min_upside_pct: float = 0.0
) -> List[str]:
    """Filter symbols by analyst ratings."""
    filtered = []
    summaries = get_all_summaries(symbols)

    for symbol, summary in summaries.items():
        if summary.total_analysts < min_analysts:
            continue
        if summary.bullish_pct < min_bullish_pct:
            continue
        if summary.upside_pct < min_upside_pct:
            continue
        filtered.append(symbol)

    return filtered


def get_recent_upgrades(
    symbols: List[str],
    days: int = 7
) -> List[AnalystRating]:
    """Get all recent upgrades across symbols."""
    upgrades = []
    all_ratings = get_all_ratings(symbols, days)

    for symbol, ratings in all_ratings.items():
        for r in ratings:
            if r.is_upgrade:
                upgrades.append(r)

    return sorted(upgrades, key=lambda r: r.date, reverse=True)


def get_recent_downgrades(
    symbols: List[str],
    days: int = 7
) -> List[AnalystRating]:
    """Get all recent downgrades across symbols."""
    downgrades = []
    all_ratings = get_all_ratings(symbols, days)

    for symbol, ratings in all_ratings.items():
        for r in ratings:
            if r.is_downgrade:
                downgrades.append(r)

    return sorted(downgrades, key=lambda r: r.date, reverse=True)


# ============================================
# REPORTING
# ============================================
def print_ratings_report(ratings: Dict[str, List[AnalystRating]]):
    """Print ratings report."""
    print("\n" + "="*80)
    print("ANALYST UPGRADES/DOWNGRADES")
    print("="*80)

    all_ratings = []
    for symbol, symbol_ratings in ratings.items():
        all_ratings.extend(symbol_ratings)

    if not all_ratings:
        print("No recent ratings found")
        return

    # Sort by date
    all_ratings.sort(key=lambda r: r.date, reverse=True)

    print(f"{'Date':<12} {'Symbol':<8} {'Action':<10} {'Firm':<25} {'From':<15} {'To':<15}")
    print("-"*80)

    for r in all_ratings[:50]:  # Limit to 50
        action_str = r.action.upper()
        if r.is_upgrade:
            action_str = "UPGRADE"
        elif r.is_downgrade:
            action_str = "DOWNGRADE"

        print(f"{r.date.strftime('%Y-%m-%d'):<12} {r.symbol:<8} {action_str:<10} "
              f"{r.firm[:24]:<25} {r.from_grade[:14]:<15} {r.to_grade[:14]:<15}")

    print("="*80)


def print_summary_report(summaries: Dict[str, AnalystSummary]):
    """Print recommendation summary report."""
    print("\n" + "="*100)
    print("ANALYST RECOMMENDATION SUMMARY")
    print("="*100)

    if not summaries:
        print("No summaries available")
        return

    print(f"{'Symbol':<8} {'Consensus':<12} {'Analysts':>8} {'Bullish%':>9} "
          f"{'Price':>10} {'Target':>10} {'Upside%':>9} {'SB':>4} {'B':>4} {'H':>4} {'S':>4} {'SS':>4}")
    print("-"*100)

    # Sort by bullish percentage
    sorted_summaries = sorted(summaries.values(), key=lambda s: s.bullish_pct, reverse=True)

    for s in sorted_summaries:
        upside_str = f"{s.upside_pct:+.1f}%" if s.upside_pct != 0 else "N/A"
        print(f"{s.symbol:<8} {s.consensus:<12} {s.total_analysts:>8} {s.bullish_pct:>8.1f}% "
              f"${s.current_price:>9.2f} ${s.target_mean:>9.2f} {upside_str:>9} "
              f"{s.strong_buy:>4} {s.buy:>4} {s.hold:>4} {s.sell:>4} {s.strong_sell:>4}")

    print("="*100)


def save_ratings_to_csv(ratings: Dict[str, List[AnalystRating]], filepath: str):
    """Save ratings to CSV."""
    rows = []
    for symbol, symbol_ratings in ratings.items():
        for r in symbol_ratings:
            rows.append({
                'Date': r.date.strftime('%Y-%m-%d'),
                'Symbol': r.symbol,
                'Action': r.action,
                'Firm': r.firm,
                'FromGrade': r.from_grade,
                'ToGrade': r.to_grade,
                'IsUpgrade': r.is_upgrade,
                'IsDowngrade': r.is_downgrade
            })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Ratings saved to {filepath}")


def save_summaries_to_csv(summaries: Dict[str, AnalystSummary], filepath: str):
    """Save summaries to CSV."""
    rows = []
    for symbol, s in summaries.items():
        rows.append({
            'Symbol': s.symbol,
            'Consensus': s.consensus,
            'TotalAnalysts': s.total_analysts,
            'BullishPct': s.bullish_pct,
            'StrongBuy': s.strong_buy,
            'Buy': s.buy,
            'Hold': s.hold,
            'Sell': s.sell,
            'StrongSell': s.strong_sell,
            'CurrentPrice': s.current_price,
            'TargetMean': s.target_mean,
            'TargetLow': s.target_low,
            'TargetHigh': s.target_high,
            'UpsidePct': s.upside_pct
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"Summaries saved to {filepath}")


# ============================================
# MAIN
# ============================================
def main():
    parser = argparse.ArgumentParser(description='Analyst Ratings for DOW/NASDAQ')
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_TRADING_SYMBOLS,
                        help='Symbols to check')
    parser.add_argument('--all-dow', action='store_true', help='Use all DOW 30 symbols')
    parser.add_argument('--all-nasdaq', action='store_true', help='Use all NASDAQ 100 symbols')
    parser.add_argument('--recent', type=int, default=30, help='Days to look back')
    parser.add_argument('--upgrades-only', action='store_true', help='Show only upgrades')
    parser.add_argument('--downgrades-only', action='store_true', help='Show only downgrades')
    parser.add_argument('--summary', action='store_true', help='Show recommendation summary')
    parser.add_argument('--output', default=None, help='Output CSV file')

    args = parser.parse_args()

    # Determine symbols
    if args.all_dow:
        symbols = DOW_30
    elif args.all_nasdaq:
        symbols = NASDAQ_100_TOP
    else:
        symbols = args.symbols

    print("="*60)
    print("ANALYST RATINGS - DOW/NASDAQ")
    print("="*60)
    print(f"Symbols: {len(symbols)} stocks")
    print(f"Period: Last {args.recent} days")
    print("="*60 + "\n")

    if args.summary:
        # Get recommendation summaries
        summaries = get_all_summaries(symbols)
        print_summary_report(summaries)

        if args.output:
            save_summaries_to_csv(summaries, args.output)
    else:
        # Get upgrades/downgrades
        ratings = get_all_ratings(symbols, args.recent)

        if args.upgrades_only:
            for symbol in ratings:
                ratings[symbol] = [r for r in ratings[symbol] if r.is_upgrade]
        elif args.downgrades_only:
            for symbol in ratings:
                ratings[symbol] = [r for r in ratings[symbol] if r.is_downgrade]

        print_ratings_report(ratings)

        if args.output:
            save_ratings_to_csv(ratings, args.output)


if __name__ == "__main__":
    main()
