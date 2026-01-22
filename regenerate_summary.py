#!/usr/bin/env python3
"""Regenerate trading summary from existing simulation logs (offline)."""

import json
import os
import sys
from datetime import datetime, timezone

import pandas as pd

# Import summary generation functions from paper_trader
sys.path.insert(0, os.path.dirname(__file__))

START_EQUITY = 16000.0
BERLIN_TZ = "Europe/Berlin"


def calc_symbol_stats(trades_df):
    """Calculate per-symbol statistics."""
    if trades_df.empty or "symbol" not in trades_df.columns:
        return []

    symbol_stats = []
    for symbol in sorted(trades_df["symbol"].unique()):
        sym_df = trades_df[trades_df["symbol"] == symbol].copy()
        if sym_df.empty:
            continue

        if "exit_time" in sym_df.columns:
            sym_df = sym_df.sort_values("exit_time")

        total_trades = len(sym_df)
        pnl_series = sym_df["pnl"] if "pnl" in sym_df.columns else pd.Series([0.0])
        total_pnl = float(pnl_series.sum())
        avg_pnl = float(pnl_series.mean())

        winners = len(sym_df[sym_df["pnl"] > 0]) if "pnl" in sym_df.columns else 0
        losers = len(sym_df[sym_df["pnl"] < 0]) if "pnl" in sym_df.columns else 0
        win_rate = (winners / total_trades * 100.0) if total_trades else 0.0

        best_trade = float(pnl_series.max()) if not pnl_series.empty else 0.0
        worst_trade = float(pnl_series.min()) if not pnl_series.empty else 0.0

        cumulative_pnl = pnl_series.cumsum()
        running_max = cumulative_pnl.cummax()
        drawdown = running_max - cumulative_pnl
        max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0

        long_trades = len(sym_df[sym_df["direction"].str.lower() == "long"]) if "direction" in sym_df.columns else 0
        short_trades = len(sym_df[sym_df["direction"].str.lower() == "short"]) if "direction" in sym_df.columns else 0

        long_pnl = float(sym_df[sym_df["direction"].str.lower() == "long"]["pnl"].sum()) if "direction" in sym_df.columns and "pnl" in sym_df.columns else 0.0
        short_pnl = float(sym_df[sym_df["direction"].str.lower() == "short"]["pnl"].sum()) if "direction" in sym_df.columns and "pnl" in sym_df.columns else 0.0

        gross_profit = float(pnl_series[pnl_series > 0].sum()) if not pnl_series.empty else 0.0
        gross_loss = abs(float(pnl_series[pnl_series < 0].sum())) if not pnl_series.empty else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0

        symbol_stats.append({
            "symbol": symbol,
            "trades": total_trades,
            "winners": winners,
            "losers": losers,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(avg_pnl, 2),
            "best_trade": round(best_trade, 2),
            "worst_trade": round(worst_trade, 2),
            "max_drawdown": round(max_drawdown, 2),
            "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "∞",
            "long_trades": long_trades,
            "short_trades": short_trades,
            "long_pnl": round(long_pnl, 2),
            "short_pnl": round(short_pnl, 2),
        })

    symbol_stats.sort(key=lambda x: x["total_pnl"], reverse=True)
    return symbol_stats


def build_summary(trades_df, start_ts, end_ts):
    """Build summary payload from trades DataFrame."""
    total_trades = len(trades_df)
    winners = len(trades_df[trades_df["pnl"] > 0]) if not trades_df.empty else 0
    losers = len(trades_df[trades_df["pnl"] < 0]) if not trades_df.empty else 0
    pnl_sum = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    avg_pnl = float(trades_df["pnl"].mean()) if not trades_df.empty else 0.0
    win_rate = (winners / total_trades * 100.0) if total_trades else 0.0

    final_capital = START_EQUITY + pnl_sum

    # Direction stats
    def calc_direction_stats(df, direction_name):
        if "direction" not in df.columns or df.empty:
            return {}
        dir_df = df[df["direction"].str.lower() == direction_name.lower()]
        if dir_df.empty:
            return {
                f"{direction_name}_trades": 0,
                f"{direction_name}_pnl": 0.0,
                f"{direction_name}_avg_pnl": 0.0,
                f"{direction_name}_win_rate": 0.0,
                f"{direction_name}_winners": 0,
                f"{direction_name}_losers": 0,
            }
        count = len(dir_df)
        wins = len(dir_df[dir_df["pnl"] > 0])
        losses = len(dir_df[dir_df["pnl"] < 0])
        pnl = float(dir_df["pnl"].sum())
        avg = float(dir_df["pnl"].mean())
        wr = (wins / count * 100.0) if count else 0.0
        return {
            f"{direction_name}_trades": int(count),
            f"{direction_name}_pnl": round(pnl, 6),
            f"{direction_name}_avg_pnl": round(avg, 6),
            f"{direction_name}_win_rate": round(wr, 4),
            f"{direction_name}_winners": int(wins),
            f"{direction_name}_losers": int(losses),
        }

    long_stats = calc_direction_stats(trades_df, "long")
    short_stats = calc_direction_stats(trades_df, "short")

    symbol_stats = calc_symbol_stats(trades_df)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "start": start_ts.isoformat(),
        "end": end_ts.isoformat(),
        "closed_trades": int(total_trades),
        "open_positions": 0,
        "closed_pnl": round(pnl_sum, 6),
        "avg_trade_pnl": round(avg_pnl, 6),
        "win_rate_pct": round(win_rate, 4),
        "winners": int(winners),
        "losers": int(losers),
        "open_equity": 0.0,
        "final_capital": round(final_capital, 6),
        **long_stats,
        **short_stats,
        "long_open": 0,
        "long_open_equity": 0.0,
        "short_open": 0,
        "short_open_equity": 0.0,
        "symbol_stats": symbol_stats,
    }


def generate_html(summary, trades_df, path):
    """Generate HTML summary."""
    html_parts = [
        "<html><head><meta charset='utf-8'>",
        "<title>Paper Trading Simulation Summary</title>",
        "<style>body{font-family:Arial,sans-serif;margin:20px;}table{border-collapse:collapse;margin-top:12px;width:auto;}th,td{border:1px solid #ccc;padding:6px 10px;text-align:right;}th{text-align:center;background:#f0f0f0;font-weight:bold;}td:first-child{text-align:left;}h1{margin-bottom:10px;}h2{margin-top:30px;margin-bottom:10px;}.stats-container{display:flex;gap:20px;flex-wrap:wrap;}</style>",
        "</head><body>",
        f"<h1>Simulation Summary {summary['start']} → {summary['end']}</h1>",
        "<h2>Overall Statistics</h2>",
        "<table>",
        "<tr><th>Metric</th><th>Value</th></tr>",
        f"<tr><td>Closed trades</td><td>{summary['closed_trades']}</td></tr>",
        f"<tr><td>Open positions</td><td>{summary['open_positions']}</td></tr>",
        f"<tr><td>Closed PnL (USDT)</td><td>{summary['closed_pnl']:.2f}</td></tr>",
        f"<tr><td>Avg trade PnL (USDT)</td><td>{summary['avg_trade_pnl']:.2f}</td></tr>",
        f"<tr><td>Win rate (%)</td><td>{summary['win_rate_pct']:.2f}</td></tr>",
        f"<tr><td>Winners</td><td>{summary['winners']}</td></tr>",
        f"<tr><td>Losers</td><td>{summary['losers']}</td></tr>",
        f"<tr><td>Final capital (USDT)</td><td>{summary['final_capital']:.2f}</td></tr>",
        "</table>",
        "<h2>Statistics by Direction</h2>",
        "<div class='stats-container'>",
        "<table>",
        "<tr><th colspan='2'>Long Statistics</th></tr>",
        f"<tr><td>Closed trades</td><td>{summary.get('long_trades', 0)}</td></tr>",
        f"<tr><td>PnL (USDT)</td><td>{summary.get('long_pnl', 0):.2f}</td></tr>",
        f"<tr><td>Avg PnL (USDT)</td><td>{summary.get('long_avg_pnl', 0):.2f}</td></tr>",
        f"<tr><td>Win rate (%)</td><td>{summary.get('long_win_rate', 0):.2f}</td></tr>",
        f"<tr><td>Winners</td><td>{summary.get('long_winners', 0)}</td></tr>",
        f"<tr><td>Losers</td><td>{summary.get('long_losers', 0)}</td></tr>",
        "</table>",
        "<table>",
        "<tr><th colspan='2'>Short Statistics</th></tr>",
        f"<tr><td>Closed trades</td><td>{summary.get('short_trades', 0)}</td></tr>",
        f"<tr><td>PnL (USDT)</td><td>{summary.get('short_pnl', 0):.2f}</td></tr>",
        f"<tr><td>Avg PnL (USDT)</td><td>{summary.get('short_avg_pnl', 0):.2f}</td></tr>",
        f"<tr><td>Win rate (%)</td><td>{summary.get('short_win_rate', 0):.2f}</td></tr>",
        f"<tr><td>Winners</td><td>{summary.get('short_winners', 0)}</td></tr>",
        f"<tr><td>Losers</td><td>{summary.get('short_losers', 0)}</td></tr>",
        "</table>",
        "</div>",
    ]

    symbol_stats = summary.get("symbol_stats", [])
    if symbol_stats:
        html_parts.append("<h2>Statistics by Symbol</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Symbol</th><th>Trades</th><th>Win</th><th>Loss</th><th>Win%</th><th>Total PnL</th><th>Avg PnL</th><th>Best</th><th>Worst</th><th>Max DD</th><th>PF</th><th>Long</th><th>Short</th><th>Long PnL</th><th>Short PnL</th></tr>")
        for ss in symbol_stats:
            pnl_color = "green" if ss["total_pnl"] >= 0 else "red"
            html_parts.append(
                f"<tr>"
                f"<td>{ss['symbol']}</td>"
                f"<td>{ss['trades']}</td>"
                f"<td>{ss['winners']}</td>"
                f"<td>{ss['losers']}</td>"
                f"<td>{ss['win_rate']:.1f}%</td>"
                f"<td style='color:{pnl_color}'>{ss['total_pnl']:.2f}</td>"
                f"<td>{ss['avg_pnl']:.2f}</td>"
                f"<td style='color:green'>{ss['best_trade']:.2f}</td>"
                f"<td style='color:red'>{ss['worst_trade']:.2f}</td>"
                f"<td style='color:orange'>{ss['max_drawdown']:.2f}</td>"
                f"<td>{ss['profit_factor']}</td>"
                f"<td>{ss['long_trades']}</td>"
                f"<td>{ss['short_trades']}</td>"
                f"<td>{ss['long_pnl']:.2f}</td>"
                f"<td>{ss['short_pnl']:.2f}</td>"
                f"</tr>"
            )
        html_parts.append("</table>")

    html_parts.append("</body></html>")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))
    print(f"[Summary] HTML written to {path}")


def main():
    csv_path = "simulation_logs/full_2025_trades.csv"
    if not os.path.exists(csv_path):
        print(f"[Error] {csv_path} not found")
        return

    print(f"[Summary] Loading trades from {csv_path}")
    df = pd.read_csv(csv_path, quotechar='"')

    # Normalize column names
    col_map = {
        "exit_time": ["exit_time", "ExitZeit", "Exit_Time"],
        "entry_time": ["entry_time", "Zeit", "Entry_Time"],
        "pnl": ["pnl", "PnL", "Pnl", "profit"],
        "symbol": ["symbol", "Symbol"],
        "direction": ["direction", "Direction"],
    }

    for target, sources in col_map.items():
        for src in sources:
            if src in df.columns and target not in df.columns:
                df[target] = df[src]
                break

    # Parse dates
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["exit_time", "pnl"])
    df = df.sort_values("exit_time").reset_index(drop=True)

    print(f"[Summary] Loaded {len(df)} trades")
    print(f"[Summary] Columns: {list(df.columns)}")

    # Time range
    start_ts = df["exit_time"].min()
    end_ts = df["exit_time"].max()

    # Build summary
    summary = build_summary(df, start_ts, end_ts)

    # Write outputs
    os.makedirs("report_html", exist_ok=True)

    html_path = "report_html/trading_summary.html"
    generate_html(summary, df, html_path)

    json_path = "report_html/trading_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Summary] JSON written to {json_path}")

    # Print summary
    print("\n" + "="*60)
    print("TRADING SUMMARY")
    print("="*60)
    print(f"Period: {summary['start'][:10]} → {summary['end'][:10]}")
    print(f"Total Trades: {summary['closed_trades']}")
    print(f"Total PnL: {summary['closed_pnl']:,.2f} USDT")
    print(f"Win Rate: {summary['win_rate_pct']:.1f}%")
    print(f"Final Capital: {summary['final_capital']:,.2f} USDT")
    print("-"*60)
    print(f"Long: {summary.get('long_trades', 0)} trades, PnL: {summary.get('long_pnl', 0):,.2f}")
    print(f"Short: {summary.get('short_trades', 0)} trades, PnL: {summary.get('short_pnl', 0):,.2f}")
    print("="*60)


if __name__ == "__main__":
    main()
