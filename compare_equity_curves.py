#!/usr/bin/env python3
"""
Compare equity curves from different simulation periods.
Run simulations for multiple periods and generate comparison chart.
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Default periods to compare
DEFAULT_PERIODS = [
    ("2024-06-01", "2024-12-31", "H2 2024"),
    ("2025-01-01", "2025-06-30", "H1 2025"),
    ("2025-01-01", "2025-12-23", "Full 2025"),
]

START_CAPITAL = 16_000.0


def run_simulation(start: str, end: str, output_prefix: str) -> str:
    """Run a simulation for a specific period and save with unique name."""
    output_log = f"simulation_logs/{output_prefix}_trades.csv"
    output_json = f"simulation_logs/{output_prefix}_trades.json"

    os.makedirs("simulation_logs", exist_ok=True)

    cmd = [
        sys.executable, "paper_trader.py",
        "--simulate",
        "--start", start,
        "--end", end,
        "--sim-log", output_log,
        "--sim-json", output_json,
    ]

    print(f"\n{'='*60}")
    print(f"Running simulation: {start} to {end}")
    print(f"Output: {output_log}")
    print('='*60)

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(f"[Warning] Simulation failed for {output_prefix}")
        return ""

    return output_log


def load_trades_csv(csv_path: str) -> pd.DataFrame:
    """Load trades from CSV and calculate equity curve."""
    if not os.path.exists(csv_path):
        print(f"[Warning] File not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Normalize column names (handle different formats)
    col_map = {
        "exit_time": ["exit_time", "ExitZeit", "Exit_Time"],
        "pnl": ["pnl", "PnL", "Pnl", "profit"],
    }

    for target, sources in col_map.items():
        for src in sources:
            if src in df.columns and target not in df.columns:
                df[target] = df[src]
                break

    if "exit_time" not in df.columns or "pnl" not in df.columns:
        print(f"[Warning] Missing required columns in {csv_path}")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # Parse dates and sort
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["exit_time", "pnl"])
    df = df.sort_values("exit_time").reset_index(drop=True)

    # Calculate equity
    df["cumulative_pnl"] = df["pnl"].cumsum()
    df["equity"] = START_CAPITAL + df["cumulative_pnl"]

    # Calculate drawdown
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["peak"] - df["equity"]
    df["drawdown_pct"] = (df["drawdown"] / df["peak"]) * 100

    return df


def compare_equity_curves(
    periods: List[Tuple[str, str, str]],
    run_simulations: bool = False,
    output_dir: str = "report_html/charts",
) -> None:
    """Create comparison chart for multiple periods."""

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("simulation_logs", exist_ok=True)

    # Collect data for each period
    period_data = []

    for start, end, label in periods:
        prefix = label.replace(" ", "_").lower()
        csv_path = f"simulation_logs/{prefix}_trades.csv"

        if run_simulations:
            csv_path = run_simulation(start, end, prefix)
            if not csv_path:
                continue

        if not os.path.exists(csv_path):
            print(f"[Warning] No data for {label}. Run with --run-simulations first.")
            continue

        df = load_trades_csv(csv_path)
        if df.empty:
            continue

        # Calculate statistics
        total_trades = len(df)
        total_pnl = df["pnl"].sum()
        final_equity = df["equity"].iloc[-1]
        max_dd = df["drawdown"].max()
        max_dd_pct = df["drawdown_pct"].max()
        win_rate = (df["pnl"] > 0).mean() * 100

        period_data.append({
            "label": label,
            "start": start,
            "end": end,
            "df": df,
            "trades": total_trades,
            "pnl": total_pnl,
            "final_equity": final_equity,
            "max_dd": max_dd,
            "max_dd_pct": max_dd_pct,
            "win_rate": win_rate,
            "return_pct": ((final_equity - START_CAPITAL) / START_CAPITAL) * 100,
        })

    if not period_data:
        print("[Error] No data available for comparison.")
        return

    # Create comparison figure
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        row_heights=[0.45, 0.30, 0.25],
        subplot_titles=(
            "Equity Curves Comparison",
            "Drawdown Comparison",
            "Summary Statistics"
        )
    )

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # Plot equity curves
    for i, data in enumerate(period_data):
        df = data["df"]
        color = colors[i % len(colors)]

        # Normalize time axis (days from start)
        df["days"] = (df["exit_time"] - df["exit_time"].iloc[0]).dt.total_seconds() / 86400

        fig.add_trace(
            go.Scatter(
                x=df["days"],
                y=df["equity"],
                mode="lines",
                name=f"{data['label']} ({data['trades']} trades, +{data['return_pct']:.0f}%)",
                line=dict(color=color, width=2),
            ),
            row=1, col=1
        )

        # Plot drawdown
        fig.add_trace(
            go.Scatter(
                x=df["days"],
                y=-df["drawdown"],
                mode="lines",
                name=f"{data['label']} DD",
                line=dict(color=color, width=1),
                fill="tozeroy",
                fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}",
                showlegend=False,
            ),
            row=2, col=1
        )

    # Add starting capital reference
    fig.add_hline(y=START_CAPITAL, line_dash="dash", line_color="gray",
                  annotation_text=f"Start: {START_CAPITAL:,.0f}", row=1, col=1)

    # Summary bar chart
    labels = [d["label"] for d in period_data]
    returns = [d["return_pct"] for d in period_data]
    bar_colors = ["green" if r > 0 else "red" for r in returns]

    fig.add_trace(
        go.Bar(
            x=labels,
            y=returns,
            marker_color=bar_colors,
            text=[f"{r:.0f}%<br>{d['trades']} trades<br>DD: {d['max_dd_pct']:.1f}%"
                  for r, d in zip(returns, period_data)],
            textposition="outside",
            name="Return %",
            showlegend=False,
        ),
        row=3, col=1
    )

    # Layout
    fig.update_layout(
        title=dict(
            text=f"Equity Curve Comparison | Start Capital: {START_CAPITAL:,.0f} USDT",
            font=dict(size=16)
        ),
        height=1000,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Days from Start", row=2, col=1)
    fig.update_yaxes(title_text="Equity (USDT)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (USDT)", row=2, col=1)
    fig.update_yaxes(title_text="Return %", row=3, col=1)

    # Save
    out_path = os.path.join(output_dir, "equity_comparison.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"\n[Comparison] Saved to {out_path}")

    # Print summary table
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'Period':<15} {'Trades':>8} {'PnL':>12} {'Return':>10} {'Max DD':>10} {'Win Rate':>10}")
    print("-"*80)
    for d in period_data:
        print(f"{d['label']:<15} {d['trades']:>8} {d['pnl']:>12,.0f} {d['return_pct']:>9.1f}% {d['max_dd_pct']:>9.1f}% {d['win_rate']:>9.1f}%")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare equity curves from different periods")
    parser.add_argument("--run-simulations", action="store_true",
                        help="Run simulations for all periods (takes time)")
    parser.add_argument("--periods", nargs="+",
                        help="Custom periods as 'start,end,label' (e.g., '2024-01-01,2024-06-30,H1_2024')")
    parser.add_argument("--output", default="report_html/charts",
                        help="Output directory for charts")

    args = parser.parse_args()

    if args.periods:
        periods = []
        for p in args.periods:
            parts = p.split(",")
            if len(parts) == 3:
                periods.append((parts[0], parts[1], parts[2]))
    else:
        periods = DEFAULT_PERIODS

    compare_equity_curves(
        periods=periods,
        run_simulations=args.run_simulations,
        output_dir=args.output,
    )
