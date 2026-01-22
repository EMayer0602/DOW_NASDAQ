"""Merge initial_trades.csv into paper_trading_log.csv to build full history"""
import pandas as pd
import os

log_file = "paper_trading_log.csv"
initial_file = "report_html/initial_trades.csv"

# Load existing log (if any)
if os.path.exists(log_file):
    log_df = pd.read_csv(log_file, quotechar='"')
    print(f"Existing log: {len(log_df)} trades")
else:
    log_df = pd.DataFrame()
    print("No existing log")

# Load initial trades from last simulation
if os.path.exists(initial_file):
    initial_df = pd.read_csv(initial_file)
    print(f"Initial trades: {len(initial_df)} trades")
    
    # Merge - keep all unique trades
    if not log_df.empty and not initial_df.empty:
        # Combine and deduplicate
        combined = pd.concat([initial_df, log_df], ignore_index=True)
        # Remove duplicates based on key fields
        if {"Symbol", "EntryTime", "ExitTime", "ExitPrice"}.issubset(combined.columns):
            combined = combined.drop_duplicates(subset=["Symbol", "EntryTime", "ExitTime", "ExitPrice"], keep="last")
        print(f"Combined: {len(combined)} unique trades")
        
        # Sort by exit time
        if "ExitTime" in combined.columns:
            combined["ExitTime_sort"] = pd.to_datetime(combined["ExitTime"], errors="coerce")
            combined = combined.sort_values("ExitTime_sort").drop(columns=["ExitTime_sort"])
        
        # Write back with proper CSV format (quotes for ParamDesc/Reason)
        combined.to_csv(log_file, index=False, quoting=1)  # QUOTE_ALL
        print(f"✓ Wrote {len(combined)} trades to {log_file}")
    elif not initial_df.empty:
        initial_df.to_csv(log_file, index=False, quoting=1)
        print(f"✓ Created new log with {len(initial_df)} trades")
else:
    print(f"No initial trades file found at {initial_file}")
