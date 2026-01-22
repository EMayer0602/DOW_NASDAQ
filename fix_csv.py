"""Fix paper_trading_log.csv - rewrite with proper quoting"""
import csv

# Read the broken CSV manually
with open("paper_trading_log.csv", "r", encoding="utf-8") as f:
    lines = f.readlines()

# Parse header
header = lines[0].strip().split(",")
print(f"Header: {header}")

# Manually parse the data rows (CSV is broken, so we can't use csv.reader)
fixed_rows = []
for i, line in enumerate(lines[1:], 1):
    parts = line.strip().split(",")
    print(f"\nRow {i}: {len(parts)} parts")
    print(f"Raw: {line[:100]}")
    
    # The CSV has extra commas in ParamDesc - we need to reconstruct
    # Expected format: Timestamp,Symbol,Direction,Indicator,HTF,ParamDesc,EntryTime,EntryPrice,ExitTime,ExitPrice,Stake,Fees,PnL,EquityAfter,Reason
    # ParamDesc has format: ParamA=X, ParamB=Y, ATR=Z (with commas!)
    
    if len(parts) >= 15:
        # Try to reconstruct - ParamDesc is position 5, should have 3 parts (ParamA, ParamB, ATR)
        timestamp = parts[0]
        symbol = parts[1]
        direction = parts[2]
        indicator = parts[3]
        htf = parts[4]
        # ParamDesc spans multiple parts due to commas
        param_parts = []
        param_start = 5
        param_end = param_start
        # Find where ParamDesc ends (look for ISO timestamp pattern in EntryTime)
        for j in range(param_start, len(parts)):
            if "T" in parts[j] and ":" in parts[j]:
                param_end = j
                break
        param_desc = ",".join(parts[param_start:param_end])
        
        entry_time = parts[param_end]
        entry_price = parts[param_end + 1]
        exit_time = parts[param_end + 2]
        exit_price = parts[param_end + 3]
        stake = parts[param_end + 4]
        fees = parts[param_end + 5]
        pnl = parts[param_end + 6]
        equity = parts[param_end + 7]
        reason = ",".join(parts[param_end + 8:])  # Reason might also have commas
        
        fixed_rows.append([
            timestamp, symbol, direction, indicator, htf, param_desc,
            entry_time, entry_price, exit_time, exit_price,
            stake, fees, pnl, equity, reason
        ])
        print(f"Parsed: Symbol={symbol}, Direction={direction}")

# Write fixed CSV with proper quoting
with open("paper_trading_log_FIXED.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)
    for row in fixed_rows:
        writer.writerow(row)

print(f"\n✓ Fixed CSV written to paper_trading_log_FIXED.csv ({len(fixed_rows)} rows)")
print("Please check the file, then rename it to paper_trading_log.csv")
