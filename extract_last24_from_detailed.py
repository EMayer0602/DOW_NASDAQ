import re
import datetime
import argparse
import pytz
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from io import StringIO

START_CAPITAL = 14_000.0
STAKE_DIVISOR = 14
FEE_RATE = 0.00075  # VIP Level 1

def main():
    parser = argparse.ArgumentParser(description="Extract trades from overall_best_detailed.html and recompute PnL/capital.")
    parser.add_argument("--hours", type=float, default=24, help="Lookback window in hours (default: 24)")
    args = parser.parse_args()

    html_path = Path('report_html/overall_best_detailed.html')
    if not html_path.exists():
        print('File not found:', html_path)
        return
    tz = pytz.timezone('Europe/Berlin')
    cutoff = datetime.datetime.now(tz) - datetime.timedelta(hours=args.hours)
    soup = BeautifulSoup(html_path.read_text(encoding='utf-8'), 'html.parser')
    results = []
    for section in soup.find_all('section'):
        h3 = section.find('h3')
        if not h3:
            continue
        header = h3.get_text(' ', strip=True)
        m = re.match(r'([^–-]+)[–-]', header)
        symbol = m.group(1).strip() if m else ''
        table = section.find('table')
        if table is None:
            continue
        df = pd.read_html(StringIO(str(table)))[0]
        if 'Zeit' not in df.columns:
            continue
        df['Zeit'] = pd.to_datetime(df['Zeit'])
        df = df[df['Zeit'] >= cutoff]
        if df.empty:
            continue
        df.insert(0, 'Symbol', symbol)
        # Ensure PnL and Equity columns exist; fill with NaN if missing
        for col in ("PnL (USD)", "Equity"):
            if col not in df.columns:
                df[col] = pd.NA
        results.append(df)
    if not results:
        print('No trades in last 24h')
        return
    out = pd.concat(results, ignore_index=True)

    # Sort chronologisch, dann PnL/Eigenkapital gemäß Regelwerk neu berechnen
    out = out.sort_values(by=["Zeit", "ExitZeit"], kind="mergesort").reset_index(drop=True)
    capital = START_CAPITAL
    recomputed = []
    for _, row in out.iterrows():
        entry = float(row.get("Entry", 0.0) or 0.0)
        exit_price = float(row.get("ExitPreis", 0.0) or 0.0)
        direction = str(row.get("Direction", "Long")).lower()
        stake_used = capital / STAKE_DIVISOR if STAKE_DIVISOR else capital
        shares = stake_used / entry if entry else 0.0
        # Correct fee calculation: fees based on traded volume
        fees = (entry + exit_price) * shares * FEE_RATE
        entry_fee = entry * shares * FEE_RATE
        exit_fee = exit_price * shares * FEE_RATE
        if direction == "short":
            price_diff = entry - exit_price
        else:
            price_diff = exit_price - entry
        pnl_recalc = shares * price_diff - fees
        capital_after = capital + pnl_recalc
        capital = capital_after

        recomputed.append({
            "StakeUsed": stake_used,
            "Shares": shares,
            "EntryFee": entry_fee,
            "ExitFee": exit_fee,
            "PnL (recalc)": pnl_recalc,
            "CapitalAfter": capital_after,
        })

    recomputed_df = pd.DataFrame(recomputed)
    out = pd.concat([out, recomputed_df], axis=1)

    label_hours = int(args.hours) if float(args.hours).is_integer() else args.hours
    out_path = Path(f'report_html/last{label_hours}_from_detailed_all.csv')
    out.to_csv(out_path, index=False)

    cols = [
        'Symbol', 'Zeit', 'Entry', 'ExitZeit', 'ExitPreis', 'Direction', 'ExitReason',
        'StakeUsed', 'Shares', 'EntryFee', 'ExitFee', 'PnL (recalc)', 'CapitalAfter'
    ]
    existing = [c for c in cols if c in out.columns]
    print(out[existing])
    print('Saved to', out_path)

if __name__ == '__main__':
    main()
