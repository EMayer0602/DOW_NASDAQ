import os
# placeholder
import ccxt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from ta.volatility import AverageTrueRange
from datetime import datetime
from zoneinfo import ZoneInfo

BERLIN_TZ = ZoneInfo("Europe/Berlin")


def timeframe_to_minutes(tf_str: str) -> int:
    """Convert timeframe tokens like '5m' or '1h' to minutes."""
    unit = tf_str[-1].lower()
    value = int(tf_str[:-1])
    if unit == "m":
        return value
    if unit == "h":
        return value * 60
    if unit == "d":
        return value * 1440
    raise ValueError(f"Unsupported timeframe unit in {tf_str}")

# =========================
# CONFIG
# =========================
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "XRP/USDT",
    "ADA/USDT"

    # =========================
    # Main
    # =========================
    def run_current_configuration():
        os.makedirs(OUT_DIR, exist_ok=True)

        if RUN_PARAMETER_SWEEP:
            figs_blocks = []
            sections_blocks = []
            ranking_tables = []
            best_params_summary = []

            directions = ["long", "short"] if ENABLE_SHORTS else ["long"]
            hold_day_candidates = MIN_HOLD_DAY_VALUES if USE_MIN_HOLD_FILTER else [DEFAULT_MIN_HOLD_DAYS]

            for symbol in SYMBOLS:
                df_raw = prepare_symbol_dataframe(symbol)

                results = {d: [] for d in directions}
                trades_per_combo = {d: {} for d in directions}
                df_cache = {}

                for param_a in PARAM_A_VALUES:
                    for param_b in PARAM_B_VALUES:
                        cache_key = (param_a, param_b)
                        if cache_key not in df_cache:
                            df_tmp = compute_indicator(df_raw, param_a, param_b)
                            for col in ("htf_trend", "htf_supertrend", "momentum"):
                                if col in df_raw.columns:
                                    df_tmp[col] = df_raw[col]
                            df_cache[cache_key] = df_tmp
                        df_st = df_cache[cache_key]
                        for atr_mult in ATR_STOP_MULTS:
                            for hold_days in hold_day_candidates:
                                min_hold_bars = hold_days * BARS_PER_DAY
                                for direction in directions:
                                    df_st_with_htf = df_st.copy()
                                    for col in ("htf_trend", "htf_supertrend", "momentum"):
                                        if col in df_raw.columns:
                                            df_st_with_htf[col] = df_raw[col]
                                    trades = backtest_supertrend(
                                        df_st_with_htf,
                                        atr_stop_mult=atr_mult,
                                        direction=direction,
                                        min_hold_bars=min_hold_bars,
                                        min_hold_days=hold_days,
                                    )
                                    stats = performance_report(
                                        trades,
                                        symbol,
                                        param_a,
                                        param_b,
                                        direction.capitalize(),
                                        hold_days,
                                    )
                                    stats["ATRStopMult"] = atr_mult if atr_mult is not None else "None"
                                    stats["MinHoldBars"] = min_hold_bars
                                    results[direction].append(stats)
                                    trades_per_combo[direction][(param_a, param_b, atr_mult, hold_days)] = trades

            for direction in directions:
                dir_results = results[direction]
                ranking_df = pd.DataFrame(dir_results)
                ranking_df = ranking_df.sort_values("FinalEquity", ascending=False).reset_index(drop=True) if not ranking_df.empty else ranking_df
                ranking_tables.append(df_to_html_table(
                    ranking_df,
                    title=f"Ranking: {symbol} {INDICATOR_DISPLAY_NAME} ({direction.capitalize()} nach FinalEquity)"
                ))

                if not ranking_df.empty:
                    best_row = ranking_df.iloc[0]
                    best_param_a = best_row.get("ParamA", best_row.get(PARAM_A_LABEL, DEFAULT_PARAM_A))
                    best_param_b = best_row.get("ParamB", best_row.get(PARAM_B_LABEL, DEFAULT_PARAM_B))
                    best_param_a = best_param_a if not pd.isna(best_param_a) else DEFAULT_PARAM_A
                    best_param_b = best_param_b if not pd.isna(best_param_b) else DEFAULT_PARAM_B
                    best_atr_raw = best_row["ATRStopMult"]
                    best_atr = best_atr_raw if best_atr_raw != "None" else None
                    best_hold_days = int(best_row.get("MinHoldDays", DEFAULT_MIN_HOLD_DAYS))
                    best_df = df_cache[(best_param_a, best_param_b)]
                    best_trades = trades_per_combo[direction][(best_param_a, best_param_b, best_atr, best_hold_days)]
                else:
                    best_param_a, best_param_b = DEFAULT_PARAM_A, DEFAULT_PARAM_B
                    best_atr = None
                    best_hold_days = DEFAULT_MIN_HOLD_DAYS
                    best_df = compute_indicator(df_raw, best_param_a, best_param_b)
                    best_trades = pd.DataFrame()

                atr_label = best_atr if best_atr is not None else "None"
                best_params_summary.append({
                    "Symbol": symbol,
                    "Direction": direction.capitalize(),
                    "ParamA": best_param_a,
                    "ParamB": best_param_b,
                    PARAM_A_LABEL: best_param_a,
                    PARAM_B_LABEL: best_param_b,
                    "Length": best_param_a if INDICATOR_TYPE == "supertrend" else None,
                    "Factor": best_param_b if INDICATOR_TYPE == "supertrend" else None,
                    "ATRStopMult": atr_label,
                    "MinHoldDays": best_hold_days,
                })

                fig = build_two_panel_figure(
                    symbol,
                    best_df,
                    best_trades,
                    best_param_a,
                    best_param_b,
                    direction.capitalize(),
                    min_hold_days=best_hold_days,
                )
                fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
                figs_blocks.append(
                    f"<h2>{symbol} – {direction.capitalize()} beste Parameter: {PARAM_A_LABEL}={best_param_a}, {PARAM_B_LABEL}={best_param_b}, ATRStop={atr_label}, MinHold={best_hold_days}d</h2>\n"
                    + fig_html
                )

                sections_blocks.append(
                    df_to_html_table(
                        best_trades,
                        title=f"Trade-Liste {symbol} ({direction.capitalize()} beste Parameter, MinHold={best_hold_days}d)"
                    )
                )

                csv_suffix = "" if direction == "long" else "_short"
                csv_path = os.path.join(OUT_DIR, f"trades_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_best{csv_suffix}.csv")
                best_trades.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
                print(f"[Saved] {csv_path}")

                csv_rank_suffix = "" if direction == "long" else "_short"
                csv_rank_path = os.path.join(OUT_DIR, f"ranking_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_params{csv_rank_suffix}.csv")
                ranking_df.to_csv(csv_rank_path, sep=";", decimal=",", index=False, encoding="utf-8")
                print(f"[Saved] {csv_rank_path}")

                print(f"[Info] {symbol} {direction.capitalize()} best MinHoldDays: {best_hold_days}")

        if best_params_summary:
            summary_df = pd.DataFrame(best_params_summary)
            summary_path = os.path.join(OUT_DIR, BEST_PARAMS_FILE)
            summary_df.to_csv(summary_path, sep=";", decimal=",", index=False, encoding="utf-8")
            print(f"[Saved] {summary_path}")

        report_html = build_full_report(figs_blocks, sections_blocks, ranking_tables)
        report_path = os.path.join(OUT_DIR, REPORT_FILE)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        print(f"[Saved] {report_path}")
    elif RUN_SAVED_PARAMS:
        summary_path = os.path.join(OUT_DIR, BEST_PARAMS_FILE)
        summary_df = None
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path, sep=";", decimal=",")
        else:
            derived_rows = []
            for symbol in SYMBOLS:
                rank_path = os.path.join(OUT_DIR, f"ranking_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_params.csv")
                if not os.path.exists(rank_path):
                    continue
                rank_df = pd.read_csv(rank_path, sep=";", decimal=",")
                if rank_df.empty:
                    continue
                best_row = rank_df.iloc[0]
                derived_rows.append({
                    "Symbol": symbol,
                    "Direction": best_row.get("Direction", "Long"),
                    "ParamA": best_row.get("ParamA", best_row.get(PARAM_A_LABEL, DEFAULT_PARAM_A)),
                    "ParamB": best_row.get("ParamB", best_row.get(PARAM_B_LABEL, DEFAULT_PARAM_B)),
                    PARAM_A_LABEL: best_row.get(PARAM_A_LABEL, DEFAULT_PARAM_A),
                    PARAM_B_LABEL: best_row.get(PARAM_B_LABEL, DEFAULT_PARAM_B),
                    "Length": best_row.get("Length", DEFAULT_PARAM_A if INDICATOR_TYPE == "supertrend" else None),
                    "Factor": best_row.get("Factor", DEFAULT_PARAM_B if INDICATOR_TYPE == "supertrend" else None),
                    "ATRStopMult": best_row.get("ATRStopMult", "None"),
                    "MinHoldDays": best_row.get("MinHoldDays", DEFAULT_MIN_HOLD_DAYS),
                })
            if derived_rows:
                summary_df = pd.DataFrame(derived_rows)
                summary_df.to_csv(summary_path, sep=";", decimal=",", index=False, encoding="utf-8")
                print(f"[Saved] {summary_path} (derived from ranking files)")

        if summary_df is None or summary_df.empty:
            print(f"[Skip] No saved parameters available. Run the sweep (set RUN_PARAMETER_SWEEP = True) to generate them.")
        else:
            figs_blocks = []
            sections_blocks = []
            ranking_tables = [df_to_html_table(summary_df, title="Gespeicherte Parameter (ohne Sweep)")]
            data_cache = {}
            st_cache = {}

            for _, row in summary_df.iterrows():
                symbol = row.get("Symbol")
                if not symbol:
                    continue
                direction = str(row.get("Direction", "Long")).lower()
                if direction not in {"long", "short"}:
                    print(f"[Warn] Skipping {symbol}: unsupported direction '{direction}'.")
                    continue
                param_a = row.get("ParamA", row.get(PARAM_A_LABEL, DEFAULT_PARAM_A))
                param_b = row.get("ParamB", row.get(PARAM_B_LABEL, DEFAULT_PARAM_B))
                param_a = param_a if not pd.isna(param_a) else DEFAULT_PARAM_A
                param_b = param_b if not pd.isna(param_b) else DEFAULT_PARAM_B
                atr_raw = row.get("ATRStopMult", "None")
                atr_mult = None if pd.isna(atr_raw) or str(atr_raw).lower() == "none" else float(atr_raw)
                hold_days = int(row.get("MinHoldDays", DEFAULT_MIN_HOLD_DAYS))
                min_hold_bars = hold_days * BARS_PER_DAY

                if symbol not in data_cache:
                    data_cache[symbol] = prepare_symbol_dataframe(symbol)
                df_raw = data_cache[symbol]

                st_key = (symbol, param_a, param_b)
                if st_key not in st_cache:
                    df_tmp = compute_indicator(df_raw, param_a, param_b)
                    for col in ("htf_trend", "htf_supertrend", "momentum"):
                        if col in df_raw.columns:
                            df_tmp[col] = df_raw[col]
                    st_cache[st_key] = df_tmp
                df_st = st_cache[st_key]

                trades = backtest_supertrend(
                    df_st,
                    atr_stop_mult=atr_mult,
                    direction=direction,
                    min_hold_bars=min_hold_bars,
                    min_hold_days=hold_days,
                )

                direction_title = direction.capitalize()
                atr_label = "None" if atr_mult is None else atr_mult
                param_desc = f"{PARAM_A_LABEL}={param_a}, {PARAM_B_LABEL}={param_b}"

                fig = build_two_panel_figure(
                    symbol,
                    df_st,
                    trades,
                    param_a,
                    param_b,
                    direction_title,
                    min_hold_days=hold_days,
                )
                fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
                figs_blocks.append(
                    f"<h2>{symbol} – {direction_title} gespeicherte Parameter: {param_desc}, ATRStop={atr_label}, MinHold={hold_days}d</h2>\n"
                    + fig_html
                )

                sections_blocks.append(
                    df_to_html_table(
                        trades,
                        title=f"Trade-Liste {symbol} ({direction_title} gespeicherte Parameter, MinHold={hold_days}d)"
                    )
                )

                csv_suffix = "" if direction == "long" else "_short"
                csv_path = os.path.join(OUT_DIR, f"trades_{symbol.replace('/', '_')}_{INDICATOR_SLUG}_best{csv_suffix}.csv")
                trades.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
                print(f"[Saved] {csv_path}")

                print(f"[Info] {symbol} {direction_title} MinHoldDays={hold_days} (gespeichert)")

            if figs_blocks or sections_blocks:
                report_html = build_full_report(figs_blocks, sections_blocks, ranking_tables)
                report_path = os.path.join(OUT_DIR, REPORT_FILE)
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_html)
                print(f"[Saved] {report_path}")
            else:
                print("[Skip] No valid entries in saved parameter file.")
    else:
        print("[Skip] Backtesting disabled. Set RUN_PARAMETER_SWEEP = True or RUN_SAVED_PARAMS = True to produce output.")


    if __name__ == "__main__":
        indicator_candidates = get_indicator_candidates()
        for indicator_name in indicator_candidates:
            apply_indicator_type(indicator_name)
            htf_candidates = get_highertimeframe_candidates()
            for htf_value in htf_candidates:
                apply_higher_timeframe(htf_value)
                print(f"[Run] Indicator={INDICATOR_DISPLAY_NAME}, HTF={HIGHER_TIMEFRAME}")
                run_current_configuration()
                if low[i] < ep_low:
                    ep_low = low[i]
                    af = min(af + step, max_step)

        psar_vals[i] = psar_candidate
        trend_flags[i] = 1 if bullish else -1

    psar_series = pd.Series(psar_vals, index=df.index)
    trend_series = pd.Series(trend_flags, index=df.index)
    df["psar"] = psar_series
    df["psar_trend"] = trend_series
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
    df["atr"] = atr
    df["indicator_line"] = df["psar"]
    df["trend_flag"] = df["psar_trend"]
    return df


def jurik_moving_average(series: pd.Series, length: int, phase: int) -> pd.Series:
    # Approximates Jurik smoothing with adaptive double filtering so we can stay dependency-free.
    if series.empty:
        return pd.Series(index=series.index, dtype=float)
    length = max(1, int(length))
    phase = int(np.clip(phase, -100, 100))
    beta = 0.45 * (length - 1) / (0.45 * (length - 1) + 2) if length > 1 else 0.0
    alpha = beta ** 2
    phase_ratio = (phase + 100) / 200
    jma_values = pd.Series(index=series.index, dtype=float)
    e0 = float(series.iloc[0])
    e1 = 0.0
    e2 = 0.0
    for idx, (_, price) in enumerate(series.items()):
        e0 = (1 - alpha) * price + alpha * e0
        e1 = price - e0
        e2 = (1 - beta) * e1 + beta * e2
        jma = e0 + phase_ratio * e2
        jma_values.iloc[idx] = jma
    return jma_values


def compute_jma(df, length=20, phase=0):
    df = df.copy()
    jma = jurik_moving_average(df["close"], length=length, phase=phase)
    trend = np.where(df["close"] >= jma, 1, -1)
    df["jma"] = jma
    df["jma_trend"] = trend
    atr = AverageTrueRange(df["high"], df["low"], df["close"], window=ATR_WINDOW).average_true_range()
    df["atr"] = atr
    df["indicator_line"] = df["jma"]
    df["trend_flag"] = df["jma_trend"]
    return df


def compute_indicator(df, param_a, param_b):
    if INDICATOR_TYPE == "supertrend":
        return compute_supertrend(df, length=int(param_a), factor=float(param_b))
    if INDICATOR_TYPE == "psar":
        return compute_psar(df, step=float(param_a), max_step=float(param_b))
    if INDICATOR_TYPE == "jma":
        return compute_jma(df, length=int(param_a), phase=int(param_b))
    raise ValueError(f"Unsupported INDICATOR_TYPE: {INDICATOR_TYPE}")


def attach_higher_timeframe_trend(df_low, symbol):
    if not USE_HIGHER_TIMEFRAME_FILTER:
        df_low = df_low.copy()
        df_low["htf_trend"] = 0
        df_low["htf_supertrend"] = np.nan
        return df_low

    df_high = fetch_data(symbol, HIGHER_TIMEFRAME, limit=HTF_LOOKBACK)
    df_high_st = compute_supertrend(df_high, length=HTF_LENGTH, factor=HTF_FACTOR)
    htf = df_high_st[["supertrend", "st_trend"]].rename(columns={
        "supertrend": "htf_supertrend",
        "st_trend": "htf_trend"
    })
    aligned = htf.reindex(df_low.index, method="ffill")
    df_low = df_low.copy()
    df_low["htf_trend"] = aligned["htf_trend"].fillna(0).astype(int)
    df_low["htf_supertrend"] = aligned["htf_supertrend"]
    return df_low


def attach_momentum_filter(df):
    df = df.copy()
    if not USE_MOMENTUM_FILTER:
        df["momentum"] = np.nan
        return df

    if MOMENTUM_TYPE.lower() == "rsi":
        delta = df["close"].diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_gain = pd.Series(gain, index=df.index).rolling(MOMENTUM_WINDOW).mean()
        roll_loss = pd.Series(loss, index=df.index).rolling(MOMENTUM_WINDOW).mean()
        rs = roll_gain / roll_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        df["momentum"] = rsi
    else:
        df["momentum"] = np.nan
    return df

# =========================
# Backtest: Single-Position (Long/Short)
# =========================
def backtest_supertrend(df, atr_stop_mult=None, direction="long", min_hold_bars=0, min_hold_days=None):
    direction = direction.lower()
    if direction not in {"long", "short"}:
        raise ValueError("direction must be 'long' or 'short'")
    min_hold_bars = 0 if min_hold_bars is None else max(0, int(min_hold_bars))
    min_hold_days = min_hold_days if min_hold_days is not None else 0

    long_mode = direction == "long"
    equity = START_EQUITY
    trades = []
    in_position = False
    entry_price = None
    entry_ts = None
    entry_capital = None
    entry_atr = None
    bars_in_position = 0

    for i in range(1, len(df)):
        ts = df.index[i]
        trend = int(df["trend_flag"].iloc[i])
        prev_trend = int(df["trend_flag"].iloc[i-1])

        enter_long = prev_trend == -1 and trend == +1
        enter_short = prev_trend == +1 and trend == -1

        if not in_position:
            htf_value = int(df["htf_trend"].iloc[i]) if "htf_trend" in df.columns else 0
            htf_allows = True
            if USE_HIGHER_TIMEFRAME_FILTER:
                if long_mode:
                    htf_allows = htf_value >= 1
                else:
                    htf_allows = htf_value <= -1

            momentum_allows = True
            if USE_MOMENTUM_FILTER and "momentum" in df.columns:
                mom_value = df["momentum"].iloc[i]
                if pd.isna(mom_value):
                    momentum_allows = False
                else:
                    if long_mode:
                        momentum_allows = mom_value >= RSI_LONG_THRESHOLD
                    else:
                        momentum_allows = mom_value <= RSI_SHORT_THRESHOLD

            breakout_allows = True
            if USE_BREAKOUT_FILTER:
                atr_curr = df["atr"].iloc[i]
                if atr_curr is None or np.isnan(atr_curr) or atr_curr <= 0:
                    breakout_allows = False
                else:
                    candle_range = float(df["high"].iloc[i] - df["low"].iloc[i])
                    breakout_allows = candle_range >= BREAKOUT_ATR_MULT * float(atr_curr)
                    if breakout_allows and BREAKOUT_REQUIRE_DIRECTION:
                        prev_high = float(df["high"].iloc[i-1]) if i > 0 else float(df["high"].iloc[i])
                        prev_low = float(df["low"].iloc[i-1]) if i > 0 else float(df["low"].iloc[i])
                        close_curr = float(df["close"].iloc[i])
                        if long_mode:
                            breakout_allows = close_curr > prev_high
                        else:
                            breakout_allows = close_curr < prev_low

            if long_mode and enter_long and htf_allows and momentum_allows and breakout_allows:
                in_position = True
            elif not long_mode and enter_short and htf_allows and momentum_allows and breakout_allows:
                in_position = True
            if in_position:
                entry_price = float(df["close"].iloc[i])
                entry_ts = ts
                entry_capital = equity * RISK_FRACTION
                atr_val = df["atr"].iloc[i]
                entry_atr = float(atr_val) if not np.isnan(atr_val) else 0.0
                bars_in_position = 0
            continue

        bars_in_position += 1
        stake = entry_capital if entry_capital is not None else equity * RISK_FRACTION
        atr_buffer = entry_atr if entry_atr is not None else 0.0
        stop_price = None
        if atr_stop_mult is not None and atr_buffer and atr_buffer > 0:
            if long_mode:
                stop_price = entry_price - atr_stop_mult * atr_buffer
            else:
                stop_price = entry_price + atr_stop_mult * atr_buffer

        exit_price = None
        exit_reason = None

        if stop_price is not None:
            if long_mode and float(df["low"].iloc[i]) <= stop_price:
                exit_price = stop_price
                exit_reason = "ATR stop"
            elif (not long_mode) and float(df["high"].iloc[i]) >= stop_price:
                exit_price = stop_price
                exit_reason = "ATR stop"

        if exit_price is None:
            if long_mode and prev_trend == +1 and trend == -1 and bars_in_position >= min_hold_bars:
                exit_price = float(df["close"].iloc[i])
                exit_reason = "Trend flip"
            elif (not long_mode) and prev_trend == -1 and trend == +1 and bars_in_position >= min_hold_bars:
                exit_price = float(df["close"].iloc[i])
                exit_reason = "Trend flip"

        if exit_price is None:
            continue

        price_diff = exit_price - entry_price if long_mode else entry_price - exit_price
        gross_pnl = price_diff / entry_price * stake
        fees = stake * FEE_RATE * 2.0
        pnl_usd = gross_pnl - fees
        equity += pnl_usd
        trades.append({
            "Zeit": entry_ts,
            "Entry": entry_price,
            "ExitZeit": ts,
            "ExitPreis": exit_price,
            "Stake": stake,
            "Fees": fees,
            "ExitReason": exit_reason,
            "PnL (USD)": pnl_usd,
            "Equity": equity,
            "Direction": direction.capitalize(),
            "MinHoldDays": min_hold_days
        })
        in_position = False
        entry_capital = None
        entry_atr = None
        bars_in_position = 0

    if in_position:
        last = df.iloc[-1]
        exit_ts = last.name
        exit_price = float(last["close"])
        stake = entry_capital if entry_capital is not None else equity * RISK_FRACTION
        price_diff = exit_price - entry_price if long_mode else entry_price - exit_price
        gross_pnl = price_diff / entry_price * stake
        fees = stake * FEE_RATE * 2.0
        pnl_usd = gross_pnl - fees
        equity += pnl_usd
        trades.append({
            "Zeit": entry_ts,
            "Entry": entry_price,
            "ExitZeit": exit_ts,
            "ExitPreis": exit_price,
            "Stake": stake,
            "Fees": fees,
            "ExitReason": "Final bar",
            "PnL (USD)": pnl_usd,
            "Equity": equity,
            "Direction": direction.capitalize(),
            "MinHoldDays": min_hold_days
        })

    return pd.DataFrame(trades)

# =========================
# Statistiken
# =========================
def performance_report(trades_df, symbol, param_a, param_b, direction, min_hold_days):
    base = {"Symbol": symbol,
            "ParamA": param_a,
            "ParamB": param_b,
            PARAM_A_LABEL: param_a,
            PARAM_B_LABEL: param_b}
    if INDICATOR_TYPE == "supertrend":
        base["Length"] = param_a
        base["Factor"] = param_b

    if trades_df.empty:
        return {**base, "Trades":0,"WinRate":0.0,
                "AvgPnL":0.0,"ProfitFactor":0.0,"MaxDrawdown":0.0,
                "FinalEquity":START_EQUITY,"Direction":direction,"MinHoldDays":min_hold_days}
    wins = trades_df[trades_df["PnL (USD)"] > 0]
    losses = trades_df[trades_df["PnL (USD)"] < 0]
    win_rate = len(wins)/len(trades_df)
    avg_pnl = trades_df["PnL (USD)"].mean()
    total_win = wins["PnL (USD)"].sum()
    total_loss = abs(losses["PnL (USD)"].sum())
    profit_factor = (total_win/total_loss) if total_loss > 0 else np.inf
    equity_curve = trades_df["Equity"]
    max_drawdown = (equity_curve.cummax() - equity_curve).max() if not equity_curve.empty else 0.0
    final_eq = float(equity_curve.iloc[-1]) if not equity_curve.empty else START_EQUITY
    return {**base,"Trades":len(trades_df),
        "WinRate":win_rate,"AvgPnL":avg_pnl,"ProfitFactor":profit_factor,
        "MaxDrawdown":max_drawdown,"FinalEquity":final_eq,
        "Direction":direction,"MinHoldDays":min_hold_days}

# =========================
# Chart
# =========================
def build_two_panel_figure(symbol, df, trades_df, param_a, param_b, direction, min_hold_days=None):
    direction_title = direction.capitalize()
    hold_text = f", Hold≥{min_hold_days}d" if min_hold_days else ""
    indicator_desc = f"{INDICATOR_DISPLAY_NAME} {PARAM_A_LABEL}={param_a}, {PARAM_B_LABEL}={param_b}"
    line_name = "Supertrend" if INDICATOR_TYPE == "supertrend" else INDICATOR_DISPLAY_NAME
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.06, row_heights=[0.55,0.3,0.15],
                        subplot_titles=(
                            f"{symbol} {direction_title} {indicator_desc}{hold_text}",
                            "Equity",
                            "Momentum"
                        ))

    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="Price"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["indicator_line"], mode="lines",
                             name=line_name, line=dict(color="orange")), row=1, col=1)
    if USE_HIGHER_TIMEFRAME_FILTER and "htf_supertrend" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["htf_supertrend"], mode="lines",
                                 name=f"HTF Supertrend ({HIGHER_TIMEFRAME})",
                                 line=dict(color="purple", dash="dot")), row=1, col=1)

    if not trades_df.empty:
        entry_color = "green" if direction_title == "Long" else "red"
        exit_color = "red" if direction_title == "Long" else "green"
        entry_symbol = "triangle-up" if direction_title == "Long" else "triangle-down"
        exit_symbol = "triangle-down" if direction_title == "Long" else "triangle-up"
        fig.add_trace(go.Scatter(x=trades_df["Zeit"], y=trades_df["Entry"], mode="markers",
                                 marker=dict(color=entry_color, symbol=entry_symbol, size=10),
                                 name=f"{direction_title} Entry"), row=1, col=1)
        fig.add_trace(go.Scatter(x=trades_df["ExitZeit"], y=trades_df["ExitPreis"], mode="markers",
                                 marker=dict(color=exit_color, symbol=exit_symbol, size=10),
                                 name=f"{direction_title} Exit"), row=1, col=1)

        equity_series = pd.Series(index=df.index, dtype=float)
        equity_series[:] = np.nan
        if len(df.index) > 0:
            equity_series.iloc[0] = START_EQUITY
        trade_equity = trades_df.set_index("ExitZeit")["Equity"]
        for ts, value in trade_equity.items():
            if ts in equity_series.index:
                equity_series.loc[ts] = value
        equity_series = equity_series.ffill()
        fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series, mode="lines",
                                 name="Equity"), row=2, col=1)
    else:
        equity_series = pd.Series(index=df.index, data=START_EQUITY, dtype=float) if len(df.index) else None
        if equity_series is not None:
            fig.add_trace(go.Scatter(x=equity_series.index, y=equity_series, mode="lines",
                                     name="Equity"), row=2, col=1)

    if "momentum" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["momentum"], mode="lines",
                                 name="Momentum", line=dict(color="teal")), row=3, col=1)
        fig.add_hrect(y0=RSI_SHORT_THRESHOLD, y1=RSI_LONG_THRESHOLD,
                      line_width=0, fillcolor="gray", opacity=0.15,
                      row=3, col=1)

    fig.update_layout(
        height=900,
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=False)),   # kein Slider oben
        xaxis2=dict(rangeslider=dict(visible=False)),
        xaxis3=dict(rangeslider=dict(visible=True, thickness=0.03), type="date")  # Slider nur unten
    )
    fig.update_xaxes(title_text="Zeit", row=3, col=1)
    fig.update_yaxes(title_text="Preis", row=1, col=1)
    fig.update_yaxes(title_text="Equity (USD)", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    return fig

# =========================
# HTML-Report
# =========================
def df_to_html_table(df, title=None):
    html = ""
    if title:
        html += f"<h3>{title}</h3>\n"
    html += df.to_html(index=False, justify='left', border=0)
    return html

def build_full_report(figs_html_blocks, sections_html, ranking_tables_html):
    html = []
    page_title = f"{INDICATOR_DISPLAY_NAME} Parameter Report"
    html.append(f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{page_title}</title></head><body>")
    html.append(f"<h1>{page_title}</h1>")
    now = datetime.now(BERLIN_TZ).strftime("%Y-%m-%d %H:%M:%S %Z")
    html.append(f"<p>Generiert: {now}</p>")

    # Charts + Trade-Tabellen für beste Parameter je Symbol
    for blk in figs_html_blocks:
        html.append(blk)
    for sec in sections_html:
        html.append(sec)

    # Ranking-Tabellen je Symbol
    html.append("<hr>")
    html.append("<h2>Parameter-Ranking je Symbol</h2>")
    for rtbl in ranking_tables_html:
        html.append(rtbl)

    html.append("</body></html>")
    return "\n".join(html)


def prepare_symbol_dataframe(symbol):
    df = fetch_data(symbol, TIMEFRAME, LOOKBACK)
    df = attach_higher_timeframe_trend(df, symbol)
    df = attach_momentum_filter(df)
    return df

# =========================
# Main
# =========================
if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    if RUN_PARAMETER_SWEEP:
        figs_blocks = []
        sections_blocks = []
        ranking_tables = []
        best_params_summary = []

        directions = ["long", "short"] if ENABLE_SHORTS else ["long"]
        hold_day_candidates = MIN_HOLD_DAY_VALUES if USE_MIN_HOLD_FILTER else [DEFAULT_MIN_HOLD_DAYS]

        for symbol in SYMBOLS:
            df_raw = prepare_symbol_dataframe(symbol)

            results = {d: [] for d in directions}
            trades_per_combo = {d: {} for d in directions}
            df_cache = {}

            for param_a in PARAM_A_VALUES:
                for param_b in PARAM_B_VALUES:
                    cache_key = (param_a, param_b)
                    if cache_key not in df_cache:
                        df_tmp = compute_indicator(df_raw, param_a, param_b)
                        for col in ("htf_trend", "htf_supertrend", "momentum"):
                            if col in df_raw columns:
                                df_tmp[col] = df_raw[col]
                        df_cache[cache_key] = df_tmp
                    df_st = df_cache[cache_key]
                    for atr_mult in ATR_STOP_MULTS:
                        for hold_days in hold_day_candidates:
                            min_hold_bars = hold_days * BARS_PER_DAY
                            for direction in directions:
                                df_st_with_htf = df_st.copy()
                                for col in ("htf_trend", "htf_supertrend", "momentum"):
                                    if col in df_raw.columns:
                                        df_st_with_htf[col] = df_raw[col]
                                trades = backtest_supertrend(
                                    df_st_with_htf,
                                    atr_stop_mult=atr_mult,
                                    direction=direction,
                                    min_hold_bars=min_hold_bars,
                                    min_hold_days=hold_days,
                                )
                                stats = performance_report(
                                    trades,
                                    symbol,
                                    param_a,
                                    param_b,
                                    direction.capitalize(),
                                    hold_days,
                                )
                                stats["ATRStopMult"] = atr_mult if atr_mult is not None else "None"
                                stats["MinHoldBars"] = min_hold_bars
                                results[direction].append(stats)
                                trades_per_combo[direction][(param_a, param_b, atr_mult, hold_days)] = trades

            for direction in directions:
                dir_results = results[direction]
                ranking_df = pd.DataFrame(dir_results)
                ranking_df = ranking_df.sort_values("FinalEquity", ascending=False).reset_index(drop=True) if not ranking_df.empty else ranking_df
                ranking_tables.append(df_to_html_table(
                    ranking_df,
                    title=f"Ranking: {symbol} {INDICATOR_DISPLAY_NAME} ({direction.capitalize()} nach FinalEquity)"
                ))

                if not ranking_df.empty:
                    best_row = ranking_df.iloc[0]
                    best_param_a = best_row.get("ParamA", best_row.get(PARAM_A_LABEL, DEFAULT_PARAM_A))
                    best_param_b = best_row.get("ParamB", best_row.get(PARAM_B_LABEL, DEFAULT_PARAM_B))
                    best_param_a = best_param_a if not pd.isna(best_param_a) else DEFAULT_PARAM_A
                    best_param_b = best_param_b if not pd.isna(best_param_b) else DEFAULT_PARAM_B
                    best_atr_raw = best_row["ATRStopMult"]
                    best_atr = best_atr_raw if best_atr_raw != "None" else None
                    best_hold_days = int(best_row.get("MinHoldDays", DEFAULT_MIN_HOLD_DAYS))
                    best_df = df_cache[(best_param_a, best_param_b)]
                    best_trades = trades_per_combo[direction][(best_param_a, best_param_b, best_atr, best_hold_days)]
                else:
                    best_param_a, best_param_b = DEFAULT_PARAM_A, DEFAULT_PARAM_B
                    best_atr = None
                    best_hold_days = DEFAULT_MIN_HOLD_DAYS
                    best_df = compute_indicator(df_raw, best_param_a, best_param_b)
                    best_trades = pd.DataFrame()

                atr_label = best_atr if best_atr is not None else "None"
                best_params_summary.append({
                    "Symbol": symbol,
                    "Direction": direction.capitalize(),
                    "ParamA": best_param_a,
                    "ParamB": best_param_b,
                    PARAM_A_LABEL: best_param_a,
                    PARAM_B_LABEL: best_param_b,
                    "Length": best_param_a if INDICATOR_TYPE == "supertrend" else None,
                    "Factor": best_param_b if INDICATOR_TYPE == "supertrend" else None,
                    "ATRStopMult": atr_label,
                    "MinHoldDays": best_hold_days,
                })

                fig = build_two_panel_figure(
                    symbol,
                    best_df,
                    best_trades,
                    best_param_a,
                    best_param_b,
                    direction.capitalize(),
                    min_hold_days=best_hold_days,
                )
                fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
                figs_blocks.append(
                    f"<h2>{symbol} – {direction.capitalize()} beste Parameter: {PARAM_A_LABEL}={best_param_a}, {PARAM_B_LABEL}={best_param_b}, ATRStop={atr_label}, MinHold={best_hold_days}d</h2>\n"
                    + fig_html
                )

                sections_blocks.append(
                    df_to_html_table(
                        best_trades,
                        title=f"Trade-Liste {symbol} ({direction.capitalize()} beste Parameter, MinHold={best_hold_days}d)"
                    )
                )

                csv_suffix = "" if direction == "long" else "_short"
                csv_path = os.path.join(OUT_DIR, f"trades_{symbol.replace('/','_')}_{INDICATOR_SLUG}_best{csv_suffix}.csv")
                best_trades.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
                print(f"[Saved] {csv_path}")

                csv_rank_suffix = "" if direction == "long" else "_short"
                csv_rank_path = os.path.join(OUT_DIR, f"ranking_{symbol.replace('/','_')}_{INDICATOR_SLUG}_params{csv_rank_suffix}.csv")
                ranking_df.to_csv(csv_rank_path, sep=";", decimal=",", index=False, encoding="utf-8")
                print(f"[Saved] {csv_rank_path}")

                print(f"[Info] {symbol} {direction.capitalize()} best MinHoldDays: {best_hold_days}")

        if best_params_summary:
            summary_df = pd.DataFrame(best_params_summary)
            summary_path = os.path.join(OUT_DIR, BEST_PARAMS_FILE)
            summary_df.to_csv(summary_path, sep=";", decimal=",", index=False, encoding="utf-8")
            print(f"[Saved] {summary_path}")

        report_html = build_full_report(figs_blocks, sections_blocks, ranking_tables)
        report_path = os.path.join(OUT_DIR, REPORT_FILE)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_html)
        print(f"[Saved] {report_path}")
    elif RUN_SAVED_PARAMS:
        summary_path = os.path.join(OUT_DIR, BEST_PARAMS_FILE)
        summary_df = None
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path, sep=";", decimal=",")
        else:
            derived_rows = []
            for symbol in SYMBOLS:
                rank_path = os.path.join(OUT_DIR, f"ranking_{symbol.replace('/','_')}_{INDICATOR_SLUG}_params.csv")
                if not os.path.exists(rank_path):
                    continue
                rank_df = pd.read_csv(rank_path, sep=";", decimal=",")
                if rank_df.empty:
                    continue
                best_row = rank_df.iloc[0]
                derived_rows.append({
                    "Symbol": symbol,
                    "Direction": best_row.get("Direction", "Long"),
                    "ParamA": best_row.get("ParamA", best_row.get(PARAM_A_LABEL, DEFAULT_PARAM_A)),
                    "ParamB": best_row.get("ParamB", best_row.get(PARAM_B_LABEL, DEFAULT_PARAM_B)),
                    PARAM_A_LABEL: best_row.get(PARAM_A_LABEL, DEFAULT_PARAM_A),
                    PARAM_B_LABEL: best_row.get(PARAM_B_LABEL, DEFAULT_PARAM_B),
                    "Length": best_row.get("Length", DEFAULT_PARAM_A if INDICATOR_TYPE == "supertrend" else None),
                    "Factor": best_row.get("Factor", DEFAULT_PARAM_B if INDICATOR_TYPE == "supertrend" else None),
                    "ATRStopMult": best_row.get("ATRStopMult", "None"),
                    "MinHoldDays": best_row.get("MinHoldDays", DEFAULT_MIN_HOLD_DAYS),
                })
            if derived_rows:
                summary_df = pd.DataFrame(derived_rows)
                summary_df.to_csv(summary_path, sep=";", decimal=",", index=False, encoding="utf-8")
                print(f"[Saved] {summary_path} (derived from ranking files)")

        if summary_df is None or summary_df.empty:
            print(f"[Skip] No saved parameters available. Run the sweep (set RUN_PARAMETER_SWEEP = True) to generate them.")
        else:
            figs_blocks = []
            sections_blocks = []
            ranking_tables = [df_to_html_table(summary_df, title="Gespeicherte Parameter (ohne Sweep)")]
            data_cache = {}
            st_cache = {}

            for _, row in summary_df.iterrows():
                symbol = row.get("Symbol")
                if not symbol:
                    continue
                direction = str(row.get("Direction", "Long")).lower()
                if direction not in {"long", "short"}:
                    print(f"[Warn] Skipping {symbol}: unsupported direction '{direction}'.")
                    continue
                param_a = row.get("ParamA", row.get(PARAM_A_LABEL, DEFAULT_PARAM_A))
                param_b = row.get("ParamB", row.get(PARAM_B_LABEL, DEFAULT_PARAM_B))
                param_a = param_a if not pd.isna(param_a) else DEFAULT_PARAM_A
                param_b = param_b if not pd.isna(param_b) else DEFAULT_PARAM_B
                atr_raw = row.get("ATRStopMult", "None")
                atr_mult = None if pd.isna(atr_raw) or str(atr_raw).lower() == "none" else float(atr_raw)
                hold_days = int(row.get("MinHoldDays", DEFAULT_MIN_HOLD_DAYS))
                min_hold_bars = hold_days * BARS_PER_DAY

                if symbol not in data_cache:
                    data_cache[symbol] = prepare_symbol_dataframe(symbol)
                df_raw = data_cache[symbol]

                st_key = (symbol, param_a, param_b)
                if st_key not in st_cache:
                    df_tmp = compute_indicator(df_raw, param_a, param_b)
                    for col in ("htf_trend", "htf_supertrend", "momentum"):
                        if col in df_raw.columns:
                            df_tmp[col] = df_raw[col]
                    st_cache[st_key] = df_tmp
                df_st = st_cache[st_key]

                trades = backtest_supertrend(
                    df_st,
                    atr_stop_mult=atr_mult,
                    direction=direction,
                    min_hold_bars=min_hold_bars,
                    min_hold_days=hold_days,
                )

                direction_title = direction.capitalize()
                atr_label = "None" if atr_mult is None else atr_mult
                param_desc = f"{PARAM_A_LABEL}={param_a}, {PARAM_B_LABEL}={param_b}"

                fig = build_two_panel_figure(
                    symbol,
                    df_st,
                    trades,
                    param_a,
                    param_b,
                    direction_title,
                    min_hold_days=hold_days,
                )
                fig_html = pio.to_html(fig, include_plotlyjs='cdn', full_html=False)
                figs_blocks.append(
                    f"<h2>{symbol} – {direction_title} gespeicherte Parameter: {param_desc}, ATRStop={atr_label}, MinHold={hold_days}d</h2>\n"
                    + fig_html
                )

                sections_blocks.append(
                    df_to_html_table(
                        trades,
                        title=f"Trade-Liste {symbol} ({direction_title} gespeicherte Parameter, MinHold={hold_days}d)"
                    )
                )

                csv_suffix = "" if direction == "long" else "_short"
                csv_path = os.path.join(OUT_DIR, f"trades_{symbol.replace('/','_')}_{INDICATOR_SLUG}_best{csv_suffix}.csv")
                trades.to_csv(csv_path, sep=";", decimal=",", index=False, encoding="utf-8")
                print(f"[Saved] {csv_path}")

                print(f"[Info] {symbol} {direction_title} MinHoldDays={hold_days} (gespeichert)")

            if figs_blocks or sections_blocks:
                report_html = build_full_report(figs_blocks, sections_blocks, ranking_tables)
                report_path = os.path.join(OUT_DIR, REPORT_FILE)
                with open(report_path, "w", encoding="utf-8") as f:
                    f.write(report_html)
                print(f"[Saved] {report_path}")
            else:
                print("[Skip] No valid entries in saved parameter file.")
    else:
        print("[Skip] Backtesting disabled. Set RUN_PARAMETER_SWEEP = True or RUN_SAVED_PARAMS = True to produce output.")
