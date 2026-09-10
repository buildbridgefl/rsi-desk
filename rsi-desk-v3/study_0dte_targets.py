#!/usr/bin/env python3
"""
Is the flow signal tradeable with SAME-DAY (0DTE) options?

WHAT THIS FIXES FROM study_option_targets.py
--------------------------------------------
That script walked forward N hourly bars regardless of session boundary.
There are only ~7 hourly bars in a US session, so an 8-bar hold crossed
into the NEXT DAY. On a 0DTE contract that is impossible - the option
expires at the close. Every number it produced for long holds was
therefore measuring something you cannot trade with same-day expiry.

This version:
  - NEVER holds past the session close. Every trade is capped at the
    last bar of the same date.
  - Breaks results down BY HOUR OF DAY, because a 10:00 signal and a
    14:30 signal are completely different trades on 0DTE.
  - Reports how many bars of runway were actually left, which is the
    single most important variable and the one the previous script
    silently averaged away.

WHY TIME OF DAY DOMINATES 0DTE
------------------------------
Theta on a same-day contract is not linear - it accelerates into the
close. An ATM 0DTE option can lose most of its extrinsic value in the
final two hours. So the same +0.15% underlying move that pays well at
10:00 may not cover decay at 14:30. A signal that fires late may be
mechanically untradeable no matter how good the directional edge is.

LEVERAGE ON 0DTE
----------------
ATM same-day contracts are far more leveraged than the 100x assumed
previously - often 300-500x versus the underlying percent move intraday,
because the premium is so small. That cuts both ways: a smaller move
hits your target, but decay is eating the position the whole time and
gamma makes the delta unstable. --leverage defaults to 300 here. It is
still a crude approximation, NOT an options pricing model.

WHAT THIS STILL CANNOT TELL YOU
-------------------------------
  - real 0DTE option prices (yfinance has no historical chains)
  - IV behaviour, which on 0DTE moves violently
  - the option bid-ask spread, which on 0DTE is frequently 2-5% round
    trip and can consume a 10% target by itself
  - fills at size, gamma risk near the strike, or pin behaviour
These gaps are LARGE. A good result here is permission to paper trade
and check real fills - not a green light.

Usage:
    python study_0dte_targets.py
    python study_0dte_targets.py --option-target 10 --leverage 300
    python study_0dte_targets.py --max-hold 4 --min-runway 3
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

MIN_EVENTS = 20


# ----------------------------------------------------------------- data
def load_hourly(ticker, period="2y"):
    df = yf.download(ticker, period=period, interval="1h",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def load_daily(ticker):
    df = yf.download(ticker, period="5y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


def daily_trend(daily, ma_len=200):
    ma = daily["Close"].rolling(ma_len).mean()
    up = (daily["Close"] > ma).fillna(False)
    up.index = up.index.date
    return up


def flow_signal(df, imb_thr, volz_thr, vol_window=168):
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    clv = clv.fillna(0.0)
    imbalance = ((clv * df["Volume"]) / df["Volume"].replace(0, np.nan)).fillna(0.0)
    vmean = df["Volume"].rolling(vol_window).mean()
    vstd = df["Volume"].rolling(vol_window).std()
    vol_z = (df["Volume"] - vmean) / vstd.replace(0, np.nan)
    return (imbalance < -imb_thr) & (vol_z > volz_thr)


# --------------------------------------------- same-session excursions
def session_excursions(df, positions, max_hold, min_runway):
    """Entry at the NEXT bar's open. Hold until whichever comes first:
    max_hold bars, or the LAST BAR OF THE SAME SESSION. Trades with less
    than min_runway bars left in the session are skipped entirely -
    on 0DTE those are the ones theta destroys."""
    op = df["Open"].to_numpy()
    hi = df["High"].to_numpy()
    lo = df["Low"].to_numpy()
    dates = np.array([ts.date() for ts in df.index])
    hours = np.array([ts.hour + ts.minute / 60.0 for ts in df.index])
    n = len(df)

    rows, skipped_late = [], 0
    for i in positions:
        e = i + 1
        if e >= n:
            continue
        day = dates[e]
        # last bar index belonging to the same session
        same = np.where(dates == day)[0]
        last_of_day = int(same.max())
        runway = last_of_day - e + 1
        if runway < min_runway:
            skipped_late += 1
            continue
        end = min(e + max_hold, last_of_day + 1)
        if end <= e:
            continue

        entry_px = op[e]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue
        up_path = (hi[e:end] / entry_px - 1.0) * 100.0
        dn_path = (lo[e:end] / entry_px - 1.0) * 100.0
        rows.append({
            "hour": float(hours[e]),
            "runway": int(runway),
            "held": int(end - e),
            "mfe": float(np.nanmax(up_path)),
            "mae": float(np.nanmin(dn_path)),
            "up_path": up_path,
            "dn_path": dn_path,
        })
    return rows, skipped_late


def target_stats(rows, targets):
    out = []
    for t in targets:
        hits, speeds, pre_dd = 0, [], []
        for r in rows:
            idx = np.where(r["up_path"] >= t)[0]
            if len(idx):
                hits += 1
                k = int(idx[0])
                speeds.append(k + 1)
                pre_dd.append(float(np.nanmin(r["dn_path"][:k + 1])))
        n = len(rows)
        out.append({
            "target_pct": t,
            "hit_rate": hits / n if n else np.nan,
            "med_bars": np.median(speeds) if speeds else np.nan,
            "med_dd_before": np.median(pre_dd) if pre_dd else np.nan,
        })
    return pd.DataFrame(out)


def by_hour(rows, target):
    """Hit rate for one target, split by entry hour. This is the table
    that decides whether late-day signals are tradeable at all."""
    buckets = {}
    for r in rows:
        h = int(r["hour"])
        buckets.setdefault(h, []).append(r)
    out = []
    for h in sorted(buckets):
        grp = buckets[h]
        hits = sum(1 for r in grp
                   if np.any(r["up_path"] >= target))
        speeds = [int(np.where(r["up_path"] >= target)[0][0]) + 1
                  for r in grp if np.any(r["up_path"] >= target)]
        out.append({
            "entry_hour": h,
            "n": len(grp),
            "med_runway": float(np.median([r["runway"] for r in grp])),
            "hit_rate": hits / len(grp),
            "med_bars_to_hit": np.median(speeds) if speeds else np.nan,
        })
    return pd.DataFrame(out)


def summarize(rows):
    mfe = np.array([r["mfe"] for r in rows])
    mae = np.array([r["mae"] for r in rows])
    return {"n": len(rows), "med_mfe": float(np.median(mfe)),
            "med_mae": float(np.median(mae)),
            "p25_mae": float(np.percentile(mae, 25)),
            "med_held": float(np.median([r["held"] for r in rows]))}


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,QQQ,IWM,SMH,DIA,XLK,XLF,XLE")
    p.add_argument("--period", default="2y")
    p.add_argument("--max-hold", type=int, default=6,
                   help="max bars held, also capped by session close")
    p.add_argument("--min-runway", type=int, default=2,
                   help="skip signals with fewer bars than this left "
                        "in the session")
    p.add_argument("--imb", type=float, default=0.4)
    p.add_argument("--volz", type=float, default=1.0)
    p.add_argument("--split", type=float, default=0.6)
    p.add_argument("--targets", default="0.03,0.05,0.1,0.2,0.3")
    p.add_argument("--leverage", type=float, default=300.0,
                   help="rough 0DTE leverage vs underlying %% move")
    p.add_argument("--option-target", type=float, default=10.0)
    p.add_argument("--n-random", type=int, default=3000)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    targets = [float(t) for t in a.targets.split(",") if t.strip()]
    implied = a.option_target / a.leverage
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]

    print("SAME-DAY (0DTE) MODE - no position is held past the close")
    print(f"max_hold={a.max_hold} bars, min_runway={a.min_runway} bars")
    print(f"goal: +{a.option_target:.0f}% on the option")
    print(f"at assumed {a.leverage:.0f}x that needs about "
          f"+{implied:.3f}% in the underlying, BEFORE theta and spread.")
    print("Theta accelerates into the close, so treat that number as a "
          "FLOOR, not the requirement.\n")

    rng = np.random.default_rng(a.seed)
    ins_rows, oos_rows, rnd_rows = [], [], []
    total_skipped = 0

    for tk in tickers:
        try:
            h = load_hourly(tk, a.period)
            if h.empty or len(h) < 500:
                print(f"  {tk}: insufficient data")
                continue
            up = daily_trend(load_daily(tk))
            dates = pd.Series(h.index.date, index=h.index)
            uptrend = dates.map(up).fillna(False).to_numpy()

            sig = flow_signal(h, a.imb, a.volz).fillna(False).to_numpy() & uptrend
            cut = int(len(h) * a.split)
            pos = np.where(sig)[0]

            r1, s1 = session_excursions(h, pos[pos < cut], a.max_hold, a.min_runway)
            r2, s2 = session_excursions(h, pos[pos >= cut], a.max_hold, a.min_runway)
            ins_rows += r1
            oos_rows += r2
            total_skipped += s1 + s2

            up_pos = np.where(uptrend)[0]
            up_pos = up_pos[(up_pos > 0) & (up_pos < len(h) - 2)]
            if len(up_pos):
                k = min(a.n_random // len(tickers), len(up_pos))
                pick = rng.choice(up_pos, size=k, replace=False)
                r3, _ = session_excursions(h, pick, a.max_hold, a.min_runway)
                rnd_rows += r3

            print(f"  {tk}: {len(r1)} in-sample, {len(r2)} out-of-sample "
                  f"tradeable ({s1+s2} skipped - too late in session)")
        except Exception as e:
            print(f"  {tk}: failed ({type(e).__name__}: {e})")

    print(f"\nSKIPPED AS UNTRADEABLE (under {a.min_runway} bars of runway): "
          f"{total_skipped}")

    results = {}
    for label, rows in [("SIGNAL - IN-SAMPLE", ins_rows),
                        ("SIGNAL - OUT-OF-SAMPLE", oos_rows),
                        ("RANDOM UPTREND BARS (baseline)", rnd_rows)]:
        print(f"\n{'='*68}")
        print(f"=== {label} ===")
        if len(rows) < MIN_EVENTS:
            print(f"only {len(rows)} events - too few to read")
            continue
        s = summarize(rows)
        print(f"n={s['n']}  median held={s['med_held']:.1f} bars  "
              f"median MFE={s['med_mfe']:+.3f}%  "
              f"median MAE={s['med_mae']:+.3f}%  "
              f"25th pct MAE={s['p25_mae']:+.3f}%")
        ts = target_stats(rows, targets)
        results[label] = ts
        print(ts.to_string(index=False, formatters={
            "target_pct": "{:.2f}".format, "hit_rate": "{:.1%}".format,
            "med_bars": "{:.1f}".format,
            "med_dd_before": "{:+.3f}%".format}))

    key = min(targets, key=lambda t: abs(t - implied))
    if oos_rows and len(oos_rows) >= MIN_EVENTS:
        print(f"\n{'='*68}")
        print(f"=== BY ENTRY HOUR (target {key:.2f}%, out-of-sample) ===")
        print(by_hour(oos_rows, key).to_string(index=False, formatters={
            "med_runway": "{:.1f}".format, "hit_rate": "{:.1%}".format,
            "med_bars_to_hit": "{:.1f}".format}))
        print("Low hit rate late in the day is the expected shape. If the "
              "signal mostly fires late, 0DTE is the wrong instrument for "
              "it regardless of the directional edge.")

    if "SIGNAL - OUT-OF-SAMPLE" in results and \
       "RANDOM UPTREND BARS (baseline)" in results:
        sg = results["SIGNAL - OUT-OF-SAMPLE"].set_index("target_pct")
        rd = results["RANDOM UPTREND BARS (baseline)"].set_index("target_pct")
        print(f"\n{'='*68}")
        print("=== SIGNAL vs RANDOM, OUT-OF-SAMPLE ===")
        print(pd.DataFrame({
            "signal_hit": sg["hit_rate"], "random_hit": rd["hit_rate"],
            "edge": sg["hit_rate"] - rd["hit_rate"],
            "sig_speed": sg["med_bars"], "rnd_speed": rd["med_bars"],
        }).to_string(formatters={
            "signal_hit": "{:.1%}".format, "random_hit": "{:.1%}".format,
            "edge": "{:+.1%}".format, "sig_speed": "{:.1f}".format,
            "rnd_speed": "{:.1f}".format}))
        print("\n'edge' near zero means this entry is no better than a "
              "random uptrend hour - and on 0DTE you would be paying "
              "spread and theta for that coin flip.")

    print(f"\n{'='*68}")
    print("NOT TESTED: real 0DTE prices, IV moves, option bid-ask "
          "(often 2-5% round trip on 0DTE - enough to eat a 10% target "
          "by itself), fills at size, gamma near the strike.")
    print("A good result here means PAPER TRADE IT and compare real "
          "fills. It does not mean the edge survives execution.")
    print("Not financial advice - I am not a licensed advisor, and "
          "position sizing is your call.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
