#!/usr/bin/env python3
"""
BRACKET TEST - does the target get hit BEFORE the stop?

THE GAP THIS CLOSES
-------------------
study_0dte_targets.py reported that 74% of signals TOUCHED +0.30% in the
session. That is not the tradeable number. A trade that dips to -0.50%
and then rallies to +0.30% counts as a "hit" in that script - but if
your stop was at -0.40%, you were out long before the rally. You did not
get the win. You got the loss.

This script races the two levels against each other, bar by bar, in
order. For every (target, stop) pair it reports:

    win_rate      how often target was hit FIRST
    loss_rate     how often stop was hit first
    timeout       how often neither hit before the session close
    expectancy    the actual per-trade result in R (risk units)

Expectancy is what decides whether a system makes money. A 90% win rate
with a stop 10x wider than the target is a losing system. A 40% win rate
with a 3:1 payoff is a good one. Hit rate alone tells you nothing.

INTRABAR AMBIGUITY - THE HONEST CAVEAT
--------------------------------------
Hourly bars give High and Low but not the ORDER they occurred in. If a
bar's range spans both your target and your stop, we cannot know which
came first from this data. This script counts those as LOSSES
(--ambiguous loss, the default) because that is the conservative
assumption. Run with --ambiguous win to see the optimistic bound, and
--ambiguous split to see the midpoint. The truth is between them, and
the gap tells you how much the result depends on an assumption rather
than on evidence. If the two bounds disagree wildly, the result is not
trustworthy at this bar resolution and needs finer data.

BASELINE
--------
Every grid is also run on RANDOM uptrend entries. If random entries show
the same expectancy, the signal contributes nothing and you are paying
spread for a coin flip.

Usage:
    python study_brackets.py
    python study_brackets.py --targets 0.15,0.20,0.30,0.40 --stops 0.20,0.30,0.40,0.60
    python study_brackets.py --ambiguous win        # optimistic bound
    python study_brackets.py --leverage 300 --spread-pct 3
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

MIN_EVENTS = 30


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


# ------------------------------------------------------- trade capture
def capture_trades(df, positions, max_hold, min_runway):
    """Entry at NEXT bar open. Capture the bar-by-bar high/low path,
    never crossing the session close."""
    op = df["Open"].to_numpy()
    hi = df["High"].to_numpy()
    lo = df["Low"].to_numpy()
    cl = df["Close"].to_numpy()
    dates = np.array([ts.date() for ts in df.index])
    hours = np.array([ts.hour for ts in df.index])
    n = len(df)

    out, skipped = [], 0
    for i in positions:
        e = i + 1
        if e >= n:
            continue
        day = dates[e]
        same = np.where(dates == day)[0]
        last = int(same.max())
        runway = last - e + 1
        if runway < min_runway:
            skipped += 1
            continue
        end = min(e + max_hold, last + 1)
        entry_px = op[e]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue
        out.append({
            "hour": int(hours[e]),
            "up": (hi[e:end] / entry_px - 1.0) * 100.0,
            "dn": (lo[e:end] / entry_px - 1.0) * 100.0,
            "final": (cl[end - 1] / entry_px - 1.0) * 100.0,
        })
    return out, skipped


# ------------------------------------------------------- bracket race
def race(trade, target, stop, ambiguous="loss"):
    """Walk bars in order. Return (+1 target first, -1 stop first,
    0 neither) plus the bar it resolved on and the realized % move.

    Ambiguity: if ONE bar's range contains both levels, we cannot know
    the order from OHLC. Handled per the `ambiguous` setting."""
    up, dn = trade["up"], trade["dn"]
    for k in range(len(up)):
        hit_t = up[k] >= target
        hit_s = dn[k] <= -stop
        if hit_t and hit_s:
            if ambiguous == "win":
                return 1, k, target
            if ambiguous == "split":
                return 0.5, k, (target - stop) / 2.0
            return -1, k, -stop
        if hit_t:
            return 1, k, target
        if hit_s:
            return -1, k, -stop
    return 0, len(up) - 1, trade["final"]


def grid(trades, targets, stops, ambiguous, spread_pct, leverage):
    rows = []
    for t in targets:
        for s in stops:
            wins = losses = timeouts = 0
            ambiguous_bars = 0
            r_sum = 0.0
            for tr in trades:
                # count how often the ambiguity actually bit
                for k in range(len(tr["up"])):
                    if tr["up"][k] >= t and tr["dn"][k] <= -s:
                        ambiguous_bars += 1
                        break
                    if tr["up"][k] >= t or tr["dn"][k] <= -s:
                        break

                res, _, move = race(tr, t, s, ambiguous)
                if res == 1:
                    wins += 1
                    r_sum += t / s
                elif res == -1:
                    losses += 1
                    r_sum -= 1.0
                elif res == 0.5:
                    r_sum += 0.0
                else:
                    timeouts += 1
                    r_sum += move / s
            n = len(trades)
            if n == 0:
                continue
            exp_r = r_sum / n
            # option-space estimate, net of round-trip spread
            opt_gross = exp_r * s * leverage
            rows.append({
                "target": t, "stop": s, "R": t / s,
                "win%": wins / n, "loss%": losses / n, "timeout%": timeouts / n,
                "exp_R": exp_r,
                "opt_net%": opt_gross - spread_pct,
                "ambig%": ambiguous_bars / n,
            })
    return pd.DataFrame(rows)


FMT = {"target": "{:.2f}".format, "stop": "{:.2f}".format,
       "R": "{:.2f}".format, "win%": "{:.1%}".format,
       "loss%": "{:.1%}".format, "timeout%": "{:.1%}".format,
       "exp_R": "{:+.3f}".format, "opt_net%": "{:+.1f}".format,
       "ambig%": "{:.1%}".format}


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,QQQ,IWM,SMH,DIA,XLK,XLF,XLE")
    p.add_argument("--period", default="2y")
    p.add_argument("--max-hold", type=int, default=6)
    p.add_argument("--min-runway", type=int, default=2)
    p.add_argument("--imb", type=float, default=0.4)
    p.add_argument("--volz", type=float, default=1.0)
    p.add_argument("--split", type=float, default=0.6)
    p.add_argument("--targets", default="0.15,0.20,0.30,0.40")
    p.add_argument("--stops", default="0.20,0.30,0.40,0.60")
    p.add_argument("--ambiguous", default="loss",
                   choices=["loss", "win", "split"],
                   help="how to treat bars containing BOTH levels")
    p.add_argument("--leverage", type=float, default=300.0)
    p.add_argument("--spread-pct", type=float, default=3.0,
                   help="round-trip option spread cost, %% of premium")
    p.add_argument("--n-random", type=int, default=2400)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    targets = [float(x) for x in a.targets.split(",") if x.strip()]
    stops = [float(x) for x in a.stops.split(",") if x.strip()]
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]

    print("BRACKET TEST - target vs stop, raced bar by bar")
    print(f"ambiguous bars counted as: {a.ambiguous.upper()}")
    print(f"option estimate assumes {a.leverage:.0f}x leverage and "
          f"{a.spread_pct:.1f}% round-trip spread\n")

    rng = np.random.default_rng(a.seed)
    ins, oos, rnd = [], [], []
    for tk in tickers:
        try:
            h = load_hourly(tk, a.period)
            if h.empty or len(h) < 500:
                continue
            up = daily_trend(load_daily(tk))
            dates = pd.Series(h.index.date, index=h.index)
            uptrend = dates.map(up).fillna(False).to_numpy()
            sig = flow_signal(h, a.imb, a.volz).fillna(False).to_numpy() & uptrend
            cut = int(len(h) * a.split)
            pos = np.where(sig)[0]

            t1, _ = capture_trades(h, pos[pos < cut], a.max_hold, a.min_runway)
            t2, _ = capture_trades(h, pos[pos >= cut], a.max_hold, a.min_runway)
            ins += t1
            oos += t2

            up_pos = np.where(uptrend)[0]
            up_pos = up_pos[(up_pos > 0) & (up_pos < len(h) - 2)]
            if len(up_pos):
                k = min(a.n_random // len(tickers), len(up_pos))
                t3, _ = capture_trades(h, rng.choice(up_pos, size=k, replace=False),
                                       a.max_hold, a.min_runway)
                rnd += t3
            print(f"  {tk}: {len(t1)} in-sample, {len(t2)} out-of-sample")
        except Exception as e:
            print(f"  {tk}: failed ({type(e).__name__}: {e})")

    for label, trades in [("IN-SAMPLE", ins), ("OUT-OF-SAMPLE", oos),
                          ("RANDOM UPTREND (baseline)", rnd)]:
        print(f"\n{'='*84}")
        print(f"=== {label}  (n={len(trades)}) ===")
        if len(trades) < MIN_EVENTS:
            print("too few trades")
            continue
        g = grid(trades, targets, stops, a.ambiguous, a.spread_pct, a.leverage)
        print(g.to_string(index=False, formatters=FMT))

    # sensitivity: how much does the ambiguity assumption change things?
    if len(oos) >= MIN_EVENTS:
        print(f"\n{'='*84}")
        print("=== AMBIGUITY SENSITIVITY (out-of-sample) ===")
        print("Same grid under both extreme assumptions. If these differ "
              "a lot, hourly bars cannot resolve your bracket and you "
              "need finer data before trusting any of it.")
        for mode in ["loss", "win"]:
            g = grid(oos, targets, stops, mode, a.spread_pct, a.leverage)
            best = g.loc[g["exp_R"].idxmax()]
            print(f"  ambiguous={mode:<5} best cell: target {best['target']:.2f} "
                  f"stop {best['stop']:.2f}  exp_R={best['exp_R']:+.3f}  "
                  f"win={best['win%']:.1%}")

    print(f"\n{'='*84}")
    print("HOW TO READ THIS")
    print("  exp_R is expectancy in RISK UNITS per trade. Positive means "
          "the bracket made money on average. This is the only column "
          "that decides anything - a high win% with negative exp_R is a "
          "losing system.")
    print("  opt_net% is a CRUDE option-space estimate: exp_R x stop x "
          "leverage, minus spread. It assumes constant delta and ignores "
          "theta entirely, so treat it as an upper bound.")
    print("  ambig% is how often a single bar contained both levels. "
          "High values mean the result leans on an assumption, not data.")
    print("  Compare every cell against the RANDOM baseline. If random "
          "entries show similar exp_R, the signal adds nothing.")
    print("\nTHETA IS NOT MODELLED. On 0DTE it is a major cost and it "
          "makes opt_net% optimistic. Real 0DTE prices, IV moves, and "
          "fills at size are all untested here.")
    print("Not financial advice - I am not a licensed advisor. Sizing "
          "and whether to trade this at all are your decisions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
