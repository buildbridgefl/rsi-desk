#!/usr/bin/env python3
"""
BRACKET TEST at MINUTE resolution - settling the ambiguity.

THE PROBLEM THIS SOLVES
-----------------------
study_brackets.py could not decide whether the target or the stop was
hit first when a single HOURLY bar's range contained both levels. That
happened on 20-43% of trades, and the assumption swung the answer from
exp_R = -0.007 (ambiguous counted as loss) to +0.488 (counted as win).
Opposite sides of zero. The hourly test produced no usable answer.

This version:
  - detects the signal on HOURLY bars, exactly as validated
  - then races target vs stop on 1-MINUTE bars inside the hold window

A 1-minute bar can still contain both levels, but far less often. The
ambig% column tells you how often it still happened, so you can see
whether the answer is now resting on data or still on an assumption.

THE HARD LIMIT - READ THIS BEFORE BELIEVING ANY NUMBER
------------------------------------------------------
Yahoo serves at most ~7 DAYS of 1-minute data. Across 8 tickers at
roughly 1-2 signals per day that is perhaps 50-100 trades TOTAL. That
means:

  - NO in-sample / out-of-sample split is possible. Everything here is
    effectively in-sample. There is no verdict block.
  - One week is ONE market character. A calm week and a volatile week
    would give different answers.
  - At n=60, a difference of a few trades moves exp_R noticeably.
    Confidence intervals will be wide and this script prints them.

So this can tell you whether the hourly ambiguity was hiding something
worth pursuing. It CANNOT confirm an edge. If the result looks good,
the correct next step is paper trading or buying real intraday history -
not sizing up.

Usage:
    python study_brackets_minute.py
    python study_brackets_minute.py --targets 0.15,0.20,0.30 --stops 0.20,0.30,0.40
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

MIN_TRADES = 20


# ----------------------------------------------------------------- data
def load_minute(ticker, days=7):
    df = yf.download(ticker, period=f"{min(days,7)}d", interval="1m",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


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


# ------------------------------------------------- signal -> minute path
def build_trades(hourly, minute, uptrend, imb, volz, max_hold_min, min_runway_min):
    """Signal fires on an hourly bar close. Entry is the open of the NEXT
    hourly bar - located in the MINUTE series by timestamp. Then walk
    minute bars forward, never past the session close."""
    sig = flow_signal(hourly, imb, volz).fillna(False).to_numpy() & uptrend
    sig_times = hourly.index[np.where(sig)[0]]

    m_idx = minute.index
    m_open = minute["Open"].to_numpy()
    m_hi = minute["High"].to_numpy()
    m_lo = minute["Low"].to_numpy()
    m_cl = minute["Close"].to_numpy()
    m_dates = np.array([ts.date() for ts in m_idx])

    trades, skipped = [], 0
    for t in sig_times:
        # entry = first minute bar strictly after this hourly bar closes
        # (hourly bar labelled t covers t .. t+1h, so entry is at t+1h)
        entry_ts = t + pd.Timedelta(hours=1)
        loc = m_idx.searchsorted(entry_ts, side="left")
        if loc >= len(m_idx):
            continue
        day = m_dates[loc]
        same = np.where(m_dates == day)[0]
        if len(same) == 0:
            continue
        last = int(same.max())
        runway = last - loc + 1
        if runway < min_runway_min:
            skipped += 1
            continue
        end = min(loc + max_hold_min, last + 1)
        entry_px = m_open[loc]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue
        trades.append({
            "up": (m_hi[loc:end] / entry_px - 1.0) * 100.0,
            "dn": (m_lo[loc:end] / entry_px - 1.0) * 100.0,
            "final": (m_cl[end - 1] / entry_px - 1.0) * 100.0,
        })
    return trades, skipped


def random_trades(minute, uptrend_dates, n, max_hold_min, min_runway_min, rng):
    m_idx = minute.index
    m_open = minute["Open"].to_numpy()
    m_hi = minute["High"].to_numpy()
    m_lo = minute["Low"].to_numpy()
    m_cl = minute["Close"].to_numpy()
    m_dates = np.array([ts.date() for ts in m_idx])
    ok = np.array([d in uptrend_dates for d in m_dates])
    cand = np.where(ok)[0]
    cand = cand[(cand > 0) & (cand < len(m_idx) - min_runway_min - 1)]
    if len(cand) == 0:
        return []
    picks = rng.choice(cand, size=min(n, len(cand)), replace=False)

    out = []
    for loc in picks:
        day = m_dates[loc]
        same = np.where(m_dates == day)[0]
        last = int(same.max())
        if last - loc + 1 < min_runway_min:
            continue
        end = min(loc + max_hold_min, last + 1)
        entry_px = m_open[loc]
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue
        out.append({
            "up": (m_hi[loc:end] / entry_px - 1.0) * 100.0,
            "dn": (m_lo[loc:end] / entry_px - 1.0) * 100.0,
            "final": (m_cl[end - 1] / entry_px - 1.0) * 100.0,
        })
    return out


# ------------------------------------------------------- bracket race
def race(trade, target, stop, ambiguous="loss"):
    up, dn = trade["up"], trade["dn"]
    for k in range(len(up)):
        hit_t = up[k] >= target
        hit_s = dn[k] <= -stop
        if hit_t and hit_s:
            return (1 if ambiguous == "win" else -1), True
        if hit_t:
            return 1, False
        if hit_s:
            return -1, False
    return 0, False


def grid(trades, targets, stops, ambiguous):
    rows = []
    n = len(trades)
    for t in targets:
        for s in stops:
            wins = losses = timeouts = ambig = 0
            rs = []
            for tr in trades:
                res, was_ambig = race(tr, t, s, ambiguous)
                ambig += int(was_ambig)
                if res == 1:
                    wins += 1
                    rs.append(t / s)
                elif res == -1:
                    losses += 1
                    rs.append(-1.0)
                else:
                    timeouts += 1
                    rs.append(tr["final"] / s)
            rs = np.array(rs)
            exp_r = rs.mean() if len(rs) else np.nan
            # 95% CI on the mean - wide samples deserve visible error bars
            se = rs.std(ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.nan
            rows.append({
                "target": t, "stop": s, "R": t / s,
                "win%": wins / n, "loss%": losses / n, "timeout%": timeouts / n,
                "exp_R": exp_r,
                "ci_lo": exp_r - 1.96 * se, "ci_hi": exp_r + 1.96 * se,
                "ambig%": ambig / n,
            })
    return pd.DataFrame(rows)


FMT = {"target": "{:.2f}".format, "stop": "{:.2f}".format, "R": "{:.2f}".format,
       "win%": "{:.1%}".format, "loss%": "{:.1%}".format,
       "timeout%": "{:.1%}".format, "exp_R": "{:+.3f}".format,
       "ci_lo": "{:+.3f}".format, "ci_hi": "{:+.3f}".format,
       "ambig%": "{:.1%}".format}


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,QQQ,IWM,SMH,DIA,XLK,XLF,XLE")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--max-hold-min", type=int, default=360,
                   help="minutes held, also capped by session close")
    p.add_argument("--min-runway-min", type=int, default=30)
    p.add_argument("--imb", type=float, default=0.4)
    p.add_argument("--volz", type=float, default=1.0)
    p.add_argument("--targets", default="0.15,0.20,0.30,0.40")
    p.add_argument("--stops", default="0.20,0.30,0.40,0.60")
    p.add_argument("--n-random", type=int, default=1200)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()

    targets = [float(x) for x in a.targets.split(",") if x.strip()]
    stops = [float(x) for x in a.stops.split(",") if x.strip()]
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]

    print("MINUTE-RESOLUTION BRACKET TEST")
    print("signal detected on HOURLY bars, target/stop raced on 1-MINUTE bars")
    print(f"WARNING: Yahoo caps 1m data at ~{a.days} days. Sample will be "
          "small and there is NO out-of-sample block. Treat as indicative "
          "only.\n")

    rng = np.random.default_rng(a.seed)
    sig_trades, rnd_trades, total_skipped = [], [], 0

    for tk in tickers:
        try:
            mi = load_minute(tk, a.days)
            if mi.empty or len(mi) < 500:
                print(f"  {tk}: no minute data")
                continue
            ho = load_hourly(tk)
            up_series = daily_trend(load_daily(tk))
            h_dates = pd.Series(ho.index.date, index=ho.index)
            uptrend = h_dates.map(up_series).fillna(False).to_numpy()
            up_days = set(d for d, v in up_series.items() if v)

            tr, sk = build_trades(ho, mi, uptrend, a.imb, a.volz,
                                  a.max_hold_min, a.min_runway_min)
            sig_trades += tr
            total_skipped += sk
            rnd_trades += random_trades(mi, up_days, a.n_random // len(tickers),
                                        a.max_hold_min, a.min_runway_min, rng)
            print(f"  {tk}: {len(tr)} signal trades in the minute window "
                  f"({sk} skipped, too late in session)")
        except Exception as e:
            print(f"  {tk}: failed ({type(e).__name__}: {e})")

    print(f"\ntotal signal trades: {len(sig_trades)}   "
          f"random baseline trades: {len(rnd_trades)}")

    if len(sig_trades) < MIN_TRADES:
        print(f"\nOnly {len(sig_trades)} trades - below {MIN_TRADES}. "
              "One week of minute data does not contain enough signals to "
              "say anything. This is a data limit, not a result. Real "
              "intraday history (Polygon, Databento, IBKR) would be "
              "needed to settle it.")
        return 0

    for label, trades in [("SIGNAL (all in-sample - no split possible)", sig_trades),
                          ("RANDOM UPTREND (baseline)", rnd_trades)]:
        print(f"\n{'='*88}")
        print(f"=== {label}  (n={len(trades)}) ===")
        if len(trades) < MIN_TRADES:
            print("too few trades")
            continue
        g = grid(trades, targets, stops, "loss")
        print(g.to_string(index=False, formatters=FMT))

    print(f"\n{'='*88}")
    print("=== AMBIGUITY CHECK ===")
    g_loss = grid(sig_trades, targets, stops, "loss")
    g_win = grid(sig_trades, targets, stops, "win")
    bl, bw = g_loss.loc[g_loss.exp_R.idxmax()], g_win.loc[g_win.exp_R.idxmax()]
    print(f"  ambiguous=loss  best: target {bl['target']:.2f} stop "
          f"{bl['stop']:.2f}  exp_R={bl['exp_R']:+.3f}")
    print(f"  ambiguous=win   best: target {bw['target']:.2f} stop "
          f"{bw['stop']:.2f}  exp_R={bw['exp_R']:+.3f}")
    print(f"  max ambig% across grid: {g_loss['ambig%'].max():.1%}")
    print("  On hourly bars this gap was -0.007 vs +0.488. If it is now "
          "small, the minute data resolved it and the numbers above mean "
          "something.")

    print(f"\n{'='*88}")
    print("READ THE CONFIDENCE INTERVAL, NOT JUST exp_R.")
    print("  ci_lo and ci_hi are the 95% interval on expectancy. If ci_lo "
          "is below zero, this sample cannot rule out a losing system - "
          "no matter how good exp_R looks.")
    print("  With one week of data, expect wide intervals. That is the "
          "honest state of the evidence, not a flaw in the test.")
    print("\nSTILL NOT MODELLED: theta, real option prices, IV, option "
          "bid-ask (2-5% round trip on 0DTE), fills at size.")
    print("Not financial advice - I am not a licensed advisor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
