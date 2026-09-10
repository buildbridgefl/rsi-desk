#!/usr/bin/env python3
"""
BRACKET TEST at MINUTE resolution - FIXED timestamp alignment.

THE BUG IN THE PREVIOUS VERSION
-------------------------------
It located each hourly signal's entry inside the minute series with:

    loc = m_idx.searchsorted(entry_ts, side="left")

When entry_ts was EARLIER than all available minute data - which is true
for almost every signal, since the hourly series spans 2 years and the
minute series only 7 days - searchsorted returns 0. So hundreds of old
signals were all silently mapped onto the FIRST minute bar of the
window. That produced 1,003 phantom trades instead of the ~60-80 real
ones, and explains why zero trades were ever "too late in session":
they were all pinned to the same early-morning bar.

The reported exp_R of +0.055 came from that broken sample. It meant
nothing.

WHAT IS FIXED
-------------
  1. The hourly signal series is CLIPPED to the minute data's actual
     span before anything else happens.
  2. After searchsorted, the matched bar's timestamp is CHECKED against
     the intended entry time. If it is off by more than --tolerance-min
     minutes, the trade is DISCARDED rather than silently accepted.
  3. Timezones on both indices are normalized, since yfinance can return
     tz-aware hourly and tz-naive minute data (or vice versa) depending
     on version and ticker.
  4. Diagnostics print the date ranges, how many signals fell inside the
     window, and how many were rejected by the tolerance check. If the
     alignment breaks again, it will be visible instead of silent.

WHAT TO EXPECT NOW
------------------
Roughly 1-2 signals per ticker per day, so about 60-100 trades total
across 8 tickers in 7 days. At that size the 95% confidence intervals
will be WIDE - likely spanning zero. That is the honest state of one
week of data, not a defect. If ci_lo sits below zero, this sample
cannot distinguish a real edge from noise, and the correct conclusion
is "unresolved", not "positive".

Usage:
    python study_brackets_minute_fixed.py
    python study_brackets_minute_fixed.py --tolerance-min 90
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

MIN_TRADES = 20


# ----------------------------------------------------------------- data
def _strip_tz(idx):
    """Normalize to tz-naive so hourly and minute indices are comparable.
    yfinance is inconsistent about this across versions and tickers."""
    try:
        if getattr(idx, "tz", None) is not None:
            return idx.tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return idx


def load_minute(ticker, days=7):
    df = yf.download(ticker, period=f"{min(days,7)}d", interval="1m",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df.index = _strip_tz(df.index)
    return df


def load_hourly(ticker, period="2y"):
    df = yf.download(ticker, period=period, interval="1h",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    df.index = _strip_tz(df.index)
    return df


def load_daily(ticker):
    df = yf.download(ticker, period="5y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    df.index = _strip_tz(df.index)
    return df


def daily_trend(daily, ma_len=200):
    ma = daily["Close"].rolling(ma_len).mean()
    up = (daily["Close"] > ma).fillna(False)
    up.index = [d.date() for d in daily.index]
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
def build_trades(hourly, minute, uptrend, imb, volz,
                 max_hold_min, min_runway_min, tolerance_min, verbose_tag=""):
    """Signal on an hourly bar; entry at the following hour, located in
    the MINUTE series and VERIFIED to actually be there."""
    sig = flow_signal(hourly, imb, volz).fillna(False).to_numpy() & uptrend
    sig_times = hourly.index[np.where(sig)[0]]

    m_idx = minute.index
    if len(m_idx) == 0:
        return [], {"in_window": 0, "rejected": 0, "late": 0}
    m_start, m_end = m_idx[0], m_idx[-1]

    # FIX 1: only signals whose ENTRY can possibly fall inside the
    # minute window are considered at all.
    entry_times = sig_times + pd.Timedelta(hours=1)
    keep = (entry_times >= m_start) & (entry_times <= m_end)
    entry_times = entry_times[keep]

    m_open = minute["Open"].to_numpy()
    m_hi = minute["High"].to_numpy()
    m_lo = minute["Low"].to_numpy()
    m_cl = minute["Close"].to_numpy()
    m_dates = np.array([ts.date() for ts in m_idx])

    trades = []
    rejected = late = 0
    tol = pd.Timedelta(minutes=tolerance_min)

    for entry_ts in entry_times:
        loc = m_idx.searchsorted(entry_ts, side="left")
        if loc >= len(m_idx):
            rejected += 1
            continue
        # FIX 2: verify the bar we landed on is actually near the time we
        # wanted. Off by more than tolerance -> the signal has no matching
        # minute bar (market closed, gap, holiday) so DISCARD it.
        if abs(m_idx[loc] - entry_ts) > tol:
            rejected += 1
            continue

        day = m_dates[loc]
        same = np.where(m_dates == day)[0]
        if len(same) == 0:
            rejected += 1
            continue
        last = int(same.max())
        runway = last - loc + 1
        if runway < min_runway_min:
            late += 1
            continue

        end = min(loc + max_hold_min, last + 1)
        entry_px = m_open[loc]
        if not np.isfinite(entry_px) or entry_px <= 0:
            rejected += 1
            continue
        trades.append({
            "up": (m_hi[loc:end] / entry_px - 1.0) * 100.0,
            "dn": (m_lo[loc:end] / entry_px - 1.0) * 100.0,
            "final": (m_cl[end - 1] / entry_px - 1.0) * 100.0,
        })

    return trades, {"in_window": int(keep.sum()), "rejected": rejected,
                    "late": late}


def random_trades(minute, uptrend_days, n, max_hold_min, min_runway_min, rng):
    m_idx = minute.index
    if len(m_idx) == 0:
        return []
    m_open = minute["Open"].to_numpy()
    m_hi = minute["High"].to_numpy()
    m_lo = minute["Low"].to_numpy()
    m_cl = minute["Close"].to_numpy()
    m_dates = np.array([ts.date() for ts in m_idx])
    ok = np.array([d in uptrend_days for d in m_dates])
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
    rows, n = [], len(trades)
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
            se = rs.std(ddof=1) / np.sqrt(len(rs)) if len(rs) > 1 else np.nan
            rows.append({
                "target": t, "stop": s, "R": t / s,
                "win%": wins / n, "loss%": losses / n, "timeout%": timeouts / n,
                "exp_R": exp_r, "ci_lo": exp_r - 1.96 * se,
                "ci_hi": exp_r + 1.96 * se, "ambig%": ambig / n,
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
    p.add_argument("--max-hold-min", type=int, default=360)
    p.add_argument("--min-runway-min", type=int, default=30)
    p.add_argument("--tolerance-min", type=int, default=90,
                   help="max minutes the matched bar may differ from the "
                        "intended entry time before the trade is discarded")
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

    print("MINUTE-RESOLUTION BRACKET TEST  (timestamp alignment FIXED)")
    print("Previous version mapped ~2 years of hourly signals onto the "
          "first minute bar, producing 1,003 phantom trades. Expect far "
          "fewer trades now - that is the fix working.\n")

    rng = np.random.default_rng(a.seed)
    sig_trades, rnd_trades = [], []
    tot = {"in_window": 0, "rejected": 0, "late": 0}

    for tk in tickers:
        try:
            mi = load_minute(tk, a.days)
            if mi.empty or len(mi) < 300:
                print(f"  {tk}: no minute data")
                continue
            ho = load_hourly(tk)
            up_series = daily_trend(load_daily(tk))
            h_dates = pd.Series([d.date() for d in ho.index], index=ho.index)
            uptrend = h_dates.map(up_series).fillna(False).to_numpy()
            up_days = set(d for d, v in up_series.items() if v)

            tr, diag = build_trades(ho, mi, uptrend, a.imb, a.volz,
                                    a.max_hold_min, a.min_runway_min,
                                    a.tolerance_min, tk)
            sig_trades += tr
            for k in tot:
                tot[k] += diag[k]
            rnd_trades += random_trades(mi, up_days, a.n_random // len(tickers),
                                        a.max_hold_min, a.min_runway_min, rng)
            print(f"  {tk}: minute data {mi.index[0].date()} -> "
                  f"{mi.index[-1].date()}   "
                  f"{diag['in_window']} signals in window -> "
                  f"{len(tr)} trades "
                  f"({diag['rejected']} rejected, {diag['late']} too late)")
        except Exception as e:
            print(f"  {tk}: failed ({type(e).__name__}: {e})")

    print(f"\nALIGNMENT DIAGNOSTICS")
    print(f"  signals falling inside the minute window: {tot['in_window']}")
    print(f"  rejected (no matching minute bar):        {tot['rejected']}")
    print(f"  skipped (too late in session):            {tot['late']}")
    print(f"  usable trades:                            {len(sig_trades)}")
    print(f"  random baseline trades:                   {len(rnd_trades)}")
    if tot["late"] == 0 and len(sig_trades) > 50:
        print("  WARNING: zero late-session skips with a large sample was "
              "the signature of the old bug. Check the numbers above.")

    if len(sig_trades) < MIN_TRADES:
        print(f"\nOnly {len(sig_trades)} usable trades - below {MIN_TRADES}.")
        print("This is a DATA LIMIT, not a verdict. Seven days of minute "
              "bars simply does not contain enough signals. Settling this "
              "properly needs real intraday history (Polygon, Databento, "
              "IBKR). Do not read the absence of a result as a negative "
              "result, or as a positive one.")
        return 0

    for label, trades in [("SIGNAL (all in-sample - no split possible)", sig_trades),
                          ("RANDOM UPTREND (baseline)", rnd_trades)]:
        print(f"\n{'='*88}")
        print(f"=== {label}  (n={len(trades)}) ===")
        if len(trades) < MIN_TRADES:
            print("too few trades")
            continue
        print(grid(trades, targets, stops, "loss").to_string(
            index=False, formatters=FMT))

    g_loss = grid(sig_trades, targets, stops, "loss")
    g_win = grid(sig_trades, targets, stops, "win")
    bl, bw = g_loss.loc[g_loss.exp_R.idxmax()], g_win.loc[g_win.exp_R.idxmax()]
    print(f"\n{'='*88}")
    print("=== AMBIGUITY CHECK ===")
    print(f"  ambiguous=loss  best exp_R={bl['exp_R']:+.3f} "
          f"(target {bl['target']:.2f} stop {bl['stop']:.2f})")
    print(f"  ambiguous=win   best exp_R={bw['exp_R']:+.3f} "
          f"(target {bw['target']:.2f} stop {bw['stop']:.2f})")
    print(f"  max ambig% across grid: {g_loss['ambig%'].max():.1%}")

    best = g_loss.loc[g_loss.exp_R.idxmax()]
    print(f"\n{'='*88}")
    print("=== THE VERDICT ON THIS SAMPLE ===")
    print(f"  best cell: target {best['target']:.2f} / stop "
          f"{best['stop']:.2f}   exp_R={best['exp_R']:+.3f}   "
          f"95% CI [{best['ci_lo']:+.3f}, {best['ci_hi']:+.3f}]")
    if best["ci_lo"] <= 0:
        print("  ci_lo is at or below ZERO. This sample CANNOT rule out a "
              "losing system. Whatever exp_R says, the honest reading is "
              "UNRESOLVED - one week of data is not enough.")
    else:
        print("  ci_lo is above zero on this sample. That is encouraging "
              "but it is ONE WEEK, fully in-sample, with no out-of-sample "
              "block. It is a reason to paper trade, not to size up.")
    print("\n  Also note: picking the best of 16 cells inflates it. The "
          "single best cell out of a grid will look better than it is - "
          "that is what searching a grid does.")

    print(f"\n{'='*88}")
    print("STILL NOT MODELLED: theta, real option prices, IV, option "
          "bid-ask (2-5% round trip on 0DTE - larger than most of the "
          "edges in this grid), fills at size.")
    print("Not financial advice - I am not a licensed advisor.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
