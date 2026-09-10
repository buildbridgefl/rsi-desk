#!/usr/bin/env python3
"""
Which side has the edge - buyers or sellers?

This is the question the fib/absorption study raised without answering.
That study bought AGAINST heavy sellers (the "absorption reversal" idea)
and lost -2.14R out-of-sample. The obvious follow-up: would going WITH
the dominant side have done better?

So this tests both directly, on the same events:

  DOMINANCE EVENT = a bar where one side clearly won
     |imbalance| > threshold  AND  volume z-score > threshold
     imbalance > 0  -> buyers were dominant
     imbalance < 0  -> sellers were dominant

  Then measure RAW forward returns (not sign-flipped) after each type:
     after BUYER dominance, does price rise (continuation)
                            or fall (reversal)?
     after SELLER dominance, does price fall (continuation)
                             or rise (reversal)?

Four possible worlds, and the data picks one:
  - both continue      -> momentum works, trade with the aggressor
  - both revert        -> mean reversion works, fade the aggressor
  - asymmetric         -> one side is informed, the other is noise
  - neither            -> no edge, dominance is not predictive

It also splits by TREND REGIME (daily close above/below 200SMA), because
"buyers are dominant" plausibly means something different in an uptrend
than in a downtrend.

Pooled across tickers from the start - we learned the hard way that a
single symbol on 59 days of 5m data cannot produce a testable sample.

Every block is permutation tested against random bars drawn from the
same universe, and split in/out-of-sample by calendar date.

Usage:
    python study_side_edge.py
    python study_side_edge.py --tickers SPY,QQQ,IWM,SMH,DIA,XLK,XLF,XLE
    python study_side_edge.py --imb 0.5 --volz 1.5
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HORIZONS_BARS = [1, 3, 6, 12, 24]   # 5m bars -> 5, 15, 30, 60, 120 min
BAR_MINUTES = 5
MIN_EVENTS = 15


# ----------------------------------------------------------------- data
def load_intraday(ticker: str, days: int = 59) -> pd.DataFrame:
    df = yf.download(ticker, period=f"{min(days, 59)}d", interval="5m",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def load_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


def daily_trend_filter(daily: pd.DataFrame, ma_len: int = 200) -> pd.Series:
    ma = daily["Close"].rolling(ma_len).mean()
    up = (daily["Close"] > ma).fillna(False)
    up.index = up.index.date
    return up


# ------------------------------------------------------------- signal
def compute_imbalance(df: pd.DataFrame, window: int = 1, vol_window: int = 78):
    """CLV-weighted volume imbalance. Same proxy used across the desk.

    CLV = where the close sat inside the bar's range, -1 (at the low)
    to +1 (at the high). Weighted by volume, it approximates which side
    was doing the aggressive transacting. It is a PROXY - real footprint
    data separates bid-side from ask-side volume directly. We do not
    have that from Yahoo."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    clv = clv.fillna(0.0)
    mf = clv * df["Volume"]
    imbalance = (mf.rolling(window).sum() /
                 df["Volume"].rolling(window).sum().replace(0, np.nan))
    vmean = df["Volume"].rolling(vol_window).mean()
    vstd = df["Volume"].rolling(vol_window).std()
    vol_z = (df["Volume"] - vmean) / vstd.replace(0, np.nan)
    return imbalance, vol_z


def forward_returns(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({h: close.shift(-h) / close - 1.0
                         for h in HORIZONS_BARS}, index=close.index)


# ------------------------------------------------------- per-ticker pass
def collect_ticker(ticker, days, vol_window, imb_thr, volz_thr):
    """Returns a tidy frame of dominance events for one ticker:
    side (+1 buyers / -1 sellers), trend regime, raw forward returns."""
    intr = load_intraday(ticker, days)
    if intr.empty or len(intr) < 300:
        print(f"  {ticker}: no usable data, skipped")
        return None, None

    daily = load_daily(ticker)
    trend = daily_trend_filter(daily)
    dates = pd.Series(intr.index.date, index=intr.index)
    trend_up = dates.map(trend).fillna(False)

    imbalance, vol_z = compute_imbalance(intr, vol_window=vol_window)
    fwd = forward_returns(intr["Close"])

    heavy = (vol_z > volz_thr)
    buyers = heavy & (imbalance > imb_thr)
    sellers = heavy & (imbalance < -imb_thr)

    frames = []
    for label, mask, side in [("buyers", buyers, 1), ("sellers", sellers, -1)]:
        m = mask.fillna(False)
        if m.sum() == 0:
            continue
        rows = fwd.loc[m].copy()
        rows.columns = [f"h{h}" for h in HORIZONS_BARS]
        rows["ticker"] = ticker
        rows["side"] = side
        rows["side_label"] = label
        rows["uptrend"] = trend_up[m].values
        rows["date"] = [ts.date() for ts in rows.index]
        frames.append(rows)

    n_b, n_s = int(buyers.fillna(False).sum()), int(sellers.fillna(False).sum())
    print(f"  {ticker}: {n_b} buyer-dominant, {n_s} seller-dominant bars")

    universe = fwd.copy()
    universe.columns = [f"h{h}" for h in HORIZONS_BARS]
    return (pd.concat(frames) if frames else None), universe.dropna()


# -------------------------------------------------------------- stats
def raw_stats(seg: pd.DataFrame) -> pd.DataFrame:
    """RAW forward returns - no sign flipping. Positive mean = price
    went UP after these events, whatever side triggered them."""
    out = []
    for h in HORIZONS_BARS:
        r = seg[f"h{h}"].dropna()
        if len(r) == 0:
            out.append({"horizon_min": h * BAR_MINUTES, "n": 0,
                        "mean_ret": np.nan, "pct_up": np.nan, "t_stat": np.nan})
            continue
        sd = r.std(ddof=1)
        t = r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else 0.0
        out.append({"horizon_min": h * BAR_MINUTES, "n": len(r),
                    "mean_ret": r.mean(), "pct_up": (r > 0).mean(), "t_stat": t})
    return pd.DataFrame(out)


def max_abs_t(seg: pd.DataFrame) -> float:
    v = raw_stats(seg)["t_stat"].to_numpy(dtype=float)
    return 0.0 if np.all(np.isnan(v)) else float(np.nanmax(np.abs(v)))


def permutation_test(seg, universe, n_perm=500, seed=7):
    rng = np.random.default_rng(seed)
    obs = max_abs_t(seg)
    k = len(seg)
    if k >= len(universe):
        return np.nan, obs
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(len(universe), size=k, replace=False)
        null[i] = max_abs_t(universe.iloc[idx])
    return (np.sum(null >= obs) + 1) / (n_perm + 1), obs


def verdict(seg, side):
    """Translate raw returns into plain language: after this side
    dominated, did price continue their way or go against them?"""
    s = raw_stats(seg)
    valid = s.dropna(subset=["mean_ret"])
    if valid.empty:
        return "no data"
    # weight by the longest horizon with a real sample
    mean_dir = np.sign(valid["mean_ret"].mean())
    if mean_dir == 0:
        return "flat - no directional tendency"
    if side == 1:
        return ("CONTINUATION - price kept rising after buyers dominated"
                if mean_dir > 0 else
                "REVERSAL - price fell after buyers dominated (buyers trapped)")
    return ("CONTINUATION - price kept falling after sellers dominated"
            if mean_dir < 0 else
            "REVERSAL - price rose after sellers dominated (sellers absorbed)")


def report_block(seg, universe, title, n_perm):
    print(f"\n--- {title} (n={len(seg)}) ---")
    if len(seg) < MIN_EVENTS:
        print(f"only {len(seg)} events - under {MIN_EVENTS}, not tested")
        return
    print(raw_stats(seg).to_string(index=False, formatters={
        "mean_ret": "{:+.3%}".format, "pct_up": "{:.1%}".format,
        "t_stat": "{:.2f}".format}))
    p, obs = permutation_test(seg, universe, n_perm)
    if not np.isnan(p):
        sig = "[SIGNIFICANT at 0.05]" if p < 0.05 else "[not significant]"
        print(f"permutation: max|t|={obs:.2f}  p={p:.4f}  {sig}")
    print(f"reading: {verdict(seg, seg['side'].iloc[0])}")


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,QQQ,IWM,SMH,DIA,XLK,XLF,XLE")
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--vol-window", type=int, default=78)
    p.add_argument("--imb", type=float, default=0.4,
                   help="how one-sided a bar must be to count as dominance")
    p.add_argument("--volz", type=float, default=1.0,
                   help="how elevated volume must be")
    p.add_argument("--split", type=float, default=0.6)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--by-trend", action="store_true",
                   help="also break results down by daily trend regime")
    a = p.parse_args()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    print(f"Pooling {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"dominance = |imbalance|>{a.imb} AND volz>{a.volz}\n")

    frames, universes = [], []
    for tk in tickers:
        try:
            rows, uni = collect_ticker(tk, a.days, a.vol_window, a.imb, a.volz)
        except Exception as e:
            print(f"  {tk}: failed ({type(e).__name__}: {e})")
            continue
        if rows is not None:
            frames.append(rows)
        if uni is not None:
            universes.append(uni)

    if not frames:
        print("\nNo dominance events fired. Loosen --imb or --volz.")
        return 0

    pooled = pd.concat(frames).sort_values("date")
    universe = pd.concat(universes, ignore_index=True)

    dates = sorted(pooled["date"].unique())
    cut = dates[int(len(dates) * a.split)]
    print(f"\npooled events: {len(pooled)}   split date: {cut}")
    print(pooled.groupby("side_label").size().to_string())

    for blk_label, blk in [("IN-SAMPLE (design)", pooled[pooled.date < cut]),
                           ("OUT-OF-SAMPLE (verdict)", pooled[pooled.date >= cut])]:
        print(f"\n{'='*58}")
        print(f"=== {blk_label} ===")
        for side, lbl in [(1, "AFTER BUYERS DOMINATED"),
                          (-1, "AFTER SELLERS DOMINATED")]:
            seg = blk[blk.side == side]
            report_block(seg, universe, lbl, a.n_perm)

            if a.by_trend and len(seg) >= MIN_EVENTS:
                for up, tl in [(True, "in daily UPTREND"), (False, "in daily DOWNTREND")]:
                    sub = seg[seg.uptrend == up]
                    if len(sub) >= MIN_EVENTS:
                        report_block(sub, universe, f"{lbl} - {tl}", a.n_perm)

    print(f"\n{'='*58}")
    print("How to read this: mean_ret is the RAW forward return, not "
          "flipped by side. Positive after seller dominance means price "
          "ROSE - sellers were absorbed. Negative after seller dominance "
          "means price kept FALLING - sellers were right.")
    print("\nThe out-of-sample block is the only one that counts. A side "
          "'having an edge' requires: significant p-value, consistent "
          "sign across horizons, AND the same story in-sample and out. "
          "Two out of three is not an edge.")
    print("\nCLV/volume is a PROXY for order flow, not real bid-ask "
          "footprint data. No costs applied. Not trading or financial "
          "advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
