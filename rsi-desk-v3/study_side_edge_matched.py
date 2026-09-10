#!/usr/bin/env python3
"""
Which side has the edge - HOURLY, with a REGIME-MATCHED null.

WHY THIS VERSION EXISTS
-----------------------
study_side_edge_hourly.py found seller-dominance-in-an-uptrend looking
like a real edge: significant in BOTH blocks (p=0.008 in, p=0.014 out),
5/5 horizons agreeing in both, effect growing with horizon, and stronger
out-of-sample than in. Buyers showed nothing anywhere. Clean asymmetry.

But that test had a flaw that could manufacture exactly that result.

The permutation null drew its random comparison bars from the ENTIRE
universe - uptrend and downtrend bars pooled together. The segment being
tested was uptrend-only. In an uptrend, the average bar has a higher
forward return simply because the market is going up. So an uptrend-only
sample gets scored against a mixed-regime baseline and looks good even
with zero real edge. The test was measuring market drift, at least in
part, and calling it signal.

THE FIX
-------
Regime-matched null. When testing uptrend seller events, draw the random
comparison bars ONLY from uptrend bars of the same tickers. Same for
downtrend. Now the question becomes the right one:

  "Given that the market was rising anyway, did seller-dominant bars
   do better than a randomly chosen moment in that same rising market?"

If the effect survives, it is real edge on top of drift.
If it collapses toward p~0.5, it was drift all along.

Also added: a DRIFT BASELINE line printed for every block, showing the
mean forward return of ALL bars in that regime. Read the event returns
against that number, not against zero. An event mean of +0.30% means
nothing if every bar in that regime averaged +0.28%.

Everything else is unchanged from the previous version.

Usage:
    python study_side_edge_matched.py --by-trend
    python study_side_edge_matched.py --by-trend --tickers SPY,QQQ,IWM
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HORIZONS_BARS = [1, 2, 4, 8, 16]   # hourly bars
MIN_EVENTS = 30


# ----------------------------------------------------------------- data
def load_hourly(ticker: str, period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1h",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])


def load_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="5y", interval="1d",
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
def compute_imbalance(df: pd.DataFrame, window: int = 1, vol_window: int = 168):
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
def collect_ticker(ticker, period, vol_window, imb_thr, volz_thr):
    """Returns (events, universe). The universe now carries an 'uptrend'
    column so the null can be regime-matched."""
    intr = load_hourly(ticker, period)
    if intr.empty or len(intr) < 500:
        print(f"  {ticker}: no usable hourly data, skipped")
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

    n_b = int(buyers.fillna(False).sum())
    n_s = int(sellers.fillna(False).sum())
    print(f"  {ticker}: {len(intr)} bars, {n_b} buyer / {n_s} seller events, "
          f"{trend_up.mean():.0%} of bars in daily uptrend")

    uni = fwd.copy()
    uni.columns = [f"h{h}" for h in HORIZONS_BARS]
    uni["uptrend"] = trend_up.values
    return (pd.concat(frames) if frames else None), uni.dropna()


# -------------------------------------------------------------- stats
def raw_stats(seg: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h in HORIZONS_BARS:
        r = seg[f"h{h}"].dropna()
        if len(r) == 0:
            out.append({"horizon_hours": h, "n": 0, "mean_ret": np.nan,
                        "pct_up": np.nan, "t_stat": np.nan})
            continue
        sd = r.std(ddof=1)
        t = r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else 0.0
        out.append({"horizon_hours": h, "n": len(r), "mean_ret": r.mean(),
                    "pct_up": (r > 0).mean(), "t_stat": t})
    return pd.DataFrame(out)


def excess_stats(seg: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Event mean MINUS the drift baseline for the same regime. This is
    the number that actually matters - how much better than just being
    in the market at a random moment."""
    s = raw_stats(seg)
    rows = []
    for _, row in s.iterrows():
        h = int(row["horizon_hours"])
        b = baseline[f"h{h}"].dropna()
        drift = b.mean() if len(b) else np.nan
        rows.append({"horizon_hours": h,
                     "event_mean": row["mean_ret"],
                     "drift_mean": drift,
                     "excess": row["mean_ret"] - drift})
    return pd.DataFrame(rows)


def max_abs_t(seg: pd.DataFrame) -> float:
    v = raw_stats(seg)["t_stat"].to_numpy(dtype=float)
    return 0.0 if np.all(np.isnan(v)) else float(np.nanmax(np.abs(v)))


def permutation_matched(seg, pool, n_perm=500, seed=7):
    """THE FIX: `pool` must already be filtered to the same regime as
    `seg`. Draws random bars from that matched pool only."""
    rng = np.random.default_rng(seed)
    obs = max_abs_t(seg)
    k = len(seg)
    if k >= len(pool) or len(pool) == 0:
        return np.nan, obs
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(len(pool), size=k, replace=False)
        null[i] = max_abs_t(pool.iloc[idx])
    return (np.sum(null >= obs) + 1) / (n_perm + 1), obs


def consistency(seg) -> str:
    s = raw_stats(seg).dropna(subset=["mean_ret"])
    if s.empty:
        return "n/a"
    signs = np.sign(s["mean_ret"].to_numpy())
    return f"{max(int((signs>0).sum()), int((signs<0).sum()))}/{len(signs)} agree"


def excess_consistency(ex: pd.DataFrame) -> str:
    v = ex["excess"].dropna().to_numpy()
    if len(v) == 0:
        return "n/a"
    signs = np.sign(v)
    return f"{max(int((signs>0).sum()), int((signs<0).sum()))}/{len(v)} agree"


def report_block(seg, pool, title, n_perm):
    print(f"\n--- {title} (n={len(seg)}) ---")
    if len(seg) < MIN_EVENTS:
        print(f"only {len(seg)} events - under {MIN_EVENTS}, not tested")
        return
    if len(pool) == 0:
        print("no regime-matched bars available for the null, skipped")
        return

    print(raw_stats(seg).to_string(index=False, formatters={
        "mean_ret": "{:+.3%}".format, "pct_up": "{:.1%}".format,
        "t_stat": "{:.2f}".format}))

    ex = excess_stats(seg, pool)
    print("\nvs regime drift baseline:")
    print(ex.to_string(index=False, formatters={
        "event_mean": "{:+.3%}".format, "drift_mean": "{:+.3%}".format,
        "excess": "{:+.3%}".format}))

    p, obs = permutation_matched(seg, pool, n_perm)
    if not np.isnan(p):
        sig = "[SIGNIFICANT at 0.05]" if p < 0.05 else "[not significant]"
        print(f"\nregime-matched permutation: max|t|={obs:.2f}  "
              f"p={p:.4f}  {sig}   (null drawn from {len(pool)} same-regime bars)")
    print(f"raw consistency: {consistency(seg)}   "
          f"excess consistency: {excess_consistency(ex)}")


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,QQQ,IWM,SMH,DIA,XLK,XLF,XLE")
    p.add_argument("--period", default="2y")
    p.add_argument("--vol-window", type=int, default=168)
    p.add_argument("--imb", type=float, default=0.4)
    p.add_argument("--volz", type=float, default=1.0)
    p.add_argument("--split", type=float, default=0.6)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--by-trend", action="store_true")
    a = p.parse_args()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    print(f"HOURLY bars, period={a.period}, REGIME-MATCHED null")
    print(f"Pooling {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"dominance = |imbalance|>{a.imb} AND volz>{a.volz}\n")

    frames, universes = [], []
    for tk in tickers:
        try:
            rows, uni = collect_ticker(tk, a.period, a.vol_window, a.imb, a.volz)
        except Exception as e:
            print(f"  {tk}: failed ({type(e).__name__}: {e})")
            continue
        if rows is not None:
            frames.append(rows)
        if uni is not None:
            universes.append(uni)

    if not frames:
        print("\nNo dominance events fired.")
        return 0

    pooled = pd.concat(frames).sort_values("date")
    universe = pd.concat(universes, ignore_index=True)
    up_pool = universe[universe.uptrend == True]
    dn_pool = universe[universe.uptrend == False]

    dates = sorted(pooled["date"].unique())
    cut = dates[int(len(dates) * a.split)]
    n_up = int(pooled["uptrend"].sum())
    print(f"\npooled events: {len(pooled)}")
    print(f"date range: {dates[0]} -> {dates[-1]}   split at: {cut}")
    print(f"regime mix: {n_up} events in uptrend, {len(pooled)-n_up} in downtrend")
    print(f"null pools: {len(up_pool)} uptrend bars, {len(dn_pool)} downtrend bars")
    print(pooled.groupby("side_label").size().to_string())

    for blk_label, blk in [("IN-SAMPLE (design)", pooled[pooled.date < cut]),
                           ("OUT-OF-SAMPLE (verdict)", pooled[pooled.date >= cut])]:
        print(f"\n{'='*62}")
        print(f"=== {blk_label} ===")
        for side, lbl in [(1, "AFTER BUYERS DOMINATED"),
                          (-1, "AFTER SELLERS DOMINATED")]:
            seg = blk[blk.side == side]
            # whole-segment test uses the full universe (mixed regime is
            # correct here, since the segment itself is mixed)
            report_block(seg, universe, lbl, a.n_perm)

            if a.by_trend and len(seg) >= MIN_EVENTS:
                for up, tl, pool in [(True, "daily UPTREND", up_pool),
                                     (False, "daily DOWNTREND", dn_pool)]:
                    sub = seg[seg.uptrend == up]
                    if len(sub) >= MIN_EVENTS:
                        report_block(sub, pool, f"{lbl} - {tl}", a.n_perm)
                    else:
                        print(f"\n--- {lbl} - {tl} (n={len(sub)}) --- too few")

    print(f"\n{'='*62}")
    print("WHAT CHANGED: the null for each regime-split block is now drawn "
          "only from bars in that SAME regime. The previous version "
          "compared uptrend-only events against a mixed-regime baseline, "
          "which flatters any uptrend sample because the market was rising "
          "anyway.")
    print("\nRead the 'excess' column, not the raw mean. Excess is how much "
          "better the event did than a random moment in the same regime. "
          "If excess is near zero, the effect was drift.")
    print("\nAn edge still needs all three: significant p under the matched "
          "null, consistent EXCESS sign across horizons, and the same story "
          "in-sample and out.")
    print("\nCLV/volume is a PROXY for order flow, not real bid-ask "
          "footprint. No costs or slippage applied - at these effect sizes "
          "costs matter enormously. Not trading or financial advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
