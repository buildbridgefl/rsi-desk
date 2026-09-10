#!/usr/bin/env python3
"""
Which side has the edge - HOURLY bars, 2 years of history.

Why this exists: study_side_edge.py found that on 5-minute bars, seller
dominance looked like a real reversal edge IN-SAMPLE (p=0.006, all five
horizons agreeing, effect growing with time) and then failed to replicate
OUT-OF-SAMPLE (p=0.052, horizons disagreeing, one negative, effect 20x
smaller). The out-of-sample window was ~1 month. That is not enough time
to distinguish "the edge decayed" from "the edge was never there."

Yahoo caps 5-minute data at 59 days. Hourly data goes back ~2 years.
Same question, coarser resolution, roughly 8x the history - and crucially
enough calendar time to include DOWN-trending days, which the 5m study
had exactly zero of (every event happened above the 200MA, which is why
the --by-trend split returned identical numbers).

Same rules as the 5m version:
  DOMINANCE = |imbalance| > threshold AND volume z-score > threshold
  Measure RAW forward returns after buyer-dominant and seller-dominant
  bars separately. No sign flipping - the data says which way to trade.

What changed for hourly:
  - horizons are in HOURS (1, 2, 4, 8, 16 bars ~ 1h to ~2.5 sessions)
  - vol_window defaults to 168 bars (~1 month of trading hours) instead
    of 78 (~1 day of 5m bars)
  - split is by date across 2 years, so out-of-sample is ~9 months

Usage:
    python study_side_edge_hourly.py
    python study_side_edge_hourly.py --by-trend
    python study_side_edge_hourly.py --tickers SPY,QQQ,IWM --imb 0.5
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HORIZONS_BARS = [1, 2, 4, 8, 16]   # hourly bars
BAR_LABEL = "hours"
MIN_EVENTS = 30        # higher bar than the 5m study - we have the data now


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
    """CLV-weighted volume imbalance - same proxy as the rest of the desk.
    Still a PROXY: real footprint data separates bid-side from ask-side
    volume directly. Yahoo does not provide that."""
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
    n_up = int(trend_up.sum())
    print(f"  {ticker}: {len(intr)} bars, {n_b} buyer / {n_s} seller events, "
          f"{n_up/len(intr):.0%} of bars in daily uptrend")

    universe = fwd.copy()
    universe.columns = [f"h{h}" for h in HORIZONS_BARS]
    return (pd.concat(frames) if frames else None), universe.dropna()


# -------------------------------------------------------------- stats
def raw_stats(seg: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h in HORIZONS_BARS:
        r = seg[f"h{h}"].dropna()
        if len(r) == 0:
            out.append({f"horizon_{BAR_LABEL}": h, "n": 0, "mean_ret": np.nan,
                        "pct_up": np.nan, "t_stat": np.nan})
            continue
        sd = r.std(ddof=1)
        t = r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else 0.0
        out.append({f"horizon_{BAR_LABEL}": h, "n": len(r), "mean_ret": r.mean(),
                    "pct_up": (r > 0).mean(), "t_stat": t})
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


def consistency(seg) -> str:
    """A real effect should point the same way across horizons. A single
    spiking horizon inside an otherwise flat row is what noise looks like
    when you scan five of them - that is exactly how the 5m result failed."""
    s = raw_stats(seg).dropna(subset=["mean_ret"])
    if s.empty:
        return "n/a"
    signs = np.sign(s["mean_ret"].to_numpy())
    pos, neg = int((signs > 0).sum()), int((signs < 0).sum())
    agree = max(pos, neg)
    return f"{agree}/{len(signs)} horizons agree in sign"


def verdict(seg, side):
    s = raw_stats(seg).dropna(subset=["mean_ret"])
    if s.empty:
        return "no data"
    d = np.sign(s["mean_ret"].mean())
    if d == 0:
        return "flat"
    if side == 1:
        return ("CONTINUATION - price kept rising after buyers dominated"
                if d > 0 else
                "REVERSAL - price fell after buyers dominated (buyers trapped)")
    return ("CONTINUATION - price kept falling after sellers dominated"
            if d < 0 else
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
    print(f"consistency: {consistency(seg)}")
    print(f"reading: {verdict(seg, seg['side'].iloc[0])}")


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,QQQ,IWM,SMH,DIA,XLK,XLF,XLE")
    p.add_argument("--period", default="2y", help="1y, 2y (yahoo hourly cap)")
    p.add_argument("--vol-window", type=int, default=168)
    p.add_argument("--imb", type=float, default=0.4)
    p.add_argument("--volz", type=float, default=1.0)
    p.add_argument("--split", type=float, default=0.6)
    p.add_argument("--n-perm", type=int, default=500)
    p.add_argument("--by-trend", action="store_true")
    a = p.parse_args()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    print(f"HOURLY bars, period={a.period}")
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
        print("\nNo dominance events fired. Loosen --imb or --volz.")
        return 0

    pooled = pd.concat(frames).sort_values("date")
    universe = pd.concat(universes, ignore_index=True)

    dates = sorted(pooled["date"].unique())
    cut = dates[int(len(dates) * a.split)]
    n_up = int(pooled["uptrend"].sum())
    print(f"\npooled events: {len(pooled)}")
    print(f"date range: {dates[0]} -> {dates[-1]}   split at: {cut}")
    print(f"regime mix: {n_up} events in uptrend, "
          f"{len(pooled)-n_up} in downtrend")
    print(pooled.groupby("side_label").size().to_string())

    for blk_label, blk in [("IN-SAMPLE (design)", pooled[pooled.date < cut]),
                           ("OUT-OF-SAMPLE (verdict)", pooled[pooled.date >= cut])]:
        print(f"\n{'='*60}")
        print(f"=== {blk_label} ===")
        for side, lbl in [(1, "AFTER BUYERS DOMINATED"),
                          (-1, "AFTER SELLERS DOMINATED")]:
            seg = blk[blk.side == side]
            report_block(seg, universe, lbl, a.n_perm)

            if a.by_trend and len(seg) >= MIN_EVENTS:
                for up, tl in [(True, "daily UPTREND"), (False, "daily DOWNTREND")]:
                    sub = seg[seg.uptrend == up]
                    if len(sub) >= MIN_EVENTS:
                        report_block(sub, universe, f"{lbl} - {tl}", a.n_perm)
                    else:
                        print(f"\n--- {lbl} - {tl} (n={len(sub)}) --- too few")

    print(f"\n{'='*60}")
    print("The 5-minute version of this study found seller dominance "
          "significant in-sample (p=0.006, 5/5 horizons agreeing) and "
          "NOT significant out-of-sample (p=0.052, horizons disagreeing). "
          "This run has ~8x the history and real downtrend days. If the "
          "effect is real it should survive here. If it does not, it was "
          "noise both times.")
    print("\nAn edge needs all three: significant p, consistent sign across "
          "horizons, and the same story in-sample and out. Two out of "
          "three is not an edge.")
    print("\nCLV/volume is a PROXY for order flow, not real bid-ask "
          "footprint. No costs applied. Hourly bars mean wider stops and "
          "fewer trades than the intraday version. Not trading or "
          "financial advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
