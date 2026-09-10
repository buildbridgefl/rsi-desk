#!/usr/bin/env python3
"""
Fib-retracement absorption/reversal study - POOLED across tickers.

Same decoded rules as study_fib_absorption.py. The only change is WHY:
on any single ticker this setup fires roughly once or twice a week, and
Yahoo caps 5-minute history at 59 days. That is not enough trades to
tell signal from noise, no matter how the thresholds are tuned.

So instead of loosening filters until trades appear (which is fitting
the test to the data), this pools the SAME strict setup across several
tickers and tests the combined sample. Each ticker is processed
independently - trend, fib zone, absorption, forward returns are all
computed per symbol - and only the resulting entry/return pairs are
concatenated. No cross-ticker contamination.

The in/out-of-sample split is by CALENDAR DATE across all tickers, so
the out-of-sample block is genuinely later in time for every symbol.

Usage:
    python study_fib_pooled.py
    python study_fib_pooled.py --tickers SPY,QQQ,IWM,SMH,DIA
    python study_fib_pooled.py --tickers SPY,QQQ --imb 0.3 --volz 0.8
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HORIZONS_BARS = [1, 3, 6, 12, 24]   # 5m bars -> 5, 15, 30, 60, 120 min ahead
BAR_MINUTES = 5
MIN_SETUPS = 15


# ----------------------------------------------------------------- data
def load_intraday(ticker: str, days: int = 59) -> pd.DataFrame:
    days = min(days, 59)
    df = yf.download(ticker, period=f"{days}d", interval="5m",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    return df


def load_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


# --------------------------------------------------------------- trend
def daily_trend_filter(daily: pd.DataFrame, ma_len: int = 200) -> pd.Series:
    ma = daily["Close"].rolling(ma_len).mean()
    up = (daily["Close"] > ma).fillna(False)
    up.index = up.index.date
    return up


def merge_trend_onto_intraday(intraday: pd.DataFrame, trend_by_date: pd.Series) -> pd.Series:
    dates = pd.Series(intraday.index.date, index=intraday.index)
    return dates.map(trend_by_date).fillna(False)


# ------------------------------------------------------------- location
def fib_zone(df: pd.DataFrame, swing: int = 50):
    swing_hi = df["High"].rolling(swing).max()
    swing_lo = df["Low"].rolling(swing).min()
    rng = swing_hi - swing_lo
    fib_705 = swing_hi - 0.705 * rng
    fib_886 = swing_hi - 0.886 * rng
    return fib_705, fib_886


# ------------------------------------------------------------- signal
def compute_imbalance(df: pd.DataFrame, window: int = 1, vol_window: int = 78):
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    clv = clv.fillna(0.0)
    money_flow = clv * df["Volume"]
    imbalance = (money_flow.rolling(window).sum() /
                 df["Volume"].rolling(window).sum().replace(0, np.nan))
    vol_mean = df["Volume"].rolling(vol_window).mean()
    vol_std = df["Volume"].rolling(vol_window).std()
    vol_z = (df["Volume"] - vol_mean) / vol_std.replace(0, np.nan)
    return imbalance, vol_z


def tag_absorption_reversal(df, trend_up, fib_705, fib_886, imbalance, vol_z,
                            imb_threshold=0.4, vol_threshold=1.0):
    in_zone = (df["Close"] >= fib_886) & (df["Close"] <= fib_705)
    location_ok = trend_up & in_zone
    absorption = (imbalance < -imb_threshold) & (vol_z > vol_threshold) & location_ok
    next_bar_green = (df["Close"] > df["Open"]).shift(-1).fillna(False)
    entry = (absorption & next_bar_green).shift(1).fillna(False)
    stop_level = df["Low"].where(absorption).shift(1)
    stop_level = stop_level.ffill().where(entry)
    return entry, stop_level


def forward_returns(close: pd.Series, horizons: list = HORIZONS_BARS) -> pd.DataFrame:
    out = {h: close.shift(-h) / close - 1.0 for h in horizons}
    return pd.DataFrame(out, index=close.index)


# ------------------------------------------------------- per-ticker pass
def collect_ticker(ticker, days, swing, vol_window, imb, volz, horizon_bars=12):
    """Run the full pipeline on ONE ticker. Returns a tidy DataFrame of
    just the setups that fired: date, ticker, forward returns, R-multiple.
    Returns None if the ticker has no usable data."""
    intraday = load_intraday(ticker, days)
    if intraday.empty or len(intraday) < 300:
        print(f"  {ticker}: no usable intraday data, skipped")
        return None

    daily = load_daily(ticker)
    trend_up = merge_trend_onto_intraday(intraday, daily_trend_filter(daily))
    fib_705, fib_886 = fib_zone(intraday, swing)
    imbalance, vol_z = compute_imbalance(intraday, vol_window=vol_window)

    entry, stop_level = tag_absorption_reversal(
        intraday, trend_up, fib_705, fib_886, imbalance, vol_z, imb, volz)
    fwd = forward_returns(intraday["Close"])

    mask = entry.fillna(False)
    n = int(mask.sum())
    print(f"  {ticker}: {n} setups over {len(intraday)} bars")
    if n == 0:
        return None

    rows = fwd.loc[mask].copy()
    rows.columns = [f"h{h}" for h in HORIZONS_BARS]
    rows["ticker"] = ticker
    rows["date"] = [ts.date() for ts in rows.index]

    # R-multiple off the structural stop (absorption bar's low)
    entry_px = intraday["Close"][mask]
    risk = (entry_px - stop_level[mask]).replace(0, np.nan)
    fwd_px = intraday["Close"].shift(-horizon_bars)[mask]
    rows["r_mult"] = ((fwd_px - entry_px) / risk).values
    return rows


# -------------------------------------------------------------- stats
def signed_stats(pooled: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h in HORIZONS_BARS:
        r = pooled[f"h{h}"].dropna()
        if len(r) == 0:
            out.append({"horizon_min": h * BAR_MINUTES, "n": 0,
                        "mean_ret": np.nan, "hit_rate": np.nan, "t_stat": np.nan})
            continue
        sd = r.std(ddof=1)
        t = r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else 0.0
        out.append({"horizon_min": h * BAR_MINUTES, "n": len(r),
                    "mean_ret": r.mean(), "hit_rate": (r > 0).mean(), "t_stat": t})
    return pd.DataFrame(out)


def max_abs_t(pooled: pd.DataFrame) -> float:
    vals = signed_stats(pooled)["t_stat"].to_numpy(dtype=float)
    return 0.0 if np.all(np.isnan(vals)) else float(np.nanmax(np.abs(vals)))


def permutation_test_pooled(pooled, all_returns, n_perm=500, seed=7):
    """Null: draw the same number of return observations at random from
    the full universe of bars (all tickers, all times), and see how often
    a random draw produces a max |t| as extreme as the real setups did."""
    rng = np.random.default_rng(seed)
    observed = max_abs_t(pooled)
    k = len(pooled)
    null = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.choice(len(all_returns), size=k, replace=False)
        null[i] = max_abs_t(all_returns.iloc[idx])
    p = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return p, observed


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,QQQ,IWM,SMH",
                   help="comma separated, e.g. SPY,QQQ,IWM,SMH,DIA")
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--swing", type=int, default=50)
    p.add_argument("--vol-window", type=int, default=78)
    p.add_argument("--imb", type=float, default=0.4)
    p.add_argument("--volz", type=float, default=1.0)
    p.add_argument("--split", type=float, default=0.6)
    p.add_argument("--n-perm", type=int, default=500)
    a = p.parse_args()

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    print(f"Pooling {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"thresholds: imb<{-a.imb}  volz>{a.volz}  swing={a.swing}\n")

    frames, universes = [], []
    for tk in tickers:
        try:
            rows = collect_ticker(tk, a.days, a.swing, a.vol_window, a.imb, a.volz)
        except Exception as e:
            print(f"  {tk}: failed ({type(e).__name__}: {e})")
            continue
        if rows is not None:
            frames.append(rows)
        # build the null universe from every bar of this ticker
        try:
            intr = load_intraday(tk, a.days)
            if not intr.empty:
                u = forward_returns(intr["Close"])
                u.columns = [f"h{h}" for h in HORIZONS_BARS]
                universes.append(u.dropna())
        except Exception:
            pass

    if not frames:
        print("\nNo setups fired on any ticker. Nothing to test.")
        return 0

    pooled = pd.concat(frames).sort_values("date")
    all_returns = pd.concat(universes, ignore_index=True) if universes else None

    dates = sorted(pooled["date"].unique())
    cut = dates[int(len(dates) * a.split)]
    ins = pooled[pooled["date"] < cut]
    oos = pooled[pooled["date"] >= cut]

    print(f"\npooled setups: {len(pooled)}  "
          f"({len(ins)} in-sample, {len(oos)} out-of-sample)")
    print(f"split date: {cut}")
    print("\nper-ticker contribution:")
    print(pooled.groupby("ticker").size().to_string())

    for label, seg in [("IN-SAMPLE (design)", ins),
                       ("OUT-OF-SAMPLE (verdict)", oos)]:
        print(f"\n=== {label} ===")
        if len(seg) < MIN_SETUPS:
            print(f"only {len(seg)} setups - still under {MIN_SETUPS}, "
                  f"not enough to test. Add tickers rather than loosening "
                  f"thresholds.")
            continue

        s = signed_stats(seg)
        print(s.to_string(index=False, formatters={
            "mean_ret": "{:+.3%}".format, "hit_rate": "{:.1%}".format,
            "t_stat": "{:.2f}".format}))

        if all_returns is not None and len(all_returns) > len(seg):
            p_val, obs = permutation_test_pooled(seg, all_returns, a.n_perm)
            sig = "[SIGNIFICANT at 0.05]" if p_val < 0.05 else "[not significant]"
            print(f"\npermutation vs random bars: max|t|={obs:.2f}  "
                  f"p={p_val:.4f}  {sig}")

        r = seg["r_mult"].dropna()
        if len(r):
            print(f"\nR-multiple (structural stop, 60min hold): n={len(r)}  "
                  f"avg R={r.mean():+.2f}  win rate={(r > 0).mean():.1%}")
            print("(video claims ~60-65% win rate / ~1.5-2R average; an "
                  "independent backtest of that video scored 32%)")

    print("\nPooled across tickers to reach a testable sample - the setup "
          "rules are unchanged and were applied independently per symbol. "
          "Equity forward-return proxy, long side only, no costs applied. "
          "Not trading or financial advice.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
