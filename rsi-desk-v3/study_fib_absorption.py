#!/usr/bin/env python3
"""
Fib-retracement absorption/reversal study.

Decoded from a YouTube video (IQCapital, "Trading WORLD CHAMPION Reveals
the Orderflow Strategy...") claiming a 60-65% win rate / ~1.5-2R average
trade. Treat those numbers as marketing, not evidence: an independent
backtest of these exact rules on real market data came back at 32% win
rate, negative in 5 of 9 tested years, -32% max drawdown. The video is
also a funnel for a prop-firm-challenge business, which is a reason to
distrust the self-reported stats, not a reason to distrust the mechanics.

The mechanical idea, stripped of gamma/GEX talk, is:

  1. TREND   - only trade pullbacks in the direction of the higher
               timeframe trend (daily Close > 200SMA = uptrend, longs only)
  2. LOCATION - price must be inside a Fibonacci pullback zone
               (70.5%-88.6% retracement of the recent swing) - his
               "discount" zone in an uptrend
  3. CONFIRM - inside that zone, look for an ABSORPTION bar: one side
               (sellers, in an uptrend pullback) dominates volume and
               imbalance but price fails to make further progress, then
               the very next bar closes back in the trend direction -
               reuses the same CLV/volume-imbalance proxy as
               core/flow_study.py
  4. STOP    - structural: beyond the low of the absorption bar, not a
               flat % stop. engine.py doesn't support per-trade
               structural stops yet - this script reports R-multiples
               using the structural distance so you can sanity-check the
               claimed 1.5-2R average without needing that engine change.

Same discipline as the rest of the desk: permutation test (corrects for
scanning multiple forward horizons at once), in/out-of-sample split by
calendar day, honest reporting.

Usage:
    python study_fib_absorption.py --ticker SPY
    python study_fib_absorption.py --ticker QQQ --days 59 --swing 50
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

HORIZONS_BARS = [1, 3, 6, 12, 24]   # 5m bars -> 5, 15, 30, 60, 120 min ahead
BAR_MINUTES = 5


# ----------------------------------------------------------------- data
def load_intraday(ticker: str, days: int = 59) -> pd.DataFrame:
    days = min(days, 59)
    df = yf.download(ticker, period=f"{days}d", interval="5m",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        raise SystemExit(f"no intraday data for {ticker}")
    return df


def load_daily(ticker: str) -> pd.DataFrame:
    df = yf.download(ticker, period="2y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


# --------------------------------------------------------------- trend
def daily_trend_filter(daily: pd.DataFrame, ma_len: int = 200) -> pd.Series:
    """True on days the close is above its 200SMA (uptrend). Indexed by date."""
    ma = daily["Close"].rolling(ma_len).mean()
    up = (daily["Close"] > ma).fillna(False)
    up.index = up.index.date
    return up


def merge_trend_onto_intraday(intraday: pd.DataFrame, trend_by_date: pd.Series) -> pd.Series:
    dates = pd.Series(intraday.index.date, index=intraday.index)
    return dates.map(trend_by_date).fillna(False)


# ------------------------------------------------------------- location
def fib_zone(df: pd.DataFrame, swing: int = 50):
    """Trailing swing high/low over `swing` bars (no lookahead - only
    past bars including current)."""
    swing_hi = df["High"].rolling(swing).max()
    swing_lo = df["Low"].rolling(swing).min()
    rng = swing_hi - swing_lo
    fib_705 = swing_hi - 0.705 * rng
    fib_886 = swing_hi - 0.886 * rng
    return fib_705, fib_886, swing_hi, swing_lo


# ------------------------------------------------------------- signal
def compute_imbalance(df: pd.DataFrame, window: int = 1, vol_window: int = 78):
    """Same CLV/volume-imbalance proxy as core/flow_study.py.
    window=1 here since we want PER-BAR absorption, read candle by candle."""
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
    """
    Long setup only (short/premium side is the mirror image - follow-up):
      - trend_up on the daily
      - price inside the discount zone (fib_886 <= Close <= fib_705)
      - THIS bar shows seller absorption: imbalance very negative,
        volume elevated
      - NEXT bar closes green -> entry signal fires on that bar
    Returns (entry boolean Series, structural stop level Series).
    """
    in_zone = (df["Close"] >= fib_886) & (df["Close"] <= fib_705)
    location_ok = trend_up & in_zone

    absorption = (imbalance < -imb_threshold) & (vol_z > vol_threshold) & location_ok
    next_bar_green = (df["Close"] > df["Open"]).shift(-1).fillna(False)
    entry = (absorption & next_bar_green).shift(1).fillna(False)
    stop_level = df["Low"].where(absorption).shift(1)
    stop_level = stop_level.ffill().where(entry)
    return entry, stop_level


# --------------------------------------------------------- forward returns
def forward_returns(close: pd.Series, horizons: list = HORIZONS_BARS) -> pd.DataFrame:
    out = {h: close.shift(-h) / close - 1.0 for h in horizons}
    return pd.DataFrame(out, index=close.index)


# -------------------------------------------------------------- stats
def signed_stats(entry: pd.Series, fwd: pd.DataFrame, horizons: list = HORIZONS_BARS):
    mask = entry.fillna(False)
    rows = []
    for h in horizons:
        r = fwd.loc[mask, h].dropna()
        if len(r) == 0:
            rows.append({"horizon_min": h * BAR_MINUTES, "n": 0,
                         "mean_ret": np.nan, "hit_rate": np.nan, "t_stat": np.nan})
            continue
        sd = r.std(ddof=1)
        t = r.mean() / (sd / np.sqrt(len(r))) if sd > 0 else 0.0
        rows.append({"horizon_min": h * BAR_MINUTES, "n": len(r),
                     "mean_ret": r.mean(), "hit_rate": (r > 0).mean(), "t_stat": t})
    return pd.DataFrame(rows)


def max_abs_tstat(entry: pd.Series, fwd: pd.DataFrame, horizons: list = HORIZONS_BARS) -> float:
    s = signed_stats(entry, fwd, horizons)
    vals = s["t_stat"].to_numpy(dtype=float)
    return 0.0 if np.all(np.isnan(vals)) else float(np.nanmax(np.abs(vals)))


def permutation_test(entry: pd.Series, fwd: pd.DataFrame, horizons: list = HORIZONS_BARS,
                     n_perm: int = 500, seed: int = 7):
    rng = np.random.default_rng(seed)
    observed = max_abs_tstat(entry, fwd, horizons)
    n = len(entry)
    null_stats = np.empty(n_perm)
    vals = entry.fillna(False).to_numpy()
    for i in range(n_perm):
        shift = rng.integers(1, max(n - 1, 2))
        shuffled = pd.Series(np.roll(vals, shift), index=entry.index)
        null_stats[i] = max_abs_tstat(shuffled, fwd, horizons)
    p = (np.sum(null_stats >= observed) + 1) / (n_perm + 1)
    return p, observed, null_stats


def r_multiple_check(df, entry, stop_level, horizon_bars=12):
    """Gut-check the video's claimed ~1.5-2R average. Rough only:
    no partials, no trailing, fixed hold."""
    mask = entry.fillna(False)
    entry_px = df["Close"][mask]
    stop = stop_level[mask]
    risk = (entry_px - stop).replace(0, np.nan)
    fwd_px = df["Close"].shift(-horizon_bars)[mask]
    reward = fwd_px - entry_px
    r = (reward / risk).dropna()
    if len(r) == 0:
        return {"n": 0, "avg_r": np.nan, "win_rate": np.nan}
    return {"n": len(r), "avg_r": r.mean(), "win_rate": (r > 0).mean()}


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--days", type=int, default=59)
    p.add_argument("--swing", type=int, default=50, help="bars for trailing swing hi/lo")
    p.add_argument("--vol-window", type=int, default=78)
    p.add_argument("--imb", type=float,
