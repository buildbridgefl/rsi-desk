#!/usr/bin/env python3
"""
Backtest the validated flow_absorption signal through the real engine.

The research scripts measured FORWARD RETURNS - what price did after an
event. That is not the same as what a tradeable strategy earns. This runs
the same signal through engine.py, which enforces:
  - next-bar fills (no lookahead)
  - costs on every fill
  - one position at a time (so overlapping signals get dropped - this
    alone can change the picture a lot, since the research counted every
    event independently)

TWO THINGS THIS FIXES ABOUT NAIVE USAGE
---------------------------------------
1. engine.stats() hardcodes TRADING_DAYS=252. On HOURLY bars that
   annualizes by the wrong factor and reports a wildly inflated Sharpe.
   This script computes stats with ~1750 hourly bars/year instead.

2. It sweeps COSTS. The measured edge was an excess of +0.09% (1h) to
   +0.31% (16h) over drift. Round-trip cost at 2bps/side is ~0.04%.
   That is a meaningful fraction of the short-horizon edge, so the only
   honest question is: at what cost level does this stop working?

Usage:
    python run_flow_backtest.py
    python run_flow_backtest.py --tickers SPY,QQQ --max-hold 16
    python run_flow_backtest.py --cost-sweep 0,1,2,5,10
"""

import argparse
import sys

import numpy as np
import pandas as pd
import yfinance as yf

import strategies  # noqa: F401  (registers everything)
from core import engine
from strategies.base import REGISTRY

HOURS_PER_YEAR = 1750   # ~250 sessions x 7 hourly bars


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


def attach_daily_trend(hourly, daily, ma_len=200):
    """The tested definition: DAILY close above its 200-day MA, mapped
    onto each hourly bar by date."""
    ma = daily["Close"].rolling(ma_len).mean()
    up = (daily["Close"] > ma).fillna(False)
    up.index = up.index.date
    dates = pd.Series(hourly.index.date, index=hourly.index)
    hourly = hourly.copy()
    hourly["uptrend"] = dates.map(up).fillna(False).values
    return hourly


# -------------------------------------------------------------- stats
def hourly_stats(eq: pd.Series) -> dict:
    """Same shape as engine.stats but annualized for HOURLY bars."""
    ret = eq.pct_change().dropna()
    yrs = max(len(eq) / HOURS_PER_YEAR, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(HOURS_PER_YEAR)
    return {
        "total_ret": eq.iloc[-1] / eq.iloc[0] - 1,
        "cagr": cagr,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "vol": vol,
        "sharpe": (ret.mean() * HOURS_PER_YEAR) / vol if vol > 0 else 0.0,
    }


def run_one(sig, max_hold, stop_pct, cost_bps):
    eq, bh, tr = engine.run(sig, max_hold=max_hold, stop_pct=stop_pct,
                            cost_bps=cost_bps)
    s, b = hourly_stats(eq), hourly_stats(bh)
    ts = engine.trade_stats(tr)
    return s, b, ts, tr


def fmt_row(label, s, b, ts):
    return (f"{label:<14}{s['total_ret']:>9.1%}{s['cagr']:>9.1%}"
            f"{s['max_dd']:>9.1%}{s['sharpe']:>8.2f}"
            f"{ts['trades']:>8}{ts['win_rate']:>8.0%}"
            f"{s['cagr']-b['cagr']:>+9.1%}")


# ------------------------------------------------------------------ cli
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY,QQQ,IWM,SMH,DIA,XLK,XLF,XLE")
    p.add_argument("--period", default="2y")
    p.add_argument("--strategy", default="flow_absorption")
    p.add_argument("--max-hold", type=int, default=8,
                   help="bars (hours) to hold. Research showed the edge "
                        "grows out to ~16.")
    p.add_argument("--stop-pct", type=float, default=0.0)
    p.add_argument("--split", type=float, default=0.6)
    p.add_argument("--cost-sweep", default="0,1,2,5,10",
                   help="bps per side to test")
    p.add_argument("--imb", type=float, default=0.4)
    p.add_argument("--volz", type=float, default=1.0)
    a = p.parse_args()

    if a.strategy not in REGISTRY:
        print(f"unknown strategy {a.strategy!r}. have: {list(REGISTRY)}")
        return 1
    strat = REGISTRY[a.strategy]
    costs = [float(c) for c in a.cost_sweep.split(",") if c.strip()]
    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]

    print(f"strategy: {a.strategy}   max_hold={a.max_hold} hours   "
          f"stop={a.stop_pct:.1%}")
    print(f"NOTE: stats annualized at {HOURS_PER_YEAR} hourly bars/year, "
          f"not engine.stats' 252 (which is for daily bars).\n")

    per_ticker = {}
    for tk in tickers:
        try:
            h = load_hourly(tk, a.period)
            if h.empty or len(h) < 500:
                print(f"{tk}: insufficient data, skipped")
                continue
            h = attach_daily_trend(h, load_daily(tk))
            sig = strat(h, imb_threshold=a.imb, vol_threshold=a.volz)
        except Exception as e:
            print(f"{tk}: failed ({type(e).__name__}: {e})")
            continue

        n_sig = int(sig["long_entry"].sum())
        cut = int(len(sig) * a.split)
        ins, oos = sig.iloc[:cut], sig.iloc[cut:]
        per_ticker[tk] = (ins, oos, n_sig)
        print(f"{tk}: {len(sig)} bars, {n_sig} raw signals")

    if not per_ticker:
        print("\nno tickers produced data")
        return 0

    for cost in costs:
        print(f"\n{'='*78}")
        print(f"=== COST = {cost} bps/side  (round trip ~{cost*2/100:.3f}%) ===")
        for block_name, idx in [("IN-SAMPLE", 0), ("OUT-OF-SAMPLE", 1)]:
            print(f"\n  {block_name}")
            print(f"  {'ticker':<14}{'total':>9}{'CAGR':>9}{'maxDD':>9}"
                  f"{'Sharpe':>8}{'trades':>8}{'win':>8}{'vs hold':>9}")
            agg_cagr, agg_bh, agg_tr, agg_win = [], [], 0, []
            for tk, (ins, oos, _) in per_ticker.items():
                seg = (ins, oos)[idx]
                if int(seg["long_entry"].sum()) == 0:
                    print(f"  {tk:<14}{'no signals':>9}")
                    continue
                try:
                    s, b, ts, tr = run_one(seg, a.max_hold, a.stop_pct, cost)
                except Exception as e:
                    print(f"  {tk:<14}error: {type(e).__name__}")
                    continue
                print("  " + fmt_row(tk, s, b, ts))
                if ts["trades"] > 0:
                    agg_cagr.append(s["cagr"])
                    agg_bh.append(b["cagr"])
                    agg_tr += ts["trades"]
                    agg_win.append(ts["win_rate"])
            if agg_cagr:
                print(f"  {'-'*74}")
                print(f"  {'AVERAGE':<14}{'':>9}{np.mean(agg_cagr):>9.1%}"
                      f"{'':>9}{'':>8}{agg_tr:>8}{np.mean(agg_win):>8.0%}"
                      f"{np.mean(agg_cagr)-np.mean(agg_bh):>+9.1%}")

    print(f"\n{'='*78}")
    print("HOW TO READ THIS")
    print("  'vs hold' is the number that matters. Positive means the "
          "strategy beat buy-and-hold on CAGR for that ticker/block.")
    print("  Watch how it decays as cost rises. The breakeven cost tells "
          "you how much execution quality this needs to survive.")
    print("  OUT-OF-SAMPLE is the verdict. In-sample looking good is "
          "expected and means nothing on its own.")
    print("\n  Being flat most of the time means low exposure, so a lower "
          "CAGR than buy-and-hold is not automatically a failure - but "
          "check Sharpe and drawdown before concluding that.")
    print("\n  Engine holds ONE position at a time, so overlapping signals "
          "are dropped. Trade count here will be lower than the raw signal "
          "count, and that is realistic.")
    print("\n  Not trading or financial advice. Equity backtest only - "
          "says nothing about how options on these moves would behave.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
