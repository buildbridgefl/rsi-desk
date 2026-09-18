#!/usr/bin/env python3
"""
Option chain snapshot collector.

Records the chain and its max pain so the "does price gravitate to max
pain?" question can eventually be answered with your own data.

    python -m alerts.snapshot_chain --tickers SPY,QQQ --expiries 2

Writes two things under data/:
  chains/<ticker>_<expiry>_<stamp>.csv   full chain (one row per strike)
  maxpain_log.csv                        one row per ticker/expiry/run

Open interest updates overnight, so intraday runs mostly re-record the
same OI with fresher volume and spot. That is fine — the spot column is
what makes the later test possible.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

DATA = os.environ.get("DATA_DIR", "data")
LOG = os.path.join(DATA, "maxpain_log.csv")
CHAINS = os.path.join(DATA, "chains")


def max_pain_from(strikes, call_oi, put_oi):
    k = np.asarray(strikes, dtype=float)
    c = np.asarray(call_oi, dtype=float)
    p = np.asarray(put_oi, dtype=float)
    if len(k) == 0 or (c.sum() + p.sum()) == 0:
        return np.nan, np.nan
    totals = np.array([
        (np.maximum(s - k, 0) * c).sum() + (np.maximum(k - s, 0) * p).sum()
        for s in k
    ]) * 100
    i = int(totals.argmin())
    return float(k[i]), float(totals[i])


def fetch_chain(ticker: str, expiry: str) -> pd.DataFrame:
    ch = yf.Ticker(ticker).option_chain(expiry)
    calls = ch.calls.rename(columns={
        "openInterest": "call_oi", "volume": "call_vol",
        "lastPrice": "call_last", "impliedVolatility": "call_iv"})
    puts = ch.puts.rename(columns={
        "openInterest": "put_oi", "volume": "put_vol",
        "lastPrice": "put_last", "impliedVolatility": "put_iv"})
    cols_c = ["strike", "call_oi", "call_vol", "call_last", "call_iv"]
    cols_p = ["strike", "put_oi", "put_vol", "put_last", "put_iv"]
    df = calls[cols_c].merge(puts[cols_p], on="strike", how="outer")
    return df.fillna(0).sort_values("strike").reset_index(drop=True)


def spot_of(ticker: str) -> float:
    h = yf.Ticker(ticker).history(period="1d", interval="1m")
    return float(h["Close"].iloc[-1]) if len(h) else float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY")
    p.add_argument("--expiries", type=int, default=2,
                   help="how many of the nearest expirations to record")
    p.add_argument("--no-chain-files", action="store_true",
                   help="only append to maxpain_log.csv")
    a = p.parse_args()

    os.makedirs(CHAINS, exist_ok=True)
    now = datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M")
    rows = []

    for tk in [t.strip().upper() for t in a.tickers.split(",") if t.strip()]:
        try:
            exps = list(yf.Ticker(tk).options)[:a.expiries]
            spot = spot_of(tk)
        except Exception as e:
            print(f"{tk}: could not list expirations: {e}", file=sys.stderr)
            continue

        for exp in exps:
            try:
                ch = fetch_chain(tk, exp)
            except Exception as e:
                print(f"{tk} {exp}: chain failed: {e}", file=sys.stderr)
                continue
            if ch.empty:
                continue

            mp, total = max_pain_from(ch.strike, ch.call_oi, ch.put_oi)
            call_oi, put_oi = ch.call_oi.sum(), ch.put_oi.sum()
            rows.append({
                "ts_utc": now.strftime("%Y-%m-%d %H:%M"),
                "ticker": tk, "expiry": exp, "spot": round(spot, 2),
                "max_pain": mp,
                "gap_pct": round((mp / spot - 1) * 100, 3) if spot else np.nan,
                "total_value_at_mp": round(total, 0),
                "call_oi": int(call_oi), "put_oi": int(put_oi),
                "pc_ratio": round(put_oi / call_oi, 3) if call_oi else np.nan,
                "strikes": len(ch),
            })
            print(f"{tk} {exp}: spot {spot:.2f} max pain {mp:.0f} "
                  f"({(mp/spot-1)*100:+.2f}%) pc {put_oi/max(call_oi,1):.2f}")

            if not a.no_chain_files:
                ch.insert(0, "ts_utc", now.strftime("%Y-%m-%d %H:%M"))
                ch.insert(1, "spot", round(spot, 2))
                ch.to_csv(os.path.join(CHAINS, f"{tk}_{exp}_{stamp}.csv"),
                          index=False)

    if not rows:
        print("nothing recorded", file=sys.stderr)
        return 1

    new = pd.DataFrame(rows)
    if os.path.exists(LOG):
        new = pd.concat([pd.read_csv(LOG), new], ignore_index=True)
    os.makedirs(DATA, exist_ok=True)
    new.to_csv(LOG, index=False)
    print(f"{len(rows)} rows -> {LOG} ({len(new)} total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
