#!/usr/bin/env python3
"""
Absorption alerts -> your phone (ntfy, optional Telegram).

Same rule as SPY Levels (time-of-day volume version, adopted 2026-10-26):
  - hourly imbalance (CLV) below -0.40
  - volume z above 1.0, judged against the SAME clock hour on the last 20 sessions
  - prior-day close above its 200-day average
  - only finished bars count

Also sends a WATCH alert when sellers press (imbalance below -0.40) within
0.3% of one of this week's levels in levels.json but volume hasn't confirmed.

    NTFY_TOPIC=your-topic python -m alerts.absorption_alerts
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data" / "absorption_alert_state.json"
LEVELS = ROOT / "levels.json"
TICKERS = ["SPY", "QQQ"]
IMB_THR, VOLZ_THR, NEAR = -0.40, 1.0, 0.003
MAX_AGE = pd.Timedelta(hours=2)   # don't alert on stale bars (weekends, late runs)


def now_et():
    return pd.Timestamp.now(tz="America/New_York").tz_localize(None)


def load(ticker, interval, period):
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df[~df.index.duplicated(keep="last")].sort_index()


def evaluate(h, d):
    """Add imb / z / gate / end columns to hourly bars h, using daily bars d."""
    h = h.copy()
    rng = (h["High"] - h["Low"]).replace(0, np.nan)
    h["imb"] = (((h["Close"] - h["Low"]) - (h["High"] - h["Close"])) / rng).fillna(0.0)
    # time-of-day volume z: previous 20 sessions at this clock slot (needs 10), sample std
    h["z"] = np.nan
    slot = h.index.strftime("%H:%M")
    for _, idx in h.groupby(slot).groups.items():
        v = h.loc[idx, "Volume"].astype(float)
        prev = v.shift(1)
        m = prev.rolling(20, min_periods=10).mean()
        sd = prev.rolling(20, min_periods=10).std().replace(0, np.nan)
        h.loc[idx, "z"] = ((v - m) / sd).to_numpy()
    # prior-day close vs its 200-day average
    dc = d["Close"].copy()
    dc.index = pd.to_datetime(dc.index).normalize()
    ma = dc.rolling(200).mean()
    above = (dc > ma).where(ma.notna())
    gates = []
    for day in h.index.normalize():
        k = above.index.searchsorted(day) - 1
        gates.append(bool(above.iloc[k]) if k >= 0 and pd.notna(above.iloc[k]) else None)
    h["gate"] = gates
    h["end"] = [min(t + pd.Timedelta(hours=1), t.normalize() + pd.Timedelta(hours=16)) for t in h.index]
    return h


def this_week_levels(ticker, today):
    try:
        lv = json.loads(LEVELS.read_text())
    except Exception:
        return [], "no levels.json"
    monday = (today - pd.Timedelta(days=today.weekday())).strftime("%Y-%m-%d")
    if today.weekday() >= 5:
        monday = (today + pd.Timedelta(days=7 - today.weekday())).strftime("%Y-%m-%d")
    if lv.get("week") != monday:
        return [], f"levels.json is for {lv.get('week')}, not {monday}"
    return [float(x) for x in lv.get(ticker, [])], ""


def notify(title, body):
    sent = False
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        try:
            requests.post(f"https://ntfy.sh/{topic}", data=body.encode(),
                          headers={"Title": title, "Priority": "high", "Tags": "chart_with_upwards_trend"},
                          timeout=10)
            sent = True
        except requests.RequestException as e:
            print(f"[warn] ntfy: {e}", file=sys.stderr)
    tok, chat = os.environ.get("TG_BOT_TOKEN"), os.environ.get("TG_CHAT_ID")
    if tok and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": f"{title}\n{body}"}, timeout=10)
            sent = True
        except requests.RequestException as e:
            print(f"[warn] telegram: {e}", file=sys.stderr)
    if not sent:
        print(f"[no channel]
