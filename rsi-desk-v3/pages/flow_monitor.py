"""
Flow Absorption Monitor — live scanner for the validated signal.

Drop this in a `pages/` folder next to app.py and Streamlit picks it up
automatically as a second page. No changes to app.py needed.

WHAT IT SHOWS
  For each ticker: whether the daily trend gate is open, the current
  hourly imbalance and volume z-score, how close it is to firing, and
  every fire in the recent window with what price did afterward.

WHAT THE SIGNAL MEANS
  Sellers are pressing hard — high volume, strongly negative imbalance —
  and price is NOT going anywhere. That is absorption: the passive buyer
  underneath is stronger than the aggressive seller on top. In a daily
  uptrend that has predicted higher forward returns out-of-sample.

  It is NOT "sellers giving up." They are trying and failing. That
  distinction matters: you are reading strength on the passive side, not
  exhaustion on the active side.

HONESTY NOTE BUILT INTO THE UI
  The signal is validated. The STRATEGY is not. See FLOW_ABSORPTION.md —
  three attempts to turn this into a tradeable system failed or stalled
  on data. This page is a monitor, not a trade recommendation.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Flow Absorption", page_icon="🌊", layout="wide")

VOL_WINDOW = 168


# ------------------------------------------------------------------ data
@st.cache_data(ttl=300, show_spinner=False)
def get_hourly(ticker, period="60d"):
    df = yf.download(ticker, period=period, interval="1h",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    except (AttributeError, TypeError):
        pass
    return df


@st.cache_data(ttl=900, show_spinner=False)
def get_daily(ticker):
    df = yf.download(ticker, period="2y", interval="1d",
                     progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Close"])


def compute(df, vol_window=VOL_WINDOW):
    """CLV-weighted imbalance and volume z-score. Same math as the
    validated study scripts — do not change without re-validating."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    clv = clv.fillna(0.0)
    imbalance = ((clv * df["Volume"]) / df["Volume"].replace(0, np.nan)).fillna(0.0)
    vmean = df["Volume"].rolling(vol_window).mean()
    vstd = df["Volume"].rolling(vol_window).std()
    vol_z = (df["Volume"] - vmean) / vstd.replace(0, np.nan)
    return imbalance, vol_z


def uptrend_flag(daily, ma_len=200):
    if len(daily) < ma_len:
        return None
    ma = float(daily["Close"].rolling(ma_len).mean().iloc[-1])
    px = float(daily["Close"].iloc[-1])
    return px > ma, px, ma


# ---------------------------------------------------------------- sidebar
st.sidebar.title("Flow Absorption")
tickers = [t.strip().upper() for t in st.sidebar.text_area(
    "Watchlist", "SPY, QQQ, IWM, SMH, DIA, XLK, XLF, XLE").split(",") if t.strip()]

imb_thr = st.sidebar.slider("Imbalance threshold", 0.1, 0.9, 0.4, 0.05,
                            help="How one-sided the bar must be. "
                                 "0.4 is the validated value.")
volz_thr = st.sidebar.slider("Volume z-score threshold", 0.0, 3.0, 1.0, 0.1,
                             help="How elevated volume must be. "
                                  "1.0 is the validated value.")
lookback_days = st.sidebar.slider("History shown (days)", 5, 60, 20)

if imb_thr != 0.4 or volz_thr != 1.0:
    st.sidebar.warning("You are off the validated thresholds (0.4 / 1.0). "
                       "Results below are not the tested configuration.")

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.caption("Yahoo data, 15-min delayed. Hourly bars.")


# ------------------------------------------------------------------- head
st.title("Flow Absorption Monitor")
st.caption("Sellers pressing hard and getting nowhere, inside a daily "
           "uptrend. Absorption — the passive buyer is stronger than the "
           "aggressive seller.")

with st.expander("What this is and is not", expanded=False):
    st.markdown("""
**Validated:** on hourly bars, seller-dominant bars inside a daily
uptrend showed higher forward returns than random bars in the *same*
uptrend. Significant in-sample and out-of-sample (p=0.008 / p=0.006)
under a regime-matched null, excess over drift +0.09% to +0.31% across
all five horizons tested. Buyer-dominant bars showed nothing anywhere.

**Not validated:** that you can make money from it. Three attempts —
long-only equity, hourly bracket test, minute bracket test — failed or
stalled on data. The measured excess is +0.1–0.3%; 0DTE option spread
alone is 2–5%.

**Regime matters completely.** Below the 200-day MA the same pattern
flipped to *continuation*. This monitor only flags setups above it.

See `FLOW_ABSORPTION.md` for the full chain.
""")

st.info("This is a monitor, not a trade recommendation. Nothing here has "
        "cleared execution testing.", icon="ℹ️")


# ------------------------------------------------------------------ scan
rows = []
detail = {}

for tk in tickers:
    try:
        h = get_hourly(tk, f"{max(lookback_days, 45)}d")
        d = get_daily(tk)
        if h.empty or len(h) < VOL_WINDOW + 5:
            rows.append({"ticker": tk, "status": "insufficient data"})
            continue

        up = uptrend_flag(d)
        if up is None:
            rows.append({"ticker": tk, "status": "no 200MA yet"})
            continue
        is_up, px, ma = up

        imb, vz = compute(h)
        h = h.assign(imbalance=imb, vol_z=vz)

        last = h.iloc[-1]
        prev = h.iloc[-2]

        # the last COMPLETED bar is the one to judge; the newest bar may
        # still be forming
        fired = (bool(prev.imbalance < -imb_thr) and
                 bool(prev.vol_z > volz_thr) and is_up)

        # how close is the forming bar
        imb_pct = min(abs(float(last.imbalance)) / imb_thr, 1.5) if imb_thr else 0
        vz_pct = min(float(last.vol_z) / volz_thr, 1.5) if volz_thr else 0

        if not is_up:
            status = "gate closed (below 200MA)"
        elif fired:
            status = "FIRED"
        elif last.imbalance < -imb_thr and last.vol_z > volz_thr:
            status = "forming"
        elif last.imbalance < -imb_thr * 0.7 and last.vol_z > volz_thr * 0.7:
            status = "close"
        else:
            status = "—"

        rows.append({
            "ticker": tk,
            "price": round(float(last.Close), 2),
            "vs 200MA": f"{(px/ma - 1)*100:+.1f}%",
            "imbalance (forming)": round(float(last.imbalance), 2),
            "vol z (forming)": round(float(last.vol_z), 2),
            "status": status,
            # stats of the bar that ACTUALLY triggered "FIRED", if it did -
            # this is prev, not last. Keeping it separate is what fixes the
            # mismatch (e.g. showing a forming bar's low vol_z next to a
            # FIRED banner that was earned by the completed bar before it).
            "fired_imbalance": round(float(prev.imbalance), 2) if fired else None,
            "fired_vol_z": round(float(prev.vol_z), 2) if fired else None,
        })
        detail[tk] = h
    except Exception as e:
        rows.append({"ticker": tk, "status": f"error: {type(e).__name__}"})

df = pd.DataFrame(rows)


def paint(v):
    return {"FIRED": "background-color:#10361f;color:#4ade80;font-weight:bold",
            "forming": "background-color:#1e3a2f;color:#86efac",
            "close": "color:#fbbf24",
            "gate closed (below 200MA)": "color:#6b7280"}.get(v, "")


st.subheader("Current state")
show_cols = [c for c in df.columns if c not in ("fired_imbalance", "fired_vol_z")]
if "status" in df.columns:
    st.dataframe(df[show_cols].style.map(paint, subset=["status"]),
                 use_container_width=True, hide_index=True)
else:
    st.dataframe(df[show_cols], use_container_width=True, hide_index=True)
st.caption("imbalance/vol z above are the CURRENTLY FORMING bar, updating "
           "live. They can look nothing like the numbers that actually "
           "triggered a FIRED banner below — that used the bar BEFORE it, "
           "which has already closed.")

hits = df[df.get("status", pd.Series(dtype=str)) == "FIRED"]
if not hits.empty:
    for _, r in hits.iterrows():
        # use fired_imbalance/fired_vol_z (the bar that actually triggered
        # this), NOT the "(forming)" columns, which are the NEW bar now
        # building and can look completely different.
        st.success(f"**{r.ticker}** — absorption fired on the last "
                   f"completed bar @ {r.price} "
                   f"(imbalance {r['fired_imbalance']}, "
                   f"vol z {r['fired_vol_z']})")
else:
    st.caption("No fires on the most recent completed bar.")

st.caption(f"Updated {datetime.now():%Y-%m-%d %H:%M} · the newest hourly "
           f"bar may still be forming — 'FIRED' refers to the last "
           f"COMPLETED bar.")


# ---------------------------------------------------------------- detail
st.divider()
st.subheader("Recent fires")

pick = st.selectbox("Ticker", [t for t in tickers if t in detail])
if pick:
    h = detail[pick]
    d = get_daily(pick)
    up = uptrend_flag(d)
    is_up = up[0] if up else False

    cutoff = h.index[-1] - pd.Timedelta(days=lookback_days)
    win = h[h.index >= cutoff].copy()
    win["fire"] = ((win.imbalance < -imb_thr) & (win.vol_z > volz_thr)
                   & is_up).fillna(False)

    n_fires = int(win.fire.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Fires in window", n_fires)
    c2.metric("Bars scanned", len(win))
    c3.metric("Daily trend", "UP — gate open" if is_up else "DOWN — gate shut")

    if not is_up:
        st.warning("This ticker is below its 200-day MA. The gate is shut, "
                   "so no setups are flagged. Below the 200MA this pattern "
                   "flipped to continuation in testing — the opposite trade.")

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=win.index, open=win.Open, high=win.High,
                                 low=win.Low, close=win.Close, name=pick))
    pts = win.index[win.fire]
    if len(pts):
        fig.add_trace(go.Scatter(
            x=pts, y=win.loc[pts, "Low"] * 0.997, mode="markers",
            name="absorption", marker=dict(symbol="triangle-up", size=13,
                                           color="#4ade80")))
    fig.update_layout(height=460, template="plotly_dark",
                      xaxis_rangeslider_visible=False, margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)

    if n_fires:
        fires = win[win.fire].copy()
        out = []
        closes = h["Close"]
        for ts in fires.index:
            loc = closes.index.get_loc(ts)
            row = {"time": ts.strftime("%Y-%m-%d %H:%M"),
                   "price": round(float(closes.iloc[loc]), 2),
                   "imbalance": round(float(fires.loc[ts, "imbalance"]), 2),
                   "vol z": round(float(fires.loc[ts, "vol_z"]), 2)}
            for label, k in [("+1h", 1), ("+4h", 4), ("+8h", 8), ("+16h", 16)]:
                if loc + k < len(closes):
                    fwd = float(closes.iloc[loc + k]) / float(closes.iloc[loc]) - 1
                    row[label] = f"{fwd*100:+.2f}%"
                else:
                    row[label] = "—"
            out.append(row)
        st.dataframe(pd.DataFrame(out), use_container_width=True,
                     hide_index=True)
        st.caption("Forward returns are what price did after the fire — "
                   "raw, no costs, no option translation. A small sample "
                   "of recent fires tells you nothing on its own; the "
                   "validation used 440 out-of-sample events.")
    else:
        st.info("No fires in this window.")
