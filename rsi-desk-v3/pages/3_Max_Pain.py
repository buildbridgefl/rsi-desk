"""
Max Pain — the strike where expiring options are worth the least in total.

Self-contained page (no imports from core/ or ui/). Drop in pages/.

WHAT IT IS
  For every candidate settle price, add up what all calls and all puts
  would be worth if the underlying finished there. The price with the
  smallest total is "max pain": the most premium expires worthless.

WHAT IT IS NOT
  Proof that price gets pulled there. Dealers hedge both ways, and SPY
  is far too big to steer. Pinning effects in the research are weak and
  mostly limited to single stocks on monthly expirations.

  UNVALIDATED. Nothing here has been tested on your own data. The
  snapshot collector (alerts/snapshot_chain.py) records max pain every
  day so this can eventually be tested: does the close land nearer to
  max pain than chance?
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Max Pain", page_icon="🎯", layout="wide")


# ==================================================================== math
def max_pain_curve(chain: pd.DataFrame) -> pd.DataFrame:
    """chain: strike, call_oi, put_oi. Returns total payout at each strike."""
    k = chain["strike"].to_numpy(float)
    c = chain["call_oi"].to_numpy(float)
    p = chain["put_oi"].to_numpy(float)
    rows = []
    for s in k:
        call_val = (np.maximum(s - k, 0) * c).sum() * 100
        put_val = (np.maximum(k - s, 0) * p).sum() * 100
        rows.append({"settle": s, "call_value": call_val,
                     "put_value": put_val, "total": call_val + put_val})
    return pd.DataFrame(rows)


def max_pain(chain: pd.DataFrame) -> float:
    curve = max_pain_curve(chain)
    if curve.empty:
        return float("nan")
    return float(curve.loc[curve["total"].idxmin(), "settle"])


def summarize(chain: pd.DataFrame, spot: float) -> dict:
    mp = max_pain(chain)
    call_oi, put_oi = chain["call_oi"].sum(), chain["put_oi"].sum()
    return {
        "max_pain": mp,
        "spot": spot,
        "gap": mp - spot,
        "gap_pct": (mp / spot - 1) * 100 if spot else np.nan,
        "call_oi": int(call_oi),
        "put_oi": int(put_oi),
        "pc_ratio": put_oi / call_oi if call_oi else np.nan,
        "strikes": len(chain),
    }


# ==================================================================== data
@st.cache_data(ttl=600, show_spinner=False)
def expiries(ticker: str):
    return list(yf.Ticker(ticker).options)


@st.cache_data(ttl=600, show_spinner=False)
def load_chain(ticker: str, expiry: str) -> pd.DataFrame:
    ch = yf.Ticker(ticker).option_chain(expiry)
    calls = ch.calls[["strike", "openInterest", "volume", "lastPrice"]]
    puts = ch.puts[["strike", "openInterest", "volume", "lastPrice"]]
    calls = calls.rename(columns={"openInterest": "call_oi", "volume": "call_vol",
                                  "lastPrice": "call_last"})
    puts = puts.rename(columns={"openInterest": "put_oi", "volume": "put_vol",
                                "lastPrice": "put_last"})
    df = calls.merge(puts, on="strike", how="outer").fillna(0)
    return df.sort_values("strike").reset_index(drop=True)


@st.cache_data(ttl=120, show_spinner=False)
def spot_price(ticker: str) -> float:
    h = yf.Ticker(ticker).history(period="1d", interval="1m")
    return float(h["Close"].iloc[-1]) if len(h) else float("nan")


# =================================================================== layout
st.title("Max Pain")
st.caption("Where expiring options are worth the least in total. A snapshot "
           "of positioning — not a prediction.")

st.info("UNVALIDATED. Nothing here has been tested on your own data. Treat it "
        "as a hypothesis the snapshot collector is gathering evidence for.",
        icon="ℹ️")

c = st.columns(3)
ticker = c[0].text_input("Ticker", "SPY").upper().strip()

try:
    exp_list = expiries(ticker)
except Exception as e:
    st.error(f"Couldn't load expirations for {ticker}: {e}")
    st.stop()
if not exp_list:
    st.error(f"No option expirations found for {ticker}.")
    st.stop()

expiry = c[1].selectbox("Expiration", exp_list)
window = c[2].slider("Strikes shown (± % of spot)", 1, 15, 5,
                     help="Max pain is computed on the FULL chain. This only "
                          "controls how much of it is charted.")

try:
    chain = load_chain(ticker, expiry)
    spot = spot_price(ticker)
except Exception as e:
    st.error(f"Couldn't load the chain: {e}")
    st.stop()

if chain.empty or chain[["call_oi", "put_oi"]].to_numpy().sum() == 0:
    st.warning("No open interest in this chain yet. Yahoo updates open "
               "interest overnight, so a brand-new expiration can read zero.")
    st.stop()

s = summarize(chain, spot)

m = st.columns(5)
m[0].metric("Max pain", f"{s['max_pain']:,.0f}")
m[1].metric("Spot", f"{spot:,.2f}", f"{-s['gap']:+.2f} vs max pain",
            delta_color="off")
m[2].metric("Distance", f"{-s['gap_pct']:+.2f}%",
            help="How far spot sits above (+) or below (−) max pain.")
m[3].metric("Put/call OI", f"{s['pc_ratio']:.2f}",
            help="Total put open interest divided by call. Above 1 means more "
                 "puts are open — it does NOT mean price goes up.")
m[4].metric("Strikes in chain", s["strikes"])

# ---- payout curve
curve = max_pain_curve(chain)
lo, hi = spot * (1 - window / 100), spot * (1 + window / 100)
view = curve[(curve.settle >= lo) & (curve.settle <= hi)]

fig = go.Figure()
fig.add_trace(go.Scatter(x=view.settle, y=view.total / 1e6, name="total",
                         line=dict(color="#60a5fa", width=2)))
fig.add_trace(go.Scatter(x=view.settle, y=view.call_value / 1e6, name="calls",
                         line=dict(color="#4ade80", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=view.settle, y=view.put_value / 1e6, name="puts",
                         line=dict(color="#f87171", width=1, dash="dot")))
fig.add_vline(x=s["max_pain"], line=dict(color="#fbbf24", dash="dash"),
              annotation_text="max pain")
fig.add_vline(x=spot, line=dict(color="#e5e7eb", dash="dot"),
              annotation_text="spot")
fig.update_layout(height=380, template="plotly_dark",
                  yaxis_title="option value at settle ($M)",
                  xaxis_title="settle price", margin=dict(t=30))
st.plotly_chart(fig, width="stretch")

# ---- open interest by strike
oi = chain[(chain.strike >= lo) & (chain.strike <= hi)]
f2 = go.Figure()
f2.add_trace(go.Bar(x=oi.strike, y=oi.call_oi, name="call OI",
                    marker_color="#4ade80"))
f2.add_trace(go.Bar(x=oi.strike, y=-oi.put_oi, name="put OI",
                    marker_color="#f87171"))
f2.add_vline(x=s["max_pain"], line=dict(color="#fbbf24", dash="dash"))
f2.update_layout(height=320, template="plotly_dark", barmode="relative",
                 yaxis_title="open interest (puts shown negative)",
                 xaxis_title="strike", margin=dict(t=20))
st.plotly_chart(f2, width="stretch")

with st.expander("The strike table"):
    tbl = oi[["strike", "call_oi", "call_vol", "put_oi", "put_vol"]].copy()
    tbl["net_oi"] = tbl.call_oi - tbl.put_oi
    st.dataframe(tbl, hide_index=True, width="stretch")

# =================================================================== history
st.divider()
st.subheader("History — max pain vs where price actually closed")

LOG = os.environ.get("MAXPAIN_LOG", os.path.join("data", "maxpain_log.csv"))


@st.cache_data(ttl=300, show_spinner=False)
def closes(ticker: str, start: str, end: str) -> pd.Series:
    h = yf.Ticker(ticker).history(start=start, end=end, interval="1d")
    if h.empty:
        return pd.Series(dtype=float)
    h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
    return h["Close"]


if not os.path.exists(LOG):
    st.info("No history yet. Once the snapshot workflow has run "
            "(.github/workflows/chain_snapshot.yml), each run appends a row "
            "here and this section fills in. You can also add rows by hand to "
            "data/maxpain_log.csv to record a day you watched live.")
else:
    log = pd.read_csv(LOG)
    log["ts_utc"] = pd.to_datetime(log["ts_utc"], errors="coerce")
    log["date"] = log["ts_utc"].dt.normalize()
    log = log[log.ticker == ticker].dropna(subset=["date", "max_pain"])

    if log.empty:
        st.info(f"No recorded runs for {ticker} yet.")
    else:
        same_day = st.checkbox("Only same-day expirations (0DTE)", value=True,
                               help="The cleanest test: max pain for options "
                                    "expiring that very day, against that "
                                    "day's close.")
        h = log.copy()
        h["expiry_d"] = pd.to_datetime(h["expiry"], errors="coerce")
        if same_day:
            h = h[h.expiry_d.dt.normalize() == h.date]
        if h.empty:
            st.info("No rows match that filter yet.")
        else:
            # first reading of each day vs that day's actual close
            first = h.sort_values("ts_utc").groupby("date").first().reset_index()
            px = closes(ticker, str(first.date.min().date()),
                        str((first.date.max() + pd.Timedelta(days=1)).date()))
            first["close"] = first.date.map(px)
            d = first.dropna(subset=["close"]).copy()

            if d.empty:
                st.info("Waiting on closing prices for these dates.")
            else:
                d["dist_before"] = (d.spot - d.max_pain).abs()
                d["dist_close"] = (d.close - d.max_pain).abs()
                d["moved_toward"] = d.dist_close < d.dist_before
                d["drift"] = d.dist_before - d.dist_close

                k = st.columns(4)
                k[0].metric("Days recorded", len(d))
                k[1].metric("Closed nearer max pain",
                            f"{d.moved_toward.mean():.0%}",
                            help="Coin-flip is ~50%. Needs 30+ days before "
                                 "this means anything.")
                k[2].metric("Avg move toward it", f"{d.drift.mean():+.2f}",
                            help="Dollars of distance closed. Positive means "
                                 "price ended nearer max pain than it started.")
                k[3].metric("Avg gap at close",
                            f"{d.dist_close.mean():.2f}")

                if len(d) < 30:
                    st.warning(f"{len(d)} days recorded. Under 30 this is "
                               "noise — do not trade off it yet.")

                f3 = go.Figure()
                f3.add_trace(go.Scatter(x=d.date, y=d.max_pain, name="max pain",
                                        mode="lines+markers",
                                        line=dict(color="#fbbf24")))
                f3.add_trace(go.Scatter(x=d.date, y=d.spot, name="spot at first run",
                                        mode="markers",
                                        marker=dict(color="#9ca3af", size=6)))
                f3.add_trace(go.Scatter(x=d.date, y=d.close, name="actual close",
                                        mode="lines+markers",
                                        line=dict(color="#60a5fa")))
                f3.update_layout(height=340, template="plotly_dark",
                                 margin=dict(t=20), yaxis_title="price")
                st.plotly_chart(f3, width="stretch")

                tbl = d[["date", "expiry", "spot", "max_pain", "close",
                         "dist_before", "dist_close", "moved_toward",
                         "pc_ratio"]].copy()
                tbl["date"] = tbl.date.dt.strftime("%Y-%m-%d")
                st.dataframe(tbl.iloc[::-1].round(2), hide_index=True,
                             width="stretch")
                st.caption("dist_before: how far spot was from max pain at the "
                           "first run of the day. dist_close: how far the close "
                           "ended up. moved_toward is True when the close was "
                           "nearer — which is the whole claim being tested.")


with st.expander("How to read this — and what to distrust"):
    st.markdown("""
**The curve** is what every open contract would be worth if the underlying
settled at each price. The low point is max pain.

**Volume vs open interest.** Max pain uses open interest, which Yahoo
updates overnight. On 0DTE, same-day volume dwarfs open interest, so the
number can be badly stale during the session. Treat 0DTE readings as rough.

**Put/call ratio is not direction.** Every put has a buyer and a seller, and
many puts are hedges on stock people already own. A high ratio does not
mean price rises.

**What would make this real:** the collector records max pain and the close
every day. After a few months, we test whether the close lands nearer to
max pain than a random strike would predict. If it doesn't, we drop it —
the same standard everything else here has to clear.
""")

st.caption(f"Updated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · "
           "Yahoo data, delayed. Research tool, not trading advice.")
