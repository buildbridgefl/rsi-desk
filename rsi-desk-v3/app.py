"""RSI Desk — streamlit run app.py"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import strategies  # noqa: F401  (registers everything)
from core import engine
from core.data import fetch, split
from strategies.base import REGISTRY, names

st.set_page_config(page_title="RSI Desk", page_icon="📊", layout="wide")
JOURNAL = os.environ.get("JOURNAL_PATH",
                         os.path.expanduser("~/.rsi_desk_journal.csv"))


@st.cache_data(ttl=300, show_spinner=False)
def load(ticker, period="5y", interval="1d"):
    return fetch(ticker, period, interval)


# --------------------------------------------------------------- sidebar
st.sidebar.title("RSI Desk")
watchlist = [t.strip().upper() for t in st.sidebar.text_area(
    "Watchlist", "SPY, QQQ, IWM, SMH").split(",") if t.strip()]

strat_name = st.sidebar.selectbox("Strategy", names())
strat = REGISTRY[strat_name]
st.sidebar.caption(strat.description)

st.sidebar.markdown("**Rule parameters**")
params = {}
for k, v in strat.params.items():
    if isinstance(v, bool):
        params[k] = st.sidebar.checkbox(k, v)
    elif isinstance(v, int):
        params[k] = st.sidebar.number_input(k, 0, 500, v, 1)
    elif isinstance(v, float):
        params[k] = st.sidebar.number_input(k, 0.0, 500.0, v, 0.5)

st.sidebar.markdown("**Execution**")
max_hold = st.sidebar.number_input("max hold (bars)", 1, 2000, 250)
stop_pct = st.sidebar.number_input("stop loss %", 0.0, 25.0, 0.0, 0.5) / 100
take_profit = st.sidebar.number_input(
    "take profit %", 0.0, 50.0, 0.0, 0.5,
    help="0 = disabled. Pair this with a stop loss. A target without a "
         "stop caps your winners while letting losers run.") / 100
cost_bps = st.sidebar.number_input("cost bps/side", 0.0, 50.0, 2.0, 0.5)
cash_apr = st.sidebar.number_input(
    "cash yield % (T-bill)", 0.0, 10.0, 4.0, 0.25,
    help="Yield earned while flat. Set to roughly the risk-free rate over "
         "your test window. Leaving it at 0 unfairly penalises every rule "
         "that spends time out of the market.") / 100

if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

exec_kw = dict(max_hold=max_hold, stop_pct=stop_pct, cost_bps=cost_bps,
               cash_apr=cash_apr, take_profit_pct=take_profit)

if take_profit and not stop_pct:
    st.sidebar.warning("Target set with no stop. Winners capped, losers "
                       "uncapped — expect a high win rate and poor returns.")
t_today, t_chart, t_bt, t_log = st.tabs(["Today", "Chart", "Backtest", "Journal"])


# ----------------------------------------------------------------- today
with t_today:
    st.subheader(f"Watchlist — {strat_name}")
    rows = []
    for tk in watchlist:
        try:
            d = strat(load(tk, "3y"), **params)
            last, prev = d.iloc[-1], d.iloc[-2]
            if last.long_entry:
                sig = "LONG"
            elif last.short_entry:
                sig = "SHORT"
            else:
                sig = "—"
            rows.append({
                "ticker": tk,
                "price": round(float(last.Close), 2),
                "chg%": round((last.Close / prev.Close - 1) * 100, 2),
                "RSI": round(float(last.rsi), 1) if "rsi" in d else None,
                "signal": sig,
            })
        except Exception as e:
            rows.append({"ticker": tk, "signal": f"error: {type(e).__name__}"})

    df = pd.DataFrame(rows)

    def paint(v):
        return {"LONG": "background-color:#10361f;color:#4ade80",
                "SHORT": "background-color:#3b1219;color:#f87171"}.get(v, "")

    st.dataframe(df.style.map(paint, subset=["signal"]),
                 use_container_width=True, hide_index=True)

    hits = df[df.signal.isin(["LONG", "SHORT"])]
    if not hits.empty:
        for _, r in hits.iterrows():
            st.success(f"**{r.ticker}** {r.signal} @ {r.price}")
    else:
        st.info("No signals today.")
    st.caption(f"Updated {datetime.now():%Y-%m-%d %H:%M} · "
               "Yahoo data, 15-min delayed")


# ----------------------------------------------------------------- chart
with t_chart:
    tk = st.selectbox("Ticker", watchlist, key="ct")
    per = st.radio("Period", ["6mo", "1y", "2y", "5y"], 1, horizontal=True)
    d = strat(load(tk, per), **params)
    has_rsi = "rsi" in d.columns

    fig = make_subplots(rows=2 if has_rsi else 1, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3] if has_rsi else [1.0],
                        vertical_spacing=0.04)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High,
                                 low=d.Low, close=d.Close, name=tk), row=1, col=1)
    for n, c in [(50, "#fbbf24"), (200, "#ef4444")]:
        if len(d) >= n:
            fig.add_trace(go.Scatter(x=d.index, y=d.Close.rolling(n).mean(),
                                     name=f"{n}MA", line=dict(color=c, width=1.1)),
                          row=1, col=1)
    for col, sym, colr, nm in [("long_entry", "triangle-up", "#4ade80", "long"),
                               ("short_entry", "triangle-down", "#f87171", "short")]:
        pts = d.index[d[col].fillna(False)]
        if len(pts):
            y = d.loc[pts, "Low"] * 0.985 if "up" in sym else d.loc[pts, "High"] * 1.015
            fig.add_trace(go.Scatter(x=pts, y=y, mode="markers", name=nm,
                                     marker=dict(symbol=sym, size=7, color=colr)),
                          row=1, col=1)
    if has_rsi:
        fig.add_trace(go.Scatter(x=d.index, y=d.rsi, name="RSI",
                                 line=dict(color="#60a5fa", width=1.1)), row=2, col=1)
        for lv, c in [(70, "#f87171"), (30, "#4ade80")]:
            fig.add_hline(y=lv, line=dict(color=c, dash="dash", width=1),
                          row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)

    fig.update_layout(height=640, template="plotly_dark",
                      xaxis_rangeslider_visible=False, margin=dict(t=25))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{int(d.long_entry.sum())} long / {int(d.short_entry.sum())} "
               f"short signal-bars in this window")


# -------------------------------------------------------------- backtest
with t_bt:
    c1, c2, c3 = st.columns(3)
    bt_tk = c1.selectbox("Ticker", watchlist, key="bt")
    years = c2.slider("Years", 2, 25, 20)
    frac = c3.slider("In-sample fraction", 0.3, 0.9, 0.6, 0.05)

    if st.button("Run backtest", type="primary"):
        raw = load(bt_tk, f"{years}y")
        sig = strat(raw, **params)
        ins, oos = split(sig, frac)

        for label, seg in [("In-sample (design)", ins),
                           ("Out-of-sample (verdict)", oos)]:
            eq, bh, tr = engine.run(seg, **exec_kw)
            s, b = engine.stats(eq), engine.stats(bh)
            ts = engine.trade_stats(tr)
            st.markdown(f"### {label}")
            st.caption(f"{seg.index[0]:%Y-%m-%d} → {seg.index[-1]:%Y-%m-%d}")
            m = st.columns(5)
            m[0].metric("CAGR", f"{s['cagr']:.1%}",
                        f"{s['cagr']-b['cagr']:+.1%} vs hold")
            m[1].metric("Max DD", f"{s['max_dd']:.1%}",
                        f"{s['max_dd']-b['max_dd']:+.1%}", delta_color="inverse")
            m[2].metric("Sharpe", f"{s['sharpe']:.2f}",
                        f"{s['sharpe']-b['sharpe']:+.2f}")
            m[3].metric("Trades", ts["trades"])
            m[4].metric("Win rate", f"{ts['win_rate']:.0%}")

            f = go.Figure()
            f.add_trace(go.Scatter(x=eq.index, y=eq, name="strategy",
                                   line=dict(color="#60a5fa")))
            f.add_trace(go.Scatter(x=bh.index, y=bh, name="buy & hold",
                                   line=dict(color="#9ca3af", dash="dot")))
            f.update_layout(height=260, template="plotly_dark",
                            margin=dict(t=8, b=8))
            st.plotly_chart(f, use_container_width=True)
            if not tr.empty:
                with st.expander("trade log"):
                    st.dataframe(tr, use_container_width=True, hide_index=True)

        st.warning("Only the out-of-sample block counts. If it doesn't beat "
                   "hold on Sharpe **and** drawdown, the edge isn't there — "
                   "change the rule, don't change the test.")


# --------------------------------------------------------------- journal
with t_log:
    st.subheader("Paper trade journal")
    cols = ["date", "ticker", "side", "qty", "entry", "exit", "rsi_at_entry",
            "thesis", "pnl"]
    j = pd.read_csv(JOURNAL) if os.path.exists(JOURNAL) else pd.DataFrame(columns=cols)

    with st.expander("Log a trade", expanded=j.empty):
        c = st.columns(4)
        e = {"date": str(c[0].date_input("Date", datetime.today())),
             "ticker": c[0].text_input("Ticker", "SPY").upper(),
             "side": c[1].selectbox("Side", ["long", "short"]),
             "qty": c[1].number_input("Qty", 1, 100000, 100),
             "entry": c[2].number_input("Entry", 0.0, 1e6, 0.0, 0.01),
             "exit": c[2].number_input("Exit (0 = still open)", 0.0, 1e6, 0.0, 0.01),
             "rsi_at_entry": c[3].number_input("RSI at entry", 0.0, 100.0, 30.0, 0.1)}
        e["thesis"] = st.text_input("Thesis — one line, written BEFORE you know the outcome")
        if st.button("Save"):
            e["pnl"] = (round((e["exit"] - e["entry"]) * e["qty"] *
                              (1 if e["side"] == "long" else -1), 2)
                        if e["exit"] > 0 else np.nan)
            j = pd.concat([j, pd.DataFrame([e])], ignore_index=True)
            j.to_csv(JOURNAL, index=False)
            st.rerun()

    if not j.empty:
        st.dataframe(j, use_container_width=True, hide_index=True)
        closed = pd.to_numeric(j.pnl, errors="coerce").dropna()
        if len(closed):
            k = st.columns(4)
            k[0].metric("Closed", len(closed))
            k[1].metric("Net P&L", f"${closed.sum():,.0f}")
            k[2].metric("Win rate", f"{(closed>0).mean():.0%}")
            k[3].metric("Avg", f"${closed.mean():,.0f}")
        st.download_button("Download CSV", j.to_csv(index=False), "journal.csv")
    st.caption("Cloud filesystems reset — download the CSV periodically.")
