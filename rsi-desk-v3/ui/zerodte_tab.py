"""0DTE tracker tab. Called from app.py: zerodte_tab.render()"""

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import zerodte as z
from core.storage import CSVStore

ET = ZoneInfo("America/New_York")


# ------------------------------------------------------------- storage
def _secret(key):
    try:
        v = st.secrets.get(key)
    except Exception:          # no secrets file at all
        v = None
    return v or os.environ.get(key)


def _store() -> CSVStore:
    if "z_store" not in st.session_state:
        token, repo = _secret("GITHUB_TOKEN"), _secret("DATA_REPO")
        default = "trades/0dte.csv" if (token and repo) else \
            os.path.expanduser("~/.rsi_desk_0dte.csv")
        st.session_state.z_store = CSVStore(
            z.COLUMNS, path=_secret("TRADES_PATH") or default,
            token=token, repo=repo, branch=_secret("DATA_BRANCH") or "main")
    return st.session_state.z_store


def _load(force=False) -> pd.DataFrame:
    if force or "z_trades" not in st.session_state:
        st.session_state.z_trades = _store().load()
    return st.session_state.z_trades


def _commit(df: pd.DataFrame, msg: str):
    try:
        st.session_state.z_trades = _store().save(df, msg)
    except Exception as e:
        st.error(f"Save failed: {e}")
        return
    st.session_state.pop("z_edit", None)   # drop stale editor state
    st.rerun()


def _pct(x):
    return "—" if x is None or pd.isna(x) else f"{x:.0%}"


def _now():
    return datetime.now(ET).replace(second=0, microsecond=0)


# ---------------------------------------------------------------- forms
def _log_form(trades):
    now = _now()
    with st.expander("➕ Log a trade", expanded=trades.empty):
        with st.form("z_log", clear_on_submit=True):
            c = st.columns(4)
            day = c[0].date_input("Date", now.date())
            t_in = c[0].time_input("Time in (ET)", now.time(), step=60)
            t_out = c[0].time_input("Time out (ET)", None, step=60)

            ticker = c[1].text_input("Ticker", "SPY")
            right = c[1].selectbox("Type", ["call", "put"])
            strike = c[1].number_input("Strike", 0.0, 100000.0, None, 1.0)
            qty = c[1].number_input("Contracts", 1, 1000, 1, 1)

            entry = c[2].number_input("Entry premium", 0.0, 1000.0, None, 0.01, format="%.2f")
            exit_ = c[2].number_input("Exit premium (blank = still open)",
                                      0.0, 1000.0, None, 0.01, format="%.2f")
            stop = c[2].number_input("Planned stop premium (blank = whole premium)",
                                     0.0, 1000.0, None, 0.01, format="%.2f")

            spread = c[3].number_input("Bid-ask width at entry", 0.0, 100.0, None,
                                       0.01, format="%.2f",
                                       help="Ask minus bid when you entered. 0.02 = 2 cents.")
            fees = c[3].number_input("Fees $ (whole trade)", 0.0, 1000.0, 0.0, 0.01, format="%.2f")
            und = c[3].number_input("SPY price at entry", 0.0, 100000.0, None, 0.01, format="%.2f")

            setup = st.selectbox("Setup", z.SETUPS)
            thesis = st.text_input("Thesis — write it BEFORE you know the outcome")
            notes = st.text_input("Notes")
            ok = st.form_submit_button("Save trade", type="primary")

    if ok:
        if not entry:
            st.error("Entry premium is required.")
            return
        row = {"date": day.isoformat(), "time_in": t_in.strftime("%H:%M"),
               "time_out": t_out.strftime("%H:%M") if t_out else "",
               "ticker": ticker.upper().strip(), "right": right, "strike": strike,
               "contracts": qty, "entry": entry, "exit": exit_, "stop": stop,
               "spread": spread, "fees": fees, "underlying_in": und,
               "setup": setup, "thesis": thesis, "notes": notes}
        new = pd.DataFrame([row])
        df = new if trades.empty else pd.concat([trades, new], ignore_index=True)
        _commit(df, f"log {row['ticker']} {strike} {right} {day}")


def _close_form(trades):
    open_ = trades[pd.to_numeric(trades["exit"], errors="coerce").isna()]
    if open_.empty:
        return
    labels = {i: f"{r.date} {r.time_in} · {r.ticker} {r.strike} {r.right} "
                 f"×{r.contracts} @ {r.entry}" for i, r in open_.iterrows()}
    with st.expander(f"🔓 Close an open trade ({len(open_)})", expanded=True):
        with st.form("z_close"):
            pick = st.selectbox("Trade", list(labels), format_func=labels.get)
            c = st.columns(2)
            px = c[0].number_input("Exit premium (0 = expired worthless)",
                                   0.0, 1000.0, None, 0.01, format="%.2f")
            t = c[1].time_input("Time out (ET)", _now().time(), step=60)
            ok = st.form_submit_button("Close trade", type="primary")
    if ok:
        if px is None:
            st.error("Enter the exit premium.")
            return
        df = trades.copy()
        df["exit"] = pd.to_numeric(df["exit"], errors="coerce")
        df["time_out"] = df["time_out"].astype("object")
        df.loc[pick, "exit"] = px
        df.loc[pick, "time_out"] = t.strftime("%H:%M")
        _commit(df, "close trade")


def _editor(trades):
    if trades.empty:
        return
    with st.expander("✏️ Edit or delete trades"):
        st.caption("Fix typos right in the table. To delete, tick the row's "
                   "checkbox and press the trash icon. Then press Save changes.")
        edited = st.data_editor(trades, num_rows="dynamic", hide_index=True,
                                width="stretch", key="z_edit")
        if st.button("Save changes", key="z_edit_save"):
            _commit(edited, "edit trades")


# ------------------------------------------------------------ dashboard
def _dashboard(trades):
    if trades.empty:
        st.info("No trades yet. Log your first one above.")
        return
    e = z.enrich(trades)
    closed = e[e["closed"]]
    open_n = int((~e["closed"]).sum())
    if closed.empty:
        st.info(f"{open_n} open trade(s). Stats appear once you close one.")
        return

    s = z.summary(closed)
    kind, msg = z.verdict(s)
    getattr(st, kind)(msg)

    m = st.columns(5)
    m[0].metric("Closed trades", s["n"],
                f"{open_n} open" if open_n else None, delta_color="off")
    m[1].metric("Win rate", _pct(s["win_rate"]))
    m[2].metric("Break-even win rate", _pct(s["breakeven"]),
                help="Win rate you need with YOUR average win and average loss. "
                     "You want your win rate above this.")
    m[3].metric("Expectancy", f"{s['exp_R']:+.2f}R",
                f"${s['exp_dollars']:+,.0f} per trade", delta_color="off",
                help="Average result per trade, in units of what you risked. "
                     "This is the number that decides if it works.")
    m[4].metric("Net P&L", f"${s['net']:,.0f}")

    m = st.columns(5)
    m[0].metric("Avg win", f"${s['avg_win']:,.0f}")
    m[1].metric("Avg loss", f"-${s['avg_loss']:,.0f}")
    pf = s["profit_factor"]
    m[2].metric("Profit factor", "∞" if np.isinf(pf) else f"{pf:.2f}")
    m[3].metric("Spread paid (est.)", f"${s['spread_cost']:,.0f}",
                help="Bid-ask width × contracts × 100. Roughly what crossing "
                     "the spread cost you, already inside your fills.")
    m[4].metric("Max drawdown", f"-${abs(s['max_dd']):,.0f}")

    st.caption(f"Likely true win rate (95% range): {_pct(s['wr_low'])} – "
               f"{_pct(s['wr_high'])}. The fewer trades, the wider this is.")

    # charts
    c = st.columns(2)
    n = np.arange(1, len(closed) + 1)
    f = go.Figure(go.Scatter(x=n, y=closed["pnl"].cumsum(), mode="lines+markers",
                             line=dict(color="#60a5fa"), name="P&L"))
    f.add_hline(y=0, line=dict(color="#9ca3af", dash="dot", width=1))
    f.update_layout(title="Running P&L ($)", xaxis_title="trade #", height=300,
                    template="plotly_dark", margin=dict(t=40, b=10))
    c[0].plotly_chart(f, width="stretch")

    colors = np.where(closed["R"] > 0, "#4ade80", "#f87171")
    g = go.Figure(go.Bar(x=n, y=closed["R"], marker_color=colors, name="R"))
    g.update_layout(title="Each trade in R", xaxis_title="trade #", height=300,
                    template="plotly_dark", margin=dict(t=40, b=10))
    c[1].plotly_chart(g, width="stretch")

    # breakdowns
    st.markdown("#### Where it makes or leaks money")
    view = st.radio("Split by", ["Time of day", "Hold time", "Setup", "Call vs put"],
                    horizontal=True, key="z_split")
    col = {"Time of day": "tod", "Hold time": "hold_bucket",
           "Setup": "setup", "Call vs put": "right"}[view]
    b = z.breakdown(closed, col)
    if b.empty:
        st.caption("No data for this split yet (hold time needs time in and out).")
    else:
        st.dataframe(b, hide_index=True, width="stretch")
        st.caption("Groups under ~10 trades are mostly noise.")

    with st.expander("Closed trade log"):
        show = closed[["date", "time_in", "time_out", "ticker", "strike", "right",
                       "contracts", "entry", "exit", "pnl", "R", "ret_pct",
                       "hold_min", "setup", "thesis"]].copy()
        show[["pnl", "R", "ret_pct", "hold_min"]] = \
            show[["pnl", "R", "ret_pct", "hold_min"]].round(2)
        st.dataframe(show.iloc[::-1], hide_index=True, width="stretch")

    st.download_button("Download trades CSV", trades.to_csv(index=False),
                       "0dte_trades.csv")


# ----------------------------------------------------------------- main
def render():
    st.subheader("0DTE trade tracker")
    store = _store()
    try:
        trades = _load()
    except Exception as e:
        st.error(f"Couldn't load trades: {e}")
        return

    top = st.columns([4, 1])
    top[0].caption(f"Saving to {store.label}")
    if top[1].button("Reload", key="z_reload"):
        _load(force=True)
        st.rerun()
    if not store.remote:
        st.warning("Trades are in a local file, which Streamlit Cloud wipes when "
                   "the app sleeps. Add GITHUB_TOKEN and DATA_REPO to the app's "
                   "Secrets to keep them.")

    _log_form(trades)
    _close_form(trades)
    _editor(trades)
    _dashboard(trades)
    st.caption("Research tool, not trading advice. Assumes you buy options "
               "(long premium).")
