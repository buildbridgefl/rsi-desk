"""
0DTE trade tracker — one self-contained page (no imports from core/ or ui/).
Lives in pages/ and shows up in the app sidebar.

Saving: add GITHUB_TOKEN and DATA_REPO to the app's Secrets to keep trades
in a private repo. Without them, trades go to a local file that Streamlit
Cloud wipes when the app sleeps.
"""

import base64
import io
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="0DTE Tracker", page_icon="🎯", layout="wide")


# =================================================================== storage
API = "https://api.github.com"


class CSVStore:
    def __init__(self, columns, path, token=None, repo=None, branch="main"):
        self.columns = list(columns)
        self.path = path
        self.token, self.repo, self.branch = token, repo, branch
        self.remote = bool(token and repo)
        self._sha = None  # GitHub needs the current file version to overwrite it

    @property
    def label(self) -> str:
        if self.remote:
            return f"GitHub · {self.repo}/{self.path}"
        return f"local file · {self.path}"

    # ------------------------------------------------------------ helpers
    def _headers(self):
        return {"Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28"}

    def _url(self):
        return f"{API}/repos/{self.repo}/contents/{self.path}"

    def _conform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for c in self.columns:
            if c not in df.columns:
                df[c] = pd.NA
        extra = [c for c in df.columns if c not in self.columns]
        return df[self.columns + extra].reset_index(drop=True)

    # --------------------------------------------------------------- API
    def load(self) -> pd.DataFrame:
        if not self.remote:
            if os.path.exists(self.path) and os.path.getsize(self.path):
                return self._conform(pd.read_csv(self.path))
            return self._conform(pd.DataFrame())

        r = requests.get(self._url(), headers=self._headers(),
                         params={"ref": self.branch}, timeout=15)
        if r.status_code == 404:
            self._sha = None
            return self._conform(pd.DataFrame())
        r.raise_for_status()
        body = r.json()
        self._sha = body["sha"]
        raw = base64.b64decode(body["content"]).decode()
        if not raw.strip():
            return self._conform(pd.DataFrame())
        return self._conform(pd.read_csv(io.StringIO(raw)))

    def save(self, df: pd.DataFrame, message: str = "update trades") -> pd.DataFrame:
        df = self._conform(df)
        text = df.to_csv(index=False)

        if not self.remote:
            folder = os.path.dirname(os.path.abspath(self.path))
            os.makedirs(folder, exist_ok=True)
            with open(self.path, "w") as f:
                f.write(text)
            return df

        payload = {"message": message, "branch": self.branch,
                   "content": base64.b64encode(text.encode()).decode()}
        if self._sha:
            payload["sha"] = self._sha
        r = requests.put(self._url(), headers=self._headers(),
                         json=payload, timeout=15)
        if r.status_code in (409, 422):
            raise RuntimeError("the trades file changed somewhere else — "
                               "press Reload, then try again")
        if r.status_code in (401, 403):
            raise RuntimeError("GitHub refused the save — check that the token "
                               "has Contents: Read and write on this repo")
        r.raise_for_status()
        self._sha = r.json()["content"]["sha"]
        return df


# ====================================================================== math
COLUMNS = ["date", "time_in", "time_out", "ticker", "right", "strike",
           "contracts", "entry", "exit", "stop", "spread", "fees",
           "underlying_in", "setup", "thesis", "notes"]
NUMERIC = ["strike", "contracts", "entry", "exit", "stop", "spread",
           "fees", "underlying_in"]
SETUPS = ["first touch", "VWAP", "opening range", "11:00 turn",
          "headline / news", "trend", "other"]
MULT = 100          # shares per contract
MIN_SAMPLE = 30     # closed trades before any verdict means much


def _stamp(date: pd.Series, t: pd.Series) -> pd.Series:
    t = t.fillna("").astype(str).str.strip()
    s = date.fillna("").astype(str) + " " + t
    return pd.to_datetime(s.where(t != ""), errors="coerce")


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add P&L, R, hold time and bucket columns to the raw log."""
    d = df.copy()
    for c in COLUMNS:
        if c not in d.columns:
            d[c] = np.nan
    for c in NUMERIC:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["contracts"] = d["contracts"].fillna(1)
    d["fees"] = d["fees"].fillna(0.0)
    d["spread"] = d["spread"].fillna(0.0)
    for c in ("setup", "right"):
        d[c] = d[c].fillna("").astype(str).str.strip().replace("", "(blank)")

    d["entered"] = _stamp(d["date"], d["time_in"])
    d["exited"] = _stamp(d["date"], d["time_out"])
    d["closed"] = d["exit"].notna() & d["entry"].gt(0)

    size = d["contracts"] * MULT
    d["pnl"] = (d["exit"] - d["entry"]) * size - d["fees"]
    has_stop = d["stop"].gt(0) & d["stop"].lt(d["entry"])
    d["risk"] = np.where(has_stop, d["entry"] - d["stop"], d["entry"]) * size
    d["R"] = d["pnl"] / d["risk"]
    d["ret_pct"] = (d["exit"] / d["entry"] - 1) * 100
    d["spread_cost"] = d["spread"] * size   # ~half at entry + half at exit

    d["hold_min"] = (d["exited"] - d["entered"]).dt.total_seconds() / 60
    d["tod"] = d["entered"].dt.floor("30min").dt.strftime("%H:%M")
    d["hold_bucket"] = pd.cut(d["hold_min"], [-0.01, 5, 15, 30, 60, np.inf],
                              labels=["≤5m", "5–15m", "15–30m", "30–60m", "60m+"])
    return d.sort_values("entered", na_position="last", kind="stable")


def wilson(k: int, n: int, z: float = 1.96):
    """95% range for a win rate. Wide when n is small — that's the point."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def summary(closed: pd.DataFrame) -> dict:
    n = len(closed)
    pnl = closed["pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]          # scratches count as losses
    avg_w = wins.mean() if len(wins) else 0.0
    avg_l = -losses.mean() if len(losses) else 0.0
    lo, hi = wilson(len(wins), n)
    gross_l = -losses.sum()

    curve = pnl.cumsum()
    peak = curve.cummax().clip(lower=0)
    max_dd = float(min((curve - peak).min(), 0.0)) if n else 0.0

    return {
        "n": n,
        "wins": len(wins),
        "win_rate": len(wins) / n if n else np.nan,
        "wr_low": lo, "wr_high": hi,
        # win rate needed to break even with YOUR average win and loss
        "breakeven": avg_l / (avg_w + avg_l) if (avg_w + avg_l) > 0 else np.nan,
        "avg_win": avg_w, "avg_loss": avg_l,
        "profit_factor": wins.sum() / gross_l if gross_l > 0 else np.inf,
        "exp_dollars": pnl.mean() if n else np.nan,
        "exp_R": closed["R"].mean() if n else np.nan,
        "net": pnl.sum(),
        "spread_cost": closed["spread_cost"].sum(),
        "max_dd": max_dd,
    }


def verdict(s: dict):
    """Returns (streamlit message kind, text)."""
    n = s["n"]
    if n < MIN_SAMPLE:
        return ("info", f"{n} closed trade{'s' if n != 1 else ''} — too few to judge. "
                f"Keep logging until {MIN_SAMPLE}+; early numbers swing a lot.")
    if s["exp_R"] <= 0:
        return ("error", f"Negative expectancy ({s['exp_R']:+.2f}R per trade). "
                "On average this approach is losing money.")
    if s["wr_low"] > s["breakeven"]:
        return ("success", "Positive expectancy, and your win rate stays above "
                "break-even even at the low end of its range.")
    return ("warning", "Positive expectancy so far, but the win-rate range still "
            "overlaps break-even. Not proven yet.")


def breakdown(closed: pd.DataFrame, col: str) -> pd.DataFrame:
    sub = closed.dropna(subset=[col])
    if sub.empty:
        return pd.DataFrame()
    g = sub.groupby(col, observed=True)
    out = pd.DataFrame({
        "trades": g.size(),
        "win %": g["pnl"].apply(lambda s: round((s > 0).mean() * 100)),
        "net $": g["pnl"].sum().round(0),
        "avg R": g["R"].mean().round(2),
    })
    return out.reset_index().rename(columns={col: "group"})


# ======================================================================== UI
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
            COLUMNS, path=_secret("TRADES_PATH") or default,
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

            setup = st.selectbox("Setup", SETUPS)
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
    e = enrich(trades)
    closed = e[e["closed"]]
    open_n = int((~e["closed"]).sum())
    if closed.empty:
        st.info(f"{open_n} open trade(s). Stats appear once you close one.")
        return

    s = summary(closed)
    kind, msg = verdict(s)
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
    b = breakdown(closed, col)
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


render()
