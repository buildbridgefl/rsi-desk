"""
Backtest engine.

Consumes the boolean signal columns a strategy produces. Guarantees:
  - No lookahead: signal on bar t close -> fill at bar t+1 open
  - Costs applied to every fill
  - One position at a time
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def run(df: pd.DataFrame, max_hold: int = 20, stop_pct: float = 0.0,
        cost_bps: float = 2.0, capital: float = 100_000.0,
        cash_apr: float = 0.0, take_profit_pct: float = 0.0):
    """
    df must already have long_entry/short_entry/long_exit/short_exit.

    cash_apr: annual yield credited on uninvested cash.
    take_profit_pct: exit when unrealised gain reaches this fraction.
      Checked against the bar HIGH for longs / LOW for shorts, so it fires
      intrabar like a resting limit order.

    Note on optimism: when both the stop and the target are touched inside
    the same bar, this fills the TARGET. Daily bars can't tell you which
    came first. That biases results in the strategy's favour, so treat any
    edge that depends on a tight target as an upper bound, not a forecast.
    """
    need = {"long_entry", "short_entry", "long_exit", "short_exit"}
    if not need.issubset(df.columns):
        raise ValueError(f"missing signal columns: {sorted(need - set(df.columns))}")

    d = df.dropna(subset=["Open", "Close"]).copy()
    op, cl = d["Open"].to_numpy(), d["Close"].to_numpy()
    lo, hi = d["Low"].to_numpy(), d["High"].to_numpy()
    le = d["long_entry"].fillna(False).to_numpy()
    se = d["short_entry"].fillna(False).to_numpy()
    lx = d["long_exit"].fillna(False).to_numpy()
    sx = d["short_exit"].fillna(False).to_numpy()
    idx = d.index

    cost = cost_bps / 10_000.0
    equity = np.full(len(d), capital, dtype=float)
    cash, pos, shares, entry_px, entry_i = capital, 0, 0.0, 0.0, 0
    trades = []

    daily_cash = (1 + cash_apr) ** (1 / TRADING_DAYS) - 1 if cash_apr else 0.0

    for i in range(1, len(d)):
        if daily_cash:
            cash *= 1 + daily_cash
        equity[i] = cash + shares * cl[i] if pos != 0 else cash

        if pos != 0 and i + 1 < len(d):
            held = i - entry_i
            stopped = (stop_pct > 0 and (
                (pos == 1 and lo[i] <= entry_px * (1 - stop_pct)) or
                (pos == -1 and hi[i] >= entry_px * (1 + stop_pct))))
            hit_target = (take_profit_pct > 0 and (
                (pos == 1 and hi[i] >= entry_px * (1 + take_profit_pct)) or
                (pos == -1 and lo[i] <= entry_px * (1 - take_profit_pct))))
            reverted = (pos == 1 and lx[i]) or (pos == -1 and sx[i])

            if stopped or hit_target or reverted or held >= max_hold:
                if hit_target:
                    # resting limit order fills at the target, not next open
                    tgt = entry_px * (1 + take_profit_pct if pos == 1
                                      else 1 - take_profit_pct)
                    fill = tgt * (1 - cost if pos == 1 else 1 + cost)
                    exit_date = idx[i]
                else:
                    fill = op[i + 1] * (1 - cost if pos == 1 else 1 + cost)
                    exit_date = idx[i + 1]
                cash += shares * fill
                pnl = (fill - entry_px) * shares
                trades.append({
                    "side": "long" if pos == 1 else "short",
                    "entry_date": idx[entry_i], "exit_date": exit_date,
                    "entry": round(entry_px, 2), "exit": round(fill, 2),
                    "bars": held + 1, "pnl": round(pnl, 2),
                    "ret_pct": round(pnl / (abs(shares) * entry_px) * 100, 2),
                    "reason": ("target" if hit_target else
                               "stop" if stopped else
                               "signal" if reverted else "time"),
                })
                pos, shares, entry_px = 0, 0.0, 0.0
                continue

        if pos == 0 and i + 1 < len(d) and (le[i] or se[i]):
            pos = 1 if le[i] else -1
            entry_px = op[i + 1] * (1 + cost if pos == 1 else 1 - cost)
            shares = (cash / entry_px) * pos
            cash -= shares * entry_px
            entry_i = i + 1

    if pos != 0:
        fill = cl[-1] * (1 - cost if pos == 1 else 1 + cost)
        cash += shares * fill
        trades.append({
            "side": "long" if pos == 1 else "short",
            "entry_date": idx[entry_i], "exit_date": idx[-1],
            "entry": round(entry_px, 2), "exit": round(fill, 2),
            "bars": len(d) - 1 - entry_i,
            "pnl": round((fill - entry_px) * shares, 2),
            "ret_pct": round((fill - entry_px) * shares / (abs(shares) * entry_px) * 100, 2),
            "reason": "eod",
        })
    equity[-1] = cash

    eq = pd.Series(equity, index=idx)
    bh = capital * (d["Close"] / d["Close"].iloc[0])
    return eq, bh, pd.DataFrame(trades)


def stats(eq: pd.Series) -> dict:
    ret = eq.pct_change().dropna()
    yrs = max(len(eq) / TRADING_DAYS, 1e-9)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    vol = ret.std() * np.sqrt(TRADING_DAYS)
    return {
        "total_ret": eq.iloc[-1] / eq.iloc[0] - 1,
        "cagr": cagr,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "vol": vol,
        "sharpe": (ret.mean() * TRADING_DAYS) / vol if vol > 0 else 0.0,
    }


def trade_stats(t: pd.DataFrame) -> dict:
    if t.empty:
        return {"trades": 0, "win_rate": 0.0, "profit_factor": 0.0,
                "avg_win": 0.0, "avg_loss": 0.0, "avg_bars": 0.0}
    w, l = t[t.pnl > 0], t[t.pnl <= 0]
    gl = abs(l.pnl.sum())
    return {
        "trades": len(t),
        "win_rate": len(w) / len(t),
        "avg_win": w.ret_pct.mean() if len(w) else 0.0,
        "avg_loss": l.ret_pct.mean() if len(l) else 0.0,
        "profit_factor": w.pnl.sum() / gl if gl > 0 else float("inf"),
        "avg_bars": t.bars.mean(),
    }
