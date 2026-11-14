"""
Overnight Desk engine: SIMULATED trades only. It never places an order.

What it tests (the two candidates that survived out-of-sample in the project):
  ABS  : hourly seller absorption. CLV < -0.40 and time-of-day volume z > 1.0
  RSI5 : hourly RSI(5) < 20
Both: long only, prior-day close above the 200-day average, first fire per
day per ticker, entry at the NEXT hourly bar's open, exit at the NEXT
session's close. No stop, no target, no adding.
  BASE : benchmark. Buys every uptrend day at 10:30 (open of the 2nd bar) and holds
         to the next session's close. A signal has to beat this, not zero.

No lookahead: a signal uses only bars that have closed; the volume z uses the
same clock slot on the previous 20 sessions; the trend gate uses daily bars
strictly before the session; a bar still forming is ignored (its open may be
used as an entry price, nothing else).

Inputs
  hourly : Webull get_stock_bars JSON (M60, RTH) saved as-is, or CSV
           symbol,time_utc,open,high,low,close,volume
  daily  : Webull get_stock_bars JSON (D) saved as-is, or CSV (same columns)
  config : JSON (see DEFAULT_CONFIG)
  events : JSON {"events": {"YYYY-MM-DD": "FOMC", ...}}   (optional)
Output   : JSON {"trades": [...], "status": {...}}

Usage: python3 engine.py hourly daily config.json events.json out.json
                         [--all] [--now 2026-11-12T10:37]
"""
import csv, json, sys, math
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

DEFAULT_CONFIG = {
    "version": "o1",
    "starts": "2026-11-12",      # forward test begins; earlier dates are replays
    "tickers": ["SPY", "QQQ"],
    "strategies": ["ABS", "RSI5", "BASE", "LVLS", "LVLL"],
    "shares": 100,
    "cost_bps": 2.0,             # per side, as in the study
    "sma_len": 200,
    "abs_clv_max": -0.40,
    "abs_z_min": 1.0,
    "tod_sessions": 20,          # same clock slot on this many prior sessions
    "tod_min": 10,
    "rsi_len": 5,
    "rsi_lo": 20.0,
    "rsi_tickers": ["SPY"],      # the RSI study was SPY only
    # level alerts (Muse's rules, hourly version). Exit = SAME day's close.
    "lvl_tickers": ["SPY"],
    "lvl_near_pct": 0.3,
    "lvl_press": 0.40,           # (close-open)/(high-low)
    "lvl_z_min": 1.0,
    "lvl_round": 10.0,
    "levels": [],                # weekly levels, used only in the week starting levels_week (a Monday)
    "levels_week": "",
    "write_days": 7,             # only emit trades this recent (unless --all)
    # estimated option leg (no real quotes; Black-Scholes)
    "contracts": 1,
    "otm_pct": 0.5,
    "min_dte": 7,
    "rv_days": 20,
    "iv_mult": 1.2,
    "iv_floor": 0.12,
    "rate": 0.04,
    "spread_pct": 1.5,
    "min_spread": 0.02,
    "fee": 0.65,                 # $ per contract per side
}


# ---------- loading ----------

def _parse_time(s):
    s = s.strip().replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    if len(s) > 6 and s[-3] == ":" and s[-6] in "+-":
        return _parse_time(s[:-3] + s[-2:])
    t = datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def load_bars(path):
    """Returns {symbol: [ {t (ET), o,h,l,c,v} ... ]} sorted, de-duplicated."""
    txt = open(path).read().strip()
    out = {}
    if txt[:1] in "{[":
        d = json.loads(txt)
        groups = d["result"] if isinstance(d, dict) else d
        for g in groups:
            sym = g["symbol"].upper()
            for b in g["result"]:
                out.setdefault(sym, []).append(
                    {"t": _parse_time(b["time"]).astimezone(ET), "o": float(b["open"]),
                     "h": float(b["high"]), "l": float(b["low"]), "c": float(b["close"]),
                     "v": float(b["volume"]), "sess": b.get("trading_session") or "RTH"})
    else:
        for row in csv.reader(txt.splitlines()):
            if not row or row[0].lower().startswith("sym"):
                continue
            o, h, l, c, v = map(float, row[2:7])
            out.setdefault(row[0].upper(), []).append(
                {"t": _parse_time(row[1]).astimezone(ET), "o": o, "h": h, "l": l, "c": c, "v": v,
                 "sess": row[7] if len(row) > 7 and row[7] else "RTH"})
    for sym, bars in out.items():
        bars.sort(key=lambda b: b["t"])
        seen, ded = set(), []
        for b in bars:
            if b["t"] not in seen:
                seen.add(b["t"]); ded.append(b)
        out[sym] = ded
    return out


def prep_hourly(bars, now):
    """Adds start/end/slot/date. Webull labels the 15:30-16:00 bar '15:00'."""
    out = []
    for b in bars:
        st = b["t"]
        if st.hour == 15 and st.minute == 0:
            st = st.replace(minute=30)
        close = st.replace(hour=16, minute=0, second=0, microsecond=0)
        en = min(st + timedelta(minutes=60), close)
        out.append({**b, "st": st, "en": en, "slot": st.strftime("%H:%M"),
                    "d": st.date().isoformat()})
    for i, b in enumerate(out):
        later = i + 1 < len(out)
        b["done"] = later or b["en"] <= now
    return out


def daily_rows(bars):
    return [{"d": b["t"].date().isoformat(), "c": b["c"]} for b in bars]


# ---------- maths ----------

def rsi_series(closes, n):
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    ag = sum(max(closes[i] - closes[i - 1], 0) for i in range(1, n + 1)) / n
    al = sum(max(closes[i - 1] - closes[i], 0) for i in range(1, n + 1)) / n
    out[n] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(ch, 0)) / n
        al = (al * (n - 1) + max(-ch, 0)) / n
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(S, K, T, sigma, r, call=True):
    if T <= 0:
        return max(0.0, S - K) if call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        return S * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def years_left(t, expiry):
    return max((expiry - t).total_seconds(), 0) / (365 * 24 * 3600)


def half_spread(mid, cfg):
    return max(cfg["min_spread"], mid * cfg["spread_pct"] / 100) / 2


def gate(daily, date_str, cfg):
    """Uses only daily bars strictly BEFORE date_str."""
    prior = [r["c"] for r in daily if r["d"] < date_str]
    n = cfg["sma_len"]
    if len(prior) < n:
        return None
    sma = sum(prior[-n:]) / n
    rv = None
    k = cfg["rv_days"]
    if len(prior) >= k + 1:
        cl = prior[-(k + 1):]
        rets = [math.log(cl[i] / cl[i - 1]) for i in range(1, len(cl))]
        m = sum(rets) / len(rets)
        rv = math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1) * 252)
    return {"prior_close": prior[-1], "sma200": sma, "uptrend": prior[-1] > sma, "rv": rv}


# ---------- engine ----------

def run(hourly_path, daily_path, cfg_path, events_path, out_path, emit_all=False, now=None):
    cfg = dict(DEFAULT_CONFIG)
    try:
        cfg.update(json.load(open(cfg_path)))
    except FileNotFoundError:
        pass
    try:
        events = json.load(open(events_path)).get("events", {})
    except (FileNotFoundError, ValueError):
        events = {}
    now = now or datetime.now(ET)
    H = load_bars(hourly_path)
    D = load_bars(daily_path)
    trades, status = [], {"tickers": {}, "warnings": []}
    cost = cfg["cost_bps"] / 1e4

    for sym in cfg["tickers"]:
        if sym not in H or sym not in D:
            status["warnings"].append(f"{sym}: bars missing")
            continue
        bars = prep_hourly([b for b in H[sym] if b["sess"] == "RTH"], now)
        pre = {}
        for b in H[sym]:
            if b["sess"] == "PRE":
                x = pre.setdefault(b["t"].date().isoformat(), [b["h"], b["l"]])
                x[0], x[1] = max(x[0], b["h"]), min(x[1], b["l"])
        daily = daily_rows(D[sym])
        done = [b for b in bars if b["done"]]
        rsi = rsi_series([b["c"] for b in done], cfg["rsi_len"])
        for i, b in enumerate(done):
            b["rsi"] = rsi[i]
        sessions = sorted({b["d"] for b in bars})
        by_day = {d: [b for b in bars if b["d"] == d] for d in sessions}
        today = now.date().isoformat()

        def session_closed(d):
            if d < sessions[-1]:
                return True
            end = datetime.fromisoformat(d).replace(hour=16, tzinfo=ET)
            return now >= end and all(x["done"] for x in by_day[d])

        g_last = gate(daily, sessions[-1], cfg)
        status["tickers"][sym] = {
            "last_bar": done[-1]["en"].strftime("%b %d %H:%M") if done else None,
            "session": sessions[-1],
            "uptrend": g_last["uptrend"] if g_last else None,
            "prior_close": round(g_last["prior_close"], 2) if g_last else None,
            "sma200": round(g_last["sma200"], 2) if g_last else None,
        }
        if sessions[-1] != today and now.weekday() < 5 and now.hour >= 11:
            status["warnings"].append(f"{sym}: no bars for today ({today}); last session {sessions[-1]}")

        last_exit = {}   # strategy -> exit date of its latest trade (for the overlap tag)
        need = cfg["tod_sessions"]
        for si, d in enumerate(sessions):
            if si < need:
                continue                      # not enough prior sessions for the volume z
            g = gate(daily, d, cfg)
            if not g:
                continue
            day = by_day[d]
            prev_days = sessions[si - need:si]
            nxt_d = sessions[si + 1] if si + 1 < len(sessions) else None

            def make(strat, k, why, side="long", exit_mode="next"):
                """Signal on bar k of the day (already closed). Entry = bar k+1 open."""
                base = {"id": f"{cfg['version']}-{strat}-{sym}-{d}", "strategy": strat,
                        "ticker": sym, "version": cfg["version"], "date": d, "side": side,
                        "why": why, "signal_time": day[k]["en"].strftime("%H:%M"),
                        "shares": cfg["shares"], "replay": d < cfg["starts"],
                        "uptrend": True}
                if d in events:
                    base["event_entry"] = events[d]
                prev_exit = last_exit.get(strat)
                base["overlap"] = bool(prev_exit and prev_exit >= d)
                if k + 1 >= len(day):
                    if session_closed(d):
                        return None           # signal on the day's last bar: no entry
                    return {**base, "status": "pending",
                            "entry_time": day[k]["en"].strftime("%H:%M")}
                e = day[k + 1]
                S_in = e["o"]
                base.update({"entry_time": e["st"].strftime("%H:%M"), "entry": round(S_in, 2)})
                # estimated option leg
                opt = None
                sgn = 1 if side == "long" else -1
                call = side == "long"
                if g["rv"]:
                    K = math.ceil(S_in * (1 + cfg["otm_pct"] / 100)) if call else math.floor(S_in * (1 - cfg["otm_pct"] / 100))
                    x = e["st"].date() + timedelta(days=cfg["min_dte"])
                    while x.weekday() != 4:
                        x += timedelta(days=1)
                    expiry = datetime(x.year, x.month, x.day, 16, 0, tzinfo=ET)
                    iv = max(cfg["iv_floor"], g["rv"] * cfg["iv_mult"])
                    mid_in = bs_price(S_in, K, years_left(e["st"], expiry), iv, cfg["rate"], call)
                    prem_in = mid_in + half_spread(mid_in, cfg)
                    opt = (K, expiry, iv, prem_in)
                    base.update({"contract": f"{sym} {expiry.strftime('%b %d')} {K} {'C' if call else 'P'}",
                                 "strike": K, "expiry": expiry.date().isoformat(),
                                 "iv_used": round(iv, 4), "opt_entry": round(prem_in, 2),
                                 "contracts": cfg["contracts"], "estimated": True})
                if e["done"]:
                    base["ret_1h_pct"] = round((sgn * (e["c"] - S_in) / S_in - 2 * cost) * 100, 3)
                if strat == "BASE" and session_closed(d):
                    base["ret_day_pct"] = round(((day[-1]["c"] - S_in) / S_in - 2 * cost) * 100, 3)
                x_d = d if exit_mode == "day" else nxt_d
                if exit_mode != "day":
                    last_exit[strat] = nxt_d or "9999-12-31"
                if not x_d or not session_closed(x_d):
                    return {**base, "status": "open"}
                xbars = [x for x in by_day[x_d] if x["done"]]
                S_out = xbars[-1]["c"]
                t_out = xbars[-1]["en"]
                ret = sgn * (S_out - S_in) / S_in - 2 * cost
                base.update({"status": "closed", "exit": round(S_out, 2), "exit_date": x_d,
                             "exit_time": t_out.strftime("%H:%M"),
                             "ret_pct": round(ret * 100, 3),
                             "pnl": round(ret * S_in * cfg["shares"], 2)})
                if x_d in events:
                    base["event_exit"] = events[x_d]
                if opt:
                    K, expiry, iv, prem_in = opt
                    mid_out = bs_price(S_out, K, years_left(t_out, expiry), iv, cfg["rate"], call)
                    prem_out = max(0.0, mid_out - half_spread(mid_out, cfg))
                    per_c = (prem_out - prem_in) * 100 - 2 * cfg["fee"]
                    base.update({"opt_exit": round(prem_out, 2),
                                 "opt_pnl": round(per_c * cfg["contracts"], 2),
                                 "opt_ret_pct": round(per_c / (prem_in * 100) * 100, 1)})
                return base

            if not g["uptrend"]:
                continue
            fired = {}
            for k, b in enumerate(day):
                if not b["done"]:
                    break
                if "ABS" in cfg["strategies"] and "ABS" not in fired:
                    rng = b["h"] - b["l"]
                    clv = ((b["c"] - b["l"]) - (b["h"] - b["c"])) / rng if rng > 0 else 0.0
                    vols = [x["v"] for pd_ in prev_days for x in by_day[pd_] if x["slot"] == b["slot"]]
                    if clv < cfg["abs_clv_max"] and len(vols) >= cfg["tod_min"]:
                        m = sum(vols) / len(vols)
                        sd = math.sqrt(sum((v - m) ** 2 for v in vols) / (len(vols) - 1))
                        z = (b["v"] - m) / sd if sd > 0 else 0.0
                        if z > cfg["abs_z_min"]:
                            fired["ABS"] = make("ABS", k,
                                f"Seller absorption {b['slot']} bar: close near low (imbalance {clv:.2f}), "
                                f"volume z {z:.2f} vs same hour on prior {len(vols)} days, above 200-day avg")
                if "RSI5" in cfg["strategies"] and "RSI5" not in fired and sym in cfg["rsi_tickers"]:
                    if b.get("rsi") is not None and b["rsi"] < cfg["rsi_lo"]:
                        fired["RSI5"] = make("RSI5", k,
                            f"Hourly RSI(5) {b['rsi']:.1f} below {cfg['rsi_lo']:.0f} on the {b['slot']} bar, above 200-day avg")
                if sym in cfg["lvl_tickers"] and ("LVLS" in cfg["strategies"] or "LVLL" in cfg["strategies"]):
                    rng = b["h"] - b["l"]
                    press = (b["c"] - b["o"]) / rng if rng > 0 else 0.0
                    side = "short" if press >= cfg["lvl_press"] else "long" if press <= -cfg["lvl_press"] else None
                    strat = {"short": "LVLS", "long": "LVLL"}.get(side)
                    if strat and strat in cfg["strategies"] and strat not in fired:
                        vols = [x["v"] for pd_ in prev_days for x in by_day[pd_] if x["slot"] == b["slot"]]
                        z = None
                        if len(vols) >= cfg["tod_min"]:
                            m = sum(vols) / len(vols)
                            sd = math.sqrt(sum((v - m) ** 2 for v in vols) / (len(vols) - 1))
                            z = (b["v"] - m) / sd if sd > 0 else 0.0
                        if z is not None and z > cfg["lvl_z_min"]:
                            pb = by_day[prev_days[-1]]
                            lv = [("yesterday's high", max(x["h"] for x in pb)), ("yesterday's low", min(x["l"] for x in pb))]
                            if d in pre:
                                lv += [("premarket high", pre[d][0]), ("premarket low", pre[d][1])]
                            if k > 0:
                                lv += [("today's high", max(x["h"] for x in day[:k])), ("today's low", min(x["l"] for x in day[:k]))]
                            dc = [r["c"] for r in daily if r["d"] < d]
                            if len(dc) >= 50:
                                lv.append(("50-day average", sum(dc[-50:]) / 50))
                            pv = sum((x["h"] + x["l"] + x["c"]) / 3 * x["v"] for x in day[:k + 1])
                            vv = sum(x["v"] for x in day[:k + 1])
                            if vv > 0:
                                lv.append(("VWAP (hourly est.)", pv / vv))
                            step = cfg["lvl_round"]
                            lv.append(("round number", round(b["c"] / step) * step))
                            wk = cfg.get("levels_week") or ""
                            if wk and wk <= d <= (datetime.fromisoformat(wk) + timedelta(days=4)).date().isoformat():
                                lv += [("your weekly level", float(x)) for x in cfg.get("levels", [])]
                            tol = b["c"] * cfg["lvl_near_pct"] / 100
                            hits = []
                            for name, L in lv:
                                right_side = L >= b["c"] if side == "short" else L <= b["c"]
                                near = abs(L - b["c"]) <= tol or b["l"] <= L <= b["h"]
                                if right_side and near:
                                    hits.append((abs(L - b["c"]), name, L))
                            if hits:
                                _, name, L = min(hits)
                                who = "buyers" if side == "short" else "sellers"
                                kind = "ceiling" if side == "short" else "floor"
                                fired[strat] = make(strat, k,
                                    f"{side.upper()} setup: {who} pressing ({press:+.2f}) into {kind} {name} {L:.2f}, "
                                    f"bar close {b['c']:.2f}, volume z {z:.2f}, above 200-day avg",
                                    side=side, exit_mode="day")
                                if fired[strat]:
                                    fired[strat]["level"] = name
                if "BASE" in cfg["strategies"] and "BASE" not in fired:
                    if k == 0 and b["slot"] == "09:30":
                        fired["BASE"] = make("BASE", k, "Benchmark: bought every uptrend day at 10:30, held to next close")
            trades += [t for t in fired.values() if t]

    all_last = [v["session"] for v in status["tickers"].values()]
    if all_last and not emit_all:
        cut = (datetime.fromisoformat(max(all_last)) - timedelta(days=cfg["write_days"])).date().isoformat()
        trades = [t for t in trades if t["date"] >= cut]
        status["write_cutoff"] = cut
    lb = [v["last_bar"] for v in status["tickers"].values() if v["last_bar"]]
    status["bars_through"] = lb[0] if lb else "no bars"
    status["last_run"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    json.dump({"trades": trades, "status": status}, open(out_path, "w"), indent=1)
    return trades, status


if __name__ == "__main__":
    a = sys.argv[1:]
    emit_all = "--all" in a
    if emit_all:
        a.remove("--all")
    now = None
    if "--now" in a:
        k = a.index("--now")
        now = datetime.fromisoformat(a[k + 1]).replace(tzinfo=ET)
        del a[k:k + 2]
    ts, st = run(*a[:5], emit_all=emit_all, now=now)
    print(json.dumps(st))
    for t in ts:
        if t["strategy"] == "BASE":
            continue
        print(t["date"], t["ticker"], t["strategy"], t["status"], t.get("entry_time", ""),
              t.get("side", ""), t.get("entry", ""), "->", t.get("exit", ""), t.get("ret_pct", ""), t.get("opt_pnl", ""))
    print(f"{len(ts)} trades incl. benchmark")
