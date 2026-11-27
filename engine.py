"""
Overnight Desk engine: SIMULATED trades only. It never places an order.

What it tests (the two candidates that survived out-of-sample in the project):
  ABS  : hourly seller absorption. CLV < -0.40 and time-of-day volume z > 1.0
  RSI5 : hourly RSI(5) < 20
Both: long only, prior-day close above the 200-day average, first fire per
day per ticker, entry at the NEXT hourly bar's open, exit at the NEXT
session's close. No stop, no target, no adding.
  Every trade is also scored with a scale-out exit plan (so_* fields): see scale_out().
  LVLS/LVLL : level alerts (Muse's rules), uptrend days only, exit same-day close.
  LVLSX/LVLLX : the SAME level rules with NO 200-day gate (added 2026-11-26),
         scored separately so the gate can be judged. BASEX = 10:30-to-close
         benchmark on every day, for these two. Fires before x_starts are replay.
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
    "strategies": ["ABS", "RSI5", "BASE", "LVLS", "LVLL", "LVLSX", "LVLLX", "BASEX"],
    "x_starts": "2026-11-27",    # ungated level variants count as live from this date
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
    # scale-out exit plan, scored ALONGSIDE the hold-to-close exit (same entries)
    "so_on": True,
    "so_min_gap_pct": 0.15,      # a target level must be at least this far from the last one
    "so_max_pct": 4.0,           # look for target levels within this distance
    "so_round": 5.0,             # round-number levels every $5 for targets
    "so_pessimistic": True,      # bar touches stop and target: count the stop first
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


# ---------- scale-out exit plan ----------
#  1. Stop: beyond the signal bar's extreme (long: its low).
#  2. Target 1: next level in the trade's direction. Sell half there.
#  3. Then the stop on the other half moves to the entry price.
#  4. Runner: out at the next level after target 1, or when an hourly bar
#     closes below the previous bar's low (long; mirrored for shorts).
#  5. Whatever is left exits at the same time as the hold-to-close version.
# Hourly bars only: when one bar touches both a stop and a target, the stop
# is assumed first (pessimistic) and the trade is flagged so_amb.

def scale_out(path, S_in, side, stop, t1, t2, last_bar, prev_bar, pess=True):
    L = side == "long"
    fav = lambda p: ((p - S_in) if L else (S_in - p)) / S_in
    adv_hit = lambda b, lv: b["l"] <= lv if L else b["h"] >= lv
    fav_hit = lambda b, lv: b["h"] >= lv if L else b["l"] <= lv
    adv_gap = lambda b, lv: b["o"] <= lv if L else b["o"] >= lv
    fav_gap = lambda b, lv: b["o"] >= lv if L else b["o"] <= lv
    pos, real, hit1, st, steps, reason, amb = 1.0, 0.0, False, stop, [], None, False
    prev = prev_bar
    def when(b): return b["st"].strftime("%a %H:%M")
    for b in path:
        done = False
        if not hit1:
            if adv_gap(b, st):
                real += pos * fav(b["o"]); steps.append(f"all out {b['o']:.2f} (gapped through stop) {when(b)}"); reason = "stop"; pos = 0; done = True
            elif fav_gap(b, t1):
                real += 0.5 * fav(b["o"]); pos = 0.5; hit1 = True; st = S_in
                steps.append(f"half out {b['o']:.2f} (opened past target 1) {when(b)}")
                if fav_gap(b, t2) or fav_hit(b, t2):
                    px = b["o"] if fav_gap(b, t2) else t2
                    if adv_hit(b, S_in) and not fav_gap(b, t2): amb = True; px = S_in
                    real += 0.5 * fav(px); steps.append(f"rest out {px:.2f} {when(b)}"); reason = "target 2" if px != S_in else "stop at entry"; pos = 0; done = True
                elif adv_hit(b, S_in):
                    real += 0; steps.append(f"rest out {S_in:.2f} (stop at entry) {when(b)}"); reason = "stop at entry"; pos = 0; done = True; amb = True
            elif adv_hit(b, st) and (pess or not fav_hit(b, t1)):
                if fav_hit(b, t1): amb = True
                real += pos * fav(st); steps.append(f"all out {st:.2f} (stop) {when(b)}"); reason = "stop"; pos = 0; done = True
            elif fav_hit(b, t1):
                real += 0.5 * fav(t1); pos = 0.5; hit1 = True; st = S_in
                steps.append(f"half out {t1:.2f} (target 1) {when(b)}")
                if adv_hit(b, S_in) and pess:
                    amb = True; steps.append(f"rest out {S_in:.2f} (stop at entry) {when(b)}"); reason = "stop at entry"; pos = 0; done = True
                elif fav_hit(b, t2):
                    real += 0.5 * fav(t2); steps.append(f"rest out {t2:.2f} (target 2) {when(b)}"); reason = "target 2"; pos = 0; done = True
                prev = b
                if not done and b is last_bar:
                    real += pos * fav(b["c"]); steps.append(f"rest out {b['c']:.2f} (time) {when(b)}"); reason = "target 1 + time"; pos = 0; done = True
                if done: break
                continue
        else:
            if adv_gap(b, st):
                real += pos * fav(b["o"]); steps.append(f"rest out {b['o']:.2f} (gapped through entry) {when(b)}"); reason = "stop at entry"; pos = 0; done = True
            elif fav_gap(b, t2):
                real += pos * fav(b["o"]); steps.append(f"rest out {b['o']:.2f} (opened past target 2) {when(b)}"); reason = "target 2"; pos = 0; done = True
            elif adv_hit(b, st) and (pess or not fav_hit(b, t2)):
                if fav_hit(b, t2): amb = True
                real += pos * fav(st); steps.append(f"rest out {st:.2f} (stop at entry) {when(b)}"); reason = "stop at entry"; pos = 0; done = True
            elif fav_hit(b, t2):
                real += pos * fav(t2); steps.append(f"rest out {t2:.2f} (target 2) {when(b)}"); reason = "target 2"; pos = 0; done = True
            elif prev is not None and ((b["c"] < prev["l"]) if L else (b["c"] > prev["h"])):
                real += pos * fav(b["c"]); steps.append(f"rest out {b['c']:.2f} (trail) {when(b)}"); reason = "trail"; pos = 0; done = True
        if done:
            break
        if b is last_bar:
            real += pos * fav(b["c"]); steps.append(f"{'rest' if hit1 else 'all'} out {b['c']:.2f} (time) {when(b)}")
            reason = ("target 1 + time" if hit1 else "time"); pos = 0
            break
        prev = b
    return {"real": real, "closed": pos == 0, "steps": steps, "reason": reason, "hit1": hit1, "amb": amb}


# ---------- engine ----------

def run(hourly_path, daily_path, cfg_path, events_path, out_path, emit_all=False, now=None, logged=None):
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
    settled = []
    logged = logged or []
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

        gidx = {id(b): i for i, b in enumerate(bars)}
        last_bar_of = {d_: [b for b in by_day[d_] if b["done"]][-1] for d_ in sessions if any(b["done"] for b in by_day[d_])}

        def target_levels(d_, k_, px):
            """Levels known at the close of bar k_ on day d_ (no lookahead)."""
            si_ = sessions.index(d_)
            day_ = by_day[d_]
            lv = []
            if si_ > 0:
                pb = by_day[sessions[si_ - 1]]
                lv += [("yesterday's high", max(x["h"] for x in pb)), ("yesterday's low", min(x["l"] for x in pb)),
                       ("yesterday's close", pb[-1]["c"])]
            if d_ in pre:
                lv += [("premarket high", pre[d_][0]), ("premarket low", pre[d_][1])]
            lv += [("today's high", max(x["h"] for x in day_[:k_ + 1])), ("today's low", min(x["l"] for x in day_[:k_ + 1])),
                   ("today's open", day_[0]["o"])]
            dc = [r["c"] for r in daily if r["d"] < d_]
            if len(dc) >= 50:
                lv.append(("50-day average", sum(dc[-50:]) / 50))
            if len(dc) >= 200:
                lv.append(("200-day average", sum(dc[-200:]) / 200))
            stp = cfg["so_round"]
            base_ = math.floor(px / stp) * stp
            lv += [("round number", base_ + j * stp) for j in range(-12, 14)]
            wk = cfg.get("levels_week") or ""
            if sym in cfg["lvl_tickers"] and wk and wk <= d_ <= (datetime.fromisoformat(wk) + timedelta(days=4)).date().isoformat():
                lv += [("your weekly level", float(x)) for x in cfg.get("levels", [])]
            return lv

        def plan_for(d_, k_, S_in, side):
            sig = by_day[d_][k_]
            L = side == "long"
            raw = sig["l"] if L else sig["h"]
            ok = raw < S_in * (1 - 0.0005) if L else raw > S_in * (1 + 0.0005)
            stop = raw if ok else (S_in * (1 - 0.0025) if L else S_in * (1 + 0.0025))
            gap = cfg["so_min_gap_pct"] / 100; far = cfg["so_max_pct"] / 100
            lv = target_levels(d_, k_, S_in)
            if L:
                c1 = sorted([x for x in lv if S_in * (1 + gap) <= x[1] <= S_in * (1 + far)], key=lambda x: x[1])
            else:
                c1 = sorted([x for x in lv if S_in * (1 - far) <= x[1] <= S_in * (1 - gap)], key=lambda x: -x[1])
            R = abs(S_in - stop)
            if c1:
                t1n, t1 = c1[0]
            else:
                t1n, t1 = "1R", (S_in + R if L else S_in - R)
            c2 = [x for x in c1 if (x[1] >= t1 * (1 + gap) if L else x[1] <= t1 * (1 - gap))]
            if c2:
                t2n, t2 = c2[0]
            else:
                t2n, t2 = "2x target 1", (t1 + (t1 - S_in) if L else t1 - (S_in - t1))
            return stop, t1, t2, t1n, t2n

        def run_plan(base, d_, k_, e, S_in, side, x_d):
            if not cfg.get("so_on"):
                return
            stop, t1, t2, t1n, t2n = plan_for(d_, k_, S_in, side)
            gi = gidx[id(e)]
            if x_d and x_d in last_bar_of and session_closed(x_d):
                lb = last_bar_of[x_d]; end = gidx[id(lb)]
            else:
                lb = None; end = len(bars) - 1
            path = [b for b in bars[gi:end + 1] if b["done"]]
            prev_b = bars[gi - 1] if gi > 0 else None
            r = scale_out(path, S_in, side, stop, t1, t2, lb, prev_b, pess=cfg.get("so_pessimistic", True))
            base.update({"so_stop": round(stop, 2), "so_t1": round(t1, 2), "so_t2": round(t2, 2),
                         "so_t1_name": t1n, "so_t2_name": t2n, "so_steps": r["steps"],
                         "so_hit1": r["hit1"], "so_amb": r["amb"]})
            if r["closed"]:
                ret = r["real"] - 2 * cost
                base.update({"so_status": "closed", "so_exit_reason": r["reason"],
                             "so_ret_pct": round(ret * 100, 3), "so_pnl": round(ret * S_in * cfg["shares"], 2)})
            else:
                base.update({"so_status": "open"})

        last_exit = {}   # strategy -> exit date of its latest trade (for the overlap tag)
        need = cfg["tod_sessions"]
        for si, d in enumerate(sessions):
            if si < need:
                continue                      # not enough prior sessions for the volume z
            g = gate(daily, d, cfg)
            if not g:
                continue
            up = bool(g["uptrend"])
            day = by_day[d]
            prev_days = sessions[si - need:si]
            nxt_d = sessions[si + 1] if si + 1 < len(sessions) else None

            def make(strat, k, why, side="long", exit_mode="next"):
                """Signal on bar k of the day (already closed). Entry = bar k+1 open."""
                base = {"id": f"{cfg['version']}-{strat}-{sym}-{d}", "strategy": strat,
                        "ticker": sym, "version": cfg["version"], "date": d, "side": side,
                        "why": why, "signal_time": day[k]["en"].strftime("%H:%M"),
                        "shares": cfg["shares"],
                        "replay": d < (cfg.get("x_starts", cfg["starts"]) if strat.endswith("X") else cfg["starts"]),
                        "uptrend": up}
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
                run_plan(base, d, k, e, S_in, side, d if exit_mode == "day" else nxt_d)
                if strat in ("BASE", "BASEX") and session_closed(d):
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

            fired = {}
            for k, b in enumerate(day):
                if not b["done"]:
                    break
                if "BASEX" in cfg["strategies"] and "BASEX" not in fired and sym in cfg["lvl_tickers"]:
                    if k == 0 and b["slot"] == "09:30":
                        fired["BASEX"] = make("BASEX", k, "Benchmark for ungated level alerts: every day 10:30 to close", exit_mode="day")
                if up and "ABS" in cfg["strategies"] and "ABS" not in fired:
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
                if up and "RSI5" in cfg["strategies"] and "RSI5" not in fired and sym in cfg["rsi_tickers"]:
                    if b.get("rsi") is not None and b["rsi"] < cfg["rsi_lo"]:
                        fired["RSI5"] = make("RSI5", k,
                            f"Hourly RSI(5) {b['rsi']:.1f} below {cfg['rsi_lo']:.0f} on the {b['slot']} bar, above 200-day avg")
                if sym in cfg["lvl_tickers"] and any(x in cfg["strategies"] for x in ("LVLS", "LVLL", "LVLSX", "LVLLX")):
                    rng = b["h"] - b["l"]
                    press = (b["c"] - b["o"]) / rng if rng > 0 else 0.0
                    side = "short" if press >= cfg["lvl_press"] else "long" if press <= -cfg["lvl_press"] else None
                    base_s = {"short": "LVLS", "long": "LVLL"}.get(side)
                    want = [x for x in ([base_s] if up else []) + ([base_s + "X"] if base_s else [])
                            if x and x in cfg["strategies"] and x not in fired]
                    if want:
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
                                for strat in want:
                                    fired[strat] = make(strat, k,
                                        f"{side.upper()} setup: {who} pressing ({press:+.2f}) into {kind} {name} {L:.2f}, "
                                        f"bar close {b['c']:.2f}, volume z {z:.2f}, {'above' if up else 'BELOW'} 200-day avg"
                                        + ("" if up or not strat.endswith("X") else " (ungated variant)"),
                                        side=side, exit_mode="day")
                                    if fired[strat]:
                                        fired[strat]["level"] = name
                if up and "BASE" in cfg["strategies"] and "BASE" not in fired:
                    if k == 0 and b["slot"] == "09:30":
                        fired["BASE"] = make("BASE", k, "Benchmark: bought every uptrend day at 10:30, held to next close")
            trades += [t for t in fired.values() if t]

        # settle logged open trades this run did not re-emit (e.g. the data provider revised a bar)
        emitted = {t["id"] for t in trades}
        for lt in logged:
            if lt.get("ticker") != sym or lt.get("status") != "open" or lt["id"] in emitted or lt.get("entry") is None:
                continue
            d0 = lt["date"]
            if d0 not in by_day:
                continue
            eb = [b for b in by_day[d0] if b["st"].strftime("%H:%M") == lt.get("entry_time")]
            if not eb:
                continue
            e0 = eb[0]; S0 = float(lt["entry"]); side0 = lt.get("side", "long")
            sg0 = 1 if side0 == "long" else -1
            si0 = sessions.index(d0)
            xd0 = d0 if lt["strategy"] in ("LVLL", "LVLS", "LVLLX", "LVLSX", "BASEX") else (sessions[si0 + 1] if si0 + 1 < len(sessions) else None)
            if not xd0 or not session_closed(xd0):
                continue
            xb = [x for x in by_day[xd0] if x["done"]]
            S_out = xb[-1]["c"]; t_out = xb[-1]["en"]
            ret0 = sg0 * (S_out - S0) / S0 - 2 * cost
            nt = dict(lt)
            nt.update({"status": "closed", "exit": round(S_out, 2), "exit_date": xd0, "exit_time": t_out.strftime("%H:%M"),
                       "ret_pct": round(ret0 * 100, 3), "pnl": round(ret0 * S0 * cfg["shares"], 2), "settled": True})
            if lt.get("strike") and lt.get("expiry") and lt.get("iv_used") and lt.get("opt_entry") is not None:
                exp0 = datetime.fromisoformat(lt["expiry"]).replace(hour=16, tzinfo=ET)
                call0 = side0 == "long"
                mid0 = bs_price(S_out, lt["strike"], years_left(t_out, exp0), lt["iv_used"], cfg["rate"], call0)
                po = max(0.0, mid0 - half_spread(mid0, cfg))
                pc = (po - lt["opt_entry"]) * 100 - 2 * cfg["fee"]
                nt.update({"opt_exit": round(po, 2), "opt_pnl": round(pc * lt.get("contracts", 1), 2),
                           "opt_ret_pct": round(pc / (lt["opt_entry"] * 100) * 100, 1)})
            k0 = by_day[d0].index(e0) - 1
            if k0 >= 0:
                run_plan(nt, d0, k0, e0, S0, side0, xd0)
            settled.append(nt)

    all_last = [v["session"] for v in status["tickers"].values()]
    if all_last and not emit_all:
        cut = (datetime.fromisoformat(max(all_last)) - timedelta(days=cfg["write_days"])).date().isoformat()
        trades = [t for t in trades if t["date"] >= cut]
        status["write_cutoff"] = cut
    trades += settled
    if settled:
        status["settled"] = [t["id"] for t in settled]
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
        if t["strategy"] in ("BASE", "BASEX"):
            continue
        print(t["date"], t["ticker"], t["strategy"], t["status"], t.get("entry_time", ""),
              t.get("side", ""), t.get("entry", ""), "->", t.get("exit", ""), t.get("ret_pct", ""), t.get("opt_pnl", ""))
    print(f"{len(ts)} trades incl. benchmark")
