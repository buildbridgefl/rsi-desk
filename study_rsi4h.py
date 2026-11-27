"""
4-hour RSI + weekly options study (SIMULATED, estimated option prices).

Question: buy a call when the 4-hour RSI turns oversold, a put when it turns
overbought, using an option about 5 trading days from expiry. What happened?

Data: Yahoo hourly bars (regular hours), last 730 days. 4-hour bars are built
the TradingView way for stocks: 9:30-13:30 and 13:30-16:00 each day.

Signal: RSI crosses INTO the zone on a closed 4-hour bar (first touch only).
Entry: open of the NEXT 4-hour bar. Nothing after the signal bar is used.
Option: strike nearest the entry price, expiry 5 sessions after entry, priced
with Black-Scholes using 20-day realized vol x 1.2 (floor 12%) as IV, plus a
1.5% spread (min $0.02) and $0.65 fee per side. NO REAL OPTION QUOTES. Real
fills on fast days are worse than this.

Exits tested (same entries):
  next  : close of the next session (allowed under Alex's 60-day rule exemption)
  3d    : close 3 sessions after entry
  expiry: held to the expiry close (5 sessions)

Benchmark: the same option, same side, same exit, bought at EVERY 4-hour bar
open. A signal has to beat that, not zero. p = share of 5000 random draws of
the same number of benchmark trades whose average was as good or better.

Windows: TEST = last 6 months (what Alex asked for). CHECK = the 18 months
before it, to see whether the result holds outside the period we looked at.

Usage: python study_rsi4h.py            (writes rsi4h_results.md, rsi4h_trades.csv)
"""
import csv, math, random, sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
TICKERS = ["SPY", "QQQ"]
VARIANTS = [("RSI(14) 30/70", 14, 30, 70), ("RSI(5) 20/80", 5, 20, 80)]
EXITS = ["next", "3d", "expiry"]
DTE = 5
IV_MULT, IV_FLOOR, RATE = 1.2, 0.12, 0.04
SPREAD_PCT, MIN_SPREAD, FEE = 1.5, 0.02, 0.65
TEST_MONTHS = 6
HOLIDAYS = {"2026-11-26", "2026-12-25", "2027-01-01"}


def ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs(S, K, T, s, r, call):
    if T <= 0:
        return max(0.0, S - K) if call else max(0.0, K - S)
    d1 = (math.log(S / K) + (r + s * s / 2) * T) / (s * math.sqrt(T))
    d2 = d1 - s * math.sqrt(T)
    return S * ncdf(d1) - K * math.exp(-r * T) * ncdf(d2) if call else K * math.exp(-r * T) * ncdf(-d2) - S * ncdf(-d1)


def half_spread(mid):
    return max(MIN_SPREAD, mid * SPREAD_PCT / 100) / 2


def wilder_rsi(closes, n):
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    g = l = 0.0
    for i in range(1, n + 1):
        ch = closes[i] - closes[i - 1]
        g += max(ch, 0); l += max(-ch, 0)
    g /= n; l /= n
    out[n] = 100 - 100 / (1 + g / l) if l > 0 else 100.0
    for i in range(n + 1, len(closes)):
        ch = closes[i] - closes[i - 1]
        g = (g * (n - 1) + max(ch, 0)) / n
        l = (l * (n - 1) + max(-ch, 0)) / n
        out[i] = 100 - 100 / (1 + g / l) if l > 0 else 100.0
    return out


def fetch(tk):
    import yfinance as yf
    h = yf.Ticker(tk).history(period="730d", interval="1h", prepost=False, auto_adjust=False)
    d = yf.Ticker(tk).history(period="5y", interval="1d", auto_adjust=False)
    if h.empty or d.empty:
        raise RuntimeError(f"{tk}: no bars from Yahoo")
    hourly = []
    for ts, r in h.iterrows():
        t = ts.tz_convert(ET)
        hm = t.hour * 60 + t.minute
        if hm < 9 * 60 + 30 or hm >= 16 * 60:
            continue
        hourly.append({"t": t, "d": t.date().isoformat(), "o": float(r.Open), "h": float(r.High),
                       "l": float(r.Low), "c": float(r.Close)})
    daily = [{"d": (ts.tz_convert(ET) if ts.tzinfo else ts).date().isoformat(), "c": float(r.Close)}
             for ts, r in d.iterrows()]
    return hourly, daily


def four_hour(hourly):
    """9:30-13:30 and 13:30-16:00 buckets. Returns bars with open time, o/h/l/c."""
    out, cur, key = [], None, None
    for b in hourly:
        k = (b["d"], 0 if b["t"].hour * 60 + b["t"].minute < 13 * 60 + 30 else 1)
        if k != key:
            if cur:
                out.append(cur)
            cur = {"d": b["d"], "half": k[1], "t": b["t"], "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"]}
            key = k
        else:
            cur["h"] = max(cur["h"], b["h"]); cur["l"] = min(cur["l"], b["l"]); cur["c"] = b["c"]
    if cur:
        out.append(cur)
    return out


def rv20(daily, d):
    prior = [x["c"] for x in daily if x["d"] < d][-21:]
    if len(prior) < 21:
        return None
    rets = [math.log(prior[i] / prior[i - 1]) for i in range(1, len(prior))]
    m = sum(rets) / len(rets)
    return math.sqrt(sum((x - m) ** 2 for x in rets) / (len(rets) - 1) * 252)


def sma200(daily, d):
    prior = [x["c"] for x in daily if x["d"] < d]
    return sum(prior[-200:]) / 200 if len(prior) >= 200 else None


def simulate(bars4, daily, i, side, sessions, close_of):
    """Enter at the open of 4h bar i. Returns dict of results per exit, or None."""
    e = bars4[i]
    d_in = e["d"]
    si = sessions.index(d_in)
    sessions = list(sessions)
    while si + DTE >= len(sessions):          # future weekdays for the expiry date only
        x = datetime.fromisoformat(sessions[-1]).date() + timedelta(days=1)
        while x.weekday() >= 5 or x.isoformat() in HOLIDAYS:
            x += timedelta(days=1)
        sessions.append(x.isoformat())
    vol = rv20(daily, d_in)
    if vol is None:
        return None
    iv = max(IV_FLOOR, vol * IV_MULT)
    S = e["o"]; K = round(S); call = side == "long"
    exp_d = sessions[si + DTE]
    expiry = datetime.fromisoformat(exp_d).replace(hour=16, tzinfo=ET)
    T = (expiry - e["t"]).total_seconds() / (365 * 86400)
    mid = bs(S, K, T, iv, RATE, call)
    if mid < 0.05:
        return None
    p_in = mid + half_spread(mid)
    res = {"date": d_in, "time": e["t"].strftime("%H:%M"), "side": side, "entry": round(S, 2), "strike": K,
           "expiry": exp_d, "iv": round(iv, 3), "opt_in": round(p_in, 2)}
    for x, off in (("next", 1), ("3d", 3), ("expiry", DTE)):
        xd = sessions[si + off]
        if xd not in close_of:                # exit day not reached yet: leave unscored
            res[f"{x}_exit"] = res[f"{x}_pnl"] = res[f"{x}_ret"] = res[f"{x}_und"] = None
            continue
        So = close_of[xd]
        t_out = datetime.fromisoformat(xd).replace(hour=16, tzinfo=ET)
        m_out = bs(So, K, (expiry - t_out).total_seconds() / (365 * 86400), iv, RATE, call)
        p_out = max(0.0, m_out - half_spread(m_out)) if m_out > 0 else 0.0
        pnl = (p_out - p_in) * 100 - 2 * FEE
        res[f"{x}_exit"] = round(So, 2)
        res[f"{x}_pnl"] = round(pnl, 2)
        res[f"{x}_ret"] = round(pnl / (p_in * 100) * 100, 1)
        res[f"{x}_und"] = round((1 if call else -1) * (So - S) / S * 100, 2)
    return res


def summarize(rows, bench, x, rng):
    rows = [t for t in rows if t[f"{x}_ret"] is not None]
    bench = [t for t in bench if t[f"{x}_ret"] is not None]
    if not rows:
        return None
    r = [t[f"{x}_ret"] for t in rows]
    b = [t[f"{x}_ret"] for t in bench]
    avg = sum(r) / len(r)
    bavg = sum(b) / len(b) if b else float("nan")
    p = None
    if len(b) >= len(r) > 0:
        hits = sum(1 for _ in range(5000) if sum(rng.sample(b, len(r))) / len(r) >= avg)
        p = hits / 5000
    srt = sorted(r)
    return {"n": len(r), "win": sum(1 for v in r if v > 0) / len(r) * 100, "avg": avg,
            "med": srt[len(srt) // 2], "pnl": sum(t[f"{x}_pnl"] for t in rows),
            "und": sum(t[f"{x}_und"] for t in rows) / len(rows), "bavg": bavg, "p": p}


def main():
    rng = random.Random(7)
    all_rows, lines = [], []
    now = datetime.now(ET)
    test_start = (now - timedelta(days=round(TEST_MONTHS * 30.4))).date().isoformat()
    lines += ["# 4-hour RSI + 5-day options study", "",
              f"Run {now:%Y-%m-%d %H:%M} ET. Test window: {test_start} to today. Check window: the 18 months before.",
              "Option prices are ESTIMATED (Black-Scholes, realized vol x 1.2, 1.5% spread, $0.65/side). Not real quotes.",
              "Returns are % on the premium paid, per trade. $ is per 1 contract.", ""]
    for tk in TICKERS:
        hourly, daily = fetch(tk)
        bars4 = four_hour(hourly)
        sessions = sorted({b["d"] for b in hourly})
        close_of = {}
        for b in hourly:
            close_of[b["d"]] = b["c"]
        closes = [b["c"] for b in bars4]
        # benchmark: every 4h bar open, both sides
        bench = {"long": [], "short": []}
        for i in range(1, len(bars4)):
            for side in ("long", "short"):
                s = simulate(bars4, daily, i, side, sessions, close_of)
                if s:
                    bench[side].append(s)
        for name, n, lo, hi in VARIANTS:
            rsi = wilder_rsi(closes, n)
            sig = []
            for i in range(n + 1, len(bars4) - 1):
                a, b = rsi[i - 1], rsi[i]
                if a is None or b is None:
                    continue
                side = "long" if (b < lo <= a) else "short" if (b > hi >= a) else None
                if not side:
                    continue
                s = simulate(bars4, daily, i + 1, side, sessions, close_of)
                if s:
                    sm = sma200(daily, s["date"])
                    s.update({"ticker": tk, "variant": name, "rsi": round(b, 1),
                              "above200": None if sm is None else s["entry"] > sm})
                    sig.append(s)
            all_rows += sig
            lines += [f"## {tk} — {name}", "",
                      "| Window | Side | Exit | Trades | Win % | Avg % | Median % | $ total (1 ct) | Underlying avg | Benchmark avg % | p |",
                      "|---|---|---|---|---|---|---|---|---|---|---|"]
            for wname, cond in (("TEST (6 mo)", lambda d: d >= test_start), ("CHECK (18 mo before)", lambda d: d < test_start)):
                for side in ("long", "short"):
                    rows = [t for t in sig if t["side"] == side and cond(t["date"])]
                    bb = [t for t in bench[side] if cond(t["date"])]
                    for x in EXITS:
                        s = summarize(rows, bb, x, rng)
                        if not s:
                            lines.append(f"| {wname} | {'calls' if side == 'long' else 'puts'} | {x} | 0 | | | | | | | |")
                            continue
                        pv = "" if s["p"] is None else f"{s['p']:.3f}"
                        lines.append(f"| {wname} | {'calls' if side == 'long' else 'puts'} | {x} | {s['n']} | {s['win']:.0f} | "
                                     f"{s['avg']:+.1f} | {s['med']:+.1f} | {s['pnl']:+,.0f} | {s['und']:+.2f}% | "
                                     f"{s['bavg']:+.1f} | {pv} |")
            lines.append("")
    lines += ["## How to read this", "",
              "- Under 30 trades in a row means nothing yet, whatever the win rate.",
              "- 'Benchmark avg' is the same option bought at every 4-hour bar. Buying short-dated options loses to decay and spread on average, so the benchmark is usually negative. Beating it is the bar, not beating zero.",
              "- p under 0.05 is only interesting if the CHECK window agrees. This table has 48 cells; about 2 will look significant by luck.",
              "- A 100% win rate or a huge average on a handful of trades is a red flag, not a result.",
              "- Only the 'next' exit fits the 60-day rule exemption (SPY/QQQ sold the same or next session)."]
    open("rsi4h_results.md", "w").write("\n".join(lines) + "\n")
    if all_rows:
        keys = list(all_rows[0].keys())
        with open("rsi4h_trades.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(all_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
