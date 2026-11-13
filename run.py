#!/usr/bin/env python3
"""
Overnight Desk runner for GitHub Actions. SIMULATED paper trades only:
it never places, changes or cancels an order. No Claude usage.

Each run:
  1. downloads SPY/QQQ hourly bars (with premarket) and daily bars from Yahoo,
  2. runs overnight/engine.py (same rules as the Claude version),
  3. merges new or changed trades into docs/trades.json (never deletes),
  4. writes docs/status.json and docs/config.json for the page,
  5. pushes a phone notification through ntfy for new signals / results.

    NTFY_TOPIC=your-topic python overnight/run.py
Test without the internet, from saved Webull JSON:
    python overnight/run.py --webull hourly.json daily.json --now 2026-11-12T16:40
"""
import argparse, csv, json, os, sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DOCS = os.path.join(ROOT, "docs")
sys.path.insert(0, HERE)
import engine  # noqa: E402

ET = ZoneInfo("America/New_York")
NAMES = {"ABS": "absorption", "RSI5": "RSI dip", "LVLL": "level alert", "LVLS": "level alert"}


def fetch_yahoo(tickers, hourly_csv, daily_csv):
    import yfinance as yf
    with open(hourly_csv, "w", newline="") as fh, open(daily_csv, "w", newline="") as fd:
        wh, wd = csv.writer(fh), csv.writer(fd)
        for tk in tickers:
            h = yf.Ticker(tk).history(period="90d", interval="1h", prepost=True, auto_adjust=False)
            d = yf.Ticker(tk).history(period="2y", interval="1d", auto_adjust=False)
            if h.empty or d.empty:
                raise RuntimeError(f"{tk}: Yahoo returned no bars")
            for ts, r in h.iterrows():
                t = ts.tz_convert(ET)
                hm = t.hour * 60 + t.minute
                if hm >= 16 * 60:
                    continue                      # after hours: not used
                sess = "PRE" if hm < 9 * 60 + 30 else "RTH"
                wh.writerow([tk, t.astimezone(timezone.utc).isoformat(), r.Open, r.High, r.Low, r.Close, r.Volume, sess])
            for ts, r in d.iterrows():
                t = ts.tz_convert(ET) if ts.tzinfo else ts.tz_localize(ET)
                wd.writerow([tk, t.replace(hour=12).astimezone(timezone.utc).isoformat(), r.Open, r.High, r.Low, r.Close, r.Volume, "RTH"])


def notify(title, body):
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print(f"[no NTFY_TOPIC] {title}\n{body}")
        return
    import requests
    try:
        requests.post(f"https://ntfy.sh/{topic}", data=body.encode(),
                      headers={"Title": title, "Priority": "high", "Tags": "chart_with_upwards_trend"}, timeout=15)
    except Exception as e:  # a failed push must not lose the trade log
        print(f"[warn] ntfy: {e}", file=sys.stderr)


def line_signal(t):
    if t["strategy"] in ("LVLL", "LVLS"):
        why = t.get("why", "")
        lvl = why.split(" into ", 1)[1].split(",", 1)[0] if " into " in why else t.get("level", "")
        s = f"LEVEL ALERT, {t['ticker']} {t['side'].upper()} setup at {lvl}. In {t.get('entry_time', 'next bar')} @ {t.get('entry', '?')}"
        if t["side"] == "short":
            s += " (short side has no backtest support)"
    else:
        s = f"{t['ticker']} {NAMES.get(t['strategy'], t['strategy'])} LONG, in {t.get('entry_time', 'next bar')} @ {t.get('entry', '?')}"
        if t.get("contract"):
            s += f", est. {t['contract']}"
    tags = [x for x in ("overlap" if t.get("overlap") else "", t.get("event_entry", "")) if x]
    return s + (f" [{', '.join(tags)}]" if tags else "")


def line_result(t):
    opt = f", est. option {t['opt_ret_pct']:+.1f}%" if t.get("opt_ret_pct") is not None else ""
    return (f"{t['ticker']} {NAMES.get(t['strategy'], t['strategy'])} {t['side']}: {t['entry']} -> {t['exit']}, "
            f"{t['ret_pct']:+.2f}% (${t['pnl']:+,.0f}/100 sh){opt}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--webull", nargs=2, metavar=("HOURLY", "DAILY"))
    p.add_argument("--now")
    p.add_argument("--no-notify", action="store_true")
    a = p.parse_args()

    now = datetime.fromisoformat(a.now).replace(tzinfo=ET) if a.now else datetime.now(ET)
    if now.weekday() >= 5:
        print("Weekend, nothing to do.")
        return 0

    cfg_path = os.path.join(HERE, "config.json")
    ev_path = os.path.join(HERE, "events.json")
    cfg = dict(engine.DEFAULT_CONFIG)
    cfg.update(json.load(open(cfg_path)))
    os.makedirs(DOCS, exist_ok=True)
    tmp = os.path.join(ROOT, ".tmp_overnight")
    os.makedirs(tmp, exist_ok=True)
    if a.webull:
        hourly, daily = a.webull
    else:
        hourly, daily = os.path.join(tmp, "hourly.csv"), os.path.join(tmp, "daily.csv")
        fetch_yahoo(cfg["tickers"], hourly, daily)

    out = os.path.join(tmp, "out.json")
    trades, status = engine.run(hourly, daily, cfg_path, ev_path, out, now=now)

    tpath = os.path.join(DOCS, "trades.json")
    try:
        log = json.load(open(tpath))
    except (FileNotFoundError, ValueError):
        log = []
    by_id = {t["id"]: i for i, t in enumerate(log)}
    new, closed = [], []
    if status["warnings"]:
        print("Warnings, trade log not changed:", status["warnings"])
    else:
        for t in trades:
            i = by_id.get(t["id"])
            if i is None:
                log.append(t); by_id[t["id"]] = len(log) - 1
                if t["strategy"] != "BASE" and t["date"] == now.date().isoformat():
                    new.append(t)
                elif t["strategy"] != "BASE" and t["status"] == "closed":
                    closed.append(t)
            elif log[i].get("status") != t["status"]:
                log[i] = t
                if t["strategy"] != "BASE" and t["status"] == "closed":
                    closed.append(t)
        log.sort(key=lambda t: (t["date"], t["id"]))
        json.dump(log, open(tpath, "w"), indent=0, separators=(",", ":"))

    status["source"] = "Webull (test)" if a.webull else "Yahoo Finance"
    status["runner"] = "GitHub Actions"
    json.dump(status, open(os.path.join(DOCS, "status.json"), "w"), indent=1)
    json.dump(cfg, open(os.path.join(DOCS, "config.json"), "w"), indent=1)

    tk = status.get("tickers", {})
    gates = "Gates: " + ", ".join(f"{k} {'open' if v.get('uptrend') else 'closed'}" for k, v in tk.items())
    if status["warnings"]:
        msg = ("Overnight Desk problem", "; ".join(status["warnings"]) + "\nNo trades written. Paper only.")
    elif new:
        msg = ("PAPER SIGNAL", "\n".join(line_signal(t) for t in new) + f"\n{gates}\nPaper only.")
    elif closed:
        msg = ("PAPER RESULT", "\n".join(line_result(t) for t in closed) + f"\n{gates}\nPaper only.")
    else:
        msg = None
    print(json.dumps(status))
    if msg:
        print(msg[0]); print(msg[1])
        if not a.no_notify:
            notify(*msg)
    else:
        print(f"No new paper signals (bars through {status['bars_through']}). {gates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
