#!/usr/bin/env python3
"""
Scheduled signal check. Run by GitHub Actions or cron.

    NTFY_TOPIC=your-topic python -m alerts.run_alerts --tickers SPY,QQQ
"""
import argparse, json, os, sys
from datetime import datetime, timezone

import requests

import strategies  # noqa: F401
from core.data import fetch
from strategies.base import REGISTRY

STATE = os.environ.get("STATE_PATH", ".alert_state.json")


def notify(title, body):
    sent = False
    topic = os.environ.get("NTFY_TOPIC")
    if topic:
        try:
            requests.post(f"https://ntfy.sh/{topic}", data=body.encode(),
                          headers={"Title": title, "Priority": "high"}, timeout=10)
            sent = True
        except requests.RequestException as e:
            print(f"[warn] ntfy: {e}", file=sys.stderr)
    tok, chat = os.environ.get("TG_BOT_TOKEN"), os.environ.get("TG_CHAT_ID")
    if tok and chat:
        try:
            requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": f"{title}\n{body}"}, timeout=10)
            sent = True
        except requests.RequestException as e:
            print(f"[warn] telegram: {e}", file=sys.stderr)
    if not sent:
        print(f"[no channel] {title} | {body}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", default="SPY")
    p.add_argument("--strategy", default="rsi_reversion")
    a = p.parse_args()

    strat = REGISTRY[a.strategy]
    try:
        state = json.load(open(STATE))
    except Exception:
        state = {}

    for tk in [t.strip().upper() for t in a.tickers.split(",") if t.strip()]:
        try:
            d = strat(fetch(tk, "1y"))
        except Exception as e:
            print(f"{tk}: fetch/strategy failed: {e}", file=sys.stderr)
            continue
        last = d.iloc[-1]
        bar = str(d.index[-1].date())
        side = "LONG" if last.long_entry else ("SHORT" if last.short_entry else None)
        rsi_txt = f" RSI {last.rsi:.1f}" if "rsi" in d.columns else ""
        print(f"{datetime.now(timezone.utc):%F %H:%M}Z {tk} {last.Close:.2f}"
              f"{rsi_txt} signal={side}")
        key = f"{tk}:{a.strategy}"
        if side and state.get(key) != bar:
            notify(f"{tk} {side} — {a.strategy}",
                   f"{tk} @ {last.Close:.2f}{rsi_txt}\nBar {bar}")
            state[key] = bar

    json.dump(state, open(STATE, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
