# v3 — take-profit + short-hold testing

## What's new

**`take_profit_pct` in the engine, "take profit %" in the sidebar.**
Fires intrabar against the bar HIGH (long) or LOW (short), like a resting
limit order. Exit reason logs as `target`.

**Known optimism:** if a bar touches BOTH the stop and the target, this
engine fills the target. Daily bars don't record which came first. So any
result that leans on a tight target is an UPPER BOUND, not a forecast.
The tighter your target relative to your stop, the more this flatters you.
To see how much, re-run with the target disabled and compare.

**Sidebar warning** if you set a target with no stop.

## Testing a 7-day swing rule

    max hold (bars)   7
    take profit %     3.0      (whatever your actual target is)
    stop loss %       3.0      (do not leave this at 0)
    cost bps/side     2.0      (raise to 5 if you trade options or use market orders)
    cash yield %      4.0

Then work through the entry rules. The RSI rules are the relevant ones
here — trend rules are built to hold for months and will just hit the
7-bar cap every time, which tells you nothing.

## The thing that will bite you

Turnover. A 7-day hold means roughly 35 round trips a year if you're
always in. At 2 bps/side that's 1.4%/yr in costs. At 5 bps it's 3.5%.
Your edge has to clear that before it earns a cent.

Compare `no target, no stop` against your target/stop version on the same
entry rule. If the target version isn't clearly better, the target isn't
adding anything — you're just paying more spread for the privilege of
exiting earlier.

## The asymmetry

A target with no stop caps every winner and lets every loser run. It
generates a high win rate and a falling equity curve. If you set a 3%
target, you need a stop no wider than 3% or the arithmetic is against
you regardless of entry quality.

Nothing here is validated. Nothing here is a recommendation.
