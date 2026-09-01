# v2 — five new rules + a fixed engine

## What changed

**1. Engine now credits interest on cash (`cash_apr`).**
Previously a strategy sitting flat earned 0%. That is wrong — flat money
sits in T-bills. Over 2022-2026 that was 4-5%/yr, so every rule that
spends time out of the market was being penalised by several percent a
year for no reason. New sidebar control: "cash yield %", default 4.0.
Set it to roughly the risk-free rate over whatever window you're testing.

Note the short side is still slightly generous — the engine credits the
full short proceeds, where a real broker pays a rebate on collateral
only. Immaterial for long-only rules.

**2. Five new strategies in `strategies/trend.py`:**

| rule | idea | source |
|---|---|---|
| `trend_200ma` | long above the 200-day MA, else cash | Faber 2007 |
| `abs_momentum` | long while trailing 12-mo return > 0 | Moskowitz/Ooi/Pedersen 2012; Antonacci |
| `donchian_breakout` | buy 55-day highs, exit 20-day lows | Turtle rules |
| `vol_managed_trend` | 200MA trend + realised-vol brake | Moreira & Muir 2017 |
| `ma_cross` | 50/200 golden cross — a control, not a pick | folklore |

**3. Backtest defaults changed** — Years now goes to 25 and defaults to
20, max hold defaults to 250 bars. Trend rules hold for months; the old
20-bar cap would have chopped every position short and made them look
broken.

## Install

Drop these over your existing folder, keeping the structure:

    app.py                  -> replaces
    core/engine.py          -> replaces
    strategies/trend.py     -> new
    strategies/__init__.py  -> replaces

Then restart: Ctrl+C in the black window, then
`python -m streamlit run app.py`

## Test order

Run each on SPY, 20 years, in-sample 0.60, cash yield 4.0.
Record the OUT-OF-SAMPLE row only.

1. trend_200ma
2. abs_momentum
3. vol_managed_trend
4. donchian_breakout
5. ma_cross

## Reading the results honestly

You are about to run five tests on one asset. Expect at least one to
look good by luck alone — that is what five tries buys you. A rule is
only interesting if it ALSO holds up on QQQ and IWM without changing
its parameters.

The bar remains: beat buy-and-hold on Sharpe AND drawdown, out-of-sample,
after costs. Nothing here is validated. Nothing here is a recommendation.
