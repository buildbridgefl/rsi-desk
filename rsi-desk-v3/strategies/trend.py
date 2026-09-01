"""
Trend / momentum rules.

These have substantially better published support than RSI mean reversion.
Each docstring names the source so you can go read the original and judge
the evidence yourself rather than taking my word for it.

A shared caution: every one of these was published on data ending years
ago. Published edges decay — sometimes because they were noise, sometimes
because enough money crowded in. The out-of-sample panel is still the only
thing that counts.
"""

from core.indicators import cross_down, cross_up, ema, rsi, sma
from strategies.base import register


@register(
    "trend_200ma",
    params=dict(ma_len=200, buffer_pct=0.0),
    description="Faber (2007): hold while price is above its long moving "
                "average, sit in cash below it. The canonical tactical "
                "rule. Historically cuts drawdown hard; return advantage "
                "is much less reliable.",
)
def trend_200ma(df, ma_len=200, buffer_pct=0.0):
    ma = sma(df["Close"], ma_len)
    buf = buffer_pct / 100.0
    df["ma"] = ma
    df["long_entry"] = (df["Close"] > ma * (1 + buf)).fillna(False)
    df["short_entry"] = False
    df["long_exit"] = (df["Close"] < ma * (1 - buf)).fillna(False)
    df["short_exit"] = False
    return df


@register(
    "abs_momentum",
    params=dict(lookback=252, ma_len=0),
    description="Antonacci absolute momentum: long while trailing 12-month "
                "return is positive, else cash. Time-series momentum is one "
                "of the most replicated anomalies across asset classes "
                "(Moskowitz, Ooi & Pedersen 2012).",
)
def abs_momentum(df, lookback=252, ma_len=0):
    mom = df["Close"] / df["Close"].shift(lookback) - 1
    ok = mom > 0
    if ma_len:
        ok &= df["Close"] > sma(df["Close"], ma_len)
    df["mom"] = mom
    df["long_entry"] = ok.fillna(False)
    df["short_entry"] = False
    df["long_exit"] = (~ok).fillna(False)
    df["short_exit"] = False
    return df


@register(
    "donchian_breakout",
    params=dict(entry_len=55, exit_len=20),
    description="Turtle-style channel breakout: buy N-day highs, exit on "
                "M-day lows. Classic trend following. Works far better on "
                "futures and commodities than on equity indices — worth "
                "testing precisely because it may fail here.",
)
def donchian_breakout(df, entry_len=55, exit_len=20):
    hi = df["High"].rolling(entry_len).max().shift(1)
    lo = df["Low"].rolling(exit_len).min().shift(1)
    df["long_entry"] = (df["Close"] > hi).fillna(False)
    df["short_entry"] = False
    df["long_exit"] = (df["Close"] < lo).fillna(False)
    df["short_exit"] = False
    return df


@register(
    "vol_managed_trend",
    params=dict(ma_len=200, vol_len=20, vol_cap=25.0),
    description="Trend filter plus a volatility brake: long above the MA "
                "only while realised vol is below the cap. Volatility "
                "managed portfolios (Moreira & Muir 2017) improved "
                "risk-adjusted returns across many asset classes.",
)
def vol_managed_trend(df, ma_len=200, vol_len=20, vol_cap=25.0):
    ma = sma(df["Close"], ma_len)
    rv = df["Close"].pct_change().rolling(vol_len).std() * (252 ** 0.5) * 100
    calm = rv < vol_cap
    up = df["Close"] > ma
    df["rv"] = rv
    df["long_entry"] = (up & calm).fillna(False)
    df["long_exit"] = (~up | ~calm).fillna(False)
    df["short_entry"] = False
    df["short_exit"] = False
    return df


@register(
    "ma_cross",
    params=dict(fast=50, slow=200),
    description="Golden/death cross. Included as a control, not a "
                "recommendation — it is the most widely watched rule in "
                "retail technical analysis and a useful check on whether "
                "popularity has arbitraged it away.",
)
def ma_cross(df, fast=50, slow=200):
    f, s = sma(df["Close"], fast), sma(df["Close"], slow)
    df["long_entry"] = cross_up(f, s).fillna(False)
    df["short_entry"] = False
    df["long_exit"] = cross_down(f, s).fillna(False)
    df["short_exit"] = False
    return df
