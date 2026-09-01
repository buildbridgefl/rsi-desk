"""RSI mean-reversion rules."""

from core.indicators import cross_down, cross_up, rsi, sma
from strategies.base import register


@register(
    "rsi_reversion",
    params=dict(rsi_len=14, oversold=30, overbought=70, exit_level=50,
                trend_filter=False, ma_len=200),
    description="Enter when RSI crosses back OUT of the extreme zone. "
                "Exit when it reverts to the midline.",
)
def rsi_reversion(df, rsi_len=14, oversold=30, overbought=70,
                  exit_level=50, trend_filter=False, ma_len=200):
    r = rsi(df["Close"], rsi_len)
    df["rsi"] = r

    long_e = cross_up(r, oversold)
    short_e = cross_down(r, overbought)

    if trend_filter:
        ma = sma(df["Close"], ma_len)
        above = df["Close"] > ma
        long_e &= above
        short_e &= ~above

    df["long_entry"] = long_e.fillna(False)
    df["short_entry"] = short_e.fillna(False)
    df["long_exit"] = (r >= exit_level).fillna(False)
    df["short_exit"] = (r <= 100 - exit_level).fillna(False)
    return df


@register(
    "rsi_dip_buy",
    params=dict(rsi_len=2, threshold=10, ma_len=200, exit_level=60),
    description="Larry Connors style: short-lookback RSI dip inside an "
                "uptrend. Long only. Historically the better-supported "
                "half of RSI mean reversion on equity indices.",
)
def rsi_dip_buy(df, rsi_len=2, threshold=10, ma_len=200, exit_level=60):
    r = rsi(df["Close"], rsi_len)
    ma = sma(df["Close"], ma_len)
    df["rsi"] = r
    df["long_entry"] = ((r < threshold) & (df["Close"] > ma)).fillna(False)
    df["short_entry"] = False
    df["long_exit"] = (r > exit_level).fillna(False)
    df["short_exit"] = False
    return df
