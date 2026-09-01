"""
COPY THIS FILE to add a new rule.

  1. cp strategies/_template.py strategies/my_rule.py
  2. Edit the name, params, and logic below
  3. Add `from strategies import my_rule` to strategies/__init__.py
  4. It appears in the app's strategy dropdown automatically

You only write the RULE. The engine handles position tracking,
next-bar fills, costs, stops, and max-hold.
"""

from core.indicators import cross_down, cross_up, macd, rsi, sma
from strategies.base import register


@register(
    "my_rule",                                  # name in the dropdown
    params=dict(fast=50, slow=200, rsi_len=14),  # defaults, all tunable
    description="One line explaining what this rule does.",
)
def my_rule(df, fast=50, slow=200, rsi_len=14):
    # --- compute whatever you need
    f = sma(df["Close"], fast)
    s = sma(df["Close"], slow)
    r = rsi(df["Close"], rsi_len)
    df["rsi"] = r  # optional, but the chart will plot it if present

    # --- REQUIRED: four boolean columns
    df["long_entry"] = (cross_up(f, s) & (r < 70)).fillna(False)
    df["short_entry"] = False
    df["long_exit"] = cross_down(f, s).fillna(False)
    df["short_exit"] = False

    return df
