"""Price data. Swap the provider here if you outgrow Yahoo."""

import pandas as pd
import yfinance as yf


def fetch(ticker: str, period: str = "5y", interval: str = "1d",
          start: str | None = None, end: str | None = None) -> pd.DataFrame:
    kw = dict(interval=interval, progress=False, auto_adjust=True)
    if start:
        df = yf.download(ticker, start=start, end=end, **kw)
    else:
        df = yf.download(ticker, period=period, **kw)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def split(df: pd.DataFrame, frac: float = 0.6):
    """In-sample / out-of-sample split. Design on the first, judge on the second."""
    cut = int(len(df) * frac)
    return df.iloc[:cut], df.iloc[cut:]
