"""Indicators. Add new ones here and they're available to every strategy."""

import numpy as np
import pandas as pd


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI. Matches TradingView / ThinkOrSwim / TA-Lib."""
    delta = close.diff()
    gain = delta.clip(lower=0).to_numpy(dtype=float)
    loss = (-delta.clip(upper=0)).to_numpy(dtype=float)
    n = len(close)
    out = np.full(n, np.nan)
    if n <= length:
        return pd.Series(out, index=close.index)
    ag = np.nanmean(gain[1 : length + 1])
    al = np.nanmean(loss[1 : length + 1])
    out[length] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(length + 1, n):
        ag = (ag * (length - 1) + gain[i]) / length
        al = (al * (length - 1) + loss[i]) / length
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return pd.Series(out, index=close.index)


def sma(close: pd.Series, length: int) -> pd.Series:
    return close.rolling(length).mean()


def ema(close: pd.Series, length: int) -> pd.Series:
    return close.ewm(span=length, adjust=False).mean()


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def bbands(close: pd.Series, length=20, mult=2.0) -> pd.DataFrame:
    mid = close.rolling(length).mean()
    sd = close.rolling(length).std()
    return pd.DataFrame({"mid": mid, "upper": mid + mult * sd,
                         "lower": mid - mult * sd})


def cross_up(s: pd.Series, level) -> pd.Series:
    """True on the bar where s crosses from below `level` to at/above it."""
    lv = level if np.isscalar(level) else level
    return (s.shift(1) < lv) & (s >= lv)


def cross_down(s: pd.Series, level) -> pd.Series:
    lv = level if np.isscalar(level) else level
    return (s.shift(1) > lv) & (s <= lv)
