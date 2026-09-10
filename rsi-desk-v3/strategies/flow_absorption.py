"""
Flow absorption: seller dominance inside a daily uptrend.

This is the one hypothesis that survived every test in the research
scripts. Chain of evidence, so future-you doesn't have to re-derive it:

  study_fib_absorption.py   fib-zone version, long side: FAILED
                            (-2.14R out-of-sample, 20% win rate)
  study_side_edge.py        5m bars: significant in-sample (p=0.006),
                            failed out-of-sample (p=0.052, horizons
                            disagreeing). Also had ZERO downtrend days.
  study_side_edge_hourly.py hourly, 2y: significant in BOTH blocks -
                            but the null was mixed-regime, which
                            flatters an uptrend-only sample.
  study_side_edge_matched.py regime-matched null. SURVIVED:
                            in-sample  p=0.008, excess 5/5 positive
                            out-of-sample p=0.006, excess 5/5 positive
                            excess over drift +0.09% to +0.31% by horizon
                            buyers: never significant anywhere

THE RULE
  Long when, on an HOURLY bar:
    - the daily close is above its 200-day MA (uptrend), AND
    - sellers dominated the bar: CLV-weighted imbalance strongly
      negative, AND
    - volume was elevated (z-score above threshold)
  Exit on max_hold. The effect grows with horizon out to ~16 hours, so
  max_hold does the work, not a signal exit.

  Short side: NOT traded. Seller dominance in a DOWNTREND showed
  continuation (excess -0.20% to -0.35%), which suggests a short rule
  might exist there - but it was never significant (p=0.39) and had
  only 42 out-of-sample events. Untested, so not implemented.

CAVEATS THAT STILL APPLY
  - CLV/volume is a PROXY for order flow, not real bid-ask footprint.
  - The 8 test ETFs are correlated; effective sample < nominal sample.
  - Two years covers one broad market character.
  - Effect sizes are small enough that costs matter enormously.
  - Designed for HOURLY bars. Untested on daily.
"""

import numpy as np

from strategies.base import register


def _clv_imbalance(df, vol_window):
    """Close-location-value weighted by volume. Approximates which side
    was transacting aggressively. -1 = all selling, +1 = all buying."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    clv = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / rng
    clv = clv.fillna(0.0)
    imbalance = (clv * df["Volume"]) / df["Volume"].replace(0, np.nan)
    imbalance = imbalance.fillna(0.0)

    vmean = df["Volume"].rolling(vol_window).mean()
    vstd = df["Volume"].rolling(vol_window).std()
    vol_z = (df["Volume"] - vmean) / vstd.replace(0, np.nan)
    return imbalance, vol_z


@register(
    "flow_absorption",
    params=dict(imb_threshold=0.4, vol_threshold=1.0, vol_window=168,
                ma_bars=1400, exit_bars=8),
    description="Long when sellers dominate an hourly bar inside a daily "
                "uptrend (absorption). Validated out-of-sample with a "
                "regime-matched null. HOURLY BARS ONLY.",
)
def flow_absorption(df, imb_threshold=0.4, vol_threshold=1.0,
                    vol_window=168, ma_bars=1400, exit_bars=8):
    """
    If the caller supplies an 'uptrend' boolean column (the runner does,
    computed properly from DAILY data), it is used. Otherwise the trend
    is approximated with a rolling mean over ma_bars hourly bars
    (1400 ~ 200 days x 7 hours). The daily version is the tested one;
    the fallback is a convenience and is NOT what was validated.
    """
    imbalance, vol_z = _clv_imbalance(df, vol_window)
    df["imbalance"] = imbalance
    df["vol_z"] = vol_z

    if "uptrend" in df.columns:
        uptrend = df["uptrend"].fillna(False).astype(bool)
    else:
        ma = df["Close"].rolling(ma_bars).mean()
        uptrend = (df["Close"] > ma).fillna(False)
    df["uptrend"] = uptrend

    entry = (uptrend
             & (imbalance < -imb_threshold)
             & (vol_z > vol_threshold)).fillna(False)

    df["long_entry"] = entry
    df["short_entry"] = False
    # No signal-based exit: the effect GROWS with horizon out to ~16
    # bars, so the time-based exit (engine's max_hold) is the mechanism.
    df["long_exit"] = False
    df["short_exit"] = False
    return df
