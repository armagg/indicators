"""Shared math and warmup helpers for the indicator library.

Deliberately no third-party TA dependency. `pandas_ta` was already dropped once
here (no Python 3.14 wheel) and TA-Lib needs a C toolchain in the image; both
are heavy prices to pay for arithmetic that fits in a few hundred lines and
that we want to be able to read when a number looks wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Warmup
# --------------------------------------------------------------------------
#
# Two different things get called "warmup" and conflating them is why the same
# indicator reads differently on two charts.
#
# *Window* indicators (SMA, Bollinger, Donchian) have a hard boundary: before
# bar N the output is NaN, from bar N it is exactly right. Warmup = N.
#
# *Recursive* indicators (EMA, and everything built on one -- RSI, MACD, ATR,
# ADX, TSI) have no such boundary. `adjust=False` seeds the recursion with the
# first observation and every later value still contains a fraction of that
# seed. The output is non-NaN long before it is correct.
#
# The seed's residual weight after k bars is (1 - alpha)^k. We call it settled
# once that drops below 5%:
#
#     k > ln(0.05) / ln(1 - alpha)  ~=  3 / alpha
#
# For Wilder smoothing (alpha = 1/period) that is the familiar "3x the period"
# rule of thumb -- a 14-period RSI wants ~42 bars, not 14. For a span-n EMA
# (alpha = 2/(n+1)) it is ~1.5 * (n+1).

#: Residual weight of the seed value at which a recursion is called settled.
SEED_TOLERANCE = 0.05


def alpha_warmup(alpha: float) -> int:
    """Bars until an EMA-style recursion with this alpha has forgotten its seed."""
    if not 0 < alpha < 1:
        return 1
    return int(np.ceil(np.log(SEED_TOLERANCE) / np.log(1.0 - alpha)))


def ema_warmup(span: int) -> int:
    """Warmup for a standard span-`n` EMA (alpha = 2/(n+1))."""
    return alpha_warmup(2.0 / (span + 1.0))


def wilder_warmup(period: int) -> int:
    """Warmup for Wilder smoothing (alpha = 1/period). ~3x the period."""
    return alpha_warmup(1.0 / period)


# --------------------------------------------------------------------------
# Moving averages used as building blocks
# --------------------------------------------------------------------------


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, min_periods=span, adjust=False).mean()


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (alpha = 1/period), not `ewm(span=period)`."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()


def wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly weighted moving average -- weights 1..n, newest heaviest."""
    weights = np.arange(1, window + 1, dtype="float64")
    denom = weights.sum()
    return series.rolling(window=window, min_periods=window).apply(
        lambda values: float(np.dot(values, weights) / denom), raw=True
    )


def rolling_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).max()


def rolling_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).min()


def rolling_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).sum()


# --------------------------------------------------------------------------
# Numerics
# --------------------------------------------------------------------------


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, mapping division by zero to NaN rather than +/-inf.

    Zero denominators are not exotic here: a flat 14-bar window on an illiquid
    IRT pair makes `highest - lowest` exactly 0 for Stochastic and Williams %R,
    and a bar with no down-ticks makes RSI's average loss 0. `inf` would then
    propagate through every later smoothing step and destroy the rest of the
    series; NaN stays local to the bar that earned it.
    """
    return numerator / denominator.replace(0.0, np.nan)


def frame(index: pd.Index, **columns: pd.Series) -> pd.DataFrame:
    """Assemble an indicator's output frame with float dtype and a clean index."""
    out = pd.DataFrame(index=index)
    for name, series in columns.items():
        out[name] = pd.Series(series, index=index, dtype="float64")
    return out


def bars_per_year(index: pd.DatetimeIndex, default: float = 365 * 24 * 60) -> float:
    """Estimate how many bars of this frame's interval fit in a year.

    Crypto trades continuously, so this is plain calendar time -- no 252-day
    equity convention.
    """
    if len(index) < 2:
        return default
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    positive = deltas[deltas > 0]
    if positive.empty:
        return default
    return 365.0 * 24.0 * 3600.0 / float(positive.median())


def bar_seconds(index: pd.DatetimeIndex) -> float | None:
    if len(index) < 2:
        return None
    deltas = index.to_series().diff().dropna().dt.total_seconds()
    positive = deltas[deltas > 0]
    return float(positive.median()) if not positive.empty else None
