"""Shared math and warmup helpers for the indicator library, with no third-party TA dependency."""

from __future__ import annotations

import numpy as np
import pandas as pd


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


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, mapping division by zero to NaN rather than +/-inf."""
    return numerator / denominator.replace(0.0, np.nan)


def frame(index: pd.Index, **columns: pd.Series) -> pd.DataFrame:
    """Assemble an indicator's output frame with float dtype and a clean index."""
    out = pd.DataFrame(index=index)
    for name, series in columns.items():
        out[name] = pd.Series(series, index=index, dtype="float64")
    return out


def bars_per_year(index: pd.DatetimeIndex, default: float = 365 * 24 * 60) -> float:
    """Estimate how many bars of this frame's interval fit in a year."""
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
