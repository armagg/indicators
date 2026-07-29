"""Volatility and channel indicators.

The most operationally relevant category for this desk. `market_maker`'s
spread scaling (S2 `vol_spread_sensitivity`, S5 `spread_multiplier`) and the
Phase 2 gap guard are all volatility functions -- today fed from
`crypto_price`'s tick-based estimator with no independent check. These give a
second, candle-based estimate of the same quantity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..frame import rolling_std, true_range
from ..registry import FloatParam, IntParam, indicator
from ._common import (
    ema,
    ema_warmup,
    frame,
    rolling_max,
    rolling_min,
    rolling_sum,
    safe_ratio,
    sma,
    wilder,
    wilder_warmup,
)


@indicator(
    "atr",
    category="volatility",
    outputs=("atr_{period}",),
    params=(IntParam("period", 14, doc="Wilder smoothing period.", minimum=1, maximum=1000),),
    requires=("high", "low", "close"),
    warmup=lambda p: wilder_warmup(p["period"]),
    doc="Average True Range (Wilder), in price units.",
)
def atr_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    return frame(df.index, **{f"atr_{period}": wilder(true_range(df), period)})


@indicator(
    "natr",
    category="volatility",
    outputs=("natr_{period}",),
    params=(IntParam("period", 14, doc="Wilder smoothing period.", minimum=1, maximum=1000),),
    requires=("high", "low", "close"),
    warmup=lambda p: wilder_warmup(p["period"]),
    doc="Normalized ATR: ATR as a percentage of close.",
    notes=(
        "The one to threshold on. ATR in rials means nothing across markets and drifts "
        "with the level of the pair itself; NATR is directly comparable and is the right "
        "input to a spread multiplier."
    ),
)
def natr_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    atr = wilder(true_range(df), period)
    return frame(df.index, **{f"natr_{period}": 100.0 * safe_ratio(atr, df["close"])})


@indicator(
    "true_range",
    category="volatility",
    outputs=("true_range",),
    requires=("high", "low", "close"),
    warmup=lambda p: 2,
    doc="Per-bar true range: max(high-low, |high-prev_close|, |low-prev_close|).",
)
def true_range_indicator(df: pd.DataFrame) -> pd.DataFrame:
    return frame(df.index, true_range=true_range(df))


@indicator(
    "bbands",
    category="volatility",
    outputs=(
        "bb_upper_{window}_{std_dev}",
        "bb_mid_{window}",
        "bb_lower_{window}_{std_dev}",
        "bb_width_{window}_{std_dev}",
        "bb_percent_b_{window}_{std_dev}",
    ),
    params=(
        IntParam("window", 20, doc="SMA and deviation window.", minimum=2, maximum=5000),
        FloatParam("std_dev", 2.0, doc="Band width in standard deviations.", minimum=0.1, maximum=10.0),
    ),
    warmup=lambda p: p["window"],
    aliases=("bollinger",),
    doc=(
        "Bollinger Bands with bandwidth and %B. `bb_width` is (upper-lower)/mid; "
        "`bb_percent_b` is where close sits in the band (0 = lower, 1 = upper)."
    ),
    notes=(
        "Population standard deviation (ddof=0), matching Bollinger's definition and "
        "every charting platform. pandas' own default is the sample deviation, which "
        "makes 20-period bands about 2.6% too wide -- invisible on a chart, but enough "
        "to move which touches register as signals."
    ),
)
def bbands_indicator(
    df: pd.DataFrame, window: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    mid = sma(df["close"], window)
    deviation = rolling_std(df["close"], window, ddof=0)
    upper = mid + std_dev * deviation
    lower = mid - std_dev * deviation
    return frame(
        df.index,
        **{
            f"bb_upper_{window}_{std_dev}": upper,
            f"bb_mid_{window}": mid,
            f"bb_lower_{window}_{std_dev}": lower,
            f"bb_width_{window}_{std_dev}": safe_ratio(upper - lower, mid),
            f"bb_percent_b_{window}_{std_dev}": safe_ratio(
                df["close"] - lower, upper - lower
            ),
        },
    )


@indicator(
    "keltner",
    category="volatility",
    outputs=(
        "kc_upper_{window}_{atr_period}_{multiplier}",
        "kc_mid_{window}",
        "kc_lower_{window}_{atr_period}_{multiplier}",
    ),
    params=(
        IntParam("window", 20, doc="EMA span for the centre line.", minimum=2, maximum=5000),
        IntParam("atr_period", 10, doc="ATR period for the band width.", minimum=1, maximum=1000),
        FloatParam("multiplier", 2.0, doc="ATR multiple.", minimum=0.1, maximum=20.0),
    ),
    requires=("high", "low", "close"),
    warmup=lambda p: max(ema_warmup(p["window"]), wilder_warmup(p["atr_period"])),
    doc="Keltner Channels: an EMA centre line with ATR-width bands.",
    notes=(
        "Bollinger bands sitting *inside* Keltner channels is the standard volatility "
        "squeeze: range compression that usually precedes expansion. See "
        "`ta.analysis.regime`, which computes exactly that comparison."
    ),
)
def keltner_indicator(
    df: pd.DataFrame, window: int = 20, atr_period: int = 10, multiplier: float = 2.0
) -> pd.DataFrame:
    mid = ema(df["close"], window)
    band = multiplier * wilder(true_range(df), atr_period)
    return frame(
        df.index,
        **{
            f"kc_upper_{window}_{atr_period}_{multiplier}": mid + band,
            f"kc_mid_{window}": mid,
            f"kc_lower_{window}_{atr_period}_{multiplier}": mid - band,
        },
    )


@indicator(
    "donchian",
    category="volatility",
    outputs=("dc_upper_{window}", "dc_mid_{window}", "dc_lower_{window}", "dc_width_{window}"),
    params=(IntParam("window", 20, doc="Lookback length.", minimum=2, maximum=5000),),
    requires=("high", "low"),
    warmup=lambda p: p["window"],
    doc="Donchian Channel: highest high and lowest low over the window, plus its midpoint.",
    notes=(
        "Computed over a window ending at the *current* bar, so the current bar's own "
        "high and low are included. A breakout rule written as close > dc_upper can "
        "therefore never fire -- compare against dc_upper.shift(1) instead."
    ),
)
def donchian_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    upper = rolling_max(df["high"], window)
    lower = rolling_min(df["low"], window)
    mid = (upper + lower) / 2.0
    return frame(
        df.index,
        **{
            f"dc_upper_{window}": upper,
            f"dc_mid_{window}": mid,
            f"dc_lower_{window}": lower,
            f"dc_width_{window}": safe_ratio(upper - lower, mid),
        },
    )


@indicator(
    "stddev",
    category="volatility",
    outputs=("stddev_{window}",),
    params=(IntParam("window", 20, doc="Lookback length.", minimum=2, maximum=5000),),
    warmup=lambda p: p["window"],
    doc="Rolling population standard deviation of close, in price units.",
)
def stddev_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return frame(
        df.index, **{f"stddev_{window}": rolling_std(df["close"], window, ddof=0)}
    )


@indicator(
    "ulcer_index",
    category="volatility",
    outputs=("ulcer_{window}",),
    params=(IntParam("window", 14, doc="Lookback length.", minimum=2, maximum=5000),),
    warmup=lambda p: 2 * p["window"],
    doc=(
        "Ulcer Index: root-mean-square percentage drawdown from the rolling high. "
        "Downside-only volatility."
    ),
    notes=(
        "The right volatility measure when the risk is asymmetric, which it is here: "
        "a passive quoter on USDT/IRT is structurally short jump risk on the upside, "
        "so symmetric volatility understates what actually hurts."
    ),
)
def ulcer_index_indicator(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    peak = rolling_max(df["close"], window)
    drawdown = 100.0 * safe_ratio(df["close"] - peak, peak)
    ulcer = np.sqrt(sma(drawdown**2, window))
    return frame(df.index, **{f"ulcer_{window}": ulcer})


@indicator(
    "mass_index",
    category="volatility",
    outputs=("mass_index_{ema_period}_{sum_period}",),
    params=(
        IntParam("ema_period", 9, doc="Span of both range EMAs.", minimum=1, maximum=1000),
        IntParam("sum_period", 25, doc="Summation window for the EMA ratio.", minimum=2, maximum=1000),
    ),
    requires=("high", "low"),
    warmup=lambda p: 2 * ema_warmup(p["ema_period"]) + p["sum_period"],
    doc=(
        "Mass Index: sum of the ratio of a range EMA to its own EMA. Rises when the "
        "high-low range widens -- a reversal warning that says nothing about direction."
    ),
)
def mass_index_indicator(
    df: pd.DataFrame, ema_period: int = 9, sum_period: int = 25
) -> pd.DataFrame:
    price_range = df["high"] - df["low"]
    single = ema(price_range, ema_period)
    double = ema(single, ema_period)
    ratio = safe_ratio(single, double)
    return frame(
        df.index,
        **{f"mass_index_{ema_period}_{sum_period}": rolling_sum(ratio, sum_period)},
    )


@indicator(
    "chaikin_volatility",
    category="volatility",
    outputs=("chaikin_vol_{ema_period}_{roc_period}",),
    params=(
        IntParam("ema_period", 10, doc="EMA span of the high-low range.", minimum=1, maximum=1000),
        IntParam("roc_period", 10, doc="Rate-of-change lookback.", minimum=1, maximum=1000),
    ),
    requires=("high", "low"),
    warmup=lambda p: ema_warmup(p["ema_period"]) + p["roc_period"],
    doc="Chaikin Volatility: rate of change of an EMA of the high-low range, in percent.",
)
def chaikin_volatility_indicator(
    df: pd.DataFrame, ema_period: int = 10, roc_period: int = 10
) -> pd.DataFrame:
    smoothed = ema(df["high"] - df["low"], ema_period)
    prior = smoothed.shift(roc_period)
    value = 100.0 * safe_ratio(smoothed - prior, prior)
    return frame(df.index, **{f"chaikin_vol_{ema_period}_{roc_period}": value})
