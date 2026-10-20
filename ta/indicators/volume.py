"""Volume-based indicators.

A caveat that applies to this whole module on this desk's data: the `volume`
column comes from `market-data-collector` summing the `amount` field of trades
scraped from public endpoints. Nobitex's public trade feed is capped at 20
records per poll, so on a busy USDTIRT minute the collector *sees* only part of
the tape. Volume here is a sample, not a total, and its sampling rate changes
with activity -- exactly when you would most want to trust it.

Ratios and z-scores of volume against its own recent history survive that
reasonably well. Absolute volume levels, and anything comparing volume across
venues, do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..frame import median_price, typical_price
from ..registry import ChoiceParam, FloatParam, IntParam, indicator
from ._common import (
    ema,
    ema_warmup,
    frame,
    rolling_sum,
    safe_ratio,
    sma,
)
from ..frame import rolling_std


@indicator(
    "obv",
    category="volume",
    outputs=("obv",),
    requires=("close", "volume"),
    warmup=lambda p: 2,
    doc="On-Balance Volume: running total of volume signed by the close-to-close direction.",
    notes=(
        "A cumulative series with an arbitrary origin -- its level depends entirely on "
        "where the frame starts, so only its slope and divergences mean anything."
    ),
)
def obv_indicator(df: pd.DataFrame) -> pd.DataFrame:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return frame(df.index, obv=(direction * df["volume"]).fillna(0.0).cumsum())


@indicator(
    "vwap",
    category="volume",
    outputs=("vwap_{anchor}",),
    params=(
        ChoiceParam(
            "anchor",
            "day",
            doc="Where the cumulative sums reset.",
            choices=("day", "week", "month", "none"),
        ),
    ),
    requires=("high", "low", "close", "volume"),
    warmup=lambda p: 1,
    doc="Volume-Weighted Average Price, anchored to a calendar period (UTC).",
    notes=(
        "VWAP is only meaningful relative to an anchor. `anchor=none` accumulates from "
        "the first bar of whatever window you happened to fetch, which makes the value "
        "a function of your query rather than of the market -- offered for completeness, "
        "not recommended. Anchors are UTC; Iranian venues run on Tehran time (UTC+3:30), "
        "so a 'day' here is not a local trading day."
    ),
)
def vwap_indicator(df: pd.DataFrame, anchor: str = "day") -> pd.DataFrame:
    tp = typical_price(df)
    notional = tp * df["volume"]

    if anchor == "none":
        cumulative_notional = notional.cumsum()
        cumulative_volume = df["volume"].cumsum()
    else:
        freq = {"day": "D", "week": "W", "month": "MS"}[anchor]
        groups = df.index.tz_localize(None).to_period(freq)
        cumulative_notional = notional.groupby(groups).cumsum()
        cumulative_volume = df["volume"].groupby(groups).cumsum()

    return frame(
        df.index, **{f"vwap_{anchor}": safe_ratio(cumulative_notional, cumulative_volume)}
    )


def _money_flow_multiplier(df: pd.DataFrame) -> pd.Series:
    """((close-low) - (high-close)) / (high-low), NaN on a zero-range bar."""
    span = df["high"] - df["low"]
    return safe_ratio((df["close"] - df["low"]) - (df["high"] - df["close"]), span)


@indicator(
    "ad",
    category="volume",
    outputs=("ad",),
    requires=("high", "low", "close", "volume"),
    warmup=lambda p: 2,
    doc="Accumulation/Distribution Line: volume weighted by close's position in the bar.",
    notes=(
        "Zero-range bars (high == low) contribute nothing rather than blowing up -- "
        "these are common on illiquid IRT pairs where a whole interval prints one trade."
    ),
)
def ad_indicator(df: pd.DataFrame) -> pd.DataFrame:
    flow = (_money_flow_multiplier(df) * df["volume"]).fillna(0.0)
    return frame(df.index, ad=flow.cumsum())


@indicator(
    "cmf",
    category="volume",
    outputs=("cmf_{period}",),
    params=(IntParam("period", 20, doc="Lookback length.", minimum=2, maximum=1000),),
    requires=("high", "low", "close", "volume"),
    warmup=lambda p: p["period"],
    doc="Chaikin Money Flow: money-flow volume over total volume, -1 to +1.",
)
def cmf_indicator(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    flow = (_money_flow_multiplier(df) * df["volume"]).fillna(0.0)
    value = safe_ratio(rolling_sum(flow, period), rolling_sum(df["volume"], period))
    return frame(df.index, **{f"cmf_{period}": value})


@indicator(
    "mfi",
    category="volume",
    outputs=("mfi_{period}",),
    params=(IntParam("period", 14, doc="Lookback length.", minimum=2, maximum=1000),),
    requires=("high", "low", "close", "volume"),
    warmup=lambda p: p["period"] + 1,
    doc="Money Flow Index: a volume-weighted RSI on typical price, 0-100.",
)
def mfi_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    tp = typical_price(df)
    raw_flow = tp * df["volume"]
    direction = tp.diff()

    positive = raw_flow.where(direction > 0, 0.0)
    negative = raw_flow.where(direction < 0, 0.0)

    positive_sum = rolling_sum(positive, period)
    negative_sum = rolling_sum(negative, period)
    mfi = 100.0 - (100.0 / (1.0 + safe_ratio(positive_sum, negative_sum)))

    settled = positive_sum.notna() & negative_sum.notna()
    mfi = mfi.mask(settled & (negative_sum == 0) & (positive_sum > 0), 100.0)
    mfi = mfi.mask(settled & (positive_sum == 0) & (negative_sum > 0), 0.0)
    return frame(df.index, **{f"mfi_{period}": mfi})


@indicator(
    "force_index",
    category="volume",
    outputs=("force_index_{period}",),
    params=(IntParam("period", 13, doc="EMA span.", minimum=1, maximum=1000),),
    requires=("close", "volume"),
    warmup=lambda p: ema_warmup(p["period"]) + 1,
    doc="Elder's Force Index: EMA of (close change x volume).",
)
def force_index_indicator(df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
    raw = df["close"].diff() * df["volume"]
    return frame(df.index, **{f"force_index_{period}": ema(raw, period)})


@indicator(
    "eom",
    category="volume",
    outputs=("eom_{period}",),
    params=(
        IntParam("period", 14, doc="SMA smoothing window.", minimum=1, maximum=1000),
        FloatParam("divisor", 1e8, doc="Scaling divisor for the volume term.", minimum=1.0),
    ),
    requires=("high", "low", "volume"),
    warmup=lambda p: p["period"] + 1,
    doc=(
        "Ease of Movement: how far the midpoint travelled per unit of volume. "
        "High positive values mean price rose on light volume."
    ),
    notes=(
        "The divisor is cosmetic -- it only keeps the numbers readable. Rial-quoted "
        "pairs need a far larger one than the conventional 1e8; scale it until the "
        "output sits in a range you can eyeball, and keep it fixed after that."
    ),
)
def eom_indicator(df: pd.DataFrame, period: int = 14, divisor: float = 1e8) -> pd.DataFrame:
    mid = median_price(df)
    distance = mid.diff()
    box_ratio = safe_ratio(df["volume"] / divisor, df["high"] - df["low"])
    return frame(df.index, **{f"eom_{period}": sma(safe_ratio(distance, box_ratio), period)})


@indicator(
    "vpt",
    category="volume",
    outputs=("vpt",),
    requires=("close", "volume"),
    warmup=lambda p: 2,
    doc="Volume-Price Trend: running total of volume x fractional price change.",
)
def vpt_indicator(df: pd.DataFrame) -> pd.DataFrame:
    change = safe_ratio(df["close"].diff(), df["close"].shift(1))
    return frame(df.index, vpt=(change * df["volume"]).fillna(0.0).cumsum())


@indicator(
    "volume_sma",
    category="volume",
    outputs=("volume_sma_{window}", "volume_ratio_{window}"),
    params=(IntParam("window", 20, doc="Lookback length.", minimum=2, maximum=5000),),
    requires=("volume",),
    warmup=lambda p: p["window"],
    doc=(
        "Average volume over the window, and the current bar's volume as a multiple "
        "of it. `volume_ratio` of 3 means this bar traded three times the norm."
    ),
)
def volume_sma_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    average = sma(df["volume"], window)
    return frame(
        df.index,
        **{
            f"volume_sma_{window}": average,
            f"volume_ratio_{window}": safe_ratio(df["volume"], average),
        },
    )


@indicator(
    "volume_zscore",
    category="volume",
    outputs=("volume_zscore_{window}",),
    params=(IntParam("window", 50, doc="Lookback length.", minimum=5, maximum=5000),),
    requires=("volume",),
    warmup=lambda p: p["window"],
    doc="Standard scores of volume against its own rolling mean and deviation.",
    notes=(
        "The volume feature most robust to the sampling problem described at the top of "
        "this module, since it is measured against the same feed's own recent behaviour. "
        "A sustained z > 3 alongside a jump_score spike is the signature of a news event, "
        "which is the moment a passive quoter most needs to be out of the way."
    ),
)
def volume_zscore_indicator(df: pd.DataFrame, window: int = 50) -> pd.DataFrame:
    mean = sma(df["volume"], window)
    deviation = rolling_std(df["volume"], window, ddof=0)
    return frame(
        df.index, **{f"volume_zscore_{window}": safe_ratio(df["volume"] - mean, deviation)}
    )
