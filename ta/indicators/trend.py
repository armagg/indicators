"""Trend indicators: moving averages and directional-structure measures."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..frame import median_price, true_range
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
    wma,
)

_WINDOW = IntParam("window", 20, doc="Lookback length in bars.", minimum=2, maximum=5000)


# --------------------------------------------------------------------------
# Moving averages
# --------------------------------------------------------------------------


@indicator(
    "sma",
    category="trend",
    outputs=("sma_{window}",),
    params=(_WINDOW,),
    warmup=lambda p: p["window"],
    doc="Simple moving average of close.",
)
def sma_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return frame(df.index, **{f"sma_{window}": sma(df["close"], window)})


@indicator(
    "ema",
    category="trend",
    outputs=("ema_{window}",),
    params=(_WINDOW,),
    warmup=lambda p: ema_warmup(p["window"]),
    doc="Exponential moving average of close (alpha = 2/(window+1)).",
    notes=(
        "Warmup is ~1.5x the window, not the window: with adjust=False the "
        "recursion is seeded from the first close and needs that long to forget it."
    ),
)
def ema_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return frame(df.index, **{f"ema_{window}": ema(df["close"], window)})


@indicator(
    "wma",
    category="trend",
    outputs=("wma_{window}",),
    params=(_WINDOW,),
    warmup=lambda p: p["window"],
    doc="Linearly weighted moving average -- weights 1..n, newest heaviest.",
)
def wma_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return frame(df.index, **{f"wma_{window}": wma(df["close"], window)})


@indicator(
    "dema",
    category="trend",
    outputs=("dema_{window}",),
    params=(_WINDOW,),
    warmup=lambda p: 2 * ema_warmup(p["window"]),
    doc="Double EMA: 2*EMA - EMA(EMA). Less lag than an EMA, more overshoot.",
)
def dema_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    e1 = ema(df["close"], window)
    e2 = ema(e1, window)
    return frame(df.index, **{f"dema_{window}": 2 * e1 - e2})


@indicator(
    "tema",
    category="trend",
    outputs=("tema_{window}",),
    params=(_WINDOW,),
    warmup=lambda p: 3 * ema_warmup(p["window"]),
    doc="Triple EMA: 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)).",
)
def tema_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    e1 = ema(df["close"], window)
    e2 = ema(e1, window)
    e3 = ema(e2, window)
    return frame(df.index, **{f"tema_{window}": 3 * e1 - 3 * e2 + e3})


@indicator(
    "hma",
    category="trend",
    outputs=("hma_{window}",),
    params=(IntParam("window", 20, doc="Lookback length in bars.", minimum=4, maximum=5000),),
    warmup=lambda p: p["window"] + int(np.sqrt(p["window"])),
    doc="Hull moving average: WMA(2*WMA(n/2) - WMA(n), sqrt(n)). Fast and smooth.",
)
def hma_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    half = max(1, window // 2)
    root = max(1, int(np.sqrt(window)))
    raw = 2 * wma(df["close"], half) - wma(df["close"], window)
    return frame(df.index, **{f"hma_{window}": wma(raw, root)})


@indicator(
    "vwma",
    category="trend",
    outputs=("vwma_{window}",),
    params=(_WINDOW,),
    requires=("close", "volume"),
    warmup=lambda p: p["window"],
    doc="Volume-weighted moving average: sum(close*volume)/sum(volume).",
)
def vwma_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    numerator = rolling_sum(df["close"] * df["volume"], window)
    denominator = rolling_sum(df["volume"], window)
    return frame(df.index, **{f"vwma_{window}": safe_ratio(numerator, denominator)})


@indicator(
    "kama",
    category="trend",
    outputs=("kama_{window}",),
    params=(
        IntParam("window", 10, doc="Efficiency-ratio lookback.", minimum=2, maximum=1000),
        IntParam("fast", 2, doc="Fastest EMA span the filter can adapt to.", minimum=1),
        IntParam("slow", 30, doc="Slowest EMA span the filter can adapt to.", minimum=2),
    ),
    warmup=lambda p: p["window"] + ema_warmup(p["slow"]),
    doc=(
        "Kaufman adaptive moving average: smoothing tracks the efficiency ratio, so "
        "it moves like a fast EMA in a directional move and nearly flatlines in chop."
    ),
    notes=(
        "The most useful trend filter on this desk's pairs: USDT/IRT alternates "
        "between long pinned stretches and sharp ratchets, which is exactly the "
        "regime split KAMA is built to straddle."
    ),
)
def kama_indicator(
    df: pd.DataFrame, window: int = 10, fast: int = 2, slow: int = 30
) -> pd.DataFrame:
    close = df["close"]
    change = (close - close.shift(window)).abs()
    volatility = close.diff().abs().rolling(window=window, min_periods=window).sum()
    efficiency = safe_ratio(change, volatility).fillna(0.0)

    fast_alpha, slow_alpha = 2.0 / (fast + 1.0), 2.0 / (slow + 1.0)
    smoothing = ((efficiency * (fast_alpha - slow_alpha)) + slow_alpha) ** 2

    values = close.to_numpy(dtype="float64")
    alphas = smoothing.to_numpy(dtype="float64")
    out = np.full(len(values), np.nan)
    if len(values) > window:
        out[window] = values[window]
        for i in range(window + 1, len(values)):
            prev = out[i - 1]
            alpha = alphas[i]
            if not np.isfinite(prev) or not np.isfinite(alpha) or not np.isfinite(values[i]):
                out[i] = prev
                continue
            out[i] = prev + alpha * (values[i] - prev)
    return frame(df.index, **{f"kama_{window}": pd.Series(out, index=df.index)})


# --------------------------------------------------------------------------
# Directional structure
# --------------------------------------------------------------------------


@indicator(
    "adx",
    category="trend",
    outputs=("adx_{period}", "di_plus_{period}", "di_minus_{period}"),
    params=(IntParam("period", 14, doc="Wilder smoothing period.", minimum=2, maximum=1000),),
    requires=("high", "low", "close"),
    warmup=lambda p: 2 * wilder_warmup(p["period"]),
    doc=(
        "Average Directional Index with its +DI / -DI components. ADX measures trend "
        "*strength* regardless of direction; DI+ vs DI- gives the direction."
    ),
    notes=(
        "Warmup is doubled because ADX is a Wilder average *of* a Wilder average: "
        "DX settles first, then ADX has to settle on top of it."
    ),
)
def adx_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    atr = wilder(true_range(df), period)
    di_plus = 100.0 * safe_ratio(wilder(plus_dm, period), atr)
    di_minus = 100.0 * safe_ratio(wilder(minus_dm, period), atr)

    dx = 100.0 * safe_ratio((di_plus - di_minus).abs(), di_plus + di_minus)
    adx = wilder(dx, period)

    return frame(
        df.index,
        **{
            f"adx_{period}": adx,
            f"di_plus_{period}": di_plus,
            f"di_minus_{period}": di_minus,
        },
    )


@indicator(
    "aroon",
    category="trend",
    outputs=("aroon_up_{period}", "aroon_down_{period}", "aroon_osc_{period}"),
    params=(IntParam("period", 25, doc="Lookback length in bars.", minimum=2, maximum=1000),),
    requires=("high", "low"),
    warmup=lambda p: p["period"] + 1,
    doc=(
        "Aroon: how recently the window's high and low were set, as a percentage. "
        "Up near 100 means the highest high is the current bar. The oscillator is up - down."
    ),
)
def aroon_indicator(df: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    window = period + 1  # the window includes the current bar
    since_high = df["high"].rolling(window, min_periods=window).apply(
        lambda v: float(len(v) - 1 - np.argmax(v)), raw=True
    )
    since_low = df["low"].rolling(window, min_periods=window).apply(
        lambda v: float(len(v) - 1 - np.argmin(v)), raw=True
    )
    up = 100.0 * (period - since_high) / period
    down = 100.0 * (period - since_low) / period
    return frame(
        df.index,
        **{
            f"aroon_up_{period}": up,
            f"aroon_down_{period}": down,
            f"aroon_osc_{period}": up - down,
        },
    )


@indicator(
    "psar",
    category="trend",
    outputs=("psar_{step}_{max_step}", "psar_dir_{step}_{max_step}"),
    params=(
        FloatParam("step", 0.02, doc="Acceleration factor increment.", minimum=0.0001, maximum=1.0),
        FloatParam("max_step", 0.2, doc="Acceleration factor cap.", minimum=0.001, maximum=1.0),
    ),
    requires=("high", "low"),
    warmup=lambda p: 5,
    doc=(
        "Parabolic SAR: a trailing stop that accelerates toward price. "
        "`psar_dir` is +1 while long, -1 while short."
    ),
    notes=(
        "A stop-and-reverse system with no flat state -- it is always positioned. "
        "On a ranging pair that means it flips constantly and pays the spread every time. "
        "Use it as a trailing exit, not an entry."
    ),
)
def psar_indicator(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> pd.DataFrame:
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    n = len(df)

    sar = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    if n < 2:
        return frame(
            df.index,
            **{
                f"psar_{step}_{max_step}": pd.Series(sar, index=df.index),
                f"psar_dir_{step}_{max_step}": pd.Series(direction, index=df.index),
            },
        )

    long = high[1] >= high[0]
    accel = step
    extreme = high[1] if long else low[1]
    current = low[0] if long else high[0]
    sar[1] = current
    direction[1] = 1.0 if long else -1.0

    for i in range(2, n):
        current = current + accel * (extreme - current)
        # SAR may not penetrate either of the two prior bars' opposite extremes.
        if long:
            current = min(current, low[i - 1], low[i - 2])
        else:
            current = max(current, high[i - 1], high[i - 2])

        if long and low[i] < current:
            long, current, extreme, accel = False, extreme, low[i], step
        elif not long and high[i] > current:
            long, current, extreme, accel = True, extreme, high[i], step
        else:
            if long and high[i] > extreme:
                extreme, accel = high[i], min(accel + step, max_step)
            elif not long and low[i] < extreme:
                extreme, accel = low[i], min(accel + step, max_step)

        sar[i] = current
        direction[i] = 1.0 if long else -1.0

    return frame(
        df.index,
        **{
            f"psar_{step}_{max_step}": pd.Series(sar, index=df.index),
            f"psar_dir_{step}_{max_step}": pd.Series(direction, index=df.index),
        },
    )


@indicator(
    "supertrend",
    category="trend",
    outputs=("supertrend_{period}_{multiplier}", "supertrend_dir_{period}_{multiplier}"),
    params=(
        IntParam("period", 10, doc="ATR period.", minimum=2, maximum=1000),
        FloatParam("multiplier", 3.0, doc="ATR multiple for the bands.", minimum=0.1, maximum=20.0),
    ),
    requires=("high", "low", "close"),
    warmup=lambda p: wilder_warmup(p["period"]),
    doc=(
        "Supertrend: an ATR-width band that ratchets in the direction of the trend and "
        "flips when price closes through it. `supertrend_dir` is +1 up, -1 down."
    ),
)
def supertrend_indicator(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    atr = wilder(true_range(df), period)
    mid = median_price(df)
    upper_basic = (mid + multiplier * atr).to_numpy(dtype="float64")
    lower_basic = (mid - multiplier * atr).to_numpy(dtype="float64")
    close = df["close"].to_numpy(dtype="float64")
    n = len(df)

    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    line = np.full(n, np.nan)

    started = False
    for i in range(n):
        if not np.isfinite(upper_basic[i]) or not np.isfinite(lower_basic[i]):
            continue
        if not started:
            upper[i], lower[i], trend[i] = upper_basic[i], lower_basic[i], 1.0
            line[i] = lower[i]
            started = True
            continue

        # The bands only tighten while the trend holds; a widening band is
        # ignored until price closes through the previous one.
        upper[i] = (
            min(upper_basic[i], upper[i - 1])
            if close[i - 1] <= upper[i - 1]
            else upper_basic[i]
        )
        lower[i] = (
            max(lower_basic[i], lower[i - 1])
            if close[i - 1] >= lower[i - 1]
            else lower_basic[i]
        )

        if trend[i - 1] == 1.0:
            trend[i] = -1.0 if close[i] < lower[i] else 1.0
        else:
            trend[i] = 1.0 if close[i] > upper[i] else -1.0
        line[i] = lower[i] if trend[i] == 1.0 else upper[i]

    return frame(
        df.index,
        **{
            f"supertrend_{period}_{multiplier}": pd.Series(line, index=df.index),
            f"supertrend_dir_{period}_{multiplier}": pd.Series(trend, index=df.index),
        },
    )


@indicator(
    "ichimoku",
    category="trend",
    outputs=(
        "tenkan_{tenkan}",
        "kijun_{kijun}",
        "senkou_a_{tenkan}_{kijun}_{displacement}",
        "senkou_b_{senkou_b}_{displacement}",
    ),
    params=(
        IntParam("tenkan", 9, doc="Conversion line period.", minimum=1, maximum=1000),
        IntParam("kijun", 26, doc="Base line period.", minimum=1, maximum=1000),
        IntParam("senkou_b", 52, doc="Leading span B period.", minimum=1, maximum=2000),
        IntParam("displacement", 26, doc="Bars the cloud is pushed forward.", minimum=1, maximum=500),
    ),
    requires=("high", "low"),
    warmup=lambda p: p["senkou_b"] + p["displacement"],
    doc=(
        "Ichimoku Kinko Hyo, causal components only: conversion line, base line, and the "
        "two leading spans that form the cloud."
    ),
    notes=(
        "The leading spans are shifted *forward*, which means the value stamped on bar t "
        "was computed from bars at t-displacement and earlier -- known at the time, so safe. "
        "The chikou span is the opposite and lives in a separate, non-causal indicator."
    ),
)
def ichimoku_indicator(
    df: pd.DataFrame,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
    displacement: int = 26,
) -> pd.DataFrame:
    def midpoint(length: int) -> pd.Series:
        return (rolling_max(df["high"], length) + rolling_min(df["low"], length)) / 2.0

    tenkan_line = midpoint(tenkan)
    kijun_line = midpoint(kijun)
    span_a = ((tenkan_line + kijun_line) / 2.0).shift(displacement)
    span_b = midpoint(senkou_b).shift(displacement)

    return frame(
        df.index,
        **{
            f"tenkan_{tenkan}": tenkan_line,
            f"kijun_{kijun}": kijun_line,
            f"senkou_a_{tenkan}_{kijun}_{displacement}": span_a,
            f"senkou_b_{senkou_b}_{displacement}": span_b,
        },
    )


@indicator(
    "ichimoku_chikou",
    category="trend",
    outputs=("chikou_{displacement}",),
    params=(
        IntParam("displacement", 26, doc="Bars the close is pushed backward.", minimum=1, maximum=500),
    ),
    causal=False,
    warmup=lambda p: 1,
    doc="Ichimoku lagging span: close shifted backwards by `displacement` bars.",
    notes=(
        "SEPARATED AND FLAGGED ON PURPOSE. The value stamped on bar t is close[t+26]. "
        "It is a legitimate charting object -- you read it against price 26 bars ago -- "
        "but it is future data sitting on a past timestamp. Fold it into a feature matrix "
        "and every backtest downstream will look brilliant and lose money live. "
        "The engine refuses it unless allow_noncausal=True."
    ),
)
def ichimoku_chikou_indicator(df: pd.DataFrame, displacement: int = 26) -> pd.DataFrame:
    return frame(
        df.index, **{f"chikou_{displacement}": df["close"].shift(-displacement)}
    )


@indicator(
    "linreg_slope",
    category="trend",
    outputs=("linreg_slope_{window}",),
    params=(IntParam("window", 20, doc="Regression window in bars.", minimum=3, maximum=2000),),
    warmup=lambda p: p["window"],
    doc=(
        "Slope of an ordinary least-squares fit of close against bar number, expressed "
        "in basis points of price per bar so it is comparable across markets."
    ),
    notes=(
        "Scale-free by construction, unlike a raw slope: USDTIRT trades near 1e5 and "
        "BTCIRT near 1e10, so an absolute slope cannot be thresholded once for both."
    ),
)
def linreg_slope_indicator(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    x = np.arange(window, dtype="float64")
    x_centered = x - x.mean()
    denominator = float((x_centered**2).sum())

    def slope(values: np.ndarray) -> float:
        return float(np.dot(x_centered, values - values.mean()) / denominator)

    raw = df["close"].rolling(window=window, min_periods=window).apply(slope, raw=True)
    return frame(
        df.index, **{f"linreg_slope_{window}": 10_000.0 * safe_ratio(raw, df["close"])}
    )


@indicator(
    "ma_cross",
    category="trend",
    outputs=("ma_cross_{fast}_{slow}", "ma_spread_bps_{fast}_{slow}"),
    params=(
        IntParam("fast", 20, doc="Fast MA length.", minimum=1, maximum=2000),
        IntParam("slow", 50, doc="Slow MA length.", minimum=2, maximum=5000),
    ),
    warmup=lambda p: ema_warmup(p["slow"]),
    doc=(
        "Fast/slow EMA relationship: `ma_cross` is +1 when fast is above slow and -1 "
        "below; `ma_spread_bps` is their separation in basis points."
    ),
    notes=(
        "The spread matters more than the sign. Two EMAs a fraction of a basis point "
        "apart cross back and forth every other bar, and each crossing costs a full "
        "round-trip in fees -- which on these venues exceeds the touch spread outright."
    ),
)
def ma_cross_indicator(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.DataFrame:
    fast_ma = ema(df["close"], fast)
    slow_ma = ema(df["close"], slow)
    spread = 10_000.0 * safe_ratio(fast_ma - slow_ma, slow_ma)
    return frame(
        df.index,
        **{
            f"ma_cross_{fast}_{slow}": np.sign(fast_ma - slow_ma),
            f"ma_spread_bps_{fast}_{slow}": spread,
        },
    )
