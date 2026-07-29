"""Momentum and oscillator indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..frame import median_price, typical_price
from ..registry import IntParam, indicator
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


def _rsi_series(close: pd.Series, period: int) -> pd.Series:
    """Wilder RSI. Shared so Stochastic RSI cannot drift out of sync with RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = wilder(gain, period)
    avg_loss = wilder(loss, period)

    rsi = 100.0 - (100.0 / (1.0 + safe_ratio(avg_gain, avg_loss)))
    # A window with no down-ticks makes avg_loss exactly 0. safe_ratio turned
    # that into NaN to keep inf out of the arithmetic; the correct RSI there is
    # 100 (and 0 in the mirror case), so restore it explicitly. Both happen
    # routinely on thin IRT pairs that tick one way for a few bars.
    settled = avg_gain.notna() & avg_loss.notna()
    rsi = rsi.mask(settled & (avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask(settled & (avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask(settled & (avg_gain == 0) & (avg_loss == 0), 50.0)
    return rsi


@indicator(
    "rsi",
    category="momentum",
    outputs=("rsi_{period}",),
    params=(IntParam("period", 14, doc="Wilder smoothing period.", minimum=2, maximum=1000),),
    warmup=lambda p: wilder_warmup(p["period"]),
    doc="Relative Strength Index (Wilder smoothing).",
    notes=(
        "Reads 0-100. The classic 30/70 overbought-oversold interpretation assumes the "
        "series mean-reverts. USDT/IRT does not -- it ratchets on news and can hold "
        "above 70 for days while the rial repricies. Treat RSI here as a momentum "
        "measure, not a fade signal."
    ),
)
def rsi_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    return frame(df.index, **{f"rsi_{period}": _rsi_series(df["close"], period)})


@indicator(
    "stoch_rsi",
    category="momentum",
    outputs=("stoch_rsi_k_{period}_{smooth_k}", "stoch_rsi_d_{period}_{smooth_k}_{smooth_d}"),
    params=(
        IntParam("period", 14, doc="RSI period and stochastic lookback.", minimum=2, maximum=1000),
        IntParam("smooth_k", 3, doc="%K smoothing.", minimum=1, maximum=100),
        IntParam("smooth_d", 3, doc="%D smoothing of %K.", minimum=1, maximum=100),
    ),
    warmup=lambda p: wilder_warmup(p["period"]) + p["period"] + p["smooth_k"] + p["smooth_d"],
    aliases=("stochastic_rsi", "stochrsi"),
    doc="Stochastic RSI: where RSI sits within its own recent range, 0-100.",
    notes=(
        "Doubly derived, so it warms up slowly and is noisier than either input. "
        "When RSI is flat across the lookback the denominator is zero; that bar is NaN "
        "rather than an arbitrary 0 or 100."
    ),
)
def stoch_rsi_indicator(
    df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> pd.DataFrame:
    rsi = _rsi_series(df["close"], period)
    lowest = rolling_min(rsi, period)
    highest = rolling_max(rsi, period)
    stoch = 100.0 * safe_ratio(rsi - lowest, highest - lowest)
    k = sma(stoch, smooth_k)
    d = sma(k, smooth_d)
    return frame(
        df.index,
        **{
            f"stoch_rsi_k_{period}_{smooth_k}": k,
            f"stoch_rsi_d_{period}_{smooth_k}_{smooth_d}": d,
        },
    )


@indicator(
    "stoch",
    category="momentum",
    outputs=("stoch_k_{period}_{smooth_k}", "stoch_d_{period}_{smooth_k}_{smooth_d}"),
    params=(
        IntParam("period", 14, doc="Lookback for the high/low range.", minimum=2, maximum=1000),
        IntParam("smooth_k", 3, doc="%K smoothing (1 = fast stochastic).", minimum=1, maximum=100),
        IntParam("smooth_d", 3, doc="%D smoothing of %K.", minimum=1, maximum=100),
    ),
    requires=("high", "low", "close"),
    warmup=lambda p: p["period"] + p["smooth_k"] + p["smooth_d"],
    doc="Stochastic oscillator: close's position within the period's high/low range.",
)
def stoch_indicator(
    df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> pd.DataFrame:
    lowest = rolling_min(df["low"], period)
    highest = rolling_max(df["high"], period)
    raw_k = 100.0 * safe_ratio(df["close"] - lowest, highest - lowest)
    k = sma(raw_k, smooth_k)
    d = sma(k, smooth_d)
    return frame(
        df.index,
        **{
            f"stoch_k_{period}_{smooth_k}": k,
            f"stoch_d_{period}_{smooth_k}_{smooth_d}": d,
        },
    )


@indicator(
    "macd",
    category="momentum",
    outputs=(
        "macd_{fast}_{slow}_{signal}",
        "macd_signal_{fast}_{slow}_{signal}",
        "macd_hist_{fast}_{slow}_{signal}",
    ),
    params=(
        IntParam("fast", 12, doc="Fast EMA span.", minimum=1, maximum=2000),
        IntParam("slow", 26, doc="Slow EMA span.", minimum=2, maximum=5000),
        IntParam("signal", 9, doc="Signal-line EMA span.", minimum=1, maximum=1000),
    ),
    warmup=lambda p: ema_warmup(p["slow"]) + ema_warmup(p["signal"]),
    doc="Moving Average Convergence Divergence with signal line and histogram.",
    notes=(
        "Absolute price units, so MACD values are not comparable across markets or "
        "across a big repricing of the same market. Use PPO when you need that."
    ),
)
def macd_indicator(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    macd_line = ema(df["close"], fast) - ema(df["close"], slow)
    signal_line = ema(macd_line, signal)
    return frame(
        df.index,
        **{
            f"macd_{fast}_{slow}_{signal}": macd_line,
            f"macd_signal_{fast}_{slow}_{signal}": signal_line,
            f"macd_hist_{fast}_{slow}_{signal}": macd_line - signal_line,
        },
    )


@indicator(
    "ppo",
    category="momentum",
    outputs=(
        "ppo_{fast}_{slow}_{signal}",
        "ppo_signal_{fast}_{slow}_{signal}",
        "ppo_hist_{fast}_{slow}_{signal}",
    ),
    params=(
        IntParam("fast", 12, doc="Fast EMA span.", minimum=1, maximum=2000),
        IntParam("slow", 26, doc="Slow EMA span.", minimum=2, maximum=5000),
        IntParam("signal", 9, doc="Signal-line EMA span.", minimum=1, maximum=1000),
    ),
    warmup=lambda p: ema_warmup(p["slow"]) + ema_warmup(p["signal"]),
    doc="Percentage Price Oscillator: MACD expressed as a percentage of the slow EMA.",
    notes="Scale-free version of MACD. Prefer it whenever you compare across markets.",
)
def ppo_indicator(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    fast_ema, slow_ema = ema(df["close"], fast), ema(df["close"], slow)
    ppo = 100.0 * safe_ratio(fast_ema - slow_ema, slow_ema)
    signal_line = ema(ppo, signal)
    return frame(
        df.index,
        **{
            f"ppo_{fast}_{slow}_{signal}": ppo,
            f"ppo_signal_{fast}_{slow}_{signal}": signal_line,
            f"ppo_hist_{fast}_{slow}_{signal}": ppo - signal_line,
        },
    )


@indicator(
    "cci",
    category="momentum",
    outputs=("cci_{period}",),
    params=(IntParam("period", 20, doc="Lookback length.", minimum=2, maximum=1000),),
    requires=("high", "low", "close"),
    warmup=lambda p: p["period"],
    doc=(
        "Commodity Channel Index: typical price's deviation from its mean, scaled by "
        "mean absolute deviation. +/-100 is the conventional band."
    ),
)
def cci_indicator(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    tp = typical_price(df)
    mean = sma(tp, period)
    mad = tp.rolling(window=period, min_periods=period).apply(
        lambda v: float(np.mean(np.abs(v - v.mean()))), raw=True
    )
    return frame(df.index, **{f"cci_{period}": safe_ratio(tp - mean, 0.015 * mad)})


@indicator(
    "williams_r",
    category="momentum",
    outputs=("williams_r_{period}",),
    params=(IntParam("period", 14, doc="Lookback length.", minimum=2, maximum=1000),),
    requires=("high", "low", "close"),
    warmup=lambda p: p["period"],
    doc="Williams %R: an inverted stochastic, ranging -100 (low) to 0 (high).",
)
def williams_r_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    highest = rolling_max(df["high"], period)
    lowest = rolling_min(df["low"], period)
    value = -100.0 * safe_ratio(highest - df["close"], highest - lowest)
    return frame(df.index, **{f"williams_r_{period}": value})


@indicator(
    "roc",
    category="momentum",
    outputs=("roc_{period}",),
    params=(IntParam("period", 12, doc="Bars to look back.", minimum=1, maximum=5000),),
    warmup=lambda p: p["period"] + 1,
    doc="Rate of change: percentage change in close over `period` bars.",
)
def roc_indicator(df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
    prior = df["close"].shift(period)
    return frame(
        df.index, **{f"roc_{period}": 100.0 * safe_ratio(df["close"] - prior, prior)}
    )


@indicator(
    "momentum",
    category="momentum",
    outputs=("momentum_{period}",),
    params=(IntParam("period", 10, doc="Bars to look back.", minimum=1, maximum=5000),),
    warmup=lambda p: p["period"] + 1,
    doc="Raw price momentum: close minus close `period` bars ago, in price units.",
)
def momentum_indicator(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    return frame(
        df.index, **{f"momentum_{period}": df["close"] - df["close"].shift(period)}
    )


@indicator(
    "tsi",
    category="momentum",
    outputs=("tsi_{long}_{short}", "tsi_signal_{long}_{short}_{signal}"),
    params=(
        IntParam("long", 25, doc="First smoothing span.", minimum=2, maximum=2000),
        IntParam("short", 13, doc="Second smoothing span.", minimum=1, maximum=1000),
        IntParam("signal", 13, doc="Signal-line span.", minimum=1, maximum=1000),
    ),
    warmup=lambda p: ema_warmup(p["long"]) + ema_warmup(p["short"]) + ema_warmup(p["signal"]),
    doc=(
        "True Strength Index: doubly-smoothed momentum divided by doubly-smoothed "
        "absolute momentum, -100 to +100."
    ),
    notes="Much smoother than RSI at the cost of a long warmup -- three stacked EMAs.",
)
def tsi_indicator(
    df: pd.DataFrame, long: int = 25, short: int = 13, signal: int = 13
) -> pd.DataFrame:
    change = df["close"].diff()
    smooth = ema(ema(change, long), short)
    smooth_abs = ema(ema(change.abs(), long), short)
    tsi = 100.0 * safe_ratio(smooth, smooth_abs)
    return frame(
        df.index,
        **{
            f"tsi_{long}_{short}": tsi,
            f"tsi_signal_{long}_{short}_{signal}": ema(tsi, signal),
        },
    )


@indicator(
    "ultimate_oscillator",
    category="momentum",
    outputs=("uo_{short}_{medium}_{long}",),
    params=(
        IntParam("short", 7, doc="Short averaging period.", minimum=1, maximum=1000),
        IntParam("medium", 14, doc="Medium averaging period.", minimum=2, maximum=2000),
        IntParam("long", 28, doc="Long averaging period.", minimum=3, maximum=5000),
    ),
    requires=("high", "low", "close"),
    warmup=lambda p: p["long"] + 1,
    aliases=("uo",),
    doc=(
        "Ultimate Oscillator: buying pressure over true range at three timescales, "
        "weighted 4:2:1 toward the shortest. 0-100."
    ),
    notes="Built to avoid the false divergences a single-period oscillator produces.",
)
def ultimate_oscillator_indicator(
    df: pd.DataFrame, short: int = 7, medium: int = 14, long: int = 28
) -> pd.DataFrame:
    prev_close = df["close"].shift(1)
    true_low = pd.concat([df["low"], prev_close], axis=1).min(axis=1)
    true_high = pd.concat([df["high"], prev_close], axis=1).max(axis=1)
    buying_pressure = df["close"] - true_low
    tr = true_high - true_low

    def average(length: int) -> pd.Series:
        return safe_ratio(rolling_sum(buying_pressure, length), rolling_sum(tr, length))

    uo = 100.0 * (4 * average(short) + 2 * average(medium) + average(long)) / 7.0
    return frame(df.index, **{f"uo_{short}_{medium}_{long}": uo})


@indicator(
    "awesome_oscillator",
    category="momentum",
    outputs=("ao_{fast}_{slow}",),
    params=(
        IntParam("fast", 5, doc="Fast SMA of the median price.", minimum=1, maximum=1000),
        IntParam("slow", 34, doc="Slow SMA of the median price.", minimum=2, maximum=5000),
    ),
    requires=("high", "low"),
    warmup=lambda p: p["slow"],
    aliases=("ao",),
    doc="Awesome Oscillator: SMA(median price, fast) - SMA(median price, slow).",
)
def awesome_oscillator_indicator(
    df: pd.DataFrame, fast: int = 5, slow: int = 34
) -> pd.DataFrame:
    mid = median_price(df)
    return frame(df.index, **{f"ao_{fast}_{slow}": sma(mid, fast) - sma(mid, slow)})


@indicator(
    "cmo",
    category="momentum",
    outputs=("cmo_{period}",),
    params=(IntParam("period", 14, doc="Lookback length.", minimum=2, maximum=1000),),
    warmup=lambda p: p["period"] + 1,
    doc=(
        "Chande Momentum Oscillator: (gains - losses) / (gains + losses) over the "
        "period, -100 to +100."
    ),
    notes=(
        "Unsmoothed, so it reaches its extremes far more readily than RSI. Its warmup "
        "is a clean window boundary rather than an EMA convergence, which makes it the "
        "better choice on short frames."
    ),
)
def cmo_indicator(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    delta = df["close"].diff()
    gains = rolling_sum(delta.clip(lower=0.0), period)
    losses = rolling_sum(-delta.clip(upper=0.0), period)
    return frame(
        df.index, **{f"cmo_{period}": 100.0 * safe_ratio(gains - losses, gains + losses)}
    )
