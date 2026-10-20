"""The inventory tilt: how much USDT to hold, relative to normal, in [-1, +1]."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import compute_series
from ..signal_value import normalize_signal
from .types import AnalysisResult

DEFAULT_FAST = 20
DEFAULT_SLOW = 30

DEFAULT_CAP = 0.30


def implied_target_base(
    tilt: float, min_base: float, max_base: float, cap: float = DEFAULT_CAP
) -> float | None:
    """Where `target_base` would sit for this tilt, clamped to the band."""
    if min_base is None or max_base is None or max_base <= min_base:
        return None
    midpoint = (min_base + max_base) / 2.0
    half_band = (max_base - min_base) / 2.0
    target = midpoint + float(np.clip(tilt, -1.0, 1.0)) * cap * half_band
    return float(np.clip(target, min_base, max_base))


def run(
    frame: pd.DataFrame,
    fast: int = DEFAULT_FAST,
    slow: int = DEFAULT_SLOW,
    cap: float = DEFAULT_CAP,
) -> AnalysisResult:
    """Current tilt for one market, from its own candles."""
    warnings: list[str] = []
    if fast >= slow:
        raise ValueError(f"fast ({fast}) must be shorter than slow ({slow})")

    columns = compute_series(frame, [f"sma:window={fast}", f"sma:window={slow}"])
    fast_ma = columns[f"sma_{fast}"].to_numpy(dtype="float64")
    slow_ma = columns[f"sma_{slow}"].to_numpy(dtype="float64")

    with np.errstate(divide="ignore", invalid="ignore"):
        spread = (fast_ma - slow_ma) / slow_ma
    tilts = normalize_signal(spread)

    usable = np.isfinite(tilts)
    if not usable.any():
        warnings.append(
            f"frame too short for sma_{fast}_{slow}: no tilt available yet"
        )
        tilt = None
    else:
        tilt = float(tilts[usable][-1])
        if usable.sum() < 200:
            warnings.append(
                f"only {int(usable.sum())} bars of tilt history; the normalization "
                f"scale is still noisy"
            )

    spread_bps = float(spread[-1] * 1e4) if np.isfinite(spread[-1]) else None

    return AnalysisResult(
        name="tilt",
        params={"fast": fast, "slow": slow, "cap": cap},
        asof=frame.index[-1] if len(frame) else None,
        scalars={
            "tilt": tilt,
            "ma_spread_bps": spread_bps,
            "fast_ma": float(fast_ma[-1]) if np.isfinite(fast_ma[-1]) else None,
            "slow_ma": float(slow_ma[-1]) if np.isfinite(slow_ma[-1]) else None,
            "cap": cap,
            "direction": (
                None if tilt is None else ("hold more" if tilt > 0 else "hold less")
            ),
        },
        series={"tilt": pd.Series(tilts, index=frame.index)},
        warnings=warnings,
    )
