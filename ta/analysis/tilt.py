"""The inventory tilt: how much USDT to hold, relative to normal.

The signal from PLAN's approved proposal — `sma_20_30`, which came top on
both nobitex and ramzinex at 1H with identical parameters. It is deliberately
thin: two moving averages the project already computes, their spread
normalized to a bounded tilt. No new math, so nothing here can disagree with
what the brute force measured.

Output is `tilt` in [-1, +1]: +1 means "hold the most this market is allowed
to", -1 the least. It is *not* a price forecast and must never be read as
one — the measured edge is a covariance of about 0.34bp per bar, which is a
faint lean, not a call.

## Normalization is window-relative, on purpose

`normalize_signal` scales by the median and MAD of whatever frame it is
given. Run live on a trailing window, the scale therefore drifts slowly as
the window rolls. That is correct for shadow mode: the recorded tilt is
exactly what would have been set at that moment, using only data available
then. It does mean two runs over different windows can report slightly
different tilts for the same bar, which is why the shadow store keeps what
was actually computed rather than recomputing later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import compute_series
from ..signal_value import normalize_signal
from .types import AnalysisResult

#: The approved signal. Changing these is a research decision, not a tweak:
#: they are the parameters that replicated across two venues.
DEFAULT_FAST = 20
DEFAULT_SLOW = 30

#: Fraction of the band's half-width a tilt may move the target. PLAN sets
#: 0.30 as the standing cap and 0.15 for the first live month.
DEFAULT_CAP = 0.30


def implied_target_base(
    tilt: float, min_base: float, max_base: float, cap: float = DEFAULT_CAP
) -> float | None:
    """Where `target_base` would sit for this tilt.

    Anchored on the band midpoint, which is what the market maker uses when
    `target_base` is unset, so a tilt of zero reproduces today's behaviour
    exactly. Clamped to the band regardless of cap, because a target outside
    it would saturate `inventory_q` and quietly disable the skew.
    """
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
