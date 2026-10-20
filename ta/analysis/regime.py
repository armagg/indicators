"""Squeeze regime: Bollinger Bands inside Keltner Channels, and the bar that releases it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import compute_series
from .types import AnalysisResult

_MAX_EVENTS = 20


def run(
    frame: pd.DataFrame,
    bb_window: int = 20,
    bb_std: float = 2.0,
    kc_window: int = 20,
    kc_atr: int = 10,
    kc_multiplier: float = 1.5,
) -> AnalysisResult:
    bb_std = float(bb_std)
    kc_multiplier = float(kc_multiplier)
    computed = compute_series(
        frame,
        [
            f"bbands:window={bb_window},std_dev={bb_std}",
            f"keltner:window={kc_window},atr_period={kc_atr},multiplier={kc_multiplier}",
        ],
    )

    bb_upper = computed[f"bb_upper_{bb_window}_{bb_std}"]
    bb_lower = computed[f"bb_lower_{bb_window}_{bb_std}"]
    bb_mid = computed[f"bb_mid_{bb_window}"]
    kc_upper = computed[f"kc_upper_{kc_window}_{kc_atr}_{kc_multiplier}"]
    kc_lower = computed[f"kc_lower_{kc_window}_{kc_atr}_{kc_multiplier}"]

    known = bb_upper.notna() & bb_lower.notna() & kc_upper.notna() & kc_lower.notna()
    squeeze = (bb_upper < kc_upper) & (bb_lower > kc_lower) & known

    warnings: list[str] = []
    if not known.any():
        warnings.append(
            "frame shorter than the band/channel warmup: squeeze state unknown"
        )

    prev_squeeze = squeeze.shift(1, fill_value=False)
    release_mask = prev_squeeze & ~squeeze & known
    events = []
    for ts in frame.index[release_mask]:
        close = float(frame.loc[ts, "close"])
        mid = bb_mid.loc[ts]
        direction = "up" if pd.notna(mid) and close >= float(mid) else "down"
        events.append(
            {"time": ts.isoformat(), "event": "squeeze_release", "direction": direction}
        )
    events = events[-_MAX_EVENTS:]

    squeeze_on = bool(squeeze.iloc[-1]) if len(squeeze) else False
    released_now = bool(release_mask.iloc[-1]) if len(release_mask) else False

    bars_in_squeeze = 0
    for value in squeeze.iloc[::-1]:
        if not value:
            break
        bars_in_squeeze += 1

    scalars = {
        "squeeze_on": squeeze_on,
        "bars_in_squeeze": bars_in_squeeze,
        "released_now": released_now,
        "release_direction": events[-1]["direction"] if released_now and events else None,
        "release_count": int(release_mask.sum()),
    }

    return AnalysisResult(
        name="regime",
        params={
            "bb_window": bb_window,
            "bb_std": bb_std,
            "kc_window": kc_window,
            "kc_atr": kc_atr,
            "kc_multiplier": kc_multiplier,
        },
        asof=frame.index[-1] if len(frame) else None,
        scalars=scalars,
        series={"squeeze_on": squeeze.astype("float64")},
        events=events,
        warnings=warnings,
    )
