"""Trend analysis: direction and strength, folded into one actionable score.

Reuses registered indicators via the engine (never reimplements their math):
ADX for strength, and three direction votes -- fast/slow EMA relationship,
close vs KAMA, DI+ vs DI-. Votes are *continuous* in [-1, +1], scaled by how
large the displacement is relative to current volatility (one ATR), not raw
signs: in a market wobbling ±5bp the fast EMA sits trivially above the slow
one half the time, and a sign vote would read that as "up" with a straight
face. `ma_cross`'s own catalogue notes warn about exactly this; the analysis
must not repeat the mistake.

    trend_score = mean(votes) x min(1, ADX/40)      in [-1, +1]

Only a market that is directional, *moved further than its own noise*, and
trending per ADX produces a score the tilt overlay will act on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import compute_series
from .types import AnalysisResult

_BUCKETS = ((20.0, "absent"), (25.0, "weak"), (40.0, "trending"), (np.inf, "strong"))

_FLAT_BAND = 0.15


def run(
    frame: pd.DataFrame,
    adx_period: int = 14,
    kama_window: int = 10,
    fast: int = 20,
    slow: int = 50,
) -> AnalysisResult:
    warnings: list[str] = []

    requests = [
        f"kama:window={kama_window}",
        f"ma_cross:fast={fast},slow={slow}",
        "stddev:window=20",
    ]
    has_hl = {"high", "low"} <= set(frame.columns)
    if has_hl:
        requests.append(f"adx:period={adx_period}")
        requests.append(f"atr:period={adx_period}")
    else:
        warnings.append("no high/low columns: ADX skipped, strength weight is 0")

    computed = compute_series(frame, requests)

    def last(column: str) -> float | None:
        if column not in computed.columns:
            return None
        valid = computed[column].dropna()
        return float(valid.iloc[-1]) if not valid.empty else None

    adx = last(f"adx_{adx_period}") if has_hl else None
    di_plus = last(f"di_plus_{adx_period}") if has_hl else None
    di_minus = last(f"di_minus_{adx_period}") if has_hl else None
    kama = last(f"kama_{kama_window}")
    ma_spread_bps = last(f"ma_spread_bps_{fast}_{slow}")
    close = float(frame["close"].iloc[-1])

    scale = last(f"atr_{adx_period}") if has_hl else None
    if not scale:
        scale = last("stddev_20")
    if not scale:
        warnings.append("no volatility scale available: falling back to sign votes")

    def scaled_vote(displacement: float) -> float:
        if scale:
            return float(np.clip(displacement / scale, -1.0, 1.0))
        return float(np.sign(displacement))

    votes: dict[str, float | None] = {
        "ma_cross": scaled_vote(ma_spread_bps / 1e4 * close)
        if ma_spread_bps is not None
        else None,
        "kama": scaled_vote(close - kama) if kama is not None else None,
        "di": float(np.clip((di_plus - di_minus) / (di_plus + di_minus), -1.0, 1.0))
        if di_plus is not None and di_minus is not None and (di_plus + di_minus) > 0
        else None,
    }
    cast = [v for v in votes.values() if v is not None]
    if len(cast) < len(votes):
        missing = sorted(k for k, v in votes.items() if v is None)
        warnings.append(f"direction vote(s) unavailable (warmup or inputs): {', '.join(missing)}")

    direction_vote = float(np.mean(cast)) if cast else 0.0

    if adx is None:
        strength = 0.0
        bucket = None
        if has_hl:
            warnings.append("ADX still in warmup: strength weight is 0, score forced flat")
    else:
        strength = float(np.clip(adx / 40.0, 0.0, 1.0))
        bucket = next(label for bound, label in _BUCKETS if adx < bound)

    score = direction_vote * strength
    direction = "flat" if abs(score) < _FLAT_BAND else ("up" if score > 0 else "down")

    scalars = {
        "trend_score": score,
        "direction": direction,
        "adx": adx,
        "adx_bucket": bucket,
        "di_plus": di_plus,
        "di_minus": di_minus,
        "ma_spread_bps": ma_spread_bps,
        "vote_ma_cross": votes["ma_cross"],
        "vote_kama": votes["kama"],
        "vote_di": votes["di"],
    }

    series = {}
    if has_hl and f"adx_{adx_period}" in computed.columns:
        series["adx"] = computed[f"adx_{adx_period}"]

    return AnalysisResult(
        name="trend",
        params={
            "adx_period": adx_period,
            "kama_window": kama_window,
            "fast": fast,
            "slow": slow,
        },
        asof=frame.index[-1] if len(frame) else None,
        scalars=scalars,
        series=series,
        warnings=warnings,
    )
