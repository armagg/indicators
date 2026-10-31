"""Is a signal worth tilting inventory on? Scores Cov(signal, return), not naive P&L."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

_SECONDS_PER_YEAR = 365.0 * 24.0 * 3600.0


@dataclass(frozen=True)
class SignalValue:
    """What tilting inventory on one signal would have earned."""

    bars: int
    ic: float
    ic_t: float
    excess_bps: float
    excess_ann_pct: float
    naive_bps: float
    drift_bps: float
    mean_signal: float
    turnover: float
    hit_rate: float

    def as_dict(self) -> dict:
        return asdict(self)


def forward_returns(close, horizon: int = 1) -> np.ndarray:
    """Log return from bar `t` to bar `t + horizon`, stamped on `t`."""
    prices = np.asarray(close, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(prices)
    out = np.full(len(prices), np.nan)
    if len(prices) > horizon:
        out[:-horizon] = logs[horizon:] - logs[:-horizon]
    return out


def consecutive_mask(
    index: pd.DatetimeIndex, interval_sec: float | None = None, tolerance: float = 0.25
) -> np.ndarray:
    """True on bars whose successor is the very next bar on the nominal grid."""
    if len(index) < 2:
        return np.zeros(len(index), dtype=bool)
    deltas = index.to_series().diff().shift(-1).dt.total_seconds().to_numpy()
    if interval_sec is None:
        positive = deltas[np.isfinite(deltas) & (deltas > 0)]
        interval_sec = float(np.median(positive)) if positive.size else 0.0
    if not interval_sec:
        return np.zeros(len(index), dtype=bool)
    with np.errstate(invalid="ignore"):
        return np.abs(deltas - interval_sec) <= tolerance * interval_sec


def normalize_signal(values, clip: float = 3.0) -> np.ndarray:
    """Map an arbitrary signal onto a bounded tilt in [-1, 1]."""
    x = np.asarray(values, dtype="float64")
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.full(len(x), np.nan)

    centre = float(np.median(finite))
    # Rolling averages can leave a few hundred ULPs of dispersion in an
    # otherwise constant spread. Normalizing that roundoff invents a tilt.
    # Use a relative floor so genuinely small, varying signals still work.
    roundoff = 512 * np.finfo("float64").eps * float(np.max(np.abs(finite)))
    if np.ptp(finite) <= roundoff:
        return np.where(np.isfinite(x), 0.0, np.nan)
    mad = float(np.median(np.abs(finite - centre)))
    scale = mad * 1.4826 if mad > 0 else float(np.std(finite))
    if scale <= 0:
        return np.zeros(len(x))
    return np.clip((x - centre) / scale, -clip, clip) / clip


def evaluate(signal, forward, *, bar_seconds: float | None = None) -> SignalValue | None:
    """Score one signal against the returns that followed it."""
    s = np.asarray(signal, dtype="float64")
    r = np.asarray(forward, dtype="float64")
    if len(s) != len(r):
        raise ValueError(f"signal and forward differ in length: {len(s)} vs {len(r)}")

    mask = np.isfinite(s) & np.isfinite(r)
    if mask.sum() < 100:
        return None
    s, r = s[mask], r[mask]

    naive = float(np.mean(s * r))
    drift = float(np.mean(s) * np.mean(r))
    excess = naive - drift

    tilt_pnl = (s - s.mean()) * r
    deviation = float(np.std(tilt_pnl))
    t_stat = (
        float(np.mean(tilt_pnl) / (deviation / np.sqrt(len(tilt_pnl))))
        if deviation > 0
        else 0.0
    )

    sd_s, sd_r = float(np.std(s)), float(np.std(r))
    ic = float(excess / (sd_s * sd_r)) if sd_s > 0 and sd_r > 0 else 0.0

    bars_per_year = (
        _SECONDS_PER_YEAR / bar_seconds if bar_seconds else _SECONDS_PER_YEAR / 3600.0
    )
    moved = s[s != 0]
    return SignalValue(
        bars=int(mask.sum()),
        ic=ic,
        ic_t=t_stat,
        excess_bps=excess * 1e4,
        excess_ann_pct=excess * bars_per_year * 100.0,
        naive_bps=naive * 1e4,
        drift_bps=drift * 1e4,
        mean_signal=float(np.mean(s)),
        turnover=float(np.mean(np.abs(np.diff(s)))) if len(s) > 1 else 0.0,
        hit_rate=float(np.mean(np.sign(moved) == np.sign(r[s != 0]))) if moved.size else 0.0,
    )


def permutation_pvalue(
    best_excess: float, signals: list[np.ndarray], forward: np.ndarray, *, trials: int = 100, seed: int = 7
) -> dict:
    """How often a grid this size finds `best_excess` on data with no signal."""
    rng = np.random.default_rng(seed)
    r = np.asarray(forward, dtype="float64")
    finite = np.isfinite(r)
    best_per_trial = []

    for _ in range(trials):
        shuffled = r.copy()
        shuffled[finite] = rng.permutation(r[finite])
        best = -np.inf
        for s in signals:
            scored = evaluate(s, shuffled)
            if scored and scored.excess_bps > best:
                best = scored.excess_bps
        if np.isfinite(best):
            best_per_trial.append(best)

    null = np.array(best_per_trial)
    return {
        "trials": len(null),
        "null_mean_bps": float(null.mean()) if null.size else None,
        "null_p95_bps": float(np.percentile(null, 95)) if null.size else None,
        "null_max_bps": float(null.max()) if null.size else None,
        "p_value": float((null >= best_excess).mean()) if null.size else None,
    }
