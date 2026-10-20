"""The brute force itself, as a callable -- one implementation, three callers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import ta
from ..cache import ResearchCache, frame_fingerprint
from ..errors import FrameError
from ..ohlc import read_ohlc
from ..signal_value import (
    consecutive_mask,
    evaluate,
    forward_returns,
    permutation_pvalue,
)
from .families import FAMILIES, VOLUME_CAVEAT

JOB_NAME = "signal_brute"


def run_brute(
    *,
    market: str,
    source: str,
    interval: str = "1H",
    family: str = "ma",
    permutations: int = 200,
    split: float = 0.7,
    horizon: int = 1,
    limit: int = 100_000,
    consecutive_only: bool = True,
    cache: ResearchCache | None = None,
    refresh: bool = False,
    rerun: bool = False,
    log=None,
) -> dict[str, Any]:
    """Score one family of signals on one market. Returns the run payload."""
    say = log or (lambda *a: None)
    if family not in FAMILIES:
        raise FrameError(f"unknown family {family!r}; known: {', '.join(sorted(FAMILIES))}")

    params = {
        "family": family, "market": market, "source": source, "interval": interval,
        "permutations": permutations, "split": split, "horizon": horizon,
        "consecutive_only": bool(consecutive_only),
    }

    frame = read_ohlc(market, source, interval, limit=limit)
    if frame is None:
        raise FrameError(f"no candles for {market}/{source} {interval}")
    frame, info = ta.normalize_ohlcv(frame)
    dataset = frame_fingerprint(frame)

    if cache is not None and not rerun:
        cached = cache.get_run(JOB_NAME, dataset, params)
        if cached:
            say("cached run: identical inputs, returning stored result")
            return {**cached, "from_cache": True}

    say(f"{market}/{source} {interval}: {info.rows} bars, dataset={dataset}")
    if family == "volume":
        say(VOLUME_CAVEAT)

    fwd = forward_returns(frame["close"].to_numpy(), horizon)
    if consecutive_only:
        keep = consecutive_mask(frame.index, info.interval_sec)
        dropped = int(np.isfinite(fwd).sum() - (np.isfinite(fwd) & keep).sum())
        fwd = np.where(keep, fwd, np.nan)
        say(f"consecutive-only: dropped {dropped} gap-spanning pairs")

    signals, report = FAMILIES[family](frame, dataset, cache, refresh)
    say(f"series cache: {report['hits']} hits, {report['misses']} misses; "
        f"{len(signals)} signals")

    interval_sec = info.interval_sec or 3600.0
    split_at = int(len(frame) * split)
    rows = []
    for name, tilt in signals.items():
        full = evaluate(tilt, fwd, bar_seconds=interval_sec)
        if full is None:
            continue
        ins = evaluate(tilt[:split_at], fwd[:split_at], bar_seconds=interval_sec)
        oos = evaluate(tilt[split_at:], fwd[split_at:], bar_seconds=interval_sec)
        rows.append({
            "signal": name, "excess_bps": full.excess_bps, "ic": full.ic, "t": full.ic_t,
            "excess_ann_pct": full.excess_ann_pct, "naive_bps": full.naive_bps,
            "drift_bps": full.drift_bps, "turnover": full.turnover,
            "hit_rate": full.hit_rate, "bars": full.bars,
            "is_excess_bps": ins.excess_bps if ins else float("nan"),
            "oos_excess_bps": oos.excess_bps if oos else float("nan"),
        })
    if not rows:
        raise FrameError("no signal produced enough overlapping bars to score")

    grid = pd.DataFrame(rows).sort_values("excess_bps", ascending=False)
    best = grid.iloc[0]
    null = permutation_pvalue(
        float(grid["excess_bps"].max()), list(signals.values()), fwd, trials=permutations
    )
    is_oos = float(grid["is_excess_bps"].corr(grid["oos_excess_bps"]))

    headline = {
        "family": family, "market": market, "source": source, "interval": interval,
        "bars": int(info.rows), "consecutive_only": bool(consecutive_only),
        "drift_bps_per_bar": float(np.nanmean(fwd)) * 1e4,
        "best_signal": str(best["signal"]), "best_excess_bps": float(best["excess_bps"]),
        "best_ic": float(best["ic"]), "best_t": float(best["t"]),
        "best_turnover": float(best["turnover"]),
        "best_is_excess": float(best["is_excess_bps"]),
        "best_oos_excess": float(best["oos_excess_bps"]),
        "is_oos_correlation": is_oos, "p_value": null["p_value"],
        "null_p95_bps": null["null_p95_bps"], "signals": len(signals),
    }
    say(f"best {headline['best_signal']}: excess {headline['best_excess_bps']:.3f}bp "
        f"t={headline['best_t']:.2f} p={null['p_value']:.3f}")

    payload = {
        "headline": headline,
        "grid": grid.to_dict("records"),
        "permutation": null,
        "from_cache": False,
    }
    if cache is not None:
        cache.put_run(JOB_NAME, dataset, params, payload)
    return payload
