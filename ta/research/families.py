"""Signal families: how a raw indicator becomes a directional inventory tilt in [-1, 1]."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..cache import compute_cached
from ..signal_value import normalize_signal

MA_FAST = (3, 5, 8, 10, 15, 20, 30, 50)
MA_SLOW = (10, 20, 30, 50, 80, 120, 200, 300)
MA_KINDS = ("ema", "sma")

VOLUME_PERIODS = (7, 14, 20, 30, 50, 80, 120)

VOLUME_CAVEAT = (
    "VOLUME CAVEAT: `volume` here is the sum of trade amounts scraped from "
    "nobitex's public tape, which is capped at 20 records per poll. On a busy "
    "minute the collector sees only part of the flow, and the sampling rate "
    "changes with activity -- precisely when you would most want to trust it. "
    "Ratios and z-scores against the same feed's own history survive that; "
    "absolute levels and cross-venue comparisons do not."
)


def ma_signals(frame, dataset, cache, refresh):
    requests = [
        f"{kind}:window={length}"
        for kind in MA_KINDS
        for length in sorted(set(MA_FAST) | set(MA_SLOW))
    ]
    columns, report = compute_cached(
        frame, requests, dataset=dataset, cache=cache, refresh=refresh, warmup="mask"
    )
    signals = {}
    for kind in MA_KINDS:
        for fast in MA_FAST:
            for slow in MA_SLOW:
                if fast >= slow:
                    continue
                f, s = columns[f"{kind}_{fast}"].to_numpy(), columns[f"{kind}_{slow}"].to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    signals[f"{kind}_{fast}_{slow}"] = normalize_signal((f - s) / s)
    return signals, report


def volume_signals(frame, dataset, cache, refresh):
    """Flow indicators, each turned into a directional tilt."""
    requests = ["obv", "ad", "vpt"]
    for period in VOLUME_PERIODS:
        requests += [
            f"cmf:period={period}",
            f"mfi:period={period}",
            f"force_index:period={period}",
            f"volume_zscore:window={period}",
            f"volume_sma:window={period}",
        ]
    columns, report = compute_cached(
        frame, requests, dataset=dataset, cache=cache, refresh=refresh, warmup="mask"
    )

    close = frame["close"].to_numpy(dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        bar_return = np.concatenate([[np.nan], np.diff(np.log(close))])

    signals: dict[str, np.ndarray] = {}
    for period in VOLUME_PERIODS:
        signals[f"cmf_{period}"] = normalize_signal(columns[f"cmf_{period}"].to_numpy())
        signals[f"mfi_{period}"] = normalize_signal(columns[f"mfi_{period}"].to_numpy() - 50.0)
        signals[f"force_{period}"] = normalize_signal(columns[f"force_index_{period}"].to_numpy())
        z = columns[f"volume_zscore_{period}"].to_numpy()
        signals[f"volz_dir_{period}"] = normalize_signal(z * np.sign(bar_return))
        ratio = columns[f"volume_ratio_{period}"].to_numpy()
        signals[f"vratio_dir_{period}"] = normalize_signal(ratio * np.sign(bar_return))

    for name in ("obv", "ad", "vpt"):
        raw = columns[name].to_numpy()
        for period in (7, 20, 50):
            slope = raw - pd.Series(raw).shift(period).to_numpy()
            signals[f"{name}_slope_{period}"] = normalize_signal(slope)
    return signals, report


BROAD_PERIODS = (7, 14, 30, 80)

MACD_SETS = ((12, 26, 9), (8, 21, 5), (5, 35, 5), (19, 39, 9))


def broad_signals(frame, dataset, cache, refresh):
    """Ten different indicators, each swept over periods."""
    requests: list[str] = []
    for period in BROAD_PERIODS:
        requests += [
            f"rsi:period={period}",
            f"stoch:period={period}",
            f"cci:period={period}",
            f"williams_r:period={period}",
            f"roc:period={period}",
            f"adx:period={period}",
            f"bbands:window={period}",
            f"linreg_slope:window={period}",
            f"tsi:long={period * 2},short={period}",
        ]
    for fast, slow, signal in MACD_SETS:
        requests.append(f"macd:fast={fast},slow={slow},signal={signal}")

    columns, report = compute_cached(
        frame, requests, dataset=dataset, cache=cache, refresh=refresh, warmup="mask"
    )

    signals: dict[str, np.ndarray] = {}
    for period in BROAD_PERIODS:
        col = columns
        signals[f"rsi_{period}"] = normalize_signal(col[f"rsi_{period}"].to_numpy() - 50.0)
        signals[f"stoch_{period}"] = normalize_signal(col[f"stoch_k_{period}_3"].to_numpy() - 50.0)
        signals[f"cci_{period}"] = normalize_signal(col[f"cci_{period}"].to_numpy())
        signals[f"willr_{period}"] = normalize_signal(col[f"williams_r_{period}"].to_numpy() + 50.0)
        signals[f"roc_{period}"] = normalize_signal(col[f"roc_{period}"].to_numpy())
        signals[f"di_{period}"] = normalize_signal(
            col[f"di_plus_{period}"].to_numpy() - col[f"di_minus_{period}"].to_numpy()
        )
        signals[f"bbpct_{period}"] = normalize_signal(
            col[f"bb_percent_b_{period}_2.0"].to_numpy() - 0.5
        )
        signals[f"slope_{period}"] = normalize_signal(col[f"linreg_slope_{period}"].to_numpy())
        signals[f"tsi_{period}"] = normalize_signal(col[f"tsi_{period * 2}_{period}"].to_numpy())
    for fast, slow, signal in MACD_SETS:
        signals[f"macdh_{fast}_{slow}_{signal}"] = normalize_signal(
            columns[f"macd_hist_{fast}_{slow}_{signal}"].to_numpy()
        )
    return signals, report


FAMILIES = {"ma": ma_signals, "volume": volume_signals, "broad": broad_signals}