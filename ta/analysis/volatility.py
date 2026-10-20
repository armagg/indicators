"""Volatility analysis: realized vol, NATR, and the self-calibrating vol percentile."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import compute_series
from ..frame import rolling_std
from ..indicators._common import bars_per_year
from .types import AnalysisResult

_REGIMES = ((0.25, "calm"), (0.75, "normal"), (0.90, "elevated"), (1.01, "extreme"))


def run(
    frame: pd.DataFrame,
    window: int = 20,
    natr_period: int = 14,
) -> AnalysisResult:
    """Realized volatility (two estimators), NATR, and the vol percentile."""
    warnings: list[str] = []
    annualiser = np.sqrt(bars_per_year(frame.index))

    log_close = np.log(frame["close"].astype("float64"))
    returns = log_close.diff()
    cc = rolling_std(returns, window, ddof=0) * annualiser * 100.0

    parkinson = None
    if {"high", "low"} <= set(frame.columns):
        hl = np.log(frame["high"] / frame["low"]) ** 2
        parkinson = (
            np.sqrt(hl.rolling(window, min_periods=window).mean() / (4.0 * np.log(2.0)))
            * annualiser
            * 100.0
        )
    else:
        warnings.append("no high/low columns: Parkinson estimator skipped")

    natr_series = None
    if {"high", "low"} <= set(frame.columns):
        natr_series = compute_series(frame, f"natr:period={natr_period}")[
            f"natr_{natr_period}"
        ]
    else:
        warnings.append("no high/low columns: NATR skipped")

    history = cc.dropna()
    if history.empty:
        percentile = None
        regime = None
        warnings.append(
            f"frame too short for a {window}-bar vol window: no realized vol at all"
        )
    else:
        current = history.iloc[-1]
        percentile = float((history <= current).mean())
        regime = next(label for bound, label in _REGIMES if percentile < bound)
        if len(history) < 30:
            warnings.append(
                f"vol_percentile computed from only {len(history)} readings; "
                f"treat it as noise until more history accumulates"
            )

    def last(series: pd.Series | None) -> float | None:
        if series is None:
            return None
        valid = series.dropna()
        return float(valid.iloc[-1]) if not valid.empty else None

    scalars = {
        "realized_vol_cc": last(cc),
        "realized_vol_parkinson": last(parkinson),
        "natr": last(natr_series),
        "vol_percentile": percentile,
        "vol_regime": regime,
    }

    series = {"realized_vol_cc": cc}
    if natr_series is not None:
        series["natr"] = natr_series

    return AnalysisResult(
        name="volatility",
        params={"window": window, "natr_period": natr_period},
        asof=frame.index[-1] if len(frame) else None,
        scalars=scalars,
        series=series,
        warnings=warnings,
    )
