"""Non-indicator analyses: structured verdicts over one OHLCV frame."""

from __future__ import annotations

import pandas as pd

from ..errors import ParamError
from . import levels, regime, tilt, trend, volatility
from .types import AnalysisResult, call_with_loose_params

ANALYSES = {
    "volatility": volatility.run,
    "trend": trend.run,
    "regime": regime.run,
    "tilt": tilt.run,
    "levels": levels.run,
}


def list_analyses() -> list[str]:
    return sorted(ANALYSES)


def run_analysis(name: str, frame: pd.DataFrame, params=None) -> AnalysisResult:
    """Look up and run one analysis with loosely-typed params (CLI/query)."""
    key = str(name).strip().lower()
    fn = ANALYSES.get(key)
    if fn is None:
        raise ParamError(f"unknown analysis {name!r}; known: {', '.join(list_analyses())}")
    return call_with_loose_params(fn, dict(params or {}), frame, label=key)
