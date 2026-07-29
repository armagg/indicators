"""`ta` -- the technical-analysis toolkit for this repo.

Library first, HTTP second (ARCHITECTURE.md): everything here is importable
and pure; `app.py` is a thin adapter over this surface, and batch jobs should
import this package directly rather than going through HTTP.

Typical use:

    import ta

    result = ta.compute(candles, "rsi, macd, atr, bbands:window=20")
    result.frame          # OHLCV + indicator columns
    result.latest()       # last value of every indicator column
    result.meta()         # frame health, warmup, per-column provenance

`candles` is anything with OHLCV in it -- a DataFrame, or the list of dicts a
Mongo cursor yields. See `ta.ohlc.read_ohlc` for the source-aware reader and
`ta.store.save` for idempotent persistence of results.
"""

from .engine import (
    ComputeResult,
    IndicatorRequest,
    compute,
    compute_series,
    parse_requests,
)
from .errors import (
    FrameError,
    InsufficientData,
    NonCausalIndicator,
    ParamError,
    SourceError,
    TAError,
    UnknownIndicator,
)
from .frame import FrameInfo, drop_incomplete_bar, normalize_ohlcv
from .registry import all_specs, catalogue, get_spec, list_indicators

# Importing the subpackage is what fills the registry; keep it last so the
# registry decorator and its dependencies are fully defined first.
from . import indicators as _indicators  # noqa: F401  (registration side effect)

__all__ = [
    "ComputeResult",
    "IndicatorRequest",
    "compute",
    "compute_series",
    "parse_requests",
    "TAError",
    "UnknownIndicator",
    "ParamError",
    "FrameError",
    "InsufficientData",
    "NonCausalIndicator",
    "SourceError",
    "FrameInfo",
    "normalize_ohlcv",
    "drop_incomplete_bar",
    "catalogue",
    "get_spec",
    "list_indicators",
    "all_specs",
]
