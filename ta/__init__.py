"""`ta` -- the technical-analysis toolkit for this repo: library first, HTTP second."""

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
