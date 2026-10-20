"""Result type and parameter plumbing shared by all analyses.

An analysis differs from an indicator: an indicator returns columns aligned to
the frame; an analysis returns a *verdict* -- a handful of scalars, optionally
some supporting series and events -- that the overlay layer (and a human) can
act on without re-deriving anything.

Analyses have no Param specs (they are few, and their functions are their own
schema), so parameters coming off a CLI or query string are coerced loosely
and validated against the function signature instead of a registry entry.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..errors import ParamError


@dataclass
class AnalysisResult:
    """One analysis' structured verdict over one frame."""

    name: str
    params: dict[str, Any]
    asof: pd.Timestamp | None
    scalars: dict[str, Any] = field(default_factory=dict)
    series: dict[str, pd.Series] = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self, *, include_series: bool = False) -> dict:
        out = {
            "name": self.name,
            "params": {k: _clean(v) for k, v in self.params.items()},
            "asof": self.asof.isoformat() if self.asof is not None else None,
            "scalars": {k: _clean(v) for k, v in self.scalars.items()},
            "events": self.events,
            "warnings": list(self.warnings),
        }
        if include_series:
            out["series"] = {
                name: [
                    {"time": ts.isoformat(), "value": _clean(value)}
                    for ts, value in series.items()
                ]
                for name, series in self.series.items()
            }
        return out


def _clean(value: Any) -> Any:
    """numpy scalars -> python, non-finite floats -> None, for JSON."""
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def coerce_loose(value: Any) -> Any:
    """Best-effort typing for a parameter that has no schema.

    Query strings and argv deliver everything as text; analyses and overlays
    take real ints/floats/bools. int before float so "30" stays an int.
    """
    if not isinstance(value, str):
        return value
    text = value.strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def call_with_loose_params(
    fn: Callable, supplied: Mapping[str, Any], *args, label: str = ""
) -> Any:
    """Call `fn(*args, **supplied)` with coercion and a friendly unknown-kwarg
    error instead of a bare TypeError from deep inside Python."""
    label = label or fn.__name__
    accepted = {
        name
        for name, p in inspect.signature(fn).parameters.items()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.default is not p.empty
    }
    unknown = set(supplied) - accepted
    if unknown:
        raise ParamError(
            f"{label}: unknown parameter(s) {', '.join(sorted(unknown))}; "
            f"accepts {', '.join(sorted(accepted)) or '(none)'}"
        )
    return fn(*args, **{k: coerce_loose(v) for k, v in supplied.items()})
