"""The indicator registry.

Replaces the `if indicator_name == "sma": ... elif ...` chain that used to live
in `handler.py`. That chain had to be edited in four places to add an
indicator (the branch, the param parsing, the API docs, the defaults) and had
no way to answer "what indicators exist and what do they take?" -- which is
exactly what an API client, a config UI, and a backtest parameter sweep all
need to ask.

Here an indicator declares itself once, with its parameters, its output
columns, its input requirements, and its warmup. Everything else -- validation,
introspection, the HTTP catalogue, batch compute -- reads that declaration.

Registration is by decorator:

    @indicator(
        "rsi",
        category="momentum",
        outputs=("rsi_{period}",),
        params=(IntParam("period", 14, minimum=2),),
        requires=("close",),
        warmup=lambda p: p["period"] * 5,
    )
    def rsi(df, period=14):
        ...
        return pd.DataFrame({f"rsi_{period}": values}, index=df.index)

Indicator functions are **pure**: they take a frame, they return a *new* frame
containing only their own output columns, and they never touch the input. The
old implementation did `data["RSI"] = ...` on the caller's DataFrame, so
`compute_stochastic_rsi` silently left an `RSI` column behind in its caller's
data and two indicators computed in sequence could overwrite each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd

from .errors import ParamError, UnknownIndicator

# --------------------------------------------------------------------------
# Parameter specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Param:
    """One tunable input to an indicator.

    Carries enough metadata to validate a value coming off an HTTP query
    string, render a form, and enumerate a backtest sweep.
    """

    name: str
    default: Any
    doc: str = ""

    def coerce(self, value: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def describe(self) -> dict:
        return {
            "name": self.name,
            "type": type(self).__name__.replace("Param", "").lower(),
            "default": self.default,
            "doc": self.doc,
        }


@dataclass(frozen=True)
class IntParam(Param):
    minimum: int | None = None
    maximum: int | None = None

    def coerce(self, value: Any) -> int:
        try:
            # int("14.0") raises; float first so query strings survive.
            out = int(float(value))
        except (TypeError, ValueError):
            raise ParamError(f"{self.name!r} must be an integer, got {value!r}")
        if self.minimum is not None and out < self.minimum:
            raise ParamError(f"{self.name!r} must be >= {self.minimum}, got {out}")
        if self.maximum is not None and out > self.maximum:
            raise ParamError(f"{self.name!r} must be <= {self.maximum}, got {out}")
        return out

    def describe(self) -> dict:
        return {**super().describe(), "minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True)
class FloatParam(Param):
    minimum: float | None = None
    maximum: float | None = None

    def coerce(self, value: Any) -> float:
        try:
            out = float(value)
        except (TypeError, ValueError):
            raise ParamError(f"{self.name!r} must be a number, got {value!r}")
        if self.minimum is not None and out < self.minimum:
            raise ParamError(f"{self.name!r} must be >= {self.minimum}, got {out}")
        if self.maximum is not None and out > self.maximum:
            raise ParamError(f"{self.name!r} must be <= {self.maximum}, got {out}")
        return out

    def describe(self) -> dict:
        return {**super().describe(), "minimum": self.minimum, "maximum": self.maximum}


@dataclass(frozen=True)
class ChoiceParam(Param):
    choices: tuple[str, ...] = ()

    def coerce(self, value: Any) -> str:
        out = str(value).lower()
        if out not in self.choices:
            raise ParamError(
                f"{self.name!r} must be one of {', '.join(self.choices)}, got {value!r}"
            )
        return out

    def describe(self) -> dict:
        return {**super().describe(), "choices": list(self.choices)}


@dataclass(frozen=True)
class BoolParam(Param):
    _TRUE = frozenset({"1", "true", "yes", "y", "on"})
    _FALSE = frozenset({"0", "false", "no", "n", "off"})

    def coerce(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in self._TRUE:
            return True
        if text in self._FALSE:
            return False
        raise ParamError(f"{self.name!r} must be a boolean, got {value!r}")


# --------------------------------------------------------------------------
# Indicator specification
# --------------------------------------------------------------------------

#: Categories, ordered roughly by how a chartist would group them. Used to
#: organise the API catalogue and the `analyze()` snapshot.
CATEGORIES = (
    "trend",
    "momentum",
    "volatility",
    "volume",
    "statistical",
)


@dataclass(frozen=True)
class IndicatorSpec:
    """Everything the engine knows about one indicator."""

    name: str
    fn: Callable[..., pd.DataFrame]
    category: str
    #: Output column-name templates. `{param}` placeholders are filled from the
    #: resolved parameters, so `sma_{window}` becomes `sma_20`. Parameterised
    #: names are what let a caller request sma(20) and sma(50) in one pass
    #: without a collision.
    outputs: tuple[str, ...]
    params: tuple[Param, ...] = ()
    #: Input columns the indicator reads. Checked before the function runs.
    requires: tuple[str, ...] = ("close",)
    #: Resolved params -> number of leading bars whose output should not be
    #: trusted. For window indicators this is the window (output is NaN before
    #: it). For recursive ones (Wilder, EMA) the output is *non-NaN* much
    #: earlier than it is *correct*, because it still carries the seed value;
    #: those declare a convergence-based multiple instead. See `warmup_bars`.
    warmup: Callable[[Mapping[str, Any]], int] = lambda p: 1
    #: False for indicators that read bars ahead of the one they are stamped
    #: on. Refused by default on any path feeding signals or a backtest.
    causal: bool = True
    doc: str = ""
    #: Free-form notes surfaced in the catalogue -- caveats, conventions, and
    #: which platform's variant this matches.
    notes: str = ""

    def param_map(self) -> dict[str, Param]:
        return {p.name: p for p in self.params}

    def defaults(self) -> dict[str, Any]:
        return {p.name: p.default for p in self.params}

    def resolve_params(self, supplied: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Validate and fill in parameters, rejecting unknown names.

        Rejecting unknowns is deliberate: the old endpoint accepted
        `?indicator=rsi&perid=30`, silently used period=14, and returned a
        200 with data that did not answer the question asked.
        """
        supplied = dict(supplied or {})
        known = self.param_map()
        unknown = set(supplied) - set(known)
        if unknown:
            raise ParamError(
                f"{self.name}: unknown parameter(s) {', '.join(sorted(unknown))}; "
                f"accepts {', '.join(sorted(known)) or '(none)'}"
            )
        resolved = {}
        for name, param in known.items():
            resolved[name] = (
                param.coerce(supplied[name]) if name in supplied else param.default
            )
        return resolved

    def output_names(self, params: Mapping[str, Any]) -> tuple[str, ...]:
        """Concrete output column names for a resolved parameter set."""
        return tuple(template.format(**params) for template in self.outputs)

    def warmup_bars(self, params: Mapping[str, Any]) -> int:
        return max(1, int(self.warmup(params)))

    def describe(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "doc": (self.doc or self.fn.__doc__ or "").strip(),
            "notes": self.notes.strip(),
            "params": [p.describe() for p in self.params],
            "outputs": list(self.outputs),
            "requires": list(self.requires),
            "causal": self.causal,
            "warmup_default": self.warmup_bars(self.defaults()),
        }


_REGISTRY: dict[str, IndicatorSpec] = {}
_ALIASES: dict[str, str] = {}


def indicator(
    name: str,
    *,
    category: str,
    outputs: Sequence[str],
    params: Sequence[Param] = (),
    requires: Sequence[str] = ("close",),
    warmup: Callable[[Mapping[str, Any]], int] | None = None,
    causal: bool = True,
    aliases: Sequence[str] = (),
    doc: str = "",
    notes: str = "",
):
    """Register an indicator function under `name`."""

    def decorator(fn: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
        if name in _REGISTRY:
            raise ValueError(f"indicator {name!r} is already registered")
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r} for indicator {name!r}")
        spec = IndicatorSpec(
            name=name,
            fn=fn,
            category=category,
            outputs=tuple(outputs),
            params=tuple(params),
            requires=tuple(requires),
            warmup=warmup or (lambda p: 1),
            causal=causal,
            doc=doc or (fn.__doc__ or ""),
            notes=notes,
        )
        _REGISTRY[name] = spec
        for alias in aliases:
            if alias in _ALIASES or alias in _REGISTRY:
                raise ValueError(f"alias {alias!r} collides with an existing name")
            _ALIASES[alias] = name
        fn.spec = spec  # type: ignore[attr-defined]
        return fn

    return decorator


def get_spec(name: str) -> IndicatorSpec:
    """Look up an indicator by name or alias."""
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    try:
        return _REGISTRY[key]
    except KeyError:
        raise UnknownIndicator(
            f"unknown indicator {name!r}; known: {', '.join(list_indicators())}"
        )


def list_indicators(category: str | None = None) -> list[str]:
    names = sorted(
        n for n, s in _REGISTRY.items() if category is None or s.category == category
    )
    return names


def all_specs() -> list[IndicatorSpec]:
    return [_REGISTRY[n] for n in sorted(_REGISTRY)]


def catalogue() -> dict:
    """Machine-readable description of every indicator, grouped by category."""
    grouped: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    for spec in all_specs():
        grouped[spec.category].append(spec.describe())
    return {
        "count": len(_REGISTRY),
        "categories": {c: grouped[c] for c in CATEGORIES if grouped[c]},
        "aliases": dict(sorted(_ALIASES.items())),
    }
