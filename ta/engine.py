"""The compute engine: turn indicator requests into columns, honestly.

Three jobs beyond "call the function":

1. **Batch.** One frame, many indicators, one pass. The old service computed
   exactly one indicator per HTTP request and refetched the candles each time,
   so a five-indicator dashboard was five Mongo reads of the same 500 rows.

2. **Warmup.** Every indicator has a stretch at the start of its output that is
   arithmetically real but analytically worthless. The engine masks it by
   default. See `WarmupPolicy` below -- this is the part that most TA code
   gets wrong and it is why indicator values disagree between platforms.

3. **Causality.** Forward-looking indicators are refused unless explicitly
   allowed, so nothing that reads bar `t+k` can leak into a signal or a
   backtest by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .errors import FrameError, NonCausalIndicator, ParamError
from .frame import FrameInfo, describe_frame, normalize_ohlcv
from .registry import IndicatorSpec, get_spec

WarmupPolicy = str
WARMUP_POLICIES = ("mask", "trim", "keep")


@dataclass(frozen=True)
class IndicatorRequest:
    """One indicator to compute, with its parameters."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def parse(cls, value: "IndicatorRequest | str | Mapping[str, Any]") -> "IndicatorRequest":
        """Accept every shape a caller might reasonably send.

        - ``IndicatorRequest(...)``
        - ``"rsi"``
        - ``"rsi:period=21"`` / ``"macd:fast=8,slow=21,signal=5"`` -- the
          compact form, so a whole dashboard fits in one query string:
          ``?indicators=rsi:period=21,ema:window=50,atr``
        - ``{"name": "rsi", "params": {"period": 21}}``
        - ``{"name": "rsi", "period": 21}`` -- flat form
        """
        if isinstance(value, IndicatorRequest):
            return value
        if isinstance(value, str):
            return cls._parse_string(value)
        if isinstance(value, Mapping):
            data = dict(value)
            name = data.pop("name", None) or data.pop("indicator", None)
            if not name:
                raise ParamError(f"indicator request is missing 'name': {value!r}")
            params = data.pop("params", None)
            if params is None:
                params = data
            elif data:
                raise ParamError(
                    f"indicator request for {name!r} mixes 'params' with loose keys: "
                    f"{', '.join(sorted(data))}"
                )
            return cls(str(name).strip().lower(), dict(params))
        raise ParamError(f"cannot interpret indicator request: {value!r}")

    @classmethod
    def _parse_string(cls, text: str) -> "IndicatorRequest":
        head, _, tail = text.strip().partition(":")
        name = head.strip().lower()
        if not name:
            raise ParamError(f"empty indicator name in {text!r}")
        params: dict[str, Any] = {}
        for chunk in filter(None, (c.strip() for c in tail.split(","))):
            key, sep, val = chunk.partition("=")
            if not sep:
                raise ParamError(
                    f"malformed parameter {chunk!r} in {text!r}; expected key=value"
                )
            params[key.strip()] = val.strip()
        return cls(name, params)

    def key(self) -> str:
        """Stable identity string, used for cache keys and result labelling."""
        if not self.params:
            return self.name
        body = ",".join(f"{k}={self.params[k]}" for k in sorted(self.params))
        return f"{self.name}:{body}"


def parse_requests(value) -> list[IndicatorRequest]:
    """Parse a collection, or a comma-separated string, of indicator requests.

    Splitting a string is delicate because the compact form uses commas
    *inside* a request too (``macd:fast=8,slow=21``). A comma starts a new
    request only when the chunk after it introduces a new indicator name --
    i.e. it has no ``=`` before any ``:``.
    """
    if value is None:
        return []
    if isinstance(value, (str, Mapping, IndicatorRequest)):
        if isinstance(value, str):
            return [IndicatorRequest.parse(part) for part in _split_request_string(value)]
        return [IndicatorRequest.parse(value)]
    return [IndicatorRequest.parse(v) for v in value]


def _split_request_string(text: str) -> list[str]:
    parts: list[str] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        starts_new = "=" not in chunk or (":" in chunk and chunk.index(":") < chunk.index("="))
        if starts_new or not parts:
            parts.append(chunk)
        else:
            parts[-1] = f"{parts[-1]},{chunk}"
    return parts


@dataclass(frozen=True)
class ColumnMeta:
    """Provenance for one computed column."""

    column: str
    indicator: str
    request: str
    category: str
    params: dict[str, Any]
    warmup_bars: int
    sufficient: bool
    valid_values: int

    def as_dict(self) -> dict:
        return {
            "column": self.column,
            "indicator": self.indicator,
            "request": self.request,
            "category": self.category,
            "params": self.params,
            "warmup_bars": self.warmup_bars,
            "sufficient": self.sufficient,
            "valid_values": self.valid_values,
        }


@dataclass
class ComputeResult:
    """Computed columns plus everything needed to judge them."""

    frame: pd.DataFrame
    info: FrameInfo
    columns: list[ColumnMeta]
    warmup_policy: WarmupPolicy
    trimmed_rows: int = 0
    warnings: list[str] = field(default_factory=list)

    def meta(self) -> dict:
        return {
            "frame": self.info.as_dict(),
            "columns": [c.as_dict() for c in self.columns],
            "warmup_policy": self.warmup_policy,
            "trimmed_rows": self.trimmed_rows,
            "warnings": list(self.warnings),
        }

    def to_records(self, *, dropna: bool = False) -> list[dict]:
        """JSON-ready rows, NaN rendered as ``null`` rather than the string 'NaN'.

        `DataFrame.to_json` emits NaN unquoted, which is invalid JSON that some
        parsers accept and others reject. Going through `object` + `where`
        keeps it real `null`.
        """
        out = self.frame
        if dropna and not out.empty:
            value_cols = [c.column for c in self.columns if c.column in out.columns]
            if value_cols:
                out = out.dropna(subset=value_cols, how="all")
        out = out.reset_index()
        records = []
        for row in out.to_dict(orient="records"):
            clean = {}
            for k, v in row.items():
                if isinstance(v, pd.Timestamp):
                    clean[k] = v.isoformat()
                elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                    clean[k] = None
                elif isinstance(v, (np.integer,)):
                    clean[k] = int(v)
                elif isinstance(v, (np.floating,)):
                    clean[k] = float(v)
                else:
                    clean[k] = v
            records.append(clean)
        return records

    def latest(self) -> dict:
        """The most recent value of every computed column.

        What a live consumer (the quoter, the gap guard, an alert) actually
        wants -- and the shape the whole 500-row payload was being used for.
        Reports the *last non-NaN* value per column with its own timestamp and
        age in bars, because different indicators go stale at different rates:
        a masked-out ADX during warmup is not the same as a fresh one.
        """
        out: dict[str, Any] = {}
        for meta in self.columns:
            col = meta.column
            if col not in self.frame.columns:
                continue
            series = self.frame[col].dropna()
            if series.empty:
                out[col] = {"value": None, "time": None, "bars_ago": None}
                continue
            ts = series.index[-1]
            bars_ago = int(len(self.frame) - 1 - self.frame.index.get_loc(ts))
            out[col] = {
                "value": float(series.iloc[-1]),
                "time": ts.isoformat(),
                "bars_ago": bars_ago,
            }
        return out


def compute(
    data,
    requests,
    *,
    warmup: WarmupPolicy = "mask",
    include_ohlcv: bool = True,
    allow_noncausal: bool = False,
    normalize: bool = True,
) -> ComputeResult:
    """Compute one or many indicators over one OHLCV frame.

    `data` is anything `normalize_ohlcv` accepts. `requests` is anything
    `parse_requests` accepts.

    Set `normalize=False` only when passing a frame that has already been
    through `normalize_ohlcv` -- it skips a defensive copy on a hot path.
    """
    if warmup not in WARMUP_POLICIES:
        raise ParamError(
            f"warmup must be one of {', '.join(WARMUP_POLICIES)}, got {warmup!r}"
        )

    reqs = parse_requests(requests)
    if not reqs:
        raise ParamError("no indicators requested")

    resolved: list[tuple[IndicatorRequest, IndicatorSpec, dict[str, Any]]] = []
    required_columns: set[str] = {"close"}
    for req in reqs:
        spec = get_spec(req.name)
        if not spec.causal and not allow_noncausal:
            raise NonCausalIndicator(
                f"{spec.name} looks ahead of the bar it is stamped on and is refused "
                f"by default. Pass allow_noncausal=True if you are drawing a chart; "
                f"never pass it on a signal or backtest path."
            )
        params = spec.resolve_params(req.params)
        resolved.append((req, spec, params))
        required_columns.update(spec.requires)

    if normalize:
        df, info = normalize_ohlcv(data, require=sorted(required_columns))
    else:
        df = data
        missing = sorted(required_columns - set(df.columns))
        if missing:
            raise FrameError(f"missing required column(s): {', '.join(missing)}")
        info = describe_frame(df)

    warnings: list[str] = []
    if info.is_sparse:
        warnings.append(
            f"{info.missing_bars} of {info.expected_rows} bars are missing "
            f"({info.gap_ratio:.1%}): this market did not trade in every interval, so "
            f"rolling windows span more wall-clock time than their length implies."
        )
    if info.inconsistent_bars:
        warnings.append(
            f"{info.inconsistent_bars} bar(s) have high < low or open/close outside "
            f"[low, high]; range-based indicators (ATR, Stochastic, %R) will be wrong there."
        )

    pieces: list[pd.DataFrame] = []
    metas: list[ColumnMeta] = []
    max_warmup = 0

    for req, spec, params in resolved:
        out = spec.fn(df, **params)
        if isinstance(out, pd.Series):
            out = out.to_frame()
        if not isinstance(out, pd.DataFrame):
            raise TypeError(
                f"indicator {spec.name!r} returned {type(out).__name__}, expected DataFrame"
            )
        expected = spec.output_names(params)
        if tuple(out.columns) != expected:
            raise TypeError(
                f"indicator {spec.name!r} returned columns {tuple(out.columns)}, "
                f"but its spec declares {expected}"
            )

        warmup_bars = spec.warmup_bars(params)
        max_warmup = max(max_warmup, warmup_bars)
        sufficient = len(df) > warmup_bars
        if not sufficient:
            warnings.append(
                f"{req.key()} needs {warmup_bars} bars of warmup but the frame has "
                f"{len(df)}; its output is not yet meaningful."
            )
        if warmup == "mask" and warmup_bars > 0:
            out.iloc[: min(warmup_bars, len(out))] = np.nan

        pieces.append(out)
        for col in expected:
            metas.append(
                ColumnMeta(
                    column=col,
                    indicator=spec.name,
                    request=req.key(),
                    category=spec.category,
                    params=dict(params),
                    warmup_bars=warmup_bars,
                    sufficient=sufficient,
                    valid_values=int(out[col].notna().sum()),
                )
            )

    combined = pd.concat(pieces, axis=1)
    duplicated = combined.columns.duplicated(keep="first")
    if duplicated.any():
        dropped = sorted(set(combined.columns[duplicated]))
        warnings.append(
            f"duplicate output column(s) collapsed: {', '.join(dropped)} "
            f"(the same indicator was requested more than once with equal parameters)"
        )
        combined = combined.loc[:, ~duplicated]
        seen: set[str] = set()
        metas = [m for m in metas if not (m.column in seen or seen.add(m.column))]

    if include_ohlcv:
        base = df[[c for c in df.columns if c not in combined.columns]]
        combined = pd.concat([base, combined], axis=1)

    trimmed = 0
    if warmup == "trim" and max_warmup:
        trimmed = min(max_warmup, len(combined))
        combined = combined.iloc[trimmed:]
        metas = [
            replace(
                m,
                valid_values=int(combined[m.column].notna().sum())
                if m.column in combined.columns
                else 0,
            )
            for m in metas
        ]

    return ComputeResult(
        frame=combined,
        info=info,
        columns=metas,
        warmup_policy=warmup,
        trimmed_rows=trimmed,
        warnings=warnings,
    )


def compute_series(data, request, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: just the indicator columns, no OHLCV, no metadata.

    For internal callers (the analysis layer, the backtester) that already
    trust their input.
    """
    kwargs.setdefault("include_ohlcv", False)
    result = compute(data, request, **kwargs)
    return result.frame
