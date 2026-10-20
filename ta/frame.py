"""OHLCV frame normalization and health reporting; every entry point funnels through `normalize_ohlcv`."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .errors import FrameError

PRICE_COLUMNS = ("open", "high", "low", "close")
OHLCV_COLUMNS = PRICE_COLUMNS + ("volume",)

_COLUMN_ALIASES = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vol": "volume",
    "amount": "volume",
    "qty": "volume",
    "quantity": "volume",
    "base_volume": "volume",
    "price_first": "open",
    "price_max": "high",
    "price_min": "low",
    "price_last": "close",
    "amount_sum": "volume",
    "adj_close": "close",
}

_TIME_ALIASES = ("time", "timestamp", "datetime", "date", "ts", "open_time")

_METADATA_COLUMNS = ("_id", "market_name", "symbol", "interval", "source")


@dataclass(frozen=True)
class FrameInfo:
    """What we know about the shape and health of an OHLCV frame."""

    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    interval_sec: float | None
    expected_rows: int | None
    missing_bars: int | None
    gap_ratio: float | None
    duplicates_dropped: int
    nan_rows_dropped: int
    columns: tuple[str, ...]
    has_volume: bool
    inconsistent_bars: int

    def as_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat() if self.start is not None else None
        d["end"] = self.end.isoformat() if self.end is not None else None
        d["columns"] = list(self.columns)
        return d

    @property
    def is_sparse(self) -> bool:
        """True when enough bars are missing that window lengths are unreliable."""
        return self.gap_ratio is not None and self.gap_ratio > 0.2


def normalize_ohlcv(
    data,
    *,
    require: Iterable[str] = ("close",),
    time_column: str | None = None,
    drop_metadata: bool = True,
) -> tuple[pd.DataFrame, FrameInfo]:
    """Coerce arbitrary candle data into a canonical OHLCV frame, returning it with a `FrameInfo`."""
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        rows = list(data)
        if not rows:
            raise FrameError("no candles supplied")
        df = pd.DataFrame(rows)

    if df.empty:
        raise FrameError("no candles supplied")

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns=_COLUMN_ALIASES)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    df = _apply_time_index(df, time_column)

    if drop_metadata:
        df = df.drop(columns=[c for c in _METADATA_COLUMNS if c in df.columns])

    missing = [c for c in require if c not in df.columns]
    if missing:
        raise FrameError(
            f"missing required column(s): {', '.join(sorted(missing))}; "
            f"frame has {', '.join(df.columns)}"
        )

    for col in OHLCV_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    before = len(df)
    df = df[~df.index.duplicated(keep="last")]
    duplicates_dropped = before - len(df)

    before = len(df)
    price_cols = [c for c in require if c in PRICE_COLUMNS]
    if price_cols:
        df = df.dropna(subset=price_cols)
    nan_rows_dropped = before - len(df)

    if df.empty:
        raise FrameError("no usable candles left after dropping NaN price rows")

    df = df.sort_index()
    return df, describe_frame(df)


def _apply_time_index(df: pd.DataFrame, time_column: str | None) -> pd.DataFrame:
    """Put a sorted, tz-aware DatetimeIndex named `time` on the frame."""
    candidates = [time_column] if time_column else list(_TIME_ALIASES)
    for name in candidates:
        if name and name in df.columns:
            idx = pd.to_datetime(df[name], errors="coerce", utc=True)
            if idx.isna().all():
                raise FrameError(f"column '{name}' could not be parsed as timestamps")
            df = df.drop(columns=[name])
            df.index = pd.DatetimeIndex(idx, name="time")
            return df[df.index.notna()]

    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        df.index = idx.rename("time")
        return df

    if time_column:
        raise FrameError(f"time column '{time_column}' not found in frame")
    raise FrameError(
        "no time index found: supply a DatetimeIndex or a column named one of "
        + ", ".join(_TIME_ALIASES)
    )


def describe_frame(df: pd.DataFrame) -> FrameInfo:
    """Build a `FrameInfo` for an already-normalized frame."""
    n = len(df)
    start = df.index[0] if n else None
    end = df.index[-1] if n else None

    interval_sec = None
    expected_rows = None
    missing_bars = None
    gap_ratio = None
    if n >= 2:
        deltas = df.index.to_series().diff().dropna().dt.total_seconds()
        positive = deltas[deltas > 0]
        if not positive.empty:
            interval_sec = float(positive.median())
            span = (end - start).total_seconds()
            expected_rows = int(round(span / interval_sec)) + 1
            missing_bars = max(0, expected_rows - n)
            gap_ratio = missing_bars / expected_rows if expected_rows else 0.0

    inconsistent = 0
    if {"high", "low"} <= set(df.columns):
        bad = df["high"] < df["low"]
        for col in ("open", "close"):
            if col in df.columns:
                bad = bad | (df[col] > df["high"]) | (df[col] < df["low"])
        inconsistent = int(bad.sum())

    return FrameInfo(
        rows=n,
        start=start,
        end=end,
        interval_sec=interval_sec,
        expected_rows=expected_rows,
        missing_bars=missing_bars,
        gap_ratio=gap_ratio,
        duplicates_dropped=0,
        nan_rows_dropped=0,
        columns=tuple(df.columns),
        has_volume="volume" in df.columns and bool(df["volume"].notna().any()),
        inconsistent_bars=inconsistent,
    )


def drop_incomplete_bar(
    df: pd.DataFrame,
    interval_sec: float | None,
    *,
    now: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, bool]:
    """Drop the final bar when it is still forming. Returns `(frame, dropped)`."""
    if df.empty or not interval_sec:
        return df, False
    now = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if now.tz is None:
        now = now.tz_localize("UTC")
    last_open = df.index[-1]
    if now < last_open + pd.Timedelta(seconds=interval_sec):
        return df.iloc[:-1], True
    return df, False


def typical_price(df: pd.DataFrame) -> pd.Series:
    """(high + low + close) / 3, falling back to close when H/L are absent."""
    if {"high", "low"} <= set(df.columns):
        return (df["high"] + df["low"] + df["close"]) / 3.0
    return df["close"].astype("float64")


def median_price(df: pd.DataFrame) -> pd.Series:
    """(high + low) / 2, falling back to close."""
    if {"high", "low"} <= set(df.columns):
        return (df["high"] + df["low"]) / 2.0
    return df["close"].astype("float64")


def true_range(df: pd.DataFrame) -> pd.Series:
    """Wilder's true range: max(h-l, |h-c_prev|, |l-c_prev|)."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise divide with 0/0 and x/0 mapped to NaN instead of inf."""
    denom = denominator.replace(0.0, np.nan)
    return numerator / denom


def rolling_std(series: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    """Rolling standard deviation, population (ddof=0) by default."""
    return series.rolling(window=window, min_periods=window).std(ddof=ddof)


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: an EMA with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
