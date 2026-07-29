"""OHLCV frame normalization and health reporting.

Every entry point into `ta` funnels its input through `normalize_ohlcv`, so the
rest of the package can assume: lowercase columns, float dtypes, a unique,
sorted, timezone-aware `DatetimeIndex`, and no all-NaN price columns.

Why this file exists at all
---------------------------
The candles this service reads are built by `market-data-collector`, which
resamples raw trades and then does `dropna(subset=["open","high","low","close"])`.
A bar in which nothing traded is not written as an empty bar -- **it is not
written at all**. So the OHLC series for an illiquid market is not a regular
time grid; it is a sparse one with holes.

That matters more than it sounds. A 20-period SMA on a 1-minute series with
40% of its bars missing is not a 20-minute average, it is a ~33-minute average,
and it silently changes meaning as liquidity changes through the day. Every
rolling window in every indicator inherits this. `FrameInfo.gap_ratio` is the
number that tells you whether to believe the output, and nothing else in the
system was reporting it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .errors import FrameError

#: Canonical column names. `open/high/low/close` are prices, `volume` is base
#: quantity. Anything else in the frame is carried through untouched.
PRICE_COLUMNS = ("open", "high", "low", "close")
OHLCV_COLUMNS = PRICE_COLUMNS + ("volume",)

#: Column aliases seen in the wild, mapped to canonical names. The `price_*`
#: entries are the raw column names `market-data-collector`'s resampler
#: produces before it renames them, so a frame read straight out of an
#: intermediate step still works.
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

#: Columns that are metadata, not series data -- dropped from the working frame
#: so indicators never see them and `to_records` output stays lean.
_METADATA_COLUMNS = ("_id", "market_name", "symbol", "interval", "source")


@dataclass(frozen=True)
class FrameInfo:
    """What we know about the shape and health of an OHLCV frame.

    Returned alongside every normalized frame and echoed into API responses.
    A consumer that ignores this is quoting off numbers it cannot vouch for.
    """

    rows: int
    start: pd.Timestamp | None
    end: pd.Timestamp | None
    #: Inferred bar width in seconds (median of index deltas). `None` when
    #: there are fewer than two bars.
    interval_sec: float | None
    #: Bars that *should* exist between `start` and `end` on the inferred grid.
    expected_rows: int | None
    #: `expected_rows - rows`. Non-zero means the collector had no trades to
    #: resample for those bars. See the module docstring.
    missing_bars: int | None
    #: `missing_bars / expected_rows`, in [0, 1]. Above ~0.2 treat every
    #: rolling-window indicator on this frame as approximate.
    gap_ratio: float | None
    #: Rows dropped because they duplicated an earlier timestamp.
    duplicates_dropped: int
    #: Rows dropped because a required price column was NaN.
    nan_rows_dropped: int
    columns: tuple[str, ...]
    has_volume: bool
    #: True when high < low, or close outside [low, high], on any bar. A real
    #: sign of a broken upstream write; several indicators divide by (high-low).
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
    """Coerce arbitrary candle data into a canonical OHLCV frame.

    Accepts a DataFrame (indexed by time or with a time column) or a list of
    dicts, which is what a Mongo cursor or a JSON request body gives you.

    `require` lists the columns the caller's downstream work actually needs;
    a missing one raises `FrameError` here rather than surfacing as a
    `KeyError` from deep inside an indicator.

    Returns the frame and a `FrameInfo` describing it. Does not mutate input.
    """
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
    # A rename can collide (e.g. both `amount` and `volume` present). Keep the
    # first occurrence -- the canonical name -- rather than letting pandas
    # hand later code a DataFrame where df["volume"] is a 2-column frame.
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
    """Put a sorted, tz-aware DatetimeIndex named `time` on the frame.

    Naive timestamps are localized to UTC rather than left naive: the collector
    writes UTC, and a frame that mixes naive and aware timestamps blows up on
    comparison much later, in a place with no context about where it came from.
    """
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
    """Drop the final bar when it is still forming.

    This is the single most common source of lookahead-flavoured nonsense in
    indicator services, and it bites in both directions:

    - Live: the last bar's `close` is whatever traded a second ago and will
      keep moving. A crossover signal computed on it flickers on and off. This
      is what "repainting" means in practice.
    - Research: the same partial bar gets frozen into a dataset and then
      compared against a *complete* bar's future return, which quietly leaks a
      few seconds of the future into every sample.

    A bar starting at `t` is complete once `now >= t + interval`. With no
    reliable interval we cannot tell, so we leave the frame alone and say so.

    Returns `(frame, dropped)`.
    """
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
    """Wilder's true range: max(h-l, |h-c_prev|, |l-c_prev|).

    The first bar has no previous close, so its TR is just `high - low` rather
    than NaN -- matching Wilder and every charting package.
    """
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    ranges = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1, skipna=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise divide with 0/0 and x/0 mapped to NaN instead of inf.

    Flat windows are routine on illiquid Iranian pairs -- a 14-bar window where
    nothing moved makes `highest - lowest` exactly zero in Stochastic, Williams
    %R and Stochastic RSI. `inf` propagates into every downstream smoothing and
    poisons the whole tail of the series; NaN stops at the bar it belongs to.
    """
    denom = denominator.replace(0.0, np.nan)
    return numerator / denom


def rolling_std(series: pd.Series, window: int, ddof: int = 0) -> pd.Series:
    """Rolling standard deviation, population (ddof=0) by default.

    pandas defaults to the *sample* deviation (ddof=1); Bollinger Bands and
    every charting platform use the population one. Getting this wrong makes
    bands ~2.6% too wide at length 20 -- small enough to never notice, large
    enough to change which touches count as a signal.
    """
    return series.rolling(window=window, min_periods=window).std(ddof=ddof)


def wilder_ema(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing: an EMA with alpha = 1/period.

    Used by RSI, ATR, ADX. Note this is *not* `ewm(span=period)`, which uses
    alpha = 2/(period+1) -- a 14-period Wilder average smooths like a
    27-period standard EMA.
    """
    return series.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
