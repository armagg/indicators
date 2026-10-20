"""Source-aware OHLC reader for Mongo `market_making.ohlc`; `source` is required and times are Tehran-localized."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
from decouple import config
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from .errors import ParamError, SourceError

TEHRAN = timezone(timedelta(hours=3, minutes=30))

TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"

INTERVALS = ("1Min", "5Min", "15Min", "1H", "4H", "1D")
_INTERVAL_LOOKUP = {label.lower(): label for label in INTERVALS}

_PROJECTION = {"_id": 0, "time": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}

DEFAULT_LOOKBACK = 1000

_client: MongoClient | None = None


def _collection():
    global _client
    if _client is None:
        _client = MongoClient(config("MONGO_URI"), serverSelectionTimeoutMS=5000)
    return _client[config("DB_NAME")][config("OHLC_COLLECTION_NAME")]


def normalize_interval(value: str) -> str:
    """Map '1h' / '1H' / '15min' onto MDC's exact labels, or raise."""
    label = _INTERVAL_LOOKUP.get(str(value).strip().lower())
    if label is None:
        raise ParamError(
            f"unknown interval {value!r}; valid: {', '.join(INTERVALS)}"
        )
    return label


def query_time(value: datetime | str) -> str:
    """Render a caller-supplied bound in Mongo's stored string format."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(TEHRAN).tz_localize(None)
    return ts.strftime(TIME_FORMAT)


def docs_to_frame(docs: list[dict]) -> pd.DataFrame:
    """Mongo OHLC docs -> DataFrame with a UTC DatetimeIndex, oldest first."""
    df = pd.DataFrame(docs)
    times = pd.to_datetime(df["time"], format="ISO8601", errors="coerce")
    df = df.drop(columns=["time"])
    df.index = pd.DatetimeIndex(times, name="time").tz_localize(TEHRAN).tz_convert("UTC")
    return df[df.index.notna()].sort_index()


def read_ohlc(
    market_name: str,
    source: str,
    interval: str,
    *,
    limit: int | None = None,
    start: datetime | str | None = None,
    end: datetime | str | None = None,
) -> pd.DataFrame | None:
    """Read candles for one (market, venue, interval), oldest first; `None` when nothing matches."""
    for name, value in (("market_name", market_name), ("source", source), ("interval", interval)):
        if value is None or not str(value).strip():
            raise ParamError(f"{name} is required")

    query: dict = {
        "market_name": str(market_name).strip(),
        "source": str(source).strip(),
        "interval": normalize_interval(interval),
    }
    time_bounds = {}
    if start is not None:
        time_bounds["$gte"] = query_time(start)
    if end is not None:
        time_bounds["$lte"] = query_time(end)
    if time_bounds:
        query["time"] = time_bounds

    try:
        cursor = _collection().find(query, _PROJECTION)
        if time_bounds:
            cursor = cursor.sort("time", 1)
            if limit:
                cursor = cursor.limit(int(limit))
            docs = list(cursor)
        else:
            docs = list(cursor.sort("time", -1).limit(int(limit or DEFAULT_LOOKBACK)))
    except PyMongoError as exc:
        raise SourceError(f"cannot read OHLC from Mongo: {exc}") from exc

    if not docs:
        return None
    return docs_to_frame(docs)
