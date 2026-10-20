"""Idempotent persistence of computed indicators to `indicators_data`.

The old writer did a blind `insert_many` on every request, so recomputing
RSI for the same market appended a fresh copy of all 500 rows each time and
the collection was mostly duplicates. Writes here are upserts on the natural
key from ARCHITECTURE.md:

    (symbol, source, interval, analysis, param_hash, time)

`param_hash` is a stable digest of the *resolved* parameters, so `rsi` and
`rsi:period=14` land on the same documents while `rsi:period=21` gets its
own -- recomputing anything is always a no-op-or-overwrite, never a duplicate.

One document per indicator request per bar, carrying all of that request's
output columns in `values` (a MACD doc holds line, signal, and histogram
together -- they are one computation and are only meaningful together).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping

import pandas as pd
from decouple import config
from pymongo import MongoClient, ReplaceOne

from .engine import ComputeResult
from .errors import SourceError

NATURAL_KEY = ("symbol", "source", "interval", "analysis", "param_hash", "time")

_client: MongoClient | None = None
_index_ensured = False


def _collection():
    global _client
    if _client is None:
        _client = MongoClient(config("MONGO_URI"), serverSelectionTimeoutMS=5000)
    return _client[config("DB_NAME")][config("INDICATORS_COLLECTION_NAME")]


def param_hash(params: Mapping[str, Any]) -> str:
    """Stable short digest of a resolved parameter mapping.

    Key order must not matter (`{"fast":12,"slow":26}` and the reverse are the
    same request), hence the sorted canonical JSON.
    """
    canonical = json.dumps(dict(params), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def docs_for(
    result: ComputeResult,
    *,
    symbol: str,
    source: str,
    interval: str,
    latest_bar_only: bool = False,
) -> list[dict]:
    """Build the documents for one compute result. Pure -- no Mongo here.

    Rows where every output column is NaN (the warmup-masked head of the
    series) are skipped: a document that says "we know nothing about this
    bar" is storage spent making queries slower.

    `latest_bar_only` keeps just the last timestamp. The always-on snapshot
    writer uses it so an hourly tick does not rewrite the whole warmup window.
    """
    by_request: dict[str, list] = {}
    for meta in result.columns:
        by_request.setdefault(meta.request, []).append(meta)

    computed_at = datetime.now(timezone.utc)
    docs: list[dict] = []
    for request_key, metas in by_request.items():
        columns = [m.column for m in metas if m.column in result.frame.columns]
        if not columns:
            continue
        params = metas[0].params
        digest = param_hash(params)
        sub = result.frame[columns]
        if latest_bar_only and not result.frame.empty:
            sub = sub.loc[[result.frame.index[-1]]]
        for ts, row in sub.iterrows():
            values = {
                column: (None if pd.isna(value) else float(value))
                for column, value in row.items()
            }
            if all(v is None for v in values.values()):
                continue
            docs.append(
                {
                    "symbol": symbol,
                    "source": source,
                    "interval": interval,
                    "analysis": metas[0].indicator,
                    "param_hash": digest,
                    "time": ts.to_pydatetime(),
                    "request": request_key,
                    "params": dict(params),
                    "values": values,
                    "computed_at": computed_at,
                }
            )
    return docs


def _ensure_index(collection) -> None:
    """Create the unique natural-key index, once per process.

    Partial (only docs that *have* the new fields): the collection still
    holds old-format rows from the legacy service, which all lack
    `analysis`/`param_hash` and would collide as nulls under a plain unique
    index, making index creation itself fail.
    """
    collection.create_index(
        [(field, 1) for field in NATURAL_KEY],
        unique=True,
        name="ta_natural_key",
        partialFilterExpression={
            "analysis": {"$exists": True},
            "param_hash": {"$exists": True},
        },
    )


def save(
    result: ComputeResult,
    *,
    symbol: str,
    source: str,
    interval: str,
    collection=None,
    latest_bar_only: bool = False,
) -> int:
    """Upsert every non-empty bar of `result`. Returns the document count.

    `collection` is injectable for tests; by default the env-configured
    `indicators_data` collection is used.
    """
    docs = docs_for(
        result,
        symbol=symbol,
        source=source,
        interval=interval,
        latest_bar_only=latest_bar_only,
    )
    if not docs:
        return 0

    global _index_ensured
    target = collection if collection is not None else _collection()
    try:
        if collection is not None or not _index_ensured:
            _ensure_index(target)
            if collection is None:
                _index_ensured = True
        target.bulk_write(
            [
                ReplaceOne({k: doc[k] for k in NATURAL_KEY}, doc, upsert=True)
                for doc in docs
            ],
            ordered=False,
        )
    except SourceError:
        raise
    except Exception as exc:
        raise SourceError(f"cannot write indicators to Mongo: {exc}") from exc
    return len(docs)
