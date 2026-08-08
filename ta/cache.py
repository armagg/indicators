"""A Mongo-backed cache for computed series and finished research runs.

Research is repetitive in a specific way: a brute force recomputes the same
EMA-20 for every one of the hundred rules that happens to reference it, and
the next brute force recomputes all of them again tomorrow. The arithmetic
is deterministic, so the second computation is pure waste.

Two collections, in their own database (`ta_research` by default -- kept
away from `market_making` so a cache wipe can never touch real data):

- `indicator_columns` -- one document per (request, dataset), holding the
  computed series. Read back, it is a DataFrame again.
- `research_runs` -- one document per finished job, holding whatever the job
  wants to remember. A repeated brute force with identical inputs returns
  instantly instead of re-running its permutation test.

## The fingerprint, and why cache invalidation is the hard part

A cache that returns yesterday's EMA for today's candles is worse than no
cache: it is silently wrong, and every downstream number inherits the error.
So the key includes a `fingerprint` of the *data*, not just of the request.

`frame_fingerprint` hashes the row count, the first and last timestamps, and
a ~100-point subsample of the closes. Row count and endpoints catch the
common case (new candles arrived). The subsample catches the nastier one:
`market-data-collector` upserts, so a bar in the middle of the range can be
rewritten when late trades land, leaving count and endpoints identical. A
pure endpoint hash would serve stale series forever in exactly that case.

It is a subsample, so it is not a proof -- a rewrite that misses every
sampled point slips through. That is a deliberate trade: hashing every close
costs more than recomputing the indicator for short frames. Pass
`refresh=True` when you need certainty.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from decouple import config as env

from .engine import compute, parse_requests
from .errors import SourceError

#: Points of the close series folded into the fingerprint.
_FINGERPRINT_SAMPLES = 100

COLUMNS_COLLECTION = "indicator_columns"
RUNS_COLLECTION = "research_runs"


def _digest(*parts: Any) -> str:
    payload = "|".join(str(p) for p in parts)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def frame_fingerprint(frame: pd.DataFrame, *, column: str = "close") -> str:
    """Identity of a dataset: changes whenever the data underneath does."""
    if frame.empty:
        return _digest("empty")
    series = frame[column].to_numpy(dtype="float64")
    step = max(1, len(series) // _FINGERPRINT_SAMPLES)
    sample = np.round(series[::step], 8)
    return _digest(
        len(frame),
        frame.index[0].isoformat(),
        frame.index[-1].isoformat(),
        hashlib.sha1(sample.tobytes()).hexdigest(),
    )


def params_digest(params: Mapping[str, Any]) -> str:
    """Order-independent digest of a parameter mapping."""
    return _digest(json.dumps(dict(params), sort_keys=True, default=str))


def _encode(values: np.ndarray) -> list:
    """NaN -> None. BSON can hold NaN doubles, but a null round-trips through
    every other tool (mongoexport, JSON views, the shell) unambiguously."""
    return [None if not np.isfinite(v) else float(v) for v in values]


def _decode(values: Sequence) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype="float64")


@dataclass(frozen=True)
class ResearchCache:
    """Handles for the two cache collections. Inject fakes in tests."""

    columns: Any
    runs: Any

    def ensure_indexes(self) -> None:
        self.columns.create_index(
            [("dataset", 1), ("request", 1)], unique=True, name="ta_columns_key"
        )
        self.columns.create_index([("created_at", -1)], name="ta_columns_age")
        self.runs.create_index(
            [("job", 1), ("dataset", 1), ("params", 1)], unique=True, name="ta_runs_key"
        )

    # ------------------------------------------------------------------
    # Computed series
    # ------------------------------------------------------------------

    def get_columns(self, dataset: str, request: str) -> pd.DataFrame | None:
        doc = self.columns.find_one({"dataset": dataset, "request": request})
        if not doc:
            return None
        index = pd.DatetimeIndex(doc["index"], name="time")
        if index.tz is None:
            index = index.tz_localize("UTC")
        return pd.DataFrame(
            {name: _decode(values) for name, values in doc["values"].items()},
            index=index,
        )

    def put_columns(self, dataset: str, request: str, frame: pd.DataFrame) -> None:
        doc = {
            "_id": _digest(dataset, request),
            "dataset": dataset,
            "request": request,
            "columns": list(frame.columns),
            "rows": int(len(frame)),
            "index": [ts.to_pydatetime() for ts in frame.index],
            "values": {
                name: _encode(frame[name].to_numpy(dtype="float64"))
                for name in frame.columns
            },
            "created_at": datetime.now(timezone.utc),
        }
        self.columns.replace_one({"_id": doc["_id"]}, doc, upsert=True)

    # ------------------------------------------------------------------
    # Finished runs
    # ------------------------------------------------------------------

    def get_run(self, job: str, dataset: str, params: Mapping[str, Any]) -> dict | None:
        doc = self.runs.find_one(
            {"job": job, "dataset": dataset, "params": params_digest(params)}
        )
        return doc.get("payload") if doc else None

    def put_run(
        self, job: str, dataset: str, params: Mapping[str, Any], payload: Mapping[str, Any]
    ) -> None:
        digest = params_digest(params)
        self.runs.replace_one(
            {"_id": _digest(job, dataset, digest)},
            {
                "_id": _digest(job, dataset, digest),
                "job": job,
                "dataset": dataset,
                "params": digest,
                "params_detail": dict(params),
                "payload": dict(payload),
                "created_at": datetime.now(timezone.utc),
            },
            upsert=True,
        )

    def stats(self) -> dict:
        return {
            "indicator_columns": self.columns.count_documents({}),
            "research_runs": self.runs.count_documents({}),
        }


def compute_cached(
    frame: pd.DataFrame,
    requests,
    *,
    dataset: str,
    cache: ResearchCache | None = None,
    refresh: bool = False,
    **compute_kwargs,
) -> tuple[pd.DataFrame, dict]:
    """`ta.compute`, but every request is looked up before it is computed.

    Returns the assembled columns and a `{"hits": n, "misses": n}` report --
    a cache whose hit rate you cannot see is one you cannot trust.

    Only the misses are computed, in a single batch, so a run that already
    has three quarters of its indicators cached pays for one quarter.
    """
    reqs = parse_requests(requests)
    pieces: dict[str, pd.DataFrame] = {}
    misses: list = []
    hits = 0

    for request in reqs:
        key = request.key()
        cached = None if (cache is None or refresh) else cache.get_columns(dataset, key)
        if cached is not None and len(cached) == len(frame):
            pieces[key] = cached
            hits += 1
        else:
            misses.append(request)

    if misses:
        result = compute(frame, misses, include_ohlcv=False, **compute_kwargs)
        for request in misses:
            spec_columns = _columns_for(result, request)
            piece = result.frame[spec_columns]
            pieces[request.key()] = piece
            if cache is not None:
                cache.put_columns(dataset, request.key(), piece)

    ordered = [pieces[r.key()] for r in reqs if r.key() in pieces]
    combined = pd.concat(ordered, axis=1) if ordered else pd.DataFrame(index=frame.index)
    combined = combined.loc[:, ~combined.columns.duplicated(keep="first")]
    return combined, {"hits": hits, "misses": len(misses)}


def _columns_for(result, request) -> list[str]:
    key = request.key()
    return [meta.column for meta in result.columns if meta.request == key]


_default: ResearchCache | None = None


def default_cache() -> ResearchCache:
    """Env-configured cache. `RESEARCH_DB_NAME` defaults to `ta_research`,
    deliberately a different database from the trading data."""
    global _default
    if _default is None:
        try:
            from pymongo import MongoClient

            client = MongoClient(env("MONGO_URI"), serverSelectionTimeoutMS=5000)
            db = client[env("RESEARCH_DB_NAME", default="ta_research")]
            cache = ResearchCache(db[COLUMNS_COLLECTION], db[RUNS_COLLECTION])
            cache.ensure_indexes()
        except Exception as exc:
            raise SourceError(f"cannot reach the research cache: {exc}") from exc
        _default = cache
    return _default
