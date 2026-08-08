"""A durable job queue with in-process workers.

The service accepts jobs, runs them in the background, and hands back
results — instead of a container being started and torn down per task.

Two design choices worth stating:

**Jobs live in Mongo, not in memory.** A queue that dies with the process
loses the record of what was asked, which on a multi-hour sweep is the
expensive part. On startup, jobs left `running` by a previous process are
marked `interrupted` rather than silently retried: a half-finished run whose
outcome nobody observed should be visible as such, not quietly resurrected.

**Workers are threads, few of them.** The heavy work is numpy, which
releases the GIL, so threads get real parallelism without a second process
to supervise. The count is deliberately small — this box also runs
production services, and a research sweep must never be the reason the
collector falls behind.

Deduplication: an identical job that is already queued or running is not
enqueued twice; the existing job's id comes back instead. Combined with the
result cache, submitting the same scenario repeatedly is cheap and safe.
"""

from __future__ import annotations

import hashlib
import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from ..errors import SourceError

QUEUED, RUNNING, DONE, FAILED, INTERRUPTED = (
    "queued", "running", "done", "failed", "interrupted"
)
ACTIVE = (QUEUED, RUNNING)

#: Fields that define what a job *is*. Two jobs agreeing on all of these are
#: the same job, however they were submitted.
IDENTITY = ("market", "source", "interval", "family", "permutations",
            "consecutive_only", "split", "horizon")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def job_id(spec: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {k: spec.get(k) for k in IDENTITY}, sort_keys=True, default=str
    )
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


@dataclass
class JobQueue:
    """Mongo-backed queue plus a thread pool. One per process."""

    collection: Any
    runner: Callable[[dict], dict]
    workers: int = 2

    def __post_init__(self):
        self._pool = ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="research"
        )
        self._lock = threading.Lock()
        self.collection.create_index([("status", 1), ("submitted_at", 1)], name="ta_job_queue")

    # ------------------------------------------------------------------

    def recover(self) -> int:
        """Mark jobs orphaned by a previous process. Call once at startup."""
        result = self.collection.update_many(
            {"status": RUNNING},
            {"$set": {"status": INTERRUPTED, "finished_at": _utcnow(),
                      "error": "process restarted while this job was running"}},
        )
        return int(getattr(result, "modified_count", 0) or 0)

    def submit(self, spec: Mapping[str, Any]) -> dict:
        """Enqueue one job, or return the existing one if already active."""
        spec = dict(spec)
        identifier = job_id(spec)
        with self._lock:
            existing = self.collection.find_one({"_id": identifier})
            if existing and existing.get("status") in ACTIVE:
                return {**existing, "deduplicated": True}
            doc = {
                "_id": identifier,
                **{k: spec.get(k) for k in IDENTITY},
                "status": QUEUED,
                "submitted_at": _utcnow(),
                "started_at": None,
                "finished_at": None,
                "error": None,
                "headline": None,
                "log": [],
            }
            self.collection.replace_one({"_id": identifier}, doc, upsert=True)
        self._pool.submit(self._execute, identifier, spec)
        return {**doc, "deduplicated": False}

    def submit_many(self, specs) -> list[dict]:
        return [self.submit(spec) for spec in specs]

    def _execute(self, identifier: str, spec: dict) -> None:
        lines: list[str] = []
        self.collection.update_one(
            {"_id": identifier},
            {"$set": {"status": RUNNING, "started_at": _utcnow()}},
        )
        try:
            payload = self.runner({**spec, "log": lines.append})
            self.collection.update_one(
                {"_id": identifier},
                {"$set": {
                    "status": DONE, "finished_at": _utcnow(),
                    "headline": payload.get("headline"),
                    "from_cache": payload.get("from_cache", False),
                    "log": lines[-40:],
                }},
            )
        except Exception as exc:  # a failed job must not take the worker down
            self.collection.update_one(
                {"_id": identifier},
                {"$set": {
                    "status": FAILED, "finished_at": _utcnow(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc()[-2000:],
                    "log": lines[-40:],
                }},
            )

    # ------------------------------------------------------------------

    def get(self, identifier: str) -> dict | None:
        return self.collection.find_one({"_id": identifier})

    def list(self, status: str | None = None, limit: int = 100) -> list[dict]:
        query = {"status": status} if status else {}
        return list(
            self.collection.find(query).sort("submitted_at", -1).limit(int(limit))
        )

    def summary(self) -> dict:
        out = {state: 0 for state in (QUEUED, RUNNING, DONE, FAILED, INTERRUPTED)}
        for doc in self.collection.find({}, {"status": 1}):
            out[doc.get("status", QUEUED)] = out.get(doc.get("status", QUEUED), 0) + 1
        return out


_queue: JobQueue | None = None


def default_queue() -> JobQueue:
    """Process-wide queue, wired to Mongo and the brute-force runner."""
    global _queue
    if _queue is None:
        from decouple import config as env

        from ..cache import default_cache
        from .brute import run_brute

        def runner(spec: dict) -> dict:
            log = spec.pop("log", None)
            return run_brute(cache=default_cache(), log=log, **spec)

        try:
            from pymongo import MongoClient

            client = MongoClient(env("MONGO_URI"), serverSelectionTimeoutMS=5000)
            db = client[env("RESEARCH_DB_NAME", default="ta_research")]
            queue = JobQueue(
                collection=db["jobs"],
                runner=runner,
                workers=int(env("RESEARCH_WORKERS", default=2)),
            )
            queue.recover()
        except Exception as exc:
            raise SourceError(f"cannot start the research queue: {exc}") from exc
        _queue = queue
    return _queue
