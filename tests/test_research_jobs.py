"""Job queue lifecycle, deduplication, recovery, and scenario expansion.
No Mongo, no compute -- the runner is a stub."""

from __future__ import annotations

import threading
import time

import pytest

from ta.research import expand, job_id, list_scenarios
from ta.research.jobs import (
    ACTIVE,
    DONE,
    FAILED,
    INTERRUPTED,
    QUEUED,
    RUNNING,
    JobQueue,
)
from ta.research.scenarios import SCENARIOS


class FakeJobs:
    def __init__(self):
        self.docs: dict = {}
        self.indexes: list = []
        self._lock = threading.Lock()

    def create_index(self, keys, **kwargs):
        self.indexes.append(keys)

    def replace_one(self, filt, doc, upsert=False):
        with self._lock:
            self.docs[filt["_id"]] = dict(doc)

    def update_one(self, filt, update):
        with self._lock:
            doc = self.docs.get(filt["_id"])
            if doc:
                doc.update(update["$set"])

    def update_many(self, filt, update):
        n = 0
        with self._lock:
            for doc in self.docs.values():
                if all(doc.get(k) == v for k, v in filt.items()):
                    doc.update(update["$set"])
                    n += 1
        return type("R", (), {"modified_count": n})()

    def find_one(self, filt):
        return self.docs.get(filt["_id"])

    def find(self, query=None, projection=None):
        query = query or {}
        rows = [d for d in self.docs.values()
                if all(d.get(k) == v for k, v in query.items())]
        return _Cursor(rows)


class _Cursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, key, direction=1):
        self.rows.sort(key=lambda d: str(d.get(key) or ""), reverse=direction < 0)
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    def __iter__(self):
        return iter(self.rows)


SPEC = {"market": "USDTIRT", "source": "nobitex", "interval": "1H",
        "family": "ma", "permutations": 20, "consecutive_only": True,
        "split": 0.7, "horizon": 1}


def wait_for(queue, identifier, states, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = queue.get(identifier)
        if job and job["status"] in states:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job never reached {states}: {queue.get(identifier)}")


def test_job_id_is_identity_not_submission_order():
    assert job_id(SPEC) == job_id({**SPEC})
    assert job_id(SPEC) != job_id({**SPEC, "interval": "15Min"})
    # Fields outside IDENTITY must not change the id.
    assert job_id(SPEC) == job_id({**SPEC, "limit": 500})


def test_a_job_runs_and_records_its_headline():
    def runner(spec):
        spec["log"]("working")
        return {"headline": {"best_signal": "sma_20_30", "p_value": 0.5},
                "from_cache": False}

    queue = JobQueue(FakeJobs(), runner=runner, workers=1)
    submitted = queue.submit(SPEC)
    assert submitted["status"] == QUEUED and not submitted["deduplicated"]

    job = wait_for(queue, submitted["_id"], {DONE})
    assert job["headline"]["best_signal"] == "sma_20_30"
    assert job["finished_at"] is not None
    assert "working" in job["log"]
    assert queue.summary()[DONE] == 1


def test_a_failing_job_is_recorded_not_fatal():
    def runner(spec):
        raise ValueError("no candles")

    queue = JobQueue(FakeJobs(), runner=runner, workers=1)
    job = wait_for(queue, queue.submit(SPEC)["_id"], {FAILED})

    assert "ValueError: no candles" in job["error"]
    assert job["traceback"]
    # The worker survives: a second job still runs.
    queue.runner = lambda spec: {"headline": {}, "from_cache": True}
    second = wait_for(queue, queue.submit({**SPEC, "interval": "4H"})["_id"], {DONE})
    assert second["status"] == DONE


def test_duplicate_submissions_are_deduplicated_while_active():
    release = threading.Event()

    def runner(spec):
        release.wait(timeout=5)
        return {"headline": {}, "from_cache": False}

    queue = JobQueue(FakeJobs(), runner=runner, workers=1)
    first = queue.submit(SPEC)
    wait_for(queue, first["_id"], {RUNNING})

    second = queue.submit(SPEC)
    assert second["deduplicated"] is True
    assert second["_id"] == first["_id"]
    assert second["status"] in ACTIVE

    release.set()
    wait_for(queue, first["_id"], {DONE})

    # Once finished it is no longer deduplicated -- resubmitting re-runs it.
    third = queue.submit(SPEC)
    assert third["deduplicated"] is False


def test_recover_marks_orphans_interrupted_rather_than_retrying():
    collection = FakeJobs()
    collection.docs["abc"] = {"_id": "abc", "status": RUNNING}
    collection.docs["def"] = {"_id": "def", "status": DONE}

    queue = JobQueue(collection, runner=lambda s: {}, workers=1)
    assert queue.recover() == 1
    assert collection.docs["abc"]["status"] == INTERRUPTED
    assert "restarted" in collection.docs["abc"]["error"]
    assert collection.docs["def"]["status"] == DONE  # untouched


def test_submit_many_and_listing():
    queue = JobQueue(FakeJobs(), runner=lambda s: {"headline": {}, "from_cache": True},
                     workers=2)
    specs = [{**SPEC, "interval": i} for i in ("1H", "4H", "15Min")]
    submitted = queue.submit_many(specs)
    assert len({j["_id"] for j in submitted}) == 3

    for job in submitted:
        wait_for(queue, job["_id"], {DONE})
    assert len(queue.list()) == 3
    assert len(queue.list(status=DONE)) == 3
    assert queue.list(status=FAILED) == []


# ------------------------------------------------------------- scenarios


def test_every_scenario_expands_to_runnable_jobs():
    for spec in list_scenarios():
        jobs = expand(spec["name"])
        assert len(jobs) == spec["jobs"]
        for job in jobs:
            assert {"market", "source", "interval", "family"} <= set(job)
            assert job["permutations"] > 0


def test_scenarios_default_to_the_gap_control():
    """`consecutive_only` is a control, not a tuning knob: sparse candles make
    a forward return span a variable horizon."""
    for name in SCENARIOS:
        assert all(job["consecutive_only"] for job in expand(name))


def test_unknown_scenario_names_the_known_ones():
    with pytest.raises(KeyError, match="usdt-cross-venue"):
        expand("nope")


def test_cross_venue_scenario_covers_both_usdt_books():
    jobs = expand("usdt-cross-venue")
    assert {(j["market"], j["source"]) for j in jobs} == {
        ("USDTIRT", "nobitex"), ("USDTIRR", "ramzinex")
    }
