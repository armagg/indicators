"""Research cache: fingerprint invalidation, round-trips, and hit accounting.
No Mongo -- collections are injected fakes."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ta
from ta.cache import (
    ResearchCache,
    compute_cached,
    frame_fingerprint,
    params_digest,
)
from tests.conftest import make_ohlc


class FakeCollection:
    def __init__(self):
        self.docs: dict = {}
        self.indexes: list = []
        self.writes = 0

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def replace_one(self, filt, doc, upsert=False):
        self.writes += 1
        self.docs[doc.get("_id", len(self.docs))] = dict(doc)

    def find_one(self, filt):
        for doc in self.docs.values():
            if all(doc.get(k) == v for k, v in filt.items()):
                return doc
        return None

    def count_documents(self, query):
        return len(self.docs)


@pytest.fixture
def cache():
    c = ResearchCache(columns=FakeCollection(), runs=FakeCollection())
    c.ensure_indexes()
    return c


# ------------------------------------------------------------ fingerprint


def test_fingerprint_is_stable_for_identical_data(ohlc):
    assert frame_fingerprint(ohlc) == frame_fingerprint(ohlc.copy())


def test_fingerprint_changes_when_candles_are_appended(ohlc):
    extended = pd.concat([ohlc, ohlc.iloc[[-1]].rename(index={ohlc.index[-1]: ohlc.index[-1] + pd.Timedelta(hours=1)})])
    assert frame_fingerprint(extended) != frame_fingerprint(ohlc)


def test_fingerprint_catches_a_rewritten_bar_in_the_middle():
    """The case an endpoint-only hash misses: MDC upserts, so a bar can be
    rewritten when late trades land, leaving count and endpoints identical."""
    frame = make_ohlc(400)
    rewritten = frame.copy()
    rewritten.iloc[200, rewritten.columns.get_loc("close")] *= 1.05

    assert len(rewritten) == len(frame)
    assert rewritten.index[0] == frame.index[0] and rewritten.index[-1] == frame.index[-1]
    assert frame_fingerprint(rewritten) != frame_fingerprint(frame)


def test_params_digest_ignores_key_order():
    assert params_digest({"a": 1, "b": 2}) == params_digest({"b": 2, "a": 1})
    assert params_digest({"a": 1}) != params_digest({"a": 2})


# --------------------------------------------------------------- columns


def test_column_round_trip_preserves_values_index_and_nans(ohlc, cache):
    computed = ta.compute(ohlc, "sma:window=20", include_ohlcv=False)
    original = computed.frame

    cache.put_columns("ds1", "sma:window=20", original)
    restored = cache.get_columns("ds1", "sma:window=20")

    pd.testing.assert_index_equal(restored.index, original.index)
    assert restored["sma_20"].isna().sum() == original["sma_20"].isna().sum()
    np.testing.assert_allclose(
        restored["sma_20"].to_numpy(), original["sma_20"].to_numpy(), equal_nan=True
    )


def test_missing_entries_return_none(cache):
    assert cache.get_columns("nope", "sma:window=20") is None
    assert cache.get_run("job", "nope", {"a": 1}) is None


# ---------------------------------------------------------- compute_cached


def test_second_call_is_all_hits_and_computes_nothing_new(ohlc, cache):
    dataset = frame_fingerprint(ohlc)
    requests = "sma:window=20, ema:window=50, rsi"

    first, report = compute_cached(ohlc, requests, dataset=dataset, cache=cache)
    assert report == {"hits": 0, "misses": 3}
    writes_after_first = cache.columns.writes

    second, report = compute_cached(ohlc, requests, dataset=dataset, cache=cache)
    assert report == {"hits": 3, "misses": 0}
    assert cache.columns.writes == writes_after_first  # nothing recomputed
    # check_freq=False: a DatetimeIndex built from date_range carries an
    # inferred `freq`, and one rebuilt from stored timestamps does not. BSON
    # cannot carry it and no indicator reads it -- the values are identical.
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_partial_hits_compute_only_the_misses(ohlc, cache):
    dataset = frame_fingerprint(ohlc)
    compute_cached(ohlc, "sma:window=20", dataset=dataset, cache=cache)

    _, report = compute_cached(
        ohlc, "sma:window=20, ema:window=50", dataset=dataset, cache=cache
    )
    assert report == {"hits": 1, "misses": 1}


def test_changed_data_misses_the_cache(ohlc, cache):
    compute_cached(ohlc, "sma:window=20", dataset=frame_fingerprint(ohlc), cache=cache)

    moved = ohlc.copy()
    moved.iloc[100, moved.columns.get_loc("close")] *= 1.02
    _, report = compute_cached(
        moved, "sma:window=20", dataset=frame_fingerprint(moved), cache=cache
    )
    assert report == {"hits": 0, "misses": 1}


def test_refresh_bypasses_the_cache(ohlc, cache):
    dataset = frame_fingerprint(ohlc)
    compute_cached(ohlc, "sma:window=20", dataset=dataset, cache=cache)
    _, report = compute_cached(
        ohlc, "sma:window=20", dataset=dataset, cache=cache, refresh=True
    )
    assert report == {"hits": 0, "misses": 1}


def test_cached_values_match_uncached_exactly(ohlc, cache):
    dataset = frame_fingerprint(ohlc)
    direct = ta.compute(ohlc, "macd", include_ohlcv=False).frame

    compute_cached(ohlc, "macd", dataset=dataset, cache=cache)
    from_cache, report = compute_cached(ohlc, "macd", dataset=dataset, cache=cache)

    assert report["hits"] == 1
    for column in direct.columns:
        np.testing.assert_allclose(
            from_cache[column].to_numpy(), direct[column].to_numpy(), equal_nan=True
        )


def test_works_with_no_cache_at_all(ohlc):
    frame, report = compute_cached(ohlc, "sma:window=20", dataset="x", cache=None)
    assert report == {"hits": 0, "misses": 1}
    assert "sma_20" in frame.columns


# ------------------------------------------------------------------- runs


def test_run_round_trip(cache):
    payload = {"best": {"kind": "sma", "excess_bps": 0.27}, "p_value": 0.08}
    cache.put_run("brute", "ds1", {"family": "ma", "perms": 200}, payload)

    assert cache.get_run("brute", "ds1", {"perms": 200, "family": "ma"}) == payload
    assert cache.get_run("brute", "ds1", {"family": "ma", "perms": 100}) is None
    assert cache.get_run("brute", "ds2", {"family": "ma", "perms": 200}) is None


def test_stats_counts_both_collections(ohlc, cache):
    compute_cached(ohlc, "sma:window=20", dataset="d", cache=cache)
    cache.put_run("brute", "d", {}, {"ok": True})
    assert cache.stats() == {"indicator_columns": 1, "research_runs": 1}
