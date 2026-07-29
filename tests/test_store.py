"""Persistence: document shape, natural key, idempotency. No Mongo required --
`docs_for` is pure and `save` accepts an injected collection."""

from __future__ import annotations

from datetime import timezone as _tz

import pytest

import ta
from ta import store


@pytest.fixture
def result(ohlc):
    return ta.compute(ohlc, "rsi, macd:fast=8,slow=21,signal=5")


def test_param_hash_is_order_stable_and_value_sensitive():
    assert store.param_hash({"fast": 12, "slow": 26}) == store.param_hash({"slow": 26, "fast": 12})
    assert store.param_hash({"period": 14}) != store.param_hash({"period": 21})


def test_docs_shape_and_natural_key(result):
    docs = store.docs_for(result, symbol="USDTIRT", source="nobitex", interval="1H")
    assert docs, "a warmed-up frame must produce documents"

    analyses = {d["analysis"] for d in docs}
    assert analyses == {"rsi", "macd"}

    sample = next(d for d in docs if d["analysis"] == "macd")
    for field in store.NATURAL_KEY:
        assert field in sample
    # One doc carries the whole request: line, signal, and histogram together.
    assert set(sample["values"]) == {"macd_8_21_5", "macd_signal_8_21_5", "macd_hist_8_21_5"}
    assert sample["params"] == {"fast": 8, "slow": 21, "signal": 5}
    assert sample["time"].tzinfo is not None
    assert sample["time"].utcoffset().total_seconds() == 0  # stored in UTC

    # Natural keys are unique -- the idempotency invariant.
    keys = [tuple(d[k] for k in store.NATURAL_KEY) for d in docs]
    assert len(keys) == len(set(keys))


def test_warmup_masked_rows_are_skipped(result):
    docs = store.docs_for(result, symbol="X", source="s", interval="1H")
    rsi_meta = next(c for c in result.columns if c.indicator == "rsi")
    rsi_docs = [d for d in docs if d["analysis"] == "rsi"]
    expected_rows = int(result.frame["rsi_14"].notna().sum())
    assert len(rsi_docs) == expected_rows
    assert expected_rows == rsi_meta.valid_values


class FakeCollection:
    def __init__(self):
        self.indexes = []
        self.batches = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def bulk_write(self, ops, ordered):
        self.batches.append(ops)


def test_save_upserts_on_natural_key(result):
    fake = FakeCollection()
    count = store.save(result, symbol="USDTIRT", source="nobitex", interval="1H", collection=fake)

    assert count == len(fake.batches[0])
    _, index_kwargs = fake.indexes[0]
    assert index_kwargs["unique"] is True
    assert "partialFilterExpression" in index_kwargs  # legacy rows must not break it

    op = fake.batches[0][0]
    # ReplaceOne(filter, doc, upsert=True): filter is exactly the natural key.
    assert set(op._filter) == set(store.NATURAL_KEY)
    assert op._upsert is True

    # Saving the same result twice targets identical filters -> overwrites, not duplicates.
    store.save(result, symbol="USDTIRT", source="nobitex", interval="1H", collection=fake)
    first = [op._filter for op in fake.batches[0]]
    second = [op._filter for op in fake.batches[1]]
    assert first == second
