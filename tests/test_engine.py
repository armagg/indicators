"""Engine behaviour: request parsing, batching, warmup policies, causality."""

from __future__ import annotations

import numpy as np
import pytest

import ta
from ta.engine import IndicatorRequest, parse_requests
from ta.errors import NonCausalIndicator, ParamError, UnknownIndicator


# ------------------------------------------------------------------ parsing


def test_parse_compact_string_with_embedded_commas():
    requests = parse_requests("macd:fast=8,slow=21,signal=5,rsi,ema:window=50")
    assert [r.name for r in requests] == ["macd", "rsi", "ema"]
    assert requests[0].params == {"fast": "8", "slow": "21", "signal": "5"}
    assert requests[2].params == {"window": "50"}


def test_parse_mapping_forms():
    nested = IndicatorRequest.parse({"name": "rsi", "params": {"period": 21}})
    flat = IndicatorRequest.parse({"name": "rsi", "period": 21})
    assert nested == flat == IndicatorRequest("rsi", {"period": 21})

    with pytest.raises(ParamError):
        IndicatorRequest.parse({"name": "rsi", "params": {"period": 21}, "stray": 1})
    with pytest.raises(ParamError):
        IndicatorRequest.parse({"period": 21})  # no name


def test_request_key_is_order_stable():
    a = IndicatorRequest("macd", {"slow": 26, "fast": 12})
    b = IndicatorRequest("macd", {"fast": 12, "slow": 26})
    assert a.key() == b.key()


# ------------------------------------------------------------------ compute


def test_batch_compute_shares_one_frame(ohlc):
    result = ta.compute(ohlc, "sma:window=20, rsi, macd")
    for column in ("open", "close", "sma_20", "rsi_14", "macd_12_26_9"):
        assert column in result.frame.columns
    assert {c.indicator for c in result.columns} == {"sma", "rsi", "macd"}


def test_unknown_indicator_and_param(ohlc):
    with pytest.raises(UnknownIndicator):
        ta.compute(ohlc, "definitely_not_real")
    # The legacy service silently ignored typo'd params; this must refuse.
    with pytest.raises(ParamError, match="perid"):
        ta.compute(ohlc, "rsi:perid=30")


def test_alias_resolves(ohlc):
    result = ta.compute(ohlc, "stochastic_rsi")
    assert "stoch_rsi_k_14_3" in result.frame.columns


def test_warmup_policies_differ(ohlc):
    masked = ta.compute(ohlc, "rsi", warmup="mask")
    kept = ta.compute(ohlc, "rsi", warmup="keep")
    trimmed = ta.compute(ohlc, "rsi", warmup="trim")

    n_masked = masked.frame["rsi_14"].notna().sum()
    n_kept = kept.frame["rsi_14"].notna().sum()
    assert n_kept > n_masked, "keep must expose the seed-influenced head"

    warmup_bars = masked.columns[0].warmup_bars
    assert len(trimmed.frame) == len(ohlc) - warmup_bars
    assert trimmed.trimmed_rows == warmup_bars


def test_duplicate_requests_collapse(ohlc):
    result = ta.compute(ohlc, "rsi, rsi:period=14")
    assert list(result.frame.columns).count("rsi_14") == 1
    assert any("duplicate" in w for w in result.warnings)


def test_noncausal_refused_by_default(ohlc):
    with pytest.raises(NonCausalIndicator):
        ta.compute(ohlc, "ichimoku_chikou")
    result = ta.compute(ohlc, "ichimoku_chikou", allow_noncausal=True)
    assert "chikou_26" in result.frame.columns


# ------------------------------------------------------------------ results


def test_to_records_renders_nan_as_none(ohlc):
    result = ta.compute(ohlc, "sma:window=490")
    records = result.to_records()
    assert len(records) == len(ohlc)
    assert records[0]["sma_490"] is None  # warmup head
    assert isinstance(records[0]["time"], str)
    tail_value = records[-1]["sma_490"]
    assert tail_value is not None and np.isfinite(tail_value)


def test_latest_reports_age(ohlc):
    result = ta.compute(ohlc, "sma:window=20")
    latest = result.latest()["sma_20"]
    assert latest["bars_ago"] == 0
    assert latest["value"] == pytest.approx(
        ohlc["close"].rolling(20).mean().iloc[-1]
    )


def test_insufficient_data_is_flagged_not_fatal(tiny):
    result = ta.compute(tiny, "sma:window=200")
    meta = result.columns[0]
    assert not meta.sufficient
    assert any("warmup" in w for w in result.warnings)
