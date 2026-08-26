"""Pipeline: one fetch, combined snapshot, CLI. Data layer stubbed."""

from __future__ import annotations

import json

import pandas as pd
import pytest

import ta.pipeline as pipeline
from ta.errors import FrameError, ParamError
from tests.conftest import make_ohlc


@pytest.fixture
def stub_data(monkeypatch):
    frame = make_ohlc(300)
    monkeypatch.setattr(pipeline, "read_ohlc", lambda *a, **k: frame)
    return frame


def test_combined_snapshot(stub_data):
    snapshot = pipeline.run(
        "USDTIRT",
        "nobitex",
        "1h",  # lowercase on purpose: must normalize to MDC's label
        indicators="rsi, macd",
        analyses="volatility, trend, regime",
    )
    assert snapshot["interval"] == "1H"
    assert snapshot["dropped_forming_bar"] is False  # fixture ends long ago
    assert "rsi_14" in snapshot["indicators"]["latest"]
    assert set(snapshot["analyses"]) == {"volatility", "trend", "regime"}
    assert snapshot["analyses"]["trend"]["scalars"]["direction"] in ("up", "down", "flat")
    assert "saved" not in snapshot  # save defaults off
    assert "records" not in snapshot["indicators"]


def test_analyses_only_and_parameterised_key(stub_data):
    snapshot = pipeline.run(
        "USDTIRT", "nobitex", "1H", analyses="volatility:window=30"
    )
    assert "indicators" not in snapshot
    entry = snapshot["analyses"]["volatility:window=30"]
    assert entry["params"]["window"] == 30


def test_save_path_calls_store(stub_data, monkeypatch):
    calls = {}

    def fake_save(result, **kwargs):
        calls.update(kwargs)
        return 123

    monkeypatch.setattr(pipeline.store, "save", fake_save)
    snapshot = pipeline.run(
        "USDTIRT", "nobitex", "1H", indicators="rsi", save=True
    )
    assert snapshot["saved"] == 123
    assert calls == {
        "symbol": "USDTIRT",
        "source": "nobitex",
        "interval": "1H",
        "latest_bar_only": False,
    }


def test_forming_bar_is_dropped(monkeypatch):
    # A frame whose last bar opened just now: still forming.
    end = pd.Timestamp.now(tz="UTC").floor("1h")
    index = pd.date_range(end=end, periods=300, freq="1h", tz="UTC")
    frame = make_ohlc(300)
    frame.index = index
    monkeypatch.setattr(pipeline, "read_ohlc", lambda *a, **k: frame)

    snapshot = pipeline.run("X", "nobitex", "1H", analyses="volatility")
    assert snapshot["dropped_forming_bar"] is True
    assert snapshot["frame"]["rows"] == 299

    kept = pipeline.run("X", "nobitex", "1H", analyses="volatility", drop_forming=False)
    assert kept["frame"]["rows"] == 300


def test_errors(stub_data, monkeypatch):
    with pytest.raises(FrameError, match="nothing to do"):
        pipeline.run("X", "nobitex", "1H")
    with pytest.raises(ParamError, match="unknown analysis"):
        pipeline.run("X", "nobitex", "1H", analyses="vibes")

    monkeypatch.setattr(pipeline, "read_ohlc", lambda *a, **k: None)
    with pytest.raises(FrameError, match="no candles"):
        pipeline.run("GHOST", "nobitex", "1H", analyses="volatility")


def test_cli_prints_snapshot_json(stub_data, capsys):
    code = pipeline.main(
        ["-m", "USDTIRT", "-s", "nobitex", "-i", "1H", "--analyses", "volatility,trend"]
    )
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["market"] == "USDTIRT"
    assert set(out["analyses"]) == {"volatility", "trend"}


def test_cli_reports_errors_on_stderr(stub_data, capsys):
    code = pipeline.main(["-m", "X", "-s", "nobitex", "-i", "1H", "--analyses", "vibes"])
    assert code == 1
    err = json.loads(capsys.readouterr().err)
    assert err["type"] == "ParamError"
