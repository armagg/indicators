"""HTTP adapter contract, with the data layer stubbed out."""

from __future__ import annotations

import pytest

import app as app_module
from tests.conftest import make_ohlc


@pytest.fixture
def client(monkeypatch):
    frame = make_ohlc(300)
    monkeypatch.setattr(app_module, "read_ohlc", lambda *a, **k: frame)
    monkeypatch.setattr(app_module.store, "save", lambda result, **k: 42)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()


def test_catalogue_endpoint(client):
    response = client.get("/indicators")
    assert response.status_code == 200
    body = response.get_json()
    assert body["count"] >= 30
    assert "momentum" in body["categories"]


def test_source_is_required(client):
    response = client.get("/indicator?symbol=USDTIRT&interval=1H&indicator=rsi")
    assert response.status_code == 400
    assert "source" in response.get_json()["error"]


def test_legacy_single_indicator_form(client):
    response = client.get(
        "/indicator?symbol=USDTIRT&source=nobitex&interval=1H&indicator=rsi&period=21"
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["saved"] == 42
    assert "rsi_21" in body["latest"]
    assert body["meta"]["columns"][0]["params"] == {"period": 21}
    assert len(body["data"]) == 300


def test_batch_compact_form_and_latest_only(client):
    response = client.get(
        "/indicator?symbol=USDTIRT&source=nobitex&interval=1H"
        "&indicators=rsi:period=21,macd,atr&latest=true&save=false"
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["saved"] == 0
    assert "data" not in body  # latest=true trims the payload
    assert {"rsi_21", "macd_12_26_9", "atr_14"} <= set(body["latest"])


def test_unknown_indicator_and_bad_param_are_400(client):
    assert (
        client.get(
            "/indicator?symbol=X&source=s&interval=1H&indicator=vibes"
        ).status_code
        == 400
    )
    response = client.get(
        "/indicator?symbol=X&source=s&interval=1H&indicator=rsi&perid=30"
    )
    assert response.status_code == 400
    assert "perid" in response.get_json()["error"]


def test_missing_data_is_404(client, monkeypatch):
    monkeypatch.setattr(app_module, "read_ohlc", lambda *a, **k: None)
    response = client.get(
        "/indicator?symbol=GHOST&source=nobitex&interval=1H&indicator=rsi"
    )
    assert response.status_code == 404


def test_mixing_batch_and_loose_params_is_400(client):
    response = client.get(
        "/indicator?symbol=X&source=s&interval=1H&indicators=rsi,macd&period=9"
    )
    assert response.status_code == 400


def test_analysis_endpoint(client, monkeypatch):
    import ta.pipeline as pipeline_module

    monkeypatch.setattr(pipeline_module, "read_ohlc", lambda *a, **k: make_ohlc(300))
    response = client.get(
        "/analysis?symbol=USDTIRT&source=nobitex&interval=1H"
        "&analyses=volatility,trend&indicators=natr"
    )
    assert response.status_code == 200
    body = response.get_json()
    assert set(body["analyses"]) == {"volatility", "trend"}
    assert "natr_14" in body["indicators"]["latest"]
    assert "saved" not in body  # /analysis does not persist by default

    assert client.get("/analysis?symbol=X&source=s&interval=1H").status_code == 400
    assert (
        client.get(
            "/analysis?symbol=X&source=s&interval=1H&analyses=trend&bogus=1"
        ).status_code
        == 400
    )


def test_shadow_endpoints_survive_the_three_field_watchlist(client, monkeypatch):
    """Regression: `watchlist_from_env` returns (market, source, config_market)
    triples, and the status endpoint used to unpack them as pairs — a 500
    that only appeared once a config alias was configured."""
    import ta.research.shadow as shadow_module

    monkeypatch.setenv("SHADOW_MARKETS", "USDTIRT:nobitex:USDTRLS,USDTIRR:ramzinex")

    class FakeStore:
        def markets(self):
            return [("USDTIRT", "nobitex")]

        def rows(self, market, source, limit=5000):
            return [{"bar_time": "2026-07-31T13:30:00", "tilt": 0.1, "mid": 1.0,
                     "implied_target_base": 230.4, "live_target_base": None,
                     "live_config_version": 109}]

        def count(self):
            return 1

    monkeypatch.setattr(shadow_module, "default_store", lambda: FakeStore())
    monkeypatch.setattr(
        "ta.research.scheduler.default_scheduler", lambda: None
    )

    response = client.get("/research/shadow")
    assert response.status_code == 200
    body = response.get_json()
    assert body["watchlist"] == [
        "USDTIRT/nobitex (config: USDTRLS)", "USDTIRR/ramzinex"
    ]
    assert body["markets"]["USDTIRT/nobitex"]["observations"] == 1

    score = client.get("/research/shadow/score")
    assert score.status_code == 200


def test_snapshot_requires_source_market_interval(client):
    assert client.get("/snapshot").status_code == 400
    assert client.get("/snapshot?source=nobitex&market=USDTRLS").status_code == 400


def test_snapshot_is_read_only_and_404_when_missing(client, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("GET /snapshot must not compute")

    monkeypatch.setattr("ta.pipeline.run", boom)
    monkeypatch.setattr("ta.ohlc.read_ohlc", boom)
    monkeypatch.setattr("ta.snapshot.read_snapshot", lambda *_a, **_k: None)
    response = client.get("/snapshot?source=nobitex&market=USDTRLS&interval=1H")
    assert response.status_code == 404
    assert response.get_json()["error"] == "no snapshot"


def test_snapshot_returns_stored_payload(client, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("GET /snapshot must not compute")

    monkeypatch.setattr("ta.pipeline.run", boom)

    def fake_read(source, market, interval, **_k):
        return {
            "source": source,
            "market": market,
            "ohlc_market": "USDTIRT",
            "interval": interval,
            "asof": "2026-08-26T11:00:00+00:00",
            "computed_at": "2026-08-26T11:01:00+00:00",
            "stale": False,
            "indicators": {"rsi_14": 55.0, "natr_14": 1.2},
            "analyses": {"volatility": {"scalars": {"vol_percentile": 0.4}}},
            "dropped_forming_bar": True,
        }

    monkeypatch.setattr("ta.snapshot.read_snapshot", fake_read)
    response = client.get("/snapshot?source=nobitex&market=USDTRLS&interval=1H")
    assert response.status_code == 200
    body = response.get_json()
    assert body["stale"] is False
    assert body["indicators"]["rsi_14"] == 55.0
    assert body["ohlc_market"] == "USDTIRT"
