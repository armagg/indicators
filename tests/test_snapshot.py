"""Always-on snapshots: mapping, stale, persist, read-only GET. No Mongo."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from ta.configclient import MarketConfig
from ta.errors import SourceError
from ta.snapshot import (
    FX_15MIN_MARKETS,
    ProcessCache,
    SnapshotScheduler,
    SnapshotStore,
    capture_enabled,
    capture_one,
    extra_15min,
    intervals_for,
    is_stale,
    ohlc_market,
    public_payload,
    read_snapshot,
    sibling_market,
)
from tests.conftest import make_ohlc

NOW = datetime(2026, 8, 26, 12, 5, tzinfo=timezone.utc)
ASOF = datetime(2026, 8, 26, 11, 0, tzinfo=timezone.utc)


class FakeLatest:
    def __init__(self):
        self.docs: dict[tuple, dict] = {}
        self.indexes = []
        self.reads = 0

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def replace_one(self, filt, doc, upsert=True):
        self.docs[tuple(filt[k] for k in ("source", "market", "interval"))] = dict(doc)

    def find_one(self, filt, projection=None):
        self.reads += 1
        doc = self.docs.get((filt["source"], filt["market"], filt["interval"]))
        if doc is None:
            return None
        if projection and projection.get("_id") == 0:
            return {k: v for k, v in doc.items() if k != "_id"}
        return dict(doc)


def _cfg(source, market, enabled=True):
    return MarketConfig(
        source=source,
        market=market,
        version=1,
        config={"is_enable": enabled},
    )


def _frame(end=ASOF, n=8, freq="1h"):
    frame = make_ohlc(n)
    frame.index = pd.date_range(end=end, periods=n, freq=freq, tz="UTC")
    return frame


def _reader(frame=None):
    candles = frame if frame is not None else _frame()

    def reader(*_a, **_k):
        return candles.copy()

    return reader


def _run(calls):
    def run(market, source, interval, **kwargs):
        calls.append({"market": market, "source": source, "interval": interval, **kwargs})
        return {
            "market": market,
            "source": source,
            "interval": interval,
            "asof": ASOF.isoformat(),
            "dropped_forming_bar": True,
            "indicators": {
                "latest": {
                    "rsi_14": {"value": 55.0, "time": ASOF.isoformat(), "bars_ago": 0},
                    "natr_14": {"value": 1.2, "time": ASOF.isoformat(), "bars_ago": 0},
                }
            },
            "analyses": {
                "volatility": {
                    "asof": ASOF.isoformat(),
                    "scalars": {"vol_percentile": 0.4, "natr": 1.2},
                    "warnings": [],
                },
                "trend": {
                    "asof": ASOF.isoformat(),
                    "scalars": {"trend_score": 0.1},
                    "warnings": [],
                },
                "regime": {
                    "asof": ASOF.isoformat(),
                    "scalars": {"squeeze_on": False},
                    "warnings": [],
                },
            },
        }

    return run


def test_nobitex_config_name_maps_to_irt_candles():
    assert ohlc_market("nobitex", "USDTRLS") == "USDTIRT"
    assert ohlc_market("nobitex", "BTCRLS") == "BTCIRT"
    assert ohlc_market("ramzinex", "USDTIRR") == "USDTIRR"
    assert ohlc_market("nobitex", "USDTIRT") == "USDTIRT"
    assert sibling_market("nobitex", "USDTRLS") == "USDTIRT"
    assert sibling_market("nobitex", "USDTIRT") == "USDTRLS"
    assert sibling_market("ramzinex", "USDTIRR") is None


def test_wallex_config_name_maps_to_irt_candles():
    """The bug this fixes: every wallex market was dark.

    market-data-collector rewrites wallex TMN->IRT on ingest
    (exchanges/wallex.py:75), but the trading config asks for *TMN. Only
    nobitex had a translation, so every wallex candle read came back empty and
    the whole venue produced no snapshots -- which risk-aggregator then read as
    "no volatility risk". Verified against the store 2026-09-05: wallex ETHTMN
    = 0 docs, wallex ETHIRT = 693,292.
    """
    assert ohlc_market("wallex", "ETHTMN") == "ETHIRT"
    assert ohlc_market("wallex", "USDTTMN") == "USDTIRT"
    assert sibling_market("wallex", "ETHTMN") == "ETHIRT"
    assert sibling_market("wallex", "ETHIRT") == "ETHTMN"


def test_ramzinex_is_deliberately_not_renamed():
    """A blanket normalise-to-IRT would break the venue that works.

    ramzinex's collector keeps the config name verbatim, so its candles really
    are stored as *IRR (ETHIRR = 525,282 docs, ETHIRT = 0). The suffix map is
    per-source for this reason; ramzinex must stay absent from it.
    """
    assert ohlc_market("ramzinex", "ETHIRR") == "ETHIRR"
    assert ohlc_market("ramzinex", "BTCIRR") == "BTCIRR"
    assert sibling_market("ramzinex", "ETHIRR") is None


def test_fx_tapes_get_15min_everyone_else_hourly_only():
    assert extra_15min("USDTRLS") and extra_15min("USDTIRR") and extra_15min("USDTTMN")
    assert extra_15min("USDTIRT")
    assert not extra_15min("BTCRLS")
    assert intervals_for("USDTRLS") == ("1H", "15Min")
    assert intervals_for("BTCRLS") == ("1H",)
    assert FX_15MIN_MARKETS >= {"USDTRLS", "USDTIRR", "USDTTMN"}


def test_stale_is_two_bars_of_computed_at():
    fresh = NOW - timedelta(hours=1)
    assert is_stale(fresh, "1H", NOW) is False
    assert is_stale(NOW - timedelta(hours=2, seconds=1), "1H", NOW) is True
    assert is_stale(NOW - timedelta(minutes=20), "15Min", NOW) is False
    assert is_stale(NOW - timedelta(minutes=31), "15Min", NOW) is True


def test_capture_one_keys_config_name_and_fetches_ohlc_alias():
    store = SnapshotStore(FakeLatest())
    calls = []
    result = capture_one(
        "nobitex",
        "USDTRLS",
        "1H",
        snapshot_store=store,
        now=NOW,
        run=_run(calls),
        reader=_reader(),
    )
    assert result["status"] == "ok"
    assert result["market"] == "USDTRLS"
    assert calls[0]["market"] == "USDTIRT"
    assert calls[0]["save"] is True
    assert calls[0]["latest_bar_only"] is True
    doc = store.get("nobitex", "USDTRLS", "1H")
    assert doc["ohlc_market"] == "USDTIRT"
    assert doc["indicators"]["rsi_14"] == 55.0
    assert doc["analyses"]["volatility"]["scalars"]["vol_percentile"] == 0.4


def test_capture_one_skips_when_asof_already_stored():
    store = SnapshotStore(FakeLatest())
    calls = []
    run = _run(calls)
    capture_one(
        "nobitex", "USDTRLS", "1H", snapshot_store=store, now=NOW, run=run, reader=_reader()
    )
    again = capture_one(
        "nobitex", "USDTRLS", "1H", snapshot_store=store, now=NOW, run=run, reader=_reader()
    )
    assert again["status"] == "skipped"
    assert len(calls) == 1


def test_capture_enabled_runs_15min_on_fx_skips_disabled_and_isolates_failures():
    store = SnapshotStore(FakeLatest())
    calls = []

    class Client:
        def list_markets(self, source):
            if source == "bitpin":
                raise RuntimeError("config-manager down")
            if source == "nobitex":
                return [
                    _cfg("nobitex", "USDTRLS", True),
                    _cfg("nobitex", "BTCRLS", False),
                    _cfg("nobitex", "ETHRLS", True),
                ]
            if source == "ramzinex":
                return [_cfg("ramzinex", "USDTIRR", True)]
            return []

    def reader(market, source, interval, **_k):
        if market == "ETHIRT":
            raise SourceError("no candles")
        return _frame()

    out = capture_enabled(
        client=Client(),
        snapshot_store=store,
        sources=["nobitex", "ramzinex", "bitpin"],
        now=NOW,
        run=_run(calls),
        reader=reader,
    )
    markets = {(c["source"], c["market"], c["interval"]) for c in calls}
    assert ("nobitex", "USDTIRT", "1H") in markets
    assert ("nobitex", "USDTIRT", "15Min") in markets
    assert ("ramzinex", "USDTIRR", "1H") in markets
    assert ("ramzinex", "USDTIRR", "15Min") in markets
    assert not any(c["market"] == "BTCRLS" or c["market"] == "BTCIRT" for c in calls)
    assert out["n_failed"] >= 2
    assert "bitpin" in out["failures"]
    assert any("ETHRLS" in key for key in out["failures"])
    assert out["ok"] >= 3


def test_read_snapshot_never_needs_a_compute_and_marks_stale():
    raw = FakeLatest()
    store = SnapshotStore(raw)
    cache = ProcessCache(ttl_sec=60)
    store.put(
        {
            "source": "nobitex",
            "market": "USDTRLS",
            "ohlc_market": "USDTIRT",
            "interval": "1H",
            "asof": ASOF,
            "computed_at": NOW - timedelta(hours=3),
            "indicators": {"rsi_14": 40.0},
            "analyses": {"volatility": {"scalars": {"vol_percentile": 0.9}}},
            "dropped_forming_bar": True,
        }
    )
    payload = read_snapshot(
        "nobitex", "USDTRLS", "1H", store=store, cache=cache, now=NOW
    )
    assert payload["stale"] is True
    assert payload["indicators"]["rsi_14"] == 40.0
    via_irt = read_snapshot(
        "nobitex", "USDTIRT", "1H", store=store, cache=cache, now=NOW
    )
    assert via_irt["market"] == "USDTRLS"
    reads_after_first = raw.reads
    again = read_snapshot(
        "nobitex", "USDTRLS", "1H", store=store, cache=cache, now=NOW
    )
    assert again["stale"] is True
    assert raw.reads == reads_after_first


def test_missing_snapshot_is_none_not_invented():
    store = SnapshotStore(FakeLatest())
    cache = ProcessCache(ttl_sec=60)
    assert read_snapshot("nobitex", "GHOST", "1H", store=store, cache=cache) is None


def test_scheduler_failed_pass_does_not_kill_the_loop():
    n = {"i": 0}

    def job():
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("mongo blip")
        return {"ok": 1}

    sched = SnapshotScheduler(job, poll_sec=3600, initial_delay_sec=0)
    sched.run_once()
    assert sched.last_error and "mongo blip" in sched.last_error
    sched.run_once()
    assert sched.last_error is None
    assert sched.last_result == {"ok": 1}
    assert sched.runs == 2


def test_public_payload_is_json_safe():
    payload = public_payload(
        {
            "source": "nobitex",
            "market": "USDTRLS",
            "ohlc_market": "USDTIRT",
            "interval": "1H",
            "asof": ASOF,
            "computed_at": NOW,
            "indicators": {"rsi_14": 55.0},
            "analyses": {},
            "dropped_forming_bar": False,
        },
        now=NOW,
    )
    assert payload["asof"].endswith("+00:00") or payload["asof"].endswith("Z")
    assert payload["stale"] is False
    assert isinstance(payload["indicators"]["rsi_14"], float)
