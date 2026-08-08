"""Shadow mode: the tilt analysis, target mapping, observation, and scoring.
No Mongo, no config-manager, no network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

import ta.research.shadow as shadow_module
from ta.analysis import run_analysis
from ta.analysis.tilt import DEFAULT_CAP, implied_target_base
from ta.analysis.tilt import run as run_tilt
from ta.errors import SourceError
from ta.research.scheduler import ShadowScheduler
from ta.research.shadow import REQUIRED_DAYS, ShadowStore, observe, observe_all, score
from tests.conftest import make_ohlc

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------- the tilt


def test_tilt_is_registered_as_an_analysis(ohlc):
    result = run_analysis("tilt", ohlc)
    assert result.name == "tilt"
    assert -1.0 <= result.scalars["tilt"] <= 1.0


def _turning(direction: float, flat: int = 400, turn: int = 40) -> pd.DataFrame:
    steps = np.concatenate([np.full(flat, 0.0), np.full(turn, direction * 0.004)])
    index = pd.date_range("2026-01-01", periods=flat + turn, freq="1h", tz="UTC")
    return pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(steps))}, index=index)


def test_tilt_is_relative_to_the_markets_own_normal():
    """A market that has risen at a steady rate forever is *normal*, and the
    tilt is near neutral — "hold more than usual" has to mean more than this
    market's own usual, or the tilt would just re-express the drift the
    covariance scoring deliberately excludes."""
    steady = pd.DataFrame(
        {"close": 100.0 * np.exp(np.cumsum(np.full(440, 0.002)))},
        index=pd.date_range("2026-01-01", periods=440, freq="1h", tz="UTC"),
    )
    assert abs(run_tilt(steady).scalars["tilt"]) < 0.5


def test_tilt_responds_to_a_turn_in_either_direction():
    up = run_tilt(_turning(+1.0)).scalars
    down = run_tilt(_turning(-1.0)).scalars

    assert up["tilt"] > 0.5 and up["direction"] == "hold more"
    assert down["tilt"] < -0.5 and down["direction"] == "hold less"
    assert up["ma_spread_bps"] > 0 > down["ma_spread_bps"]


def test_tilt_is_bounded_and_warns_on_short_history(ohlc):
    short = ohlc.iloc[:120]
    result = run_tilt(short)
    assert result.scalars["tilt"] is None or -1.0 <= result.scalars["tilt"] <= 1.0
    assert any("noisy" in w or "too short" in w for w in result.warnings)

    with pytest.raises(ValueError, match="must be shorter"):
        run_tilt(ohlc, fast=30, slow=20)


# ------------------------------------------------------- target mapping


def test_zero_tilt_reproduces_todays_behaviour():
    """A tilt of zero must land on the band midpoint, which is what the
    market maker uses when target_base is unset."""
    assert implied_target_base(0.0, 1000.0, 5000.0) == 3000.0


def test_cap_limits_how_far_the_target_moves():
    # half-band 2000, cap 0.30 -> at most 600 either side of 3000.
    assert implied_target_base(1.0, 1000.0, 5000.0, cap=0.30) == pytest.approx(3600.0)
    assert implied_target_base(-1.0, 1000.0, 5000.0, cap=0.30) == pytest.approx(2400.0)
    assert implied_target_base(0.5, 1000.0, 5000.0, cap=0.30) == pytest.approx(3300.0)
    # The first live month runs at half the standing cap.
    assert implied_target_base(1.0, 1000.0, 5000.0, cap=0.15) == pytest.approx(3300.0)


def test_target_never_leaves_the_band_even_at_an_absurd_cap():
    assert implied_target_base(1.0, 1000.0, 5000.0, cap=5.0) == 5000.0
    assert implied_target_base(-1.0, 1000.0, 5000.0, cap=5.0) == 1000.0
    assert implied_target_base(0.0, None, None) is None
    assert implied_target_base(0.0, 5000.0, 1000.0) is None  # inverted band


# --------------------------------------------------------------- observe


class FakeShadowCollection:
    def __init__(self):
        self.docs: dict = {}
        self.indexes: list = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def replace_one(self, filt, doc, upsert=False):
        self.docs[(filt["market"], filt["source"], filt["bar_time"])] = dict(doc)

    def find(self, query=None, projection=None):
        query = query or {}
        rows = [d for d in self.docs.values()
                if all(d.get(k) == v for k, v in query.items())]
        return _Cursor(rows)

    def count_documents(self, query):
        return len(self.docs)


class _Cursor:
    def __init__(self, rows):
        self.rows = list(rows)

    def sort(self, key, direction=1):
        self.rows.sort(key=lambda d: d.get(key) or 0, reverse=direction < 0)
        return self

    def limit(self, n):
        self.rows = self.rows[:n]
        return self

    def __iter__(self):
        return iter(self.rows)


class FakeConfigClient:
    def __init__(self, config=None, version=41):
        self.config = config
        self.version = version

    def get_market(self, source, market):
        if self.config is None:
            return None
        return type("MC", (), {"config": self.config, "version": self.version})()


@pytest.fixture
def store():
    s = ShadowStore(FakeShadowCollection())
    s.ensure_indexes()
    return s


@pytest.fixture
def candles(monkeypatch):
    frame = make_ohlc(600)
    frame.index = pd.date_range(end=NOW - timedelta(hours=1), periods=600, freq="1h", tz="UTC")
    monkeypatch.setattr(shadow_module, "read_ohlc", lambda *a, **k: frame)
    return frame


def test_observe_records_the_tilt_and_the_live_config(store, candles):
    client = FakeConfigClient({"min_base": 1000.0, "max_base": 5000.0,
                               "target_base": None, "is_enable": True})
    row = observe("USDTIRT", "nobitex", store=store, config_client=client, now=NOW)

    assert row["market"] == "USDTIRT" and row["source"] == "nobitex"
    assert -1.0 <= row["tilt"] <= 1.0
    assert 2400.0 <= row["implied_target_base"] <= 3600.0  # inside the cap
    assert row["live_target_base"] is None  # nothing set live yet
    assert row["live_config_version"] == 41
    assert row["live_inventory"] is None  # needs ClickHouse, not available
    assert store.count() == 1


def test_observe_survives_config_manager_being_down(store, candles):
    class Broken:
        def get_market(self, *a):
            raise SourceError("connection refused")

    row = observe("USDTIRT", "nobitex", store=store, config_client=Broken(), now=NOW)
    assert row["tilt"] is not None          # the observation is still recorded
    assert row["implied_target_base"] is None  # but no band to map it onto
    assert row["live_target_base"] is None


def test_observing_the_same_bar_twice_updates_rather_than_duplicates(store, candles):
    client = FakeConfigClient({"min_base": 1000.0, "max_base": 5000.0})
    observe("USDTIRT", "nobitex", store=store, config_client=client, now=NOW)
    observe("USDTIRT", "nobitex", store=store, config_client=client, now=NOW)
    assert store.count() == 1


def test_observe_all_isolates_failures(store, candles, monkeypatch):
    def selective(market, source, interval, **kw):
        if market == "BROKEN":
            return None
        return candles

    monkeypatch.setattr(shadow_module, "read_ohlc", selective)
    out = observe_all(
        [("USDTIRT", "nobitex"), ("BROKEN", "nobitex")], store=store, now=NOW
    )
    assert len(out["recorded"]) == 1
    assert "BROKEN/nobitex" in out["failures"]


# ---------------------------------------------------------------- score


def make_rows(n, *, tilts=None, step=1, start=NOW - timedelta(hours=500)):
    """Rows where `tilt[i]` drives the move from `mid[i]` to `mid[i+1]`.

    The off-by-one here matters: a tilt that moved the *current* bar has
    already happened and predicts nothing, which is exactly the mistake the
    forward-return stamping in `signal_value` exists to prevent.
    """
    rng = np.random.default_rng(3)
    tilts = rng.uniform(-1, 1, n) if tilts is None else tilts
    mids, price = [100.0], 100.0
    for t in tilts[:-1]:
        price *= float(np.exp(0.0005 * t + rng.normal(0, 0.001)))
        mids.append(price)
    return [
        {"market": "USDTIRT", "source": "nobitex",
         "bar_time": start + timedelta(hours=i * step),
         "tilt": float(tilts[i]), "mid": mids[i]}
        for i in range(n)
    ]


def test_score_reports_coverage_before_it_reports_an_estimate():
    out = score(make_rows(2))
    assert out["scored"] == 0 and "not enough" in out["verdict"]

    out = score(make_rows(200))
    assert out["scored"] > 150
    assert out["ready"] is False  # 200 hours is ~8 days, under the 28 required
    assert str(REQUIRED_DAYS) in out["verdict"]
    assert out["backtest_excess_bps"] == 0.343


def test_score_picks_up_a_real_relationship():
    rng = np.random.default_rng(9)
    tilts = rng.uniform(-1, 1, 900)
    out = score(make_rows(900, tilts=tilts))
    assert out["live_excess_bps"] > 0
    assert out["live_t"] > 2


def test_score_refuses_pairs_that_span_a_service_outage():
    """A restart leaves a hole; a forward return measured across it spans
    hours instead of one, which is the variable-horizon trap."""
    rows = make_rows(300)
    del rows[100:140]  # the service was down for 40 hours
    out = score(rows)
    assert out["dropped_gap_pairs"] >= 1
    assert out["scored"] < len(rows) - 1


def test_score_calls_the_proposal_dead_when_below_the_backtest_floor():
    # Tilt anti-correlated with what follows: clearly worse than the floor.
    rng = np.random.default_rng(11)
    tilts = rng.uniform(-1, 1, 800)
    rows = make_rows(800, tilts=tilts)
    for row, t in zip(rows, tilts):
        row["tilt"] = -t  # invert, so the recorded tilt is wrong-footed
    out = score(rows)
    out["span_days"] = 40.0
    assert out["live_excess_bps"] < 0


# ------------------------------------------------------------ scheduler


def test_scheduler_runs_the_job_and_survives_failures():
    calls = {"n": 0}

    def job():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("mongo hiccup")
        return {"recorded": [{"market": "USDTIRT"}], "failures": {}}

    scheduler = ShadowScheduler(job, interval_sec=999, initial_delay_sec=999)

    scheduler.run_once()
    assert "RuntimeError: mongo hiccup" in scheduler.last_error

    scheduler.run_once()  # the loop keeps going after a failure
    assert scheduler.last_error is None
    assert scheduler.last_result["recorded"][0]["market"] == "USDTIRT"
    assert scheduler.runs == 2

    status = scheduler.status()
    assert status["runs"] == 2 and status["interval_sec"] == 999


def test_watchlist_parsing_handles_the_config_name_split(monkeypatch):
    """nobitex's candles are USDTIRT but the config it trades is USDTRLS."""
    monkeypatch.setenv(
        "SHADOW_MARKETS", " USDTIRT:nobitex:USDTRLS , USDTIRR:ramzinex ,bad, "
    )
    assert shadow_module.watchlist_from_env() == [
        ("USDTIRT", "nobitex", "USDTRLS"),
        ("USDTIRR", "ramzinex", "USDTIRR"),
    ]
    monkeypatch.setenv("SHADOW_MARKETS", "")
    assert shadow_module.watchlist_from_env() == []


def test_observe_looks_up_the_config_under_its_own_name(store, candles):
    """Looking up the wrong name yields no config *silently*, which is why
    the alias is explicit rather than guessed."""
    seen = {}

    class Recording(FakeConfigClient):
        def get_market(self, source, market):
            seen["market"] = market
            return super().get_market(source, market)

    client = Recording({"min_base": 1000.0, "max_base": 5000.0})
    row = observe("USDTIRT", "nobitex", config_market="USDTRLS",
                  store=store, config_client=client, now=NOW)

    assert seen["market"] == "USDTRLS"
    assert row["config_market"] == "USDTRLS"
    assert row["implied_target_base"] is not None
