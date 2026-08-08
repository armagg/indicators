"""Exchange view: risk flags, the tier-vs-live disagreement logic, rendering,
tier store, and proposal drafting. No live Mongo or config-manager."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from ta.configclient import ConfigManagerClient, MarketConfig
from ta.errors import ParamError
from ta.proposals import ProposalStore, Status
from ta.tiers import DEFAULT_TIER, MAJOR, OTHER, TierStore, normalize_tier
from ta.view import (
    ALERT_MAJOR_NO_CONFIG,
    ALERT_MAJOR_OFF,
    ALERT_RISK_ON,
    MarketRow,
    MarketStats,
    assess,
    build_view,
    classify,
    collect,
    draft_toggle,
    load_stats,
    main,
    market_stats,
    render,
)
from tests.conftest import make_ohlc
from tests.test_apply import FakeTransport
from tests.test_proposals import FakeCollection

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def cfg(market: str, *, enabled=True, version=12, **settings) -> MarketConfig:
    base = {
        "is_enable": enabled,
        "max_base": 100.0,
        "min_base": 10.0,
        "price_min_spread": 0.004,
        "price_max_spread": 0.012,
    }
    base.update(settings)
    return MarketConfig(source="nobitex", market=market, version=version, config=base)


def frame_ending(end: datetime, periods=400, freq="1h", **overrides) -> pd.DataFrame:
    df = make_ohlc(periods)
    df.index = pd.date_range(end=end, periods=periods, freq=freq, tz="UTC")
    for column, value in overrides.items():
        df[column] = value
    return df


# ------------------------------------------------------------------- stats


def test_healthy_market_earns_no_flags():
    stats = market_stats(frame_ending(NOW), now=NOW)
    assert stats.bars == 400
    assert stats.flags == ()
    assert stats.natr is not None and stats.vol_percentile is not None
    assert stats.median_volume > 0


def test_stale_market_is_flagged():
    stats = market_stats(frame_ending(NOW - timedelta(days=3)), now=NOW)
    assert "stale" in stats.flags
    assert stats.hours_stale == pytest.approx(72.0, abs=1.0)


def test_thin_history_is_flagged():
    stats = market_stats(frame_ending(NOW, periods=50), now=NOW)
    assert "thin-history" in stats.flags


def test_sparse_series_is_flagged():
    """Irregular holes -- bars where the market simply did not trade."""
    df = frame_ending(NOW, periods=400)
    keep = np.random.default_rng(5).random(len(df)) > 0.35
    keep[0] = keep[-1] = True
    stats = market_stats(df[keep], now=NOW)

    assert "sparse" in stats.flags
    assert stats.gap_ratio > 0.2


def test_a_regular_coarser_grid_is_not_sparse():
    """Every third bar of an hourly series is a 3-hourly series, not a gappy
    hourly one. Gap detection infers the interval from the median delta, so
    it must not cry sparse over a market that simply trades less often."""
    df = frame_ending(NOW, periods=400).iloc[::3]
    stats = market_stats(df, now=NOW)

    assert stats.gap_ratio == 0.0
    assert "sparse" not in stats.flags


def test_erratic_volume_is_flagged():
    df = frame_ending(NOW)
    spiky = np.full(len(df), 1.0)
    spiky[::50] = 5000.0  # bursts
    df["volume"] = spiky
    stats = market_stats(df, now=NOW)
    assert "erratic-volume" in stats.flags


def test_empty_frame_is_a_finding_not_an_error():
    assert market_stats(pd.DataFrame()).flags == ("no-data",)
    assert load_stats("nobitex", "X", reader=lambda *a, **k: None).flags == ("no-data",)


def test_assess_is_pure_and_stable():
    stats = MarketStats(bars=10, gap_ratio=0.5, hours_stale=100.0, natr=9.0, volume_cv=5.0)
    assert assess(stats) == ("thin-history", "sparse", "stale", "volatile", "erratic-volume")
    assert assess(MarketStats()) == ()


# ------------------------------------------------------- disagreement logic


def test_classify_covers_every_combination():
    assert classify(MAJOR, has_config=True, enabled=True) == ""
    assert classify(MAJOR, has_config=True, enabled=False) == ALERT_MAJOR_OFF
    assert classify(MAJOR, has_config=False, enabled=False) == ALERT_MAJOR_NO_CONFIG
    assert classify(OTHER, has_config=True, enabled=True) == ALERT_RISK_ON
    assert classify(OTHER, has_config=True, enabled=False) == ""
    assert classify(OTHER, has_config=False, enabled=False) == ""


def test_build_view_sorts_disagreements_first():
    configs = [
        cfg("USDTIRT", enabled=True),      # major + on  -> fine
        cfg("SHIBIRT", enabled=True),      # other + on  -> RISK-ON
        cfg("ETHIRT", enabled=False),      # major + off -> MAJOR-OFF
        cfg("DOGEIRT", enabled=False),     # other + off -> fine
    ]
    tiers = {
        ("nobitex", "USDTIRT"): MAJOR,
        ("nobitex", "ETHIRT"): MAJOR,
        ("nobitex", "BTCIRT"): MAJOR,  # tiered major, no config at all
    }
    rows = build_view(configs, tiers, source="nobitex")

    assert [r.market for r in rows][:3] == ["BTCIRT", "ETHIRT", "SHIBIRT"]
    assert [r.alert for r in rows][:3] == [
        ALERT_MAJOR_NO_CONFIG,
        ALERT_MAJOR_OFF,
        ALERT_RISK_ON,
    ]
    # Agreeing rows come last, majors before others.
    assert [(r.market, r.alert) for r in rows][3:] == [("USDTIRT", ""), ("DOGEIRT", "")]


def test_unclassified_markets_default_to_other():
    rows = build_view([cfg("NEWIRT", enabled=False)], {}, source="nobitex")
    assert rows[0].tier == DEFAULT_TIER == OTHER
    assert rows[0].alert == ""  # not major, not live: the safe default state


def test_major_without_config_still_gets_a_row():
    rows = build_view([], {("nobitex", "BTCIRT"): MAJOR}, source="nobitex")
    assert len(rows) == 1
    assert rows[0].has_config is False and rows[0].version is None
    assert rows[0].alert == ALERT_MAJOR_NO_CONFIG


def test_row_reports_the_band_in_basis_points():
    row = build_view([cfg("USDTIRT")], {}, source="nobitex")[0]
    assert row.band_bps == "40-120"
    assert MarketRow("s", "m", OTHER, False, False).band_bps == "-"


# --------------------------------------------------------------- rendering


def test_render_shows_rows_and_calls_out_disagreements():
    configs = [cfg("USDTIRT"), cfg("SHIBIRT", enabled=True, version=7)]
    tiers = {("nobitex", "USDTIRT"): MAJOR}
    stats = {"SHIBIRT": MarketStats(bars=50, gap_ratio=0.4, natr=6.0, flags=("sparse", "volatile"))}
    out = render(build_view(configs, tiers, stats, source="nobitex"))

    assert "MARKET" in out and "ALERT" in out
    assert "USDTIRT" in out and "SHIBIRT" in out
    assert "40-120" in out
    assert ALERT_RISK_ON in out
    assert "1 disagreement" in out
    assert "not a major market but is live" in out
    assert "sparse,volatile" in out


def test_render_says_so_when_everything_agrees():
    out = render(build_view([cfg("USDTIRT")], {("nobitex", "USDTIRT"): MAJOR}, source="nobitex"))
    assert "tier list and live config agree" in out


def test_render_handles_an_empty_exchange():
    assert "no markets" in render([])


# --------------------------------------------------------------- tier store


@pytest.fixture
def tier_store():
    class TierCollection(FakeCollection):
        def replace_one(self, filt, doc, upsert=False):
            key = f"{filt['source']}|{filt['market']}"
            self.docs[key] = dict(doc)

        def find_one(self, filt):
            return self.docs.get(f"{filt['source']}|{filt['market']}")

        def delete_one(self, filt):
            key = f"{filt['source']}|{filt['market']}"
            existed = key in self.docs
            self.docs.pop(key, None)
            return type("R", (), {"deleted_count": 1 if existed else 0})()

    store = TierStore(TierCollection())
    store.ensure_indexes()
    return store


def test_tier_store_roundtrip(tier_store):
    assert tier_store.get("nobitex", "USDTIRT") == DEFAULT_TIER  # unknown = other

    tier_store.set("nobitex", "USDTIRT", "MAJOR", by="omid", note="core pair")
    assert tier_store.get("nobitex", "USDTIRT") == MAJOR
    assert tier_store.mapping("nobitex") == {("nobitex", "USDTIRT"): MAJOR}

    assert tier_store.clear("nobitex", "USDTIRT") is True
    assert tier_store.get("nobitex", "USDTIRT") == DEFAULT_TIER
    assert tier_store.clear("nobitex", "USDTIRT") is False


def test_tier_store_requires_identity_and_valid_tier(tier_store):
    with pytest.raises(ParamError, match="identity"):
        tier_store.set("nobitex", "USDTIRT", MAJOR, by="")
    with pytest.raises(ParamError, match="unknown tier"):
        tier_store.set("nobitex", "USDTIRT", "vip", by="omid")
    assert normalize_tier(" Major ") == MAJOR


# ---------------------------------------------------------------- drafting


def test_draft_enable_lands_pending_with_evidence():
    row = build_view([cfg("SHIBIRT", enabled=False, version=7)], {}, source="nobitex")[0]
    proposal = draft_toggle(row, enable=True, max_base=5.0)

    assert proposal.status == Status.PENDING
    assert proposal.delta.is_enable is True
    assert proposal.delta.max_base == 5.0
    assert proposal.base_version == 7
    assert proposal.is_high_impact
    assert proposal.evidence[0]["name"] == "market_view"
    assert proposal.evidence[0]["scalars"]["live_enabled"] is False


def test_draft_refuses_redundant_or_impossible_toggles():
    enabled_row = build_view([cfg("USDTIRT", enabled=True)], {}, source="nobitex")[0]
    with pytest.raises(ParamError, match="already enabled"):
        draft_toggle(enabled_row, enable=True)

    disabled_row = build_view([cfg("USDTIRT", enabled=False)], {}, source="nobitex")[0]
    with pytest.raises(ParamError, match="already disabled"):
        draft_toggle(disabled_row, enable=False)

    ghost = build_view([], {("nobitex", "BTCIRT"): MAJOR}, source="nobitex")[0]
    with pytest.raises(ParamError, match="cannot create one"):
        draft_toggle(ghost, enable=True)


# --------------------------------------------------------------------- CLI


@pytest.fixture
def client():
    return ConfigManagerClient(base_url="http://cfg:8000", transport=FakeTransport())


@pytest.fixture
def proposal_store():
    s = ProposalStore(proposals=FakeCollection(), decisions=FakeCollection())
    s.ensure_indexes()
    return s


def fake_reader(market, source, interval, limit=None, **kwargs):
    return frame_ending(NOW)


def test_collect_joins_all_three_sources(client, tier_store):
    tier_store.set("nobitex", "USDTIRT", MAJOR, by="omid")
    rows = collect(
        "nobitex", client=client, tier_store=tier_store, reader=fake_reader, now=NOW
    )
    assert [r.market for r in rows] == ["USDTIRT"]
    assert rows[0].tier == MAJOR and rows[0].enabled is True
    assert rows[0].stats.bars == 400


def test_cli_bare_options_default_to_show(client, tier_store, capsys):
    assert main(["--source", "nobitex"], client=client, tier_store=tier_store,
                reader=fake_reader) == 0
    assert "USDTIRT" in capsys.readouterr().out


def test_cli_json_output(client, tier_store, capsys):
    assert main(["show", "-s", "nobitex", "--json"], client=client,
                tier_store=tier_store, reader=fake_reader) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["market"] == "USDTIRT"
    assert payload[0]["band_bps"] == "40-120"
    assert "flags" in payload[0]["stats"]


def test_cli_no_stats_skips_the_candle_read(client, tier_store, capsys):
    def explode(*a, **k):
        raise AssertionError("--no-stats must not read candles")

    assert main(["show", "-s", "nobitex", "--no-stats"], client=client,
                tier_store=tier_store, reader=explode) == 0
    assert "USDTIRT" in capsys.readouterr().out


def test_cli_tier_commands(tier_store, capsys):
    assert main(["tier", "list"], tier_store=tier_store) == 0
    assert "empty" in capsys.readouterr().out

    assert main(["tier", "set", "nobitex", "USDTIRT", "major", "--by", "omid"],
                tier_store=tier_store) == 0
    assert "major" in capsys.readouterr().out

    assert main(["tier", "list"], tier_store=tier_store) == 0
    assert "USDTIRT" in capsys.readouterr().out

    assert main(["tier", "clear", "nobitex", "USDTIRT"], tier_store=tier_store) == 0
    assert "cleared" in capsys.readouterr().out


def test_cli_propose_creates_a_pending_proposal(client, tier_store, proposal_store, capsys):
    client.transport.responses = {"/nobitex/": (200, [
        {"source": "nobitex", "market_name": "SHIBIRT", "version": 7,
         "config": {"is_enable": False, "max_base": 50.0, "min_base": 1.0,
                    "price_min_spread": 0.005, "price_max_spread": 0.02}},
    ])}

    code = main(
        ["propose", "enable", "nobitex", "SHIBIRT", "--max-base", "5"],
        client=client, tier_store=tier_store, proposal_store=proposal_store,
        reader=fake_reader,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "is_enable=True" in out and "max_base=5.0" in out
    assert "nothing is live until it is approved" in out

    stored = proposal_store.find()
    assert len(stored) == 1 and stored[0].status == Status.PENDING


def test_cli_propose_reports_unknown_market(client, tier_store, proposal_store, capsys):
    code = main(["propose", "enable", "nobitex", "NOPEIRT"], client=client,
                tier_store=tier_store, proposal_store=proposal_store, reader=fake_reader)
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_cli_propose_surfaces_a_redundant_toggle(client, tier_store, proposal_store, capsys):
    code = main(["propose", "enable", "nobitex", "USDTIRT"], client=client,
                tier_store=tier_store, proposal_store=proposal_store, reader=fake_reader)
    assert code == 1
    assert "already enabled" in capsys.readouterr().err
