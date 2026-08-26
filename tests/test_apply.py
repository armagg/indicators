"""The config-manager client and the gated apply.

No sockets: a fake transport stands in for HTTP, so the real
`ConfigManagerClient` (URL building, wrapper unwrapping, error mapping) is
under test rather than mocked away.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from ta.configclient import SERVER_OWNED, ConfigManagerClient, MarketConfig
from ta.errors import ParamError, SourceError
from ta.proposals import (
    Proposal,
    ProposalStore,
    Status,
    approve,
    create_proposal,
    reject,
)
from ta.proposals import __main__ as cli
from ta.proposals.apply import ALLOWLIST, apply, diff_configs
from ta.signals import ConfigDelta
from tests.test_proposals import FakeCollection

#: A realistic inner config, including the bookkeeping config-manager injects.
SETTINGS = {
    "market_id": "42",
    "market_name": "USDTIRT",
    "source": "nobitex",
    "is_enable": True,
    "min_base": 10.0,
    "max_base": 100.0,
    "price_min_spread": 0.004,
    "price_max_spread": 0.012,
    "number_of_order_each_sides": 3,
    "maker_fee": 0.0015,
    "spread_multiplier": 1.0,
    "base": "USDT",
    "quote": "IRT",
    "tolerant": 0.0001,
    "version": 12,
    "timestamp": "2026-07-30T10:00:00",
}


def wrapper(version: int = 12, **overrides) -> dict:
    settings = {**SETTINGS, **overrides, "version": version}
    return {
        "source": "nobitex",
        "market_name": "USDTIRT",
        "version": version,
        "timestamp": "2026-07-30T10:00:00",
        "config": settings,
    }


class FakeTransport:
    """Stands in for config-manager, routing by URL shape the way the real
    service does: a bare market path returns one wrapper, a trailing slash
    returns the exchange list, `/versions` returns history, and a PUT echoes
    back the new wrapper with a bumped version.

    `responses` overrides any of that by URL suffix; the longest matching
    suffix wins, so a rule for `/USDTIRT/versions` beats one for `/USDTIRT`.
    """

    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls: list[tuple[str, str, object]] = []
        self.version = 12
        self.next_version = 13

    def request(self, method, url, body=None):
        self.calls.append((method, url, body))

        for suffix in sorted(self.responses, key=len, reverse=True):
            if url.endswith(suffix):
                response = self.responses[suffix]
                return response(method, url, body) if callable(response) else response

        if method == "PUT":
            return 200, wrapper(version=self.next_version, **dict(body or {}))
        if url.endswith("/versions"):
            return 200, [wrapper(version=v) for v in range(self.version, 0, -1)]
        if url.endswith("/"):
            return 200, [wrapper(version=self.version)]
        return 200, wrapper(version=self.version)

    @property
    def writes(self):
        return [c for c in self.calls if c[0] == "PUT"]


@pytest.fixture
def client():
    return ConfigManagerClient(base_url="http://cfg:8000", transport=FakeTransport())


@pytest.fixture
def store():
    s = ProposalStore(proposals=FakeCollection(), decisions=FakeCollection())
    s.ensure_indexes()
    return s


def make_approved(store=None, delta=None, base_version=12) -> Proposal:
    delta = delta or ConfigDelta(
        price_max_spread=0.024, reason="band_widen: vol_percentile=0.96"
    )
    proposal = create_proposal(
        "nobitex", "USDTIRT", delta, base_version=base_version
    )
    if store is not None:
        proposal = store.save_new(proposal).proposal
        approved = approve(proposal, by="omid")
        return store.commit(proposal, approved, action="approve", actor="omid")
    return approve(proposal, by="omid")


# ------------------------------------------------------------------ client


def test_get_market_unwraps_and_strips_server_keys(client):
    got = client.get_market("nobitex", "USDTIRT")

    assert isinstance(got, MarketConfig)
    assert got.version == 12
    assert got.market == "USDTIRT"
    assert got.is_enabled is True
    # The inner config is what we get, not the wrapper.
    assert got.config["price_max_spread"] == 0.012
    assert "config" not in got.config
    # Server bookkeeping never leaks into the settings we reason about.
    assert not (SERVER_OWNED & set(got.config))


def test_url_building(client):
    client.get_market("nobitex", "USDTIRT")
    client.list_markets("nobitex")
    client.versions("nobitex", "USDTIRT")
    urls = [call[1] for call in client.transport.calls]
    assert urls == [
        "http://cfg:8000/config/market/nobitex/USDTIRT",
        "http://cfg:8000/config/market/nobitex/",  # trailing slash: the list route
        "http://cfg:8000/config/market/nobitex/USDTIRT/versions",
    ]


def test_missing_market_is_none_not_an_error():
    transport = FakeTransport(
        {
            "/USDTIRT": (404, {"detail": "not found"}),
            "/USDTIRT/versions": (404, None),
        }
    )
    client = ConfigManagerClient(base_url="http://cfg:8000", transport=transport)
    assert client.get_market("nobitex", "USDTIRT") is None
    assert client.versions("nobitex", "USDTIRT") == []


def test_list_markets_returns_every_active_config(client):
    markets = client.list_markets("nobitex")
    assert [m.market for m in markets] == ["USDTIRT"]
    assert markets[0].version == 12
    assert markets[0].is_enabled is True


def test_server_errors_surface_as_source_errors():
    transport = FakeTransport({"USDTIRT": (500, {"detail": "boom"})})
    client = ConfigManagerClient(base_url="http://cfg:8000", transport=transport)
    with pytest.raises(SourceError, match="500"):
        client.get_market("nobitex", "USDTIRT")


def test_put_sends_the_inner_config_without_server_keys(client):
    client.put_market(
        "nobitex",
        "USDTIRT",
        {**SETTINGS, "price_max_spread": 0.02, "disable_buy": True},
    )
    method, url, body = client.transport.writes[0]

    assert method == "PUT"
    assert url == "http://cfg:8000/config/market/nobitex/USDTIRT"
    assert body["price_max_spread"] == 0.02
    assert body["market_id"] == "42"  # whole document, not a patch
    assert not (SERVER_OWNED & set(body))  # version/timestamp never sent back
    assert "disable_buy" not in body


# -------------------------------------------------------------------- diff


def test_diff_reports_changes_additions_and_drops():
    assert diff_configs({"a": 1}, {"a": 1}) == {}
    assert diff_configs({"a": 1}, {"a": 2}) == {"a": (1, 2)}
    assert diff_configs({"a": 1}, {}) == {"a": (1, None)}
    assert diff_configs({}, {"b": 3}) == {"b": (None, 3)}


def test_allowlist_matches_the_delta_shape():
    assert "price_max_spread" in ALLOWLIST and "max_base" in ALLOWLIST
    assert "reason" not in ALLOWLIST
    assert "market_id" not in ALLOWLIST  # cannot be changed by any proposal


# ------------------------------------------------------------------- apply


def test_unapproved_proposals_are_refused(client, store):
    pending = store.save_new(
        create_proposal("nobitex", "USDTIRT", ConfigDelta(max_base=50.0, reason="r"),
                        base_version=12)
    ).proposal
    with pytest.raises(ParamError, match="not approved"):
        apply(pending, client=client, store=store, dry_run=False)

    rejected = reject(pending, by="omid")
    with pytest.raises(ParamError, match="not approved"):
        apply(rejected, client=client, store=store, dry_run=False)

    # A forged approval with no approver on it is refused too.
    forged = Proposal(
        id="f", source="nobitex", market="USDTIRT",
        delta=ConfigDelta(max_base=50.0, reason="r"), base_version=12,
        status=Status.APPROVED, decided_by=None,
    )
    with pytest.raises(ParamError, match="names no approver"):
        apply(forged, client=client, store=store, dry_run=False)

    assert client.transport.writes == []  # nothing reached the wire


def test_dry_run_is_the_default_and_writes_nothing(client, store):
    approved = make_approved(store)
    result = apply(approved, client=client, store=store)  # no dry_run argument

    assert result.dry_run is True
    assert result.applied is False
    assert result.diff == {"price_max_spread": (0.012, 0.024)}
    assert client.transport.writes == []
    # A dry run is a question, not a step: status is untouched.
    assert store.get(approved.id).status == Status.APPROVED
    assert "DRY RUN" in result.render()


def test_successful_apply_records_both_versions(client, store):
    approved = make_approved(store)
    result = apply(approved, client=client, store=store, dry_run=False, actor="omid")

    assert result.applied is True
    assert result.base_version == 12
    assert result.result_version == 13
    assert len(client.transport.writes) == 1

    body = client.transport.writes[0][2]
    assert body["price_max_spread"] == 0.024
    assert body["min_base"] == 10.0  # untouched fields still sent, whole-doc PUT

    stored = store.get(approved.id)
    assert stored.status == Status.APPLIED
    assert stored.result_version == 13
    assert stored.decided_by == "omid"
    assert [row["action"] for row in store.history(approved.id)][-1] == "apply"


def test_stale_config_blocks_the_write(client, store):
    approved = make_approved(store, base_version=12)
    # A calibrator wrote between approval and apply.
    client.transport.version = 15

    result = apply(approved, client=client, store=store, dry_run=False, actor="omid")

    assert result.applied is False
    assert result.proposal.status == Status.STALE
    assert "must be reviewed again" in result.note
    assert client.transport.writes == []
    assert store.get(approved.id).status == Status.STALE


def test_transport_failure_leaves_the_proposal_failed(store):
    def explode(method, url, body):
        if method == "PUT":
            raise SourceError("connection refused")
        return 200, wrapper()

    transport = FakeTransport({"USDTIRT": explode})
    client = ConfigManagerClient(base_url="http://cfg:8000", transport=transport)
    approved = make_approved(store)

    result = apply(approved, client=client, store=store, dry_run=False, actor="omid")

    assert result.applied is False
    assert result.error == "connection refused"
    stored = store.get(approved.id)
    assert stored.status == Status.FAILED
    assert stored.error == "connection refused"
    assert "ERROR" in result.render()


def test_apply_refuses_when_composition_touches_a_non_allowlisted_field(client, store, monkeypatch):
    """The guard against a whole-document PUT silently resetting a field."""
    approved = make_approved(store)

    def sneaky(self, cfg):
        out = dict(cfg)
        out["price_max_spread"] = 0.024
        out.pop("market_id")  # a field dropped between read and write
        return out

    monkeypatch.setattr(ConfigDelta, "apply_to", sneaky)
    with pytest.raises(ParamError, match="non-allowlisted field"):
        apply(approved, client=client, store=store, dry_run=False)
    assert client.transport.writes == []


def test_missing_market_cannot_be_created(store):
    transport = FakeTransport({"USDTIRT": (404, None)})
    client = ConfigManagerClient(base_url="http://cfg:8000", transport=transport)
    with pytest.raises(ParamError, match="cannot create one"):
        apply(make_approved(store), client=client, store=store, dry_run=False)


def test_already_satisfied_proposal_closes_without_writing(client, store):
    # Propose exactly what the live config already says.
    approved = make_approved(
        store, delta=ConfigDelta(price_max_spread=0.012, reason="already true")
    )
    result = apply(approved, client=client, store=store, dry_run=False, actor="omid")

    assert result.is_noop
    assert client.transport.writes == []
    assert result.applied is True  # the desired end state holds
    assert store.get(approved.id).status == Status.APPLIED
    assert "already matches" in result.note


# --------------------------------------------------------------------- CLI


def test_cli_apply_defaults_to_dry_run(client, store, capsys):
    approved = make_approved(store)
    assert cli.main(["apply", approved.id], store=store, client=client) == 0

    out = capsys.readouterr().out
    assert "would change" in out and "DRY RUN" in out
    assert "0.012" in out and "0.024" in out
    assert client.transport.writes == []
    assert store.get(approved.id).status == Status.APPROVED


def test_cli_write_requires_an_identity(client, store, capsys):
    approved = make_approved(store)
    assert cli.main(["apply", approved.id, "--write"], store=store, client=client) == 1
    assert "requires --by" in capsys.readouterr().err
    assert client.transport.writes == []


def test_cli_write_applies(client, store, capsys):
    approved = make_approved(store)
    code = cli.main(
        ["apply", approved.id, "--write", "--by", "omid"], store=store, client=client
    )
    assert code == 0
    assert "new version: 13" in capsys.readouterr().out
    assert len(client.transport.writes) == 1
    assert store.get(approved.id).status == Status.APPLIED
