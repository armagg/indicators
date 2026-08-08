"""Proposal lifecycle, store idempotency, and the approval-gate safety
property. Entirely offline -- the store takes injected fake collections."""

from __future__ import annotations

import itertools
from datetime import datetime, timedelta, timezone

import pytest

from ta.analysis.types import AnalysisResult
from ta.errors import ParamError
from ta.proposals import (
    LEGAL_TRANSITIONS,
    Proposal,
    ProposalStore,
    Status,
    TERMINAL,
    approve,
    check_version,
    create_proposal,
    mark_applied,
    mark_failed,
    mark_stale,
    reject,
    transition,
)
from ta.proposals import __main__ as cli
from ta.signals import ConfigDelta

NOW = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

DELTA = ConfigDelta(
    price_max_spread=0.024,
    reason="band_widen: vol_percentile=0.95 >= 0.75, raised price_max_spread",
)


def make_proposal(**kwargs) -> Proposal:
    params = {
        "source": "nobitex",
        "market": "USDTIRT",
        "delta": DELTA,
        "base_version": 12,
        "now": NOW,
    }
    params.update(kwargs)
    return create_proposal(
        params.pop("source"), params.pop("market"), params.pop("delta"), **params
    )


# ------------------------------------------------------------------ models


def test_identity_is_derived_so_recomputation_is_idempotent():
    a = make_proposal()
    b = make_proposal(now=NOW + timedelta(hours=1))
    assert a.id == b.id  # same change, same situation, same card

    # A different situation is a different proposal.
    assert make_proposal(base_version=13).id != a.id
    assert make_proposal(delta=ConfigDelta(price_max_spread=0.03, reason="x")).id != a.id


def test_delta_hash_ignores_prose_but_not_numbers():
    worded = ConfigDelta(max_base=50.0, reason="one phrasing")
    reworded = ConfigDelta(max_base=50.0, reason="a completely different phrasing")
    changed = ConfigDelta(max_base=51.0, reason="one phrasing")

    assert make_proposal(delta=worded).id == make_proposal(delta=reworded).id
    assert make_proposal(delta=changed).id != make_proposal(delta=worded).id


def test_proposal_must_change_something_and_justify_itself():
    with pytest.raises(ParamError, match="changes nothing"):
        make_proposal(delta=ConfigDelta())

    with pytest.raises(ParamError, match="justification"):
        make_proposal(delta=ConfigDelta(max_base=50.0))  # no reason, no evidence

    # Either form of justification is enough.
    make_proposal(delta=ConfigDelta(max_base=50.0, reason="sized down"))
    make_proposal(
        delta=ConfigDelta(max_base=50.0),
        evidence=[AnalysisResult("volatility", {}, None, {"vol_percentile": 0.97})],
    )


def test_evidence_is_stored_as_plain_dicts():
    result = AnalysisResult("volatility", {"window": 20}, None, {"vol_percentile": 0.97})
    proposal = make_proposal(evidence=[result])
    assert proposal.evidence[0]["name"] == "volatility"
    assert proposal.evidence[0]["scalars"]["vol_percentile"] == 0.97
    assert "series" not in proposal.evidence[0]  # bulk excluded


def test_document_round_trip():
    original = make_proposal(
        evidence=[AnalysisResult("volatility", {}, None, {"vol_percentile": 0.97})]
    )
    restored = Proposal.from_doc(original.to_doc())
    assert restored == original
    assert restored.delta == DELTA


def test_high_impact_surfaces_on_the_proposal():
    assert make_proposal(delta=ConfigDelta(max_base=200.0, reason="up")).is_high_impact
    assert not make_proposal().is_high_impact  # widening is not high impact


# --------------------------------------------------------------- lifecycle


def test_transition_table_is_exhaustively_enforced():
    """Every (from, to) pair not in the table must be refused."""
    for current, target in itertools.product(Status, Status):
        proposal = Proposal(
            id="x", source="s", market="m", delta=DELTA, base_version=1, status=current,
            decided_by="someone",
        )
        legal = target in LEGAL_TRANSITIONS[current]
        if legal:
            assert transition(proposal, target).status == target
        else:
            with pytest.raises(ParamError):
                transition(proposal, target)


def test_terminal_states_are_terminal():
    for status in TERMINAL:
        assert LEGAL_TRANSITIONS[status] == frozenset()


def test_approval_requires_an_identity():
    proposal = make_proposal()
    for bad in ("", "   ", None):
        with pytest.raises(ParamError, match="identity|accountable"):
            approve(proposal, by=bad)

    approved = approve(proposal, by="  omid  ", note="vol is real", now=NOW)
    assert approved.status == Status.APPROVED
    assert approved.decided_by == "omid"  # trimmed
    assert approved.decided_at == NOW
    assert approved.decision_note == "vol is real"


def test_rejection_is_terminal():
    rejected = reject(make_proposal(), by="omid", note="not convinced")
    assert rejected.status == Status.REJECTED
    with pytest.raises(ParamError, match="terminal"):
        approve(rejected, by="omid")


def test_stale_check_blocks_a_moved_config():
    proposal = make_proposal()  # base_version 12
    assert check_version(proposal, 12) is proposal  # unchanged

    stale = check_version(proposal, 13)
    assert stale.status == Status.STALE
    assert "12" in stale.decision_note and "13" in stale.decision_note
    with pytest.raises(ParamError):
        approve(stale, by="omid")

    # Terminal proposals are left alone rather than re-marked.
    applied = mark_applied(approve(make_proposal(), by="omid"), result_version=13)
    assert check_version(applied, 99) is applied


def test_approved_proposal_can_still_go_stale_before_apply():
    """The window between approval and apply is where calibrators write."""
    approved = approve(make_proposal(), by="omid")
    stale = check_version(approved, 14)
    assert stale.status == Status.STALE
    with pytest.raises(ParamError):
        mark_applied(stale, result_version=15)


def test_failed_is_terminal_so_retries_need_a_fresh_decision():
    failed = mark_failed(approve(make_proposal(), by="omid"), "connection refused")
    assert failed.status == Status.FAILED
    assert failed.error == "connection refused"
    with pytest.raises(ParamError, match="terminal"):
        mark_applied(failed, result_version=13)


# ------------------------------------------------- THE safety property


def test_applied_is_reachable_only_through_a_recorded_approval():
    """The single claim this whole package exists to make."""
    # 1. Only APPROVED has APPLIED as a legal successor.
    reachers = [s for s in Status if Status.APPLIED in LEGAL_TRANSITIONS[s]]
    assert reachers == [Status.APPROVED]

    # 2. A pending proposal cannot shortcut.
    with pytest.raises(ParamError):
        mark_applied(make_proposal(), result_version=13)

    # 3. Even a forged "approved" document without an approver is refused.
    forged = Proposal(
        id="forged", source="s", market="m", delta=DELTA, base_version=1,
        status=Status.APPROVED, decided_by=None,
    )
    with pytest.raises(ParamError, match="carries no approver"):
        mark_applied(forged, result_version=13)

    # 4. The honest path works.
    applied = mark_applied(approve(make_proposal(), by="omid"), result_version=13)
    assert applied.status == Status.APPLIED
    assert applied.decided_by == "omid"
    assert applied.result_version == 13


# ------------------------------------------------------------------- store


class FakeCollection:
    def __init__(self):
        self.docs: dict[str, dict] = {}
        self.inserts: list[dict] = []
        self.indexes: list[tuple] = []

    def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    def insert_one(self, doc):
        key = doc.get("_id", len(self.inserts))
        if key in self.docs:
            raise RuntimeError("duplicate key")
        self.docs[key] = dict(doc)
        self.inserts.append(dict(doc))

    def replace_one(self, filt, doc):
        self.docs[filt["_id"]] = dict(doc)

    def find_one(self, filt):
        return self.docs.get(filt["_id"])

    def find(self, query=None):
        query = query or {}
        rows = [
            d
            for d in self.docs.values()
            if all(d.get(k) == v for k, v in query.items())
        ]
        return _FakeCursor(rows)


class _FakeCursor:
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


@pytest.fixture
def store():
    s = ProposalStore(proposals=FakeCollection(), decisions=FakeCollection())
    s.ensure_indexes()
    return s


def test_indexes_declare_the_natural_key(store):
    keys, kwargs = store.proposals.indexes[0]
    assert kwargs["unique"] is True
    assert [k for k, _ in keys] == ["source", "market", "delta_hash", "base_version"]


def test_save_new_inserts_once_and_logs_creation(store):
    outcome = store.save_new(make_proposal())
    assert outcome.created is True
    assert store.get(outcome.proposal.id) == outcome.proposal
    assert store.decisions.inserts[0]["action"] == "created"


def test_recomputing_the_same_proposal_does_not_duplicate(store):
    first = store.save_new(make_proposal())
    second = store.save_new(make_proposal(now=NOW + timedelta(hours=3)))

    assert second.created is False
    assert "already pending" in second.note
    assert len(store.proposals.docs) == 1
    # The original creation timestamp survives -- an old ask must not look new.
    assert second.proposal.created_at == first.proposal.created_at


def test_a_rejected_proposal_is_not_resurrected(store):
    saved = store.save_new(make_proposal()).proposal
    store.commit(saved, reject(saved, by="omid"), action="reject", actor="omid")

    again = store.save_new(make_proposal())
    assert again.created is False
    assert again.proposal.status == Status.REJECTED
    assert "already rejected" in again.note


def test_a_stale_or_failed_proposal_is_replaced_by_a_fresh_one(store):
    for dead in (Status.STALE, Status.FAILED):
        store.proposals.docs.clear()
        saved = store.save_new(make_proposal()).proposal
        approved = approve(saved, by="omid")
        store.commit(saved, approved, action="approve", actor="omid")
        killed = (
            mark_stale(approved, 13)
            if dead == Status.STALE
            else mark_failed(approved, "boom")
        )
        store.commit(approved, killed, action=str(dead), actor="system")

        again = store.save_new(make_proposal())
        assert again.created is True, dead
        assert again.proposal.status == Status.PENDING
        assert store.get(again.proposal.id).status == Status.PENDING


def test_commit_writes_state_and_appends_an_immutable_decision(store):
    saved = store.save_new(make_proposal()).proposal
    approved = approve(saved, by="omid", note="vol is real")
    store.commit(saved, approved, action="approve", actor="omid", note="vol is real")

    assert store.get(saved.id).status == Status.APPROVED
    trail = store.history(saved.id)
    assert [row["action"] for row in trail] == ["created", "approve"]
    assert trail[-1]["from_status"] == str(Status.PENDING)
    assert trail[-1]["to_status"] == str(Status.APPROVED)
    assert trail[-1]["actor"] == "omid"


def test_find_filters_and_orders(store):
    store.save_new(make_proposal())
    other = make_proposal(market="BTCIRT", delta=ConfigDelta(max_base=5.0, reason="r"))
    store.save_new(other)

    assert len(store.find()) == 2
    assert len(store.find(market="BTCIRT")) == 1
    assert len(store.find(status="pending")) == 2
    assert store.find(status="approved") == []


# --------------------------------------------------------------------- CLI


def test_cli_list_show_approve_reject(store, capsys):
    proposal = store.save_new(make_proposal()).proposal

    assert cli.main(["list"], store=store) == 0
    assert proposal.id in capsys.readouterr().out

    assert cli.main(["show", proposal.id], store=store) == 0
    shown = capsys.readouterr().out
    assert "price_max_spread" in shown and "band_widen" in shown

    assert cli.main(["approve", proposal.id, "--by", "omid"], store=store) == 0
    out = capsys.readouterr().out
    assert "approved by omid" in out
    assert "not applied" in out  # approving is not applying
    assert store.get(proposal.id).status == Status.APPROVED


def test_cli_refuses_a_decision_without_an_approver(store):
    store.save_new(make_proposal())
    with pytest.raises(SystemExit):  # argparse: --by is required
        cli.main(["approve", "whatever"], store=store)


def test_cli_reports_unknown_id_and_illegal_transition(store, capsys):
    assert cli.main(["show", "nope"], store=store) == 1
    assert "no proposal" in capsys.readouterr().err

    proposal = store.save_new(make_proposal()).proposal
    store.commit(proposal, reject(proposal, by="omid"), action="reject", actor="omid")
    assert cli.main(["approve", proposal.id, "--by", "omid"], store=store) == 1
    assert "terminal" in capsys.readouterr().err


def test_cli_json_output(store, capsys):
    import json

    store.save_new(make_proposal())
    assert cli.main(["list", "--json"], store=store) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["delta"]["price_max_spread"] == 0.024
