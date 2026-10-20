"""Persistence for proposals (current state) and the append-only decision log."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable

from decouple import config

from ..errors import SourceError
from .models import Proposal, Status

_SUPPRESSING = frozenset({Status.PENDING, Status.APPROVED, Status.REJECTED, Status.APPLIED})

_REPLACEABLE = frozenset({Status.STALE, Status.FAILED})

NATURAL_KEY = ("source", "market", "delta_hash", "base_version")


@dataclass(frozen=True)
class SaveOutcome:
    """What `save_new` did, and why."""

    proposal: Proposal
    created: bool
    note: str = ""


@dataclass(frozen=True)
class ProposalStore:
    """Handles for the two collections. Inject fakes in tests."""

    proposals: Any
    decisions: Any


    def ensure_indexes(self) -> None:
        self.proposals.create_index(
            [(field, 1) for field in NATURAL_KEY], unique=True, name="ta_proposal_key"
        )
        self.proposals.create_index(
            [("status", 1), ("source", 1), ("created_at", -1)], name="ta_proposal_queue"
        )
        self.decisions.create_index(
            [("proposal_id", 1), ("at", 1)], name="ta_decision_trail"
        )


    def get(self, proposal_id: str) -> Proposal | None:
        doc = self.proposals.find_one({"_id": proposal_id})
        return Proposal.from_doc(doc) if doc else None

    def find(
        self,
        *,
        status: str | None = None,
        source: str | None = None,
        market: str | None = None,
        limit: int = 50,
    ) -> list[Proposal]:
        query: dict[str, Any] = {}
        if status:
            query["status"] = str(Status(status))
        if source:
            query["source"] = source
        if market:
            query["market"] = market
        cursor = self.proposals.find(query).sort("created_at", -1).limit(int(limit))
        return [Proposal.from_doc(doc) for doc in cursor]

    def history(self, proposal_id: str) -> list[dict]:
        return list(self.decisions.find({"proposal_id": proposal_id}).sort("at", 1))


    def save_new(self, proposal: Proposal) -> SaveOutcome:
        """Insert a pending proposal, honouring the re-proposal rules above."""
        existing_doc = self.proposals.find_one({"_id": proposal.id})
        if existing_doc:
            existing = Proposal.from_doc(existing_doc)
            if existing.status in _SUPPRESSING:
                return SaveOutcome(
                    existing,
                    created=False,
                    note=f"an identical proposal is already {existing.status}",
                )
            if existing.status in _REPLACEABLE:
                self.proposals.replace_one({"_id": proposal.id}, proposal.to_doc())
                self._record(
                    proposal,
                    action="reproposed",
                    actor="system",
                    note=f"replaced a {existing.status} proposal",
                    from_status=existing.status,
                )
                return SaveOutcome(proposal, created=True, note="replaced a dead proposal")

        self.proposals.insert_one(proposal.to_doc())
        self._record(
            proposal, action="created", actor="system", note="", from_status=None
        )
        return SaveOutcome(proposal, created=True)

    def commit(
        self,
        before: Proposal,
        after: Proposal,
        *,
        action: str,
        actor: str = "system",
        note: str = "",
    ) -> Proposal:
        """Persist a state change produced by `lifecycle`, and log it."""
        self.proposals.replace_one({"_id": after.id}, after.to_doc())
        self._record(
            after, action=action, actor=actor, note=note, from_status=before.status
        )
        return after

    def _record(
        self,
        proposal: Proposal,
        *,
        action: str,
        actor: str,
        note: str,
        from_status: Status | None,
    ) -> None:
        self.decisions.insert_one(
            {
                "proposal_id": proposal.id,
                "source": proposal.source,
                "market": proposal.market,
                "action": action,
                "actor": actor,
                "note": note,
                "from_status": str(from_status) if from_status else None,
                "to_status": str(proposal.status),
                "delta": proposal.delta.as_dict(),
                "at": datetime.now(timezone.utc),
            }
        )


_default: ProposalStore | None = None


def default_store() -> ProposalStore:
    """The env-configured store, built lazily so importing never opens a connection."""
    global _default
    if _default is None:
        try:
            from pymongo import MongoClient

            client = MongoClient(config("MONGO_URI"), serverSelectionTimeoutMS=5000)
            db = client[config("DB_NAME")]
            store = ProposalStore(
                proposals=db[config("PROPOSALS_COLLECTION_NAME", default="ta_proposals")],
                decisions=db[config("DECISIONS_COLLECTION_NAME", default="ta_decisions")],
            )
            store.ensure_indexes()
        except Exception as exc:
            raise SourceError(f"cannot reach the proposal store: {exc}") from exc
        _default = store
    return _default
