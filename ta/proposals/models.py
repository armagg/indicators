"""What a proposal is.

A proposal is this project's only unit of change: a `ConfigDelta` for one
market, the evidence that justified it, and the config version it was
computed against. It carries no authority on its own -- until a human
approves it, it is a suggestion in a queue.

Identity is derived, not random. The id is a digest of
`(source, market, delta_hash, base_version)`, which makes the same
recommendation computed twice land on the same document. An insight job
running hourly on a market that has not changed should not grow a queue of
a hundred identical "widen the band" cards for a human to wade through; it
should keep pointing at the one card already there.

`base_version` is the config-manager version the delta was composed
against. It is what makes a proposal falsifiable later: if the live config
has moved on -- the crypto_price calibrators write the same documents on
their own schedule -- then the delta was reasoned about a world that no
longer exists, and it must not be applied. See `lifecycle.check_version`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable, Mapping

from ..analysis.types import AnalysisResult
from ..errors import ParamError
from ..signals import ConfigDelta
from ..store import param_hash


class Status(str, Enum):
    """Where a proposal sits in its life.

    Only `PENDING` and `APPROVED` are live states; the rest are terminal.
    Re-proposing is cheap, so nothing is ever revived in place -- a new
    situation produces a new proposal with a new `base_version`.

    `str, Enum` rather than `StrEnum` so this runs on Python 3.10, which is
    what the germany-2 research box has. The explicit `__str__` restores
    StrEnum's behaviour -- without it, `str(Status.PENDING)` is
    "Status.PENDING" on 3.10 and "pending" on 3.11+, and the documents this
    writes to Mongo would differ by interpreter.
    """

    __str__ = str.__str__

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    #: The config moved underneath us before this could be applied.
    STALE = "stale"
    #: An approved apply reached config-manager and did not succeed.
    FAILED = "failed"


TERMINAL = frozenset(
    {Status.REJECTED, Status.APPLIED, Status.STALE, Status.FAILED}
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def delta_hash(delta: ConfigDelta) -> str:
    """Stable digest of what a delta *changes*, ignoring its prose.

    `reason` is deliberately excluded: two runs that propose the identical
    numbers are the same proposal even if one phrased its reason with a
    different percentile in it. Including the reason would defeat the
    deduplication this hash exists for.
    """
    return param_hash(delta.fields())


def make_id(source: str, market: str, delta_digest: str, base_version) -> str:
    key = f"{source}|{market}|{delta_digest}|{base_version}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class Proposal:
    """One proposed config change, with its provenance and decision trail."""

    id: str
    source: str
    market: str
    delta: ConfigDelta
    #: config-manager version this delta was composed against. None only
    #: when the market has no config yet (a first-time enable).
    base_version: int | None
    status: Status = Status.PENDING
    #: `AnalysisResult.as_dict()` payloads, series excluded. The "why".
    evidence: tuple[dict, ...] = ()
    created_at: datetime = field(default_factory=_utcnow)
    decided_by: str | None = None
    decided_at: datetime | None = None
    decision_note: str = ""
    applied_at: datetime | None = None
    #: The version config-manager returned after a successful apply.
    result_version: int | None = None
    error: str = ""

    @property
    def delta_digest(self) -> str:
        return delta_hash(self.delta)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL

    @property
    def is_high_impact(self) -> bool:
        return self.delta.is_high_impact

    def natural_key(self) -> dict:
        return {
            "source": self.source,
            "market": self.market,
            "delta_hash": self.delta_digest,
            "base_version": self.base_version,
        }

    def summary(self) -> str:
        """One-line description of what would change."""
        fields = self.delta.fields()
        return ", ".join(f"{k}={v}" for k, v in sorted(fields.items()))

    def to_doc(self) -> dict:
        """Mongo document. `_id` is the derived id, so the natural key is
        enforced by the primary key as well as by its own index."""
        return {
            "_id": self.id,
            "source": self.source,
            "market": self.market,
            "delta": self.delta.as_dict(),
            "delta_hash": self.delta_digest,
            "base_version": self.base_version,
            "status": str(self.status),
            "high_impact": self.is_high_impact,
            "evidence": [dict(e) for e in self.evidence],
            "created_at": self.created_at,
            "decided_by": self.decided_by,
            "decided_at": self.decided_at,
            "decision_note": self.decision_note,
            "applied_at": self.applied_at,
            "result_version": self.result_version,
            "error": self.error,
        }

    @classmethod
    def from_doc(cls, doc: Mapping[str, Any]) -> "Proposal":
        delta_fields = dict(doc.get("delta") or {})
        return cls(
            id=doc["_id"],
            source=doc["source"],
            market=doc["market"],
            delta=ConfigDelta(**delta_fields),
            base_version=doc.get("base_version"),
            status=Status(doc.get("status", Status.PENDING)),
            evidence=tuple(doc.get("evidence") or ()),
            created_at=doc.get("created_at") or _utcnow(),
            decided_by=doc.get("decided_by"),
            decided_at=doc.get("decided_at"),
            decision_note=doc.get("decision_note") or "",
            applied_at=doc.get("applied_at"),
            result_version=doc.get("result_version"),
            error=doc.get("error") or "",
        )


def create_proposal(
    source: str,
    market: str,
    delta: ConfigDelta,
    *,
    base_version: int | None,
    evidence: Iterable[AnalysisResult | Mapping[str, Any]] = (),
    now: datetime | None = None,
) -> Proposal:
    """Build a pending proposal.

    Refuses an empty delta (nothing to decide) and refuses one that explains
    nothing: ARCHITECTURE principle 6 says a proposal must carry the reason
    it exists, because an approver who cannot see why is being asked to
    rubber-stamp. Either analysis evidence or a written `delta.reason`
    satisfies that -- the latter is how a human-drafted proposal (an
    `is_enable` flip, say) justifies itself.
    """
    if not source or not market:
        raise ParamError("a proposal needs both source and market")
    if delta.is_empty:
        raise ParamError("refusing to create a proposal that changes nothing")

    payload = tuple(
        e.as_dict() if isinstance(e, AnalysisResult) else dict(e) for e in evidence
    )
    if not payload and not delta.reason.strip():
        raise ParamError(
            "a proposal must carry its justification: pass analysis evidence, "
            "or set delta.reason explaining the change"
        )

    digest = delta_hash(delta)
    return Proposal(
        id=make_id(source, market, digest, base_version),
        source=source,
        market=market,
        delta=delta,
        base_version=base_version,
        evidence=payload,
        created_at=now or _utcnow(),
    )
