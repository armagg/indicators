"""The approval state machine. Pure -- no I/O, no clock it does not own."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ..errors import ParamError
from .models import Proposal, Status, TERMINAL

LEGAL_TRANSITIONS: dict[Status, frozenset[Status]] = {
    Status.PENDING: frozenset({Status.APPROVED, Status.REJECTED, Status.STALE}),
    Status.APPROVED: frozenset({Status.APPLIED, Status.FAILED, Status.STALE}),
    Status.REJECTED: frozenset(),
    Status.APPLIED: frozenset(),
    Status.STALE: frozenset(),
    Status.FAILED: frozenset(),
}


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def can_transition(current: Status, target: Status) -> bool:
    return target in LEGAL_TRANSITIONS.get(Status(current), frozenset())


def transition(proposal: Proposal, target: Status, **updates: Any) -> Proposal:
    """Move a proposal to `target`, or refuse with a legible error."""
    target = Status(target)
    current = Status(proposal.status)
    if not can_transition(current, target):
        detail = (
            f"{current} is terminal"
            if current in TERMINAL
            else f"legal next states are {', '.join(sorted(LEGAL_TRANSITIONS[current])) or '(none)'}"
        )
        raise ParamError(
            f"proposal {proposal.id}: cannot go {current} -> {target}; {detail}"
        )
    return replace(proposal, status=target, **updates)


def approve(
    proposal: Proposal, by: str, note: str = "", now: datetime | None = None
) -> Proposal:
    """Record a human approval. This is the gate."""
    if not by or not str(by).strip():
        raise ParamError(
            "approval requires an approver identity: who is accountable for this change?"
        )
    return transition(
        proposal,
        Status.APPROVED,
        decided_by=str(by).strip(),
        decided_at=now or _utcnow(),
        decision_note=note,
    )


def reject(
    proposal: Proposal, by: str, note: str = "", now: datetime | None = None
) -> Proposal:
    if not by or not str(by).strip():
        raise ParamError("rejection requires an identity")
    return transition(
        proposal,
        Status.REJECTED,
        decided_by=str(by).strip(),
        decided_at=now or _utcnow(),
        decision_note=note,
    )


def mark_stale(
    proposal: Proposal, observed_version: int | None = None, note: str = ""
) -> Proposal:
    """The live config moved on; this delta was reasoned about a dead world."""
    detail = note or (
        f"config moved from version {proposal.base_version} to {observed_version} "
        f"before this was applied"
    )
    return transition(proposal, Status.STALE, decision_note=detail)


def mark_applied(
    proposal: Proposal, result_version: int | None, now: datetime | None = None
) -> Proposal:
    """Record that the change reached config-manager."""
    if not proposal.decided_by:
        raise ParamError(
            f"proposal {proposal.id} claims status {proposal.status} but carries no "
            f"approver; refusing to mark it applied"
        )
    return transition(
        proposal,
        Status.APPLIED,
        applied_at=now or _utcnow(),
        result_version=result_version,
    )


def mark_failed(proposal: Proposal, error: str) -> Proposal:
    return transition(proposal, Status.FAILED, error=str(error))


def check_version(proposal: Proposal, current_version: int | None) -> Proposal:
    """Return the proposal, marked stale if the config has moved under it."""
    if proposal.is_terminal or current_version == proposal.base_version:
        return proposal
    return mark_stale(proposal, observed_version=current_version)
