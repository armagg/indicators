"""The approval state machine. Pure -- no I/O, no clock it does not own.

The whole safety argument of this project reduces to one property, and it
lives here:

    a proposal can reach APPLIED only from APPROVED,
    and APPROVED can only be entered by naming an approver.

Everything else -- the store, the CLI, the config-manager client -- routes
its state changes through these functions, so there is exactly one place to
audit that claim and exactly one place a bug could break it.

The transition table:

    PENDING  -> APPROVED | REJECTED | STALE
    APPROVED -> APPLIED  | FAILED   | STALE
    REJECTED, APPLIED, STALE, FAILED -> (terminal)

`FAILED` is terminal on purpose. A failed apply means an approved change did
not land, and the honest response is to look at why and decide again, not to
let a retry loop keep firing at config-manager with a stale approval behind
it. Re-running the insight job produces a fresh proposal, and if the config
moved in the meantime that proposal will carry a new `base_version` -- which
is exactly the review that should happen.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from ..errors import ParamError
from .models import Proposal, Status, TERMINAL

#: Legal successor states. Absent or empty means terminal.
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
    """Record a human approval. This is the gate.

    `by` is mandatory and must be a real string: an approval nobody's name
    is on is not an approval, and the whole audit trail rests on it.
    """
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
    """Record that the change reached config-manager.

    Belt and braces: the transition table already forbids reaching APPLIED
    from anywhere but APPROVED, and this additionally refuses a proposal
    that claims APPROVED without an approver on it -- so a hand-built or
    hand-edited document cannot smuggle itself through.
    """
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
    """Return the proposal, marked stale if the config has moved under it.

    Called before approving and again immediately before applying. The
    second call is the one that matters: approval and apply are separated by
    human time, which is exactly the window a calibrator writes in.
    """
    if proposal.is_terminal or current_version == proposal.base_version:
        return proposal
    return mark_stale(proposal, observed_version=current_version)
