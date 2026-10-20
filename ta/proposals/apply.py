"""The gated apply: the only path in this project that changes a live config.

Everything else in `ta` reads, computes, or proposes. This module is the
single door, and it is deliberately narrow:

1. Refuse anything that is not `APPROVED`. Not pending, not "obviously
   fine", not high-urgency. The state machine in `lifecycle.py` is the
   authority and this re-checks it rather than trusting the caller.
2. Re-read the live config **now**, not when the proposal was made. Approval
   and apply are separated by human time, and the crypto_price calibrators
   write these same documents on their own schedule.
3. If the version moved since `base_version`, mark the proposal `stale` and
   stop. The delta was reasoned against a config that no longer exists;
   applying it would silently revert whatever the calibrator just decided.
4. Compose onto the freshly-read config, then verify that **only allowlisted
   fields changed**. This is the guard against a subtler failure than a bad
   delta: `PUT` replaces the whole document, so a field accidentally dropped
   between read and write would be silently reset to its schema default. The
   diff is computed and checked precisely so that cannot happen unnoticed.
5. Write, and record both the version we based on and the version we got.

Dry run is the default everywhere, including the CLI. Writing takes an
explicit flag.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ..configclient import ConfigManagerClient, MarketConfig
from ..errors import ParamError, SourceError, TAError
from ..signals.overlays import ConfigDelta
from . import lifecycle
from .models import Proposal, Status
from .store import ProposalStore

ALLOWLIST = frozenset(
    f.name for f in ConfigDelta.__dataclass_fields__.values() if f.name != "reason"
)


@dataclass(frozen=True)
class ApplyResult:
    """What an apply attempt did, or would have done."""

    proposal: Proposal
    dry_run: bool
    applied: bool
    diff: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    base_version: int | None = None
    live_version: int | None = None
    result_version: int | None = None
    note: str = ""
    error: str = ""

    @property
    def is_noop(self) -> bool:
        return not self.diff

    def render(self) -> str:
        """Human-readable before/after -- what a reviewer reads before writing."""
        proposal = self.proposal
        head = (
            f"{proposal.source}/{proposal.market}  "
            f"proposal {proposal.id}  status={proposal.status}"
        )
        version_line = f"base_version={self.base_version}  live_version={self.live_version}"
        lines = [head, version_line, ""]

        if self.error:
            lines.append(f"ERROR: {self.error}")
            return "\n".join(lines)

        if not self.diff:
            lines.append("no change: the live config already matches this proposal.")
        else:
            width = max(len(k) for k in self.diff)
            lines.append("would change:" if self.dry_run else "changed:")
            for key in sorted(self.diff):
                before, after = self.diff[key]
                lines.append(f"  {key:<{width}}  {before!r}  ->  {after!r}")

        if self.note:
            lines += ["", self.note]
        if self.dry_run:
            lines += ["", "DRY RUN -- nothing was written. Pass --write to apply."]
        elif self.applied:
            lines += ["", f"written. new version: {self.result_version}"]
        return "\n".join(lines)


def diff_configs(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Field-level difference, including fields added or dropped."""
    out: dict[str, tuple[Any, Any]] = {}
    for key in set(before) | set(after):
        old, new = before.get(key), after.get(key)
        if old != new:
            out[key] = (old, new)
    return out


def _assert_only_allowlisted(diff: Mapping[str, tuple[Any, Any]]) -> None:
    stray = sorted(set(diff) - ALLOWLIST)
    if stray:
        raise ParamError(
            f"refusing to write: composing the proposal would also change "
            f"non-allowlisted field(s) {', '.join(stray)}. This usually means the "
            f"read and the write disagree about the document shape, and writing "
            f"would reset those fields to their schema defaults."
        )


def apply(
    proposal: Proposal,
    *,
    client: ConfigManagerClient,
    store: ProposalStore | None = None,
    dry_run: bool = True,
    actor: str = "system",
) -> ApplyResult:
    """Attempt to apply an approved proposal. Dry run unless told otherwise.

    A dry run never changes the proposal's status -- it is a question, not a
    step. A real run always ends the proposal in a terminal state
    (`applied`, `stale`, or `failed`).
    """
    if Status(proposal.status) is not Status.APPROVED:
        raise ParamError(
            f"proposal {proposal.id} is {proposal.status}, not approved. "
            f"Only an approved proposal may be applied -- run "
            f"`python -m ta.proposals approve {proposal.id} --by <you>` first."
        )
    if not proposal.decided_by:
        raise ParamError(
            f"proposal {proposal.id} claims approval but names no approver; refusing to apply"
        )

    live = client.get_market(proposal.source, proposal.market)
    if live is None:
        raise ParamError(
            f"{proposal.source}/{proposal.market} has no config in config-manager. "
            f"This project modifies existing markets; it cannot create one."
        )

    checked = lifecycle.check_version(proposal, live.version)
    if Status(checked.status) is Status.STALE:
        if store is not None and not dry_run:
            store.commit(proposal, checked, action="stale", actor=actor)
        return ApplyResult(
            proposal=checked,
            dry_run=dry_run,
            applied=False,
            base_version=proposal.base_version,
            live_version=live.version,
            note=(
                f"config moved from version {proposal.base_version} to "
                f"{live.version} since this was proposed; it must be reviewed again."
            ),
        )

    before = dict(live.config)
    after = proposal.delta.apply_to(before)
    diff = diff_configs(before, after)
    _assert_only_allowlisted(diff)

    if not diff:
        result = ApplyResult(
            proposal=checked,
            dry_run=dry_run,
            applied=False,
            base_version=proposal.base_version,
            live_version=live.version,
            result_version=live.version,
            note="live config already matches this proposal; nothing to write.",
        )
        if dry_run or store is None:
            return result
        done = lifecycle.mark_applied(checked, result_version=live.version)
        store.commit(
            checked, done, action="apply", actor=actor, note="no-op: already satisfied"
        )
        return ApplyResult(**{**result.__dict__, "proposal": done, "applied": True})

    if dry_run:
        return ApplyResult(
            proposal=checked,
            dry_run=True,
            applied=False,
            diff=diff,
            base_version=proposal.base_version,
            live_version=live.version,
        )

    try:
        written = client.put_market(proposal.source, proposal.market, after)
    except (SourceError, TAError) as exc:
        failed = lifecycle.mark_failed(checked, str(exc))
        if store is not None:
            store.commit(checked, failed, action="apply-failed", actor=actor)
        return ApplyResult(
            proposal=failed,
            dry_run=False,
            applied=False,
            diff=diff,
            base_version=proposal.base_version,
            live_version=live.version,
            error=str(exc),
        )

    done = lifecycle.mark_applied(checked, result_version=written.version)
    if store is not None:
        store.commit(checked, done, action="apply", actor=actor)
    return ApplyResult(
        proposal=done,
        dry_run=False,
        applied=True,
        diff=diff,
        base_version=proposal.base_version,
        live_version=live.version,
        result_version=written.version,
    )
