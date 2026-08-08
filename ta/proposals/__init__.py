"""Proposals: the approval gate between an insight and a live config change.

Nothing in this package writes to config-manager. It creates proposals,
stores them, and records human decisions about them. The module that can
actually write (`ta/proposals/apply.py`) does not exist yet, by design --
the gate is built and tested before anything can pass through it.

    from ta.proposals import create_proposal, approve, default_store

    proposal = create_proposal("nobitex", "USDTIRT", delta,
                               base_version=12, evidence=[vol_result])
    outcome = default_store().save_new(proposal)
    ...
    approved = approve(outcome.proposal, by="omid", note="agreed, vol is high")

CLI: `python -m ta.proposals list|show|approve|reject`.
"""

from .lifecycle import (  # noqa: F401
    LEGAL_TRANSITIONS,
    approve,
    can_transition,
    check_version,
    mark_applied,
    mark_failed,
    mark_stale,
    reject,
    transition,
)
from .models import (  # noqa: F401
    TERMINAL,
    Proposal,
    Status,
    create_proposal,
    delta_hash,
    make_id,
)
from .store import (  # noqa: F401
    NATURAL_KEY,
    ProposalStore,
    SaveOutcome,
    default_store,
)
