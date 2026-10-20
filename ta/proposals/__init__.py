"""Proposals: the approval gate between an insight and a live config change."""

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
