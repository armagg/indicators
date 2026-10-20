"""`python -m ta.proposals` -- review and decide on proposals; `apply` is a dry run by default."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from ..errors import TAError
from . import lifecycle
from .models import Proposal, Status
from .store import ProposalStore, default_store


def _age(created: datetime | None) -> str:
    if not created:
        return "?"
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    seconds = (datetime.now(timezone.utc) - created).total_seconds()
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds >= size:
            return f"{int(seconds // size)}{unit}"
    return f"{int(seconds)}s"


def _render_table(proposals: list[Proposal]) -> str:
    if not proposals:
        return "no proposals match."
    header = f"{'ID':<14}{'STATUS':<10}{'SOURCE':<10}{'MARKET':<12}{'VER':<6}{'IMPACT':<8}{'AGE':<6}CHANGE"
    rows = [header, "-" * len(header)]
    for p in proposals:
        rows.append(
            f"{p.id:<14}{str(p.status):<10}{p.source:<10}{p.market:<12}"
            f"{str(p.base_version if p.base_version is not None else '-'):<6}"
            f"{'HIGH' if p.is_high_impact else 'low':<8}{_age(p.created_at):<6}{p.summary()}"
        )
    return "\n".join(rows)


def _render_detail(proposal: Proposal, history: list[dict]) -> str:
    lines = [
        f"id           {proposal.id}",
        f"market       {proposal.source}/{proposal.market}",
        f"status       {proposal.status}"
        + (f"  (impact: HIGH)" if proposal.is_high_impact else ""),
        f"base_version {proposal.base_version}",
        f"created      {proposal.created_at}",
        "",
        "proposed change:",
    ]
    for key, value in sorted(proposal.delta.fields().items()):
        lines.append(f"  {key} = {value}")
    if proposal.delta.reason:
        lines += ["", f"reason: {proposal.delta.reason}"]

    if proposal.evidence:
        lines += ["", "evidence:"]
        for item in proposal.evidence:
            name = item.get("name", "?")
            scalars = item.get("scalars", {})
            shown = ", ".join(
                f"{k}={v}" for k, v in list(scalars.items())[:6] if v is not None
            )
            lines.append(f"  {name}: {shown}")

    if proposal.decided_by:
        lines += [
            "",
            f"decided      {proposal.decided_by} at {proposal.decided_at}",
        ]
        if proposal.decision_note:
            lines.append(f"note         {proposal.decision_note}")
    if proposal.applied_at:
        lines.append(f"applied      {proposal.applied_at} -> version {proposal.result_version}")
    if proposal.error:
        lines.append(f"error        {proposal.error}")

    if history:
        lines += ["", "history:"]
        for row in history:
            lines.append(
                f"  {row.get('at')}  {row.get('action'):<12} "
                f"{row.get('from_status') or '-'} -> {row.get('to_status')}  "
                f"by {row.get('actor')}"
            )
    return "\n".join(lines)


def _decide(store: ProposalStore, args, action: str) -> int:
    proposal = store.get(args.id)
    if proposal is None:
        print(f"no proposal with id {args.id}", file=sys.stderr)
        return 1

    fn = lifecycle.approve if action == "approve" else lifecycle.reject
    updated = fn(proposal, by=args.by, note=args.note or "")
    store.commit(proposal, updated, action=action, actor=args.by, note=args.note or "")
    print(f"{proposal.id} {action}d by {args.by} ({updated.status})")
    if action == "approve":
        print("not applied: applying is a separate, explicit step.")
    return 0


def _apply(store: ProposalStore, args, client) -> int:
    from .apply import apply as apply_proposal

    proposal = store.get(args.id)
    if proposal is None:
        print(f"no proposal with id {args.id}", file=sys.stderr)
        return 1

    if args.write and not args.by:
        print("--write requires --by: who is pushing this button?", file=sys.stderr)
        return 1

    if client is None:
        from ..configclient import default_client

        client = default_client()

    result = apply_proposal(
        proposal,
        client=client,
        store=store,
        dry_run=not args.write,
        actor=args.by or "cli",
    )
    print(result.render())
    return 1 if result.error else 0


def main(
    argv: list[str] | None = None,
    store: ProposalStore | None = None,
    client=None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ta.proposals", description="Review and decide on proposals."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list proposals")
    p_list.add_argument("--status", default=None, choices=[str(s) for s in Status])
    p_list.add_argument("--source", default=None)
    p_list.add_argument("--market", default=None)
    p_list.add_argument("--limit", type=int, default=50)
    p_list.add_argument("--json", action="store_true")

    p_show = sub.add_parser("show", help="show one proposal in full")
    p_show.add_argument("id")
    p_show.add_argument("--json", action="store_true")

    for name in ("approve", "reject"):
        p = sub.add_parser(name, help=f"{name} a proposal")
        p.add_argument("id")
        p.add_argument("--by", required=True, help="who is accountable for this decision")
        p.add_argument("--note", default="")

    p_apply = sub.add_parser(
        "apply", help="apply an approved proposal (dry run unless --write)"
    )
    p_apply.add_argument("id")
    p_apply.add_argument(
        "--write",
        action="store_true",
        help="actually write to config-manager (default is a dry run)",
    )
    p_apply.add_argument("--by", default="", help="who is applying; required with --write")

    args = parser.parse_args(argv)

    try:
        target = store if store is not None else default_store()

        if args.command == "list":
            found = target.find(
                status=args.status, source=args.source, market=args.market, limit=args.limit
            )
            if args.json:
                json.dump([p.to_doc() for p in found], sys.stdout, indent=2, default=str)
                sys.stdout.write("\n")
            else:
                print(_render_table(found))
            return 0

        if args.command == "show":
            proposal = target.get(args.id)
            if proposal is None:
                print(f"no proposal with id {args.id}", file=sys.stderr)
                return 1
            if args.json:
                json.dump(proposal.to_doc(), sys.stdout, indent=2, default=str)
                sys.stdout.write("\n")
            else:
                print(_render_detail(proposal, target.history(args.id)))
            return 0

        if args.command == "apply":
            return _apply(target, args, client)

        return _decide(target, args, args.command)

    except TAError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
