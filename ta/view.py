"""The exchange view: one screen per venue, joining tier list, live config, and candle stats."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from .analysis import volatility as volatility_analysis
from .configclient import ConfigManagerClient, MarketConfig
from .errors import ParamError, TAError
from .frame import normalize_ohlcv
from .tiers import DEFAULT_TIER, MAJOR, TierStore


SPARSE_GAP_RATIO = 0.20

STALE_HOURS = 24.0

THIN_HISTORY_BARS = 200

HIGH_NATR_PCT = 3.0

ERRATIC_VOLUME_CV = 2.0


@dataclass(frozen=True)
class MarketStats:
    """What the candles say about one market."""

    bars: int | None = None
    gap_ratio: float | None = None
    last_bar: datetime | None = None
    hours_stale: float | None = None
    natr: float | None = None
    vol_percentile: float | None = None
    median_volume: float | None = None
    volume_cv: float | None = None
    flags: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        out = asdict(self)
        out["last_bar"] = self.last_bar.isoformat() if self.last_bar else None
        out["flags"] = list(self.flags)
        return out


def market_stats(frame: pd.DataFrame, *, now: datetime | None = None) -> MarketStats:
    """Summarise a normalized OHLCV frame into the view's columns. Pure."""
    if frame is None or frame.empty:
        return MarketStats(flags=("no-data",))

    normalized, info = normalize_ohlcv(frame)
    now = now or datetime.now(timezone.utc)
    last_bar = normalized.index[-1].to_pydatetime()
    hours_stale = max(0.0, (now - last_bar).total_seconds() / 3600.0)

    natr = vol_percentile = None
    try:
        verdict = volatility_analysis.run(normalized)
        natr = verdict.scalars.get("natr")
        vol_percentile = verdict.scalars.get("vol_percentile")
    except TAError:
        pass

    median_volume = volume_cv = None
    if "volume" in normalized.columns and normalized["volume"].notna().any():
        volume = normalized["volume"].dropna()
        median_volume = float(volume.median())
        mean = float(volume.mean())
        if mean > 0:
            volume_cv = float(volume.std(ddof=0) / mean)

    stats = MarketStats(
        bars=int(len(normalized)),
        gap_ratio=info.gap_ratio,
        last_bar=last_bar,
        hours_stale=hours_stale,
        natr=natr,
        vol_percentile=vol_percentile,
        median_volume=median_volume,
        volume_cv=volume_cv,
    )
    return replace(stats, flags=assess(stats))


def assess(stats: MarketStats) -> tuple[str, ...]:
    """Named risk flags. Order is stable so rows diff cleanly between runs."""
    flags: list[str] = []
    if stats.bars is not None and stats.bars < THIN_HISTORY_BARS:
        flags.append("thin-history")
    if stats.gap_ratio is not None and stats.gap_ratio > SPARSE_GAP_RATIO:
        flags.append("sparse")
    if stats.hours_stale is not None and stats.hours_stale > STALE_HOURS:
        flags.append("stale")
    if stats.natr is not None and stats.natr > HIGH_NATR_PCT:
        flags.append("volatile")
    if stats.volume_cv is not None and stats.volume_cv > ERRATIC_VOLUME_CV:
        flags.append("erratic-volume")
    return tuple(flags)


def load_stats(
    source: str,
    market: str,
    *,
    interval: str = "1H",
    limit: int = 500,
    reader=None,
    now: datetime | None = None,
) -> MarketStats:
    """Read candles for one market and summarise them."""
    if reader is None:
        from .ohlc import read_ohlc as reader
    try:
        frame = reader(market, source, interval, limit=limit)
    except TAError:
        return MarketStats(flags=("no-data",))
    return market_stats(frame, now=now) if frame is not None else MarketStats(flags=("no-data",))


ALERT_MAJOR_NO_CONFIG = "NO-CONFIG"
ALERT_MAJOR_OFF = "MAJOR-OFF"
ALERT_RISK_ON = "RISK-ON"
_ALERT_ORDER = {ALERT_MAJOR_NO_CONFIG: 0, ALERT_MAJOR_OFF: 1, ALERT_RISK_ON: 2, "": 3}


@dataclass(frozen=True)
class MarketRow:
    source: str
    market: str
    tier: str
    has_config: bool
    enabled: bool
    version: int | None = None
    max_base: float | None = None
    min_base: float | None = None
    price_min_spread: float | None = None
    price_max_spread: float | None = None
    stats: MarketStats = field(default_factory=MarketStats)
    alert: str = ""

    @property
    def band_bps(self) -> str:
        if self.price_min_spread is None or self.price_max_spread is None:
            return "-"
        return f"{self.price_min_spread * 1e4:.0f}-{self.price_max_spread * 1e4:.0f}"

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "market": self.market,
            "tier": self.tier,
            "has_config": self.has_config,
            "enabled": self.enabled,
            "version": self.version,
            "max_base": self.max_base,
            "min_base": self.min_base,
            "price_min_spread": self.price_min_spread,
            "price_max_spread": self.price_max_spread,
            "band_bps": self.band_bps,
            "alert": self.alert,
            "stats": self.stats.as_dict(),
        }


def classify(tier: str, has_config: bool, enabled: bool) -> str:
    """The disagreement between what we think and what is running."""
    if tier == MAJOR and not has_config:
        return ALERT_MAJOR_NO_CONFIG
    if tier == MAJOR and not enabled:
        return ALERT_MAJOR_OFF
    if tier != MAJOR and enabled:
        return ALERT_RISK_ON
    return ""


def build_view(
    configs: Iterable[MarketConfig],
    tiers: Mapping[tuple[str, str], str],
    stats: Mapping[str, MarketStats] | None = None,
    *,
    source: str = "",
) -> list[MarketRow]:
    """Join live config, the tier list, and candle stats into sorted rows."""
    stats = stats or {}
    by_market: dict[str, MarketConfig] = {c.market: c for c in configs}
    tier_markets = {market for (src, market) in tiers if not source or src == source}

    rows: list[MarketRow] = []
    for market in sorted(set(by_market) | tier_markets):
        cfg = by_market.get(market)
        tier = tiers.get((source or (cfg.source if cfg else ""), market), DEFAULT_TIER)
        settings = cfg.config if cfg else {}
        enabled = bool(settings.get("is_enable")) if cfg else False
        rows.append(
            MarketRow(
                source=source or (cfg.source if cfg else ""),
                market=market,
                tier=tier,
                has_config=cfg is not None,
                enabled=enabled,
                version=cfg.version if cfg else None,
                max_base=settings.get("max_base"),
                min_base=settings.get("min_base"),
                price_min_spread=settings.get("price_min_spread"),
                price_max_spread=settings.get("price_max_spread"),
                stats=stats.get(market, MarketStats()),
                alert=classify(tier, cfg is not None, enabled),
            )
        )

    rows.sort(
        key=lambda r: (_ALERT_ORDER.get(r.alert, 3), r.tier != MAJOR, r.market)
    )
    return rows


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    widths = [
        max(len(str(headers[i])), *(len(str(r[i])) for r in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths)).rstrip()
    out = [line, "-" * len(line)]
    for row in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())
    return "\n".join(out)


def _fmt(value, spec: str = "", dash: str = "-") -> str:
    if value is None:
        return dash
    return format(value, spec) if spec else str(value)


def render(rows: Sequence[MarketRow]) -> str:
    if not rows:
        return "no markets: neither config-manager nor the tier list knows this source."

    body = []
    for r in rows:
        s = r.stats
        body.append(
            [
                r.market,
                r.tier,
                "on" if r.enabled else ("off" if r.has_config else "none"),
                _fmt(r.version),
                _fmt(r.max_base, ".4g"),
                r.band_bps,
                _fmt(s.gap_ratio, ".0%"),
                _fmt(s.natr, ".2f"),
                _fmt(s.hours_stale, ".0f"),
                ",".join(s.flags) or "-",
                r.alert or "-",
            ]
        )
    table = _table(
        ["MARKET", "TIER", "LIVE", "VER", "MAX_BASE", "BAND(bp)", "GAP", "NATR", "STALE_H", "FLAGS", "ALERT"],
        body,
    )

    alerts = [r for r in rows if r.alert]
    footer = [""]
    if alerts:
        footer.append(f"{len(alerts)} disagreement(s) between the tier list and live config:")
        for r in alerts:
            if r.alert == ALERT_MAJOR_NO_CONFIG:
                footer.append(f"  {r.market}: tiered major but has no config at all.")
            elif r.alert == ALERT_MAJOR_OFF:
                footer.append(f"  {r.market}: tiered major but is not quoting.")
            else:
                footer.append(
                    f"  {r.market}: not a major market but is live"
                    + (f" (flags: {','.join(r.stats.flags)})" if r.stats.flags else "")
                    + "."
                )
    else:
        footer.append("tier list and live config agree.")
    return table + "\n" + "\n".join(footer)


def draft_toggle(
    row: MarketRow,
    *,
    enable: bool,
    max_base: float | None = None,
    note: str = "",
):
    """Draft an enable/disable proposal for one market."""
    from .proposals import create_proposal
    from .signals import ConfigDelta

    if enable and row.enabled:
        raise ParamError(f"{row.market} is already enabled")
    if not enable and row.has_config and not row.enabled:
        raise ParamError(f"{row.market} is already disabled")
    if not row.has_config:
        raise ParamError(
            f"{row.market} has no config in config-manager; this project can "
            f"modify existing markets but cannot create one"
        )

    reason = note or (
        f"manual: {'enable' if enable else 'disable'} {row.market} "
        f"(tier={row.tier}, flags={','.join(row.stats.flags) or 'none'})"
    )
    fields: dict[str, Any] = {"is_enable": enable, "reason": reason}
    if max_base is not None:
        fields["max_base"] = float(max_base)

    evidence = [
        {
            "name": "market_view",
            "params": {"tier": row.tier, "source": row.source, "market": row.market},
            "scalars": {
                **row.stats.as_dict(),
                "live_enabled": row.enabled,
                "live_version": row.version,
            },
        }
    ]
    return create_proposal(
        row.source,
        row.market,
        ConfigDelta(**fields),
        base_version=row.version,
        evidence=evidence,
    )


def collect(
    source: str,
    *,
    client: ConfigManagerClient,
    tier_store: TierStore,
    interval: str = "1H",
    limit: int = 500,
    with_stats: bool = True,
    reader=None,
    now: datetime | None = None,
) -> list[MarketRow]:
    """Fetch everything one view needs and build the rows."""
    configs = client.list_markets(source)
    tiers = tier_store.mapping(source)
    stats: dict[str, MarketStats] = {}
    if with_stats:
        markets = {c.market for c in configs} | {m for (s, m) in tiers if s == source}
        for market in markets:
            stats[market] = load_stats(
                source, market, interval=interval, limit=limit, reader=reader, now=now
            )
    return build_view(configs, tiers, stats, source=source)


def main(
    argv: list[str] | None = None,
    *,
    client: ConfigManagerClient | None = None,
    tier_store: TierStore | None = None,
    proposal_store=None,
    reader=None,
) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0].startswith("-"):
        argv = ["show", *argv]

    parser = argparse.ArgumentParser(
        prog="python -m ta.view",
        description="Per-exchange view of markets: tier, live config, and data health.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_show = sub.add_parser("show", help="the market table for one exchange")
    p_show.add_argument("-s", "--source", required=True)
    p_show.add_argument("-i", "--interval", default="1H")
    p_show.add_argument("--limit", type=int, default=500)
    p_show.add_argument("--no-stats", action="store_true", help="skip the candle read")
    p_show.add_argument("--json", action="store_true")

    p_tier = sub.add_parser("tier", help="read or edit the tier list")
    tier_sub = p_tier.add_subparsers(dest="tier_command", required=True)
    t_set = tier_sub.add_parser("set")
    t_set.add_argument("source")
    t_set.add_argument("market")
    t_set.add_argument("tier", choices=["major", "other"])
    t_set.add_argument("--by", required=True)
    t_set.add_argument("--note", default="")
    t_list = tier_sub.add_parser("list")
    t_list.add_argument("-s", "--source", default=None)
    t_clear = tier_sub.add_parser("clear")
    t_clear.add_argument("source")
    t_clear.add_argument("market")

    p_propose = sub.add_parser("propose", help="draft an enable/disable proposal")
    p_propose.add_argument("action", choices=["enable", "disable"])
    p_propose.add_argument("source")
    p_propose.add_argument("market")
    p_propose.add_argument("--max-base", type=float, default=None)
    p_propose.add_argument("--note", default="")

    args = parser.parse_args(argv)

    try:
        tiers = tier_store if tier_store is not None else None

        if args.command == "tier":
            if tiers is None:
                from .tiers import default_store

                tiers = default_store()
            if args.tier_command == "set":
                doc = tiers.set(
                    args.source, args.market, args.tier, by=args.by, note=args.note
                )
                print(f"{doc['source']}/{doc['market']} -> {doc['tier']} (by {doc['updated_by']})")
            elif args.tier_command == "clear":
                removed = tiers.clear(args.source, args.market)
                print(
                    f"{args.source}/{args.market} cleared; now '{DEFAULT_TIER}' by default"
                    if removed
                    else f"{args.source}/{args.market} had no explicit tier"
                )
            else:
                rows = [
                    [d.get("source", ""), d.get("market", ""), d.get("tier", ""),
                     d.get("updated_by", ""), d.get("note", "")]
                    for d in tiers.list_all(args.source)
                ]
                print(
                    _table(["SOURCE", "MARKET", "TIER", "BY", "NOTE"], rows)
                    if rows
                    else "tier list is empty; every market defaults to 'other' (disabled)."
                )
            return 0

        if client is None:
            from .configclient import default_client

            client = default_client()
        if tiers is None:
            from .tiers import default_store

            tiers = default_store()

        if args.command == "show":
            rows = collect(
                args.source,
                client=client,
                tier_store=tiers,
                interval=args.interval,
                limit=args.limit,
                with_stats=not args.no_stats,
                reader=reader,
            )
            if args.json:
                json.dump([r.as_dict() for r in rows], sys.stdout, indent=2, default=str)
                sys.stdout.write("\n")
            else:
                print(render(rows))
            return 0

        if args.command == "propose":
            rows = collect(
                args.source, client=client, tier_store=tiers, reader=reader
            )
            match = next((r for r in rows if r.market == args.market), None)
            if match is None:
                print(f"{args.source}/{args.market} not found", file=sys.stderr)
                return 1

            proposal = draft_toggle(
                match,
                enable=args.action == "enable",
                max_base=args.max_base,
                note=args.note,
            )
            if proposal_store is None:
                from .proposals import default_store as proposal_default

                proposal_store = proposal_default()
            outcome = proposal_store.save_new(proposal)
            print(
                f"proposal {outcome.proposal.id} ({outcome.proposal.status}): "
                f"{outcome.proposal.summary()}"
            )
            if not outcome.created:
                print(f"note: {outcome.note}")
            else:
                print(
                    f"review it: python -m ta.proposals show {outcome.proposal.id}\n"
                    f"nothing is live until it is approved and applied."
                )
            return 0

        return 1

    except TAError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
