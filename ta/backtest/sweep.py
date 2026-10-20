"""Grid search over the band and size knobs for one market, ranked by PnL."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..errors import ParamError, TAError
from .runner import DISCLAIMER, BacktestResult, run

SWEEPABLE = ("price_min_spread", "price_max_spread", "max_base",
             "number_of_order_each_sides", "spread_multiplier")


@dataclass(frozen=True)
class SweepRow:
    params: dict[str, Any]
    metrics: dict[str, Any]
    error: str = ""

    @property
    def pnl(self) -> float:
        return float(self.metrics.get("pnl") or 0.0)

    def as_dict(self) -> dict:
        return {"params": self.params, "metrics": self.metrics, "error": self.error}


def grid(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    """Cartesian product of the named axes, as a list of parameter dicts."""
    unknown = sorted(set(axes) - set(SWEEPABLE))
    if unknown:
        raise ParamError(
            f"cannot sweep {', '.join(unknown)}; sweepable knobs are "
            f"{', '.join(SWEEPABLE)}"
        )
    names = sorted(axes)
    return [
        dict(zip(names, values))
        for values in itertools.product(*(axes[name] for name in names))
    ]


def sweep(
    data,
    baseline: Mapping[str, Any],
    points: Iterable[Mapping[str, Any]],
    **run_kwargs,
) -> list[SweepRow]:
    """Run every point over the same bars, ranked best PnL first."""
    rows: list[SweepRow] = []
    for point in points:
        config = {**dict(baseline), **dict(point)}
        try:
            result = run(data, config, **run_kwargs)
        except TAError as exc:
            rows.append(SweepRow(params=dict(point), metrics={}, error=str(exc)))
            continue
        rows.append(SweepRow(params=dict(point), metrics=result.metrics.as_dict()))

    rows.sort(key=lambda r: (bool(r.error), -r.pnl))
    return rows


def render(rows: Sequence[SweepRow], *, limit: int = 20) -> str:
    """Readable ranking table, with the caveat attached."""
    if not rows:
        return "no sweep results."

    axes = sorted({key for row in rows for key in row.params})
    headers = [*axes, "PNL", "FILLS", "BUY%", "SELL%", "MO600", "NOTE"]

    body = []
    for row in rows[:limit]:
        metrics = row.metrics
        markout = metrics.get("markout_600s_bps")
        body.append(
            [
                *[_fmt(row.params.get(a)) for a in axes],
                _fmt(metrics.get("pnl"), ".6g"),
                _fmt(metrics.get("fills")),
                _fmt(metrics.get("buy_uptime"), ".0%"),
                _fmt(metrics.get("sell_uptime"), ".0%"),
                _fmt(markout, "+.1f"),
                row.error[:40] or "-",
            ]
        )

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in body)) for i in range(len(headers))
    ]
    line = "  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()
    out = [line, "-" * len(line)]
    out += ["  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in body]
    if len(rows) > limit:
        out.append(f"... {len(rows) - limit} more")
    out += ["", DISCLAIMER, "", "Read the ordering, not the levels."]
    return "\n".join(out)


def _fmt(value, spec: str = "", dash: str = "-") -> str:
    if value is None:
        return dash
    try:
        return format(value, spec) if spec else str(value)
    except (TypeError, ValueError):
        return str(value)
