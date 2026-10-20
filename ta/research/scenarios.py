"""Named batches of runs, saved in code so a comparison stays reproducible."""

from __future__ import annotations

from typing import Any

_USDT_VENUES = (("USDTIRT", "nobitex"), ("USDTIRR", "ramzinex"))

_MAJORS = (
    ("BTCIRT", "nobitex"), ("ETHIRT", "nobitex"),
    ("BTCUSDT", "nobitex"), ("ETHUSDT", "nobitex"),
)


def _jobs(pairs, families, intervals, **extra) -> list[dict[str, Any]]:
    return [
        {"market": m, "source": s, "interval": i, "family": f, **extra}
        for (m, s) in pairs
        for f in families
        for i in intervals
    ]


SCENARIOS: dict[str, dict[str, Any]] = {
    "usdt-cross-venue": {
        "description": (
            "The replication test that matters: same asset, two independent "
            "venues. A signal that shows on only one is a data artifact."
        ),
        "jobs": _jobs(_USDT_VENUES, ("volume", "broad", "ma"), ("15Min", "1H")),
    },
    "usdt-volume": {
        "description": "Volume family on both USDT books — where volz_dir lives.",
        "jobs": _jobs(_USDT_VENUES, ("volume",), ("15Min", "1H")),
    },
    "majors": {
        "description": "BTC/ETH in both quote currencies, all families, 15Min + 1H.",
        "jobs": _jobs(_MAJORS, ("volume", "broad", "ma"), ("15Min", "1H")),
    },
    "actionable-1h": {
        "description": (
            "Hourly only, across every market. 1H signals have ~10x lower "
            "turnover than 15Min ones, so this is the horizon where a positive "
            "result could actually be reached by quote skew."
        ),
        "jobs": _jobs(_USDT_VENUES + _MAJORS, ("broad", "ma"), ("1H", "4H")),
    },
    "smoke": {
        "description": "One fast run, to check the service is alive and wired to Mongo.",
        "jobs": [
            {"market": "USDTIRT", "source": "nobitex", "interval": "1H",
             "family": "ma", "permutations": 20}
        ],
    },
}

DEFAULTS = {"permutations": 200, "consecutive_only": True, "split": 0.7, "horizon": 1}


def list_scenarios() -> list[dict[str, Any]]:
    return [
        {"name": name, "description": spec["description"], "jobs": len(spec["jobs"])}
        for name, spec in sorted(SCENARIOS.items())
    ]


def expand(name: str) -> list[dict[str, Any]]:
    """Scenario name -> concrete job dicts, defaults filled in."""
    spec = SCENARIOS.get(name)
    if spec is None:
        raise KeyError(f"unknown scenario {name!r}; known: {', '.join(sorted(SCENARIOS))}")
    return [{**DEFAULTS, **job} for job in spec["jobs"]]
