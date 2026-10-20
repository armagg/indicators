"""CLI over `ta.research.run_brute` -- the same code path the service uses."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from ta.cache import default_cache
from ta.errors import TAError
from ta.research import DEFAULTS, FAMILIES, expand, list_scenarios, run_brute


def _print_result(payload: dict) -> None:
    headline = payload["headline"]
    grid = pd.DataFrame(payload["grid"])
    show = ["signal", "excess_bps", "ic", "t", "naive_bps", "drift_bps",
            "is_excess_bps", "oos_excess_bps", "turnover", "hit_rate"]
    print(grid[show].head(10).to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    null = payload["permutation"]
    print(f"\nIS->OOS correlation: {headline['is_oos_correlation']:+.3f}"
          f"   (negative = the backtest ranking is anti-informative)")
    print(f"permutation null: mean {null['null_mean_bps']:.4f}  "
          f"p95 {null['null_p95_bps']:.4f}   p-value {null['p_value']:.3f}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m research.signal_brute")
    parser.add_argument("--scenario", help="run a saved batch instead of one job")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument("--market")
    parser.add_argument("--source")
    parser.add_argument("--interval", default="1H")
    parser.add_argument("--family", default="ma", choices=sorted(FAMILIES))
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--split", type=float, default=0.7)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--with-gaps", action="store_true",
                        help="disable the consecutive-bar control (not advised)")
    parser.add_argument("--refresh", action="store_true", help="ignore cached series")
    parser.add_argument("--rerun", action="store_true", help="ignore a cached run")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)

    if args.list_scenarios:
        for spec in list_scenarios():
            print(f"{spec['name']:20s} {spec['jobs']:3d} jobs  {spec['description']}")
        return 0

    cache = None if args.no_cache else default_cache()
    if args.scenario:
        specs = expand(args.scenario)
    else:
        if not args.market or not args.source:
            parser.error("--market and --source are required (or use --scenario)")
        specs = [{**DEFAULTS, "market": args.market, "source": args.source,
                  "interval": args.interval, "family": args.family,
                  "permutations": args.permutations, "split": args.split,
                  "horizon": args.horizon,
                  "consecutive_only": not args.with_gaps}]

    failures = 0
    for spec in specs:
        print(f"\n===== {spec['market']}/{spec['source']} {spec['interval']} "
              f"{spec['family']} =====")
        try:
            payload = run_brute(
                cache=cache, refresh=args.refresh, rerun=args.rerun,
                limit=args.limit, log=lambda m: print(f"  {m}"), **spec
            )
            _print_result(payload)
        except TAError as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
