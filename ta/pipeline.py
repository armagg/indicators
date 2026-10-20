"""One market in, one combined snapshot out: a single OHLC pull feeding many indicators and analyses."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import store
from .analysis import run_analysis
from .engine import compute, parse_requests
from .errors import FrameError, TAError
from .frame import describe_frame, drop_incomplete_bar, normalize_ohlcv
from .ohlc import normalize_interval, read_ohlc


def run(
    market: str,
    source: str,
    interval: str,
    *,
    indicators=None,
    analyses=None,
    limit: int | None = None,
    start=None,
    end=None,
    save: bool = False,
    warmup: str = "mask",
    drop_forming: bool = True,
    include_records: bool = False,
    include_series: bool = False,
    latest_bar_only: bool = False,
) -> dict[str, Any]:
    """Fetch once, compute everything, return one JSON-ready snapshot."""
    indicator_requests = parse_requests(indicators)
    analysis_requests = parse_requests(analyses)
    if not indicator_requests and not analysis_requests:
        raise FrameError("nothing to do: supply indicators and/or analyses")

    interval = normalize_interval(interval)
    raw = read_ohlc(market, source, interval, limit=limit, start=start, end=end)
    if raw is None:
        raise FrameError(
            f"no candles for market={market!r} source={source!r} interval={interval!r}"
        )

    frame, info = normalize_ohlcv(raw)
    dropped_forming = False
    if drop_forming:
        frame, dropped_forming = drop_incomplete_bar(frame, info.interval_sec)
        if dropped_forming:
            if frame.empty:
                raise FrameError("only one candle and it is still forming")
            info = describe_frame(frame)

    snapshot: dict[str, Any] = {
        "market": market,
        "source": source,
        "interval": interval,
        "asof": frame.index[-1].isoformat(),
        "dropped_forming_bar": dropped_forming,
        "frame": info.as_dict(),
    }

    if indicator_requests:
        result = compute(frame, indicator_requests, warmup=warmup, normalize=False)
        snapshot["indicators"] = {"latest": result.latest(), "meta": result.meta()}
        if include_records:
            snapshot["indicators"]["records"] = result.to_records()
        if save:
            snapshot["saved"] = store.save(
                result,
                symbol=market,
                source=source,
                interval=interval,
                latest_bar_only=latest_bar_only,
            )

    if analysis_requests:
        snapshot["analyses"] = {
            request.key(): run_analysis(request.name, frame, request.params).as_dict(
                include_series=include_series
            )
            for request in analysis_requests
        }

    return snapshot


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ta.pipeline",
        description="Compute indicators and analyses for one market, print JSON.",
    )
    parser.add_argument("-m", "--market", required=True)
    parser.add_argument("-s", "--source", required=True, help="venue, e.g. nobitex")
    parser.add_argument("-i", "--interval", required=True, help="1Min/5Min/15Min/1H/4H/1D")
    parser.add_argument("--indicators", default=None, help='e.g. "rsi:period=21,macd,atr"')
    parser.add_argument("--analyses", default=None, help='e.g. "volatility,trend,regime"')
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", default=None, help="Tehran wall time when naive")
    parser.add_argument("--end", default=None)
    parser.add_argument("--save", action="store_true", help="persist indicator rows to Mongo")
    parser.add_argument("--warmup", choices=("mask", "trim", "keep"), default="mask")
    parser.add_argument("--records", action="store_true", help="include full indicator rows")
    parser.add_argument("--series", action="store_true", help="include analysis series")
    parser.add_argument("--keep-forming", action="store_true", help="keep the forming last bar")
    args = parser.parse_args(argv)

    try:
        snapshot = run(
            args.market,
            args.source,
            args.interval,
            indicators=args.indicators,
            analyses=args.analyses,
            limit=args.limit,
            start=args.start,
            end=args.end,
            save=args.save,
            warmup=args.warmup,
            drop_forming=not args.keep_forming,
            include_records=args.records,
            include_series=args.series,
        )
    except TAError as exc:
        json.dump({"error": str(exc), "type": type(exc).__name__}, sys.stderr, indent=2)
        sys.stderr.write("\n")
        return 1

    json.dump(snapshot, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
