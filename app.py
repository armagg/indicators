"""Thin HTTP adapter over the `ta` package.

Two endpoints and no business logic: `GET /indicators` is the machine-readable
catalogue, `GET /indicator` computes (and by default persists) indicators for
one market. Everything interesting lives in `ta/`; this file only translates
query strings and exceptions.

Breaking change from the legacy service, on purpose: `source` is now required.
The old endpoint mixed candles from every venue that traded the market name
(ARCHITECTURE.md, data contract). Existing callers must add `&source=...`.
"""

from __future__ import annotations

from flask import Flask, jsonify, request

import ta
from ta import pipeline, store
from ta.analysis import list_analyses
from ta.errors import SourceError, TAError
from ta.ohlc import read_ohlc
from ta.registry import BoolParam, ChoiceParam, IntParam

app = Flask(__name__)

_SAVE = BoolParam("save", True)
_DROPNA = BoolParam("dropna", False)
_LATEST = BoolParam("latest", False)
_LIMIT = IntParam("limit", None, minimum=2, maximum=100_000)
_WARMUP = ChoiceParam("warmup", "mask", choices=("mask", "trim", "keep"))


@app.errorhandler(TAError)
def _bad_request(exc: TAError):
    return jsonify({"error": str(exc), "type": type(exc).__name__}), 400


@app.errorhandler(SourceError)
def _upstream_failed(exc: SourceError):
    return jsonify({"error": str(exc), "type": "SourceError"}), 502


@app.get("/health")
def health():
    return jsonify({"status": "ok", "indicators": len(ta.list_indicators())})


@app.get("/indicators")
def indicator_catalogue():
    """Every registered indicator: params, outputs, warmup, caveats."""
    return jsonify(ta.catalogue())


@app.get("/indicator")
def get_indicator_endpoint():
    args = request.args.to_dict()

    symbol = args.pop("symbol", None) or args.pop("market", None) or args.pop("market_name", None)
    source = args.pop("source", None)
    interval = args.pop("interval", None)
    if not symbol or not interval:
        return jsonify({"error": "symbol and interval parameters are required"}), 400
    if not source:
        return (
            jsonify(
                {
                    "error": "source is required (e.g. source=nobitex): the same "
                    "market_name exists for several venues and mixing them "
                    "produces meaningless indicators"
                }
            ),
            400,
        )

    save_results = _SAVE.coerce(args.pop("save", True))
    dropna = _DROPNA.coerce(args.pop("dropna", False))
    latest_only = _LATEST.coerce(args.pop("latest", False))
    warmup = _WARMUP.coerce(args.pop("warmup", "mask"))
    limit = args.pop("limit", None)
    limit = _LIMIT.coerce(limit) if limit is not None else None
    start = args.pop("start", None)
    end = args.pop("end", None)

    # Two request shapes: `indicators=` takes the compact batch form
    # ("rsi:period=21,macd,atr"); `indicator=` is the legacy single-indicator
    # form whose parameters arrive as loose query args (`&period=21`).
    batch = args.pop("indicators", None)
    single = args.pop("indicator", None)
    if batch and (single or args):
        return (
            jsonify(
                {
                    "error": "use either indicators=<compact list> alone, or "
                    "indicator=<name> with loose parameter args -- not both"
                }
            ),
            400,
        )
    if batch:
        requests = batch
    elif single:
        requests = [{"name": single, **args}]
    else:
        return jsonify({"error": "indicator (or indicators) parameter is required"}), 400

    frame = read_ohlc(symbol, source, interval, limit=limit, start=start, end=end)
    if frame is None:
        return (
            jsonify({"error": f"no candles for {symbol!r} source={source!r} interval={interval!r}"}),
            404,
        )

    result = ta.compute(frame, requests, warmup=warmup)

    saved = store.save(result, symbol=symbol, source=source, interval=interval) if save_results else 0

    payload = {
        "latest": result.latest(),
        "meta": result.meta(),
        "saved": saved,
    }
    if not latest_only:
        payload["data"] = result.to_records(dropna=dropna)
    return jsonify(payload), 200


@app.get("/analysis")
def get_analysis_endpoint():
    """Combined snapshot for one market: indicators and/or analyses.

    Thin pass-through to `ta.pipeline.run`. Unlike `/indicator`, nothing is
    persisted unless `save=true` is passed explicitly -- this is the
    read-the-market endpoint.
    """
    args = request.args.to_dict()
    symbol = args.pop("symbol", None) or args.pop("market", None)
    source = args.pop("source", None)
    interval = args.pop("interval", None)
    if not symbol or not source or not interval:
        return jsonify({"error": "symbol, source, and interval are required"}), 400

    analyses = args.pop("analyses", None)
    indicators = args.pop("indicators", None)
    if not analyses and not indicators:
        return (
            jsonify(
                {
                    "error": "supply analyses and/or indicators; known analyses: "
                    + ", ".join(list_analyses())
                }
            ),
            400,
        )
    limit = args.pop("limit", None)
    start = args.pop("start", None)
    end = args.pop("end", None)
    save = _SAVE.coerce(args.pop("save", False))
    warmup = _WARMUP.coerce(args.pop("warmup", "mask"))
    series = BoolParam("series", False).coerce(args.pop("series", False))
    if args:
        return jsonify({"error": f"unknown parameter(s): {', '.join(sorted(args))}"}), 400

    snapshot = pipeline.run(
        symbol,
        source,
        interval,
        indicators=indicators,
        analyses=analyses,
        limit=_LIMIT.coerce(limit) if limit is not None else None,
        start=start,
        end=end,
        save=save,
        warmup=warmup,
        include_series=series,
    )
    return jsonify(snapshot), 200


if __name__ == "__main__":
    app.run(debug=True)
