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

import logging
import os

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


@app.get("/research/scenarios")
def research_scenarios():
    """Saved job batches. Named and versioned with the code, so re-running
    one later means the same thing it meant before."""
    from ta.research import list_scenarios

    return jsonify({"scenarios": list_scenarios()})


@app.post("/research/scenarios/<name>")
def research_run_scenario(name: str):
    """Enqueue every job in a scenario. Already-active duplicates are reused."""
    from ta.research import default_queue, expand

    try:
        specs = expand(name)
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404
    submitted = default_queue().submit_many(specs)
    return jsonify({
        "scenario": name,
        "submitted": [
            {"id": job["_id"], "market": job["market"], "source": job["source"],
             "interval": job["interval"], "family": job["family"],
             "status": job["status"], "deduplicated": job["deduplicated"]}
            for job in submitted
        ],
    }), 202


@app.post("/research/jobs")
def research_submit():
    """Submit one job, or a list of them, as JSON."""
    from ta.research import DEFAULTS, default_queue

    body = request.get_json(silent=True)
    if body is None:
        return jsonify({"error": "send a JSON job object or a list of them"}), 400
    specs = body if isinstance(body, list) else [body]
    for spec in specs:
        missing = [k for k in ("market", "source") if not spec.get(k)]
        if missing:
            return jsonify({"error": f"job missing {', '.join(missing)}"}), 400
    submitted = default_queue().submit_many([{**DEFAULTS, **s} for s in specs])
    return jsonify({"submitted": [
        {"id": j["_id"], "status": j["status"], "deduplicated": j["deduplicated"]}
        for j in submitted
    ]}), 202


@app.get("/research/jobs")
def research_list_jobs():
    from ta.research import default_queue

    queue = default_queue()
    jobs = queue.list(status=request.args.get("status"),
                      limit=int(request.args.get("limit", 100)))
    return jsonify({
        "summary": queue.summary(),
        "jobs": [
            {k: job.get(k) for k in
             ("_id", "market", "source", "interval", "family", "status",
              "submitted_at", "finished_at", "error", "headline")}
            for job in jobs
        ],
    })


@app.get("/research/jobs/<job_id>")
def research_get_job(job_id: str):
    from ta.research import default_queue

    job = default_queue().get(job_id)
    if job is None:
        return jsonify({"error": f"no job {job_id}"}), 404
    return jsonify(job)


@app.get("/research/results")
def research_results():
    """Every finished run, newest first — the comparison table.

    Reads the cache directly rather than the job list, so results submitted
    by the CLI or an earlier process show up here too.
    """
    from ta.cache import default_cache
    from ta.research import JOB_NAME

    cache = default_cache()
    family = request.args.get("family")
    rows = []
    for doc in cache.runs.find({"job": JOB_NAME}).sort("created_at", -1).limit(200):
        detail = doc.get("params_detail", {})
        if family and detail.get("family") != family:
            continue
        headline = doc.get("payload", {}).get("headline", {})
        if request.args.get("full"):
            rows.append({"params": detail, "payload": doc.get("payload")})
        else:
            rows.append({"params": detail, "headline": headline})
    return jsonify({"count": len(rows), "results": rows, "cache": cache.stats()})


@app.post("/research/shadow/record")
def research_shadow_record():
    """Take one shadow observation now, for every watched market."""
    from ta.configclient import ConfigManagerClient
    from ta.research.shadow import default_store, observe_all, watchlist_from_env

    watchlist = watchlist_from_env()
    if not watchlist:
        return jsonify({"error": "SHADOW_MARKETS is not set"}), 400
    try:
        client = ConfigManagerClient()
    except Exception:
        client = None
    return jsonify(observe_all(watchlist, store=default_store(), config_client=client))


@app.get("/research/shadow")
def research_shadow_status():
    """Coverage and the most recent observations."""
    from ta.research.scheduler import default_scheduler
    from ta.research.shadow import default_store, watchlist_from_env

    store = default_store()
    scheduler = default_scheduler()
    markets = store.markets()
    recent = {}
    for market, source in markets:
        rows = store.rows(market, source)
        recent[f"{market}/{source}"] = {
            "observations": len(rows),
            "latest": {
                k: rows[-1].get(k)
                for k in ("bar_time", "tilt", "mid", "implied_target_base",
                          "live_target_base", "live_config_version")
            } if rows else None,
        }
    return jsonify({
        "watchlist": [
            f"{entry[0]}/{entry[1]}"
            + (f" (config: {entry[2]})" if len(entry) > 2 and entry[2] != entry[0] else "")
            for entry in watchlist_from_env()
        ],
        "total_observations": store.count(),
        "markets": recent,
        "scheduler": scheduler.status() if scheduler else "not configured",
    })


@app.get("/research/shadow/score")
def research_shadow_score():
    """Live-sampled estimate, against the backtest it has to justify."""
    from ta.research.shadow import default_store, score

    store = default_store()
    market, source = request.args.get("market"), request.args.get("source")
    targets = [(market, source)] if market and source else store.markets()
    return jsonify({
        f"{m}/{s}": score(store.rows(m, s)) for m, s in targets
    })


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


@app.get("/snapshot")
def get_snapshot_endpoint():
    """Latest stored TA snapshot. Never computes — Mongo + short cache only."""
    source = request.args.get("source")
    market = request.args.get("market") or request.args.get("symbol")
    interval = request.args.get("interval")
    if not source or not market or not interval:
        return jsonify({"error": "source, market, and interval are required"}), 400

    from ta.snapshot import read_snapshot

    payload = read_snapshot(source, market, interval)
    if payload is None:
        return (
            jsonify(
                {
                    "error": "no snapshot",
                    "source": source,
                    "market": market,
                    "interval": interval,
                }
            ),
            404,
        )
    return jsonify(payload), 200


def _autostart_snapshot_writer() -> None:
    if os.environ.get("SNAPSHOT_DISABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    if (
        os.environ.get("FLASK_DEBUG") == "1"
        and os.environ.get("WERKZEUG_RUN_MAIN") != "true"
    ):
        return
    try:
        from ta.snapshot import default_scheduler

        default_scheduler()
    except Exception as exc:
        logging.getLogger("ta.snapshot").warning("snapshot writer not started: %s", exc)


_autostart_snapshot_writer()


if __name__ == "__main__":
    app.run(debug=True)
