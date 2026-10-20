"""Always-on TA snapshots for risk. Compute here; GET /snapshot never does.

Risk-aggregator reads this service. It does not recompute RSI/MACD/vol.
That only works if a writer is always on, and the read path never touches
OHLC. Architecture (approved 2026-08-26):

- Universe: config-manager markets with is_enable=true.
- Cadence: each closed 1H bar. Extra 15Min only for the FX tapes
  USDTRLS / USDTIRR / USDTTMN (USDTIRT is the nobitex candle name for
  the same USDTRLS book).
- Set is fixed: rsi:14, stoch_rsi 14/3/3, macd 12/26/9, volume_zscore,
  volume_sma, natr, plus analyses volatility / trend / regime.
- Store: indicators_data natural-key upsert of the last closed bar, plus
  one latest-analysis document per (source, market, interval).
- Read: Mongo + a short in-process cache. If computed_at is older than
  two bars, the payload is marked stale. The caller fail-softs.

This path does not write config-manager.

Nobitex candles are *IRT; trading configs are *RLS. Snapshots are keyed
by the config name so risk can look up USDTRLS. OHLC is fetched under
the IRT alias. GET /snapshot accepts either name.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

import pandas as pd
from decouple import config as env

from . import pipeline
from .configclient import ConfigManagerClient
from .errors import SourceError, TAError
from .frame import drop_incomplete_bar, normalize_ohlcv
from .ohlc import normalize_interval, read_ohlc

logger = logging.getLogger(__name__)

SNAPSHOT_INDICATORS = "rsi:period=14,stoch_rsi,macd,volume_zscore,volume_sma,natr"
SNAPSHOT_ANALYSES = "volatility,trend,regime"

FX_15MIN_MARKETS = frozenset({"USDTRLS", "USDTIRT", "USDTIRR", "USDTTMN"})

BAR_SECONDS = {
    "1Min": 60,
    "5Min": 300,
    "15Min": 900,
    "1H": 3600,
    "4H": 4 * 3600,
    "1D": 24 * 3600,
}

DEFAULT_SOURCES = "nobitex,ramzinex,wallex,bitpin"
DEFAULT_POLL_SEC = 60.0
DEFAULT_CACHE_TTL_SEC = 15.0
STALE_BARS = 2
LATEST_COLLECTION = "ta_analysis_latest"


_OHLC_SUFFIX: dict[str, tuple[str, str]] = {
    "nobitex": ("RLS", "IRT"),
    "wallex": ("TMN", "IRT"),
}


def ohlc_market(source: str, config_market: str) -> str:
    """Candle name for a trading-config name."""
    rename = _OHLC_SUFFIX.get(source)
    if rename is None:
        return config_market
    config_suffix, candle_suffix = rename
    if config_market.endswith(config_suffix):
        return config_market[: -len(config_suffix)] + candle_suffix
    return config_market


def sibling_market(source: str, market: str) -> str | None:
    """The other name for the same book on this source, or None."""
    rename = _OHLC_SUFFIX.get(source)
    if rename is None:
        return None
    config_suffix, candle_suffix = rename
    if market.endswith(config_suffix):
        return market[: -len(config_suffix)] + candle_suffix
    if market.endswith(candle_suffix):
        return market[: -len(candle_suffix)] + config_suffix
    return None


def extra_15min(config_market: str) -> bool:
    return config_market.upper() in FX_15MIN_MARKETS


def intervals_for(config_market: str) -> tuple[str, ...]:
    if extra_15min(config_market):
        return ("1H", "15Min")
    return ("1H",)


def bar_seconds(interval: str) -> int:
    return BAR_SECONDS[normalize_interval(interval)]


def is_stale(
    computed_at: datetime,
    interval: str,
    now: datetime | None = None,
    *,
    bars: int = STALE_BARS,
) -> bool:
    now = now or datetime.now(timezone.utc)
    age = _as_utc(now) - _as_utc(computed_at)
    return age > timedelta(seconds=bar_seconds(interval) * bars)


def _as_utc(value) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _iso(value) -> str:
    return _as_utc(value).isoformat()


def last_closed_asof(
    market: str,
    source: str,
    interval: str,
    *,
    now: datetime | None = None,
    reader=read_ohlc,
) -> datetime:
    """Timestamp of the last complete bar. Cheap: a handful of candles."""
    interval = normalize_interval(interval)
    raw = reader(market, source, interval, limit=8)
    if raw is None:
        raise SourceError(
            f"no candles for market={market!r} source={source!r} interval={interval!r}"
        )
    frame, info = normalize_ohlcv(raw)
    frame, _dropped = drop_incomplete_bar(frame, info.interval_sec, now=now)
    if frame.empty:
        raise SourceError("no complete candle to snapshot")
    return _as_utc(frame.index[-1])


def _flatten_latest(latest: Mapping[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name, payload in latest.items():
        if isinstance(payload, Mapping):
            value = payload.get("value")
        else:
            value = payload
        if value is None:
            out[name] = None
        else:
            try:
                out[name] = float(value)
            except (TypeError, ValueError):
                out[name] = None
    return out


def _compact_analyses(analyses: Mapping[str, Any]) -> dict[str, dict]:
    compact = {}
    for name, payload in analyses.items():
        if not isinstance(payload, Mapping):
            continue
        compact[name] = {
            "asof": payload.get("asof"),
            "scalars": dict(payload.get("scalars") or {}),
            "warnings": list(payload.get("warnings") or []),
        }
    return compact


def public_payload(
    doc: Mapping[str, Any], *, now: datetime | None = None
) -> dict[str, Any]:
    interval = doc["interval"]
    computed_at = _as_utc(doc["computed_at"])
    return {
        "source": doc["source"],
        "market": doc["market"],
        "ohlc_market": doc.get("ohlc_market") or doc["market"],
        "interval": interval,
        "asof": _iso(doc["asof"]),
        "computed_at": _iso(computed_at),
        "stale": is_stale(computed_at, interval, now),
        "indicators": dict(doc.get("indicators") or {}),
        "analyses": dict(doc.get("analyses") or {}),
        "dropped_forming_bar": bool(doc.get("dropped_forming_bar")),
    }


@dataclass
class SnapshotStore:
    collection: Any

    def ensure_indexes(self) -> None:
        self.collection.create_index(
            [("source", 1), ("market", 1), ("interval", 1)],
            unique=True,
            name="ta_snapshot_latest_key",
        )

    def put(self, doc: Mapping[str, Any]) -> None:
        key = {k: doc[k] for k in ("source", "market", "interval")}
        self.collection.replace_one(key, dict(doc), upsert=True)

    def get(self, source: str, market: str, interval: str) -> dict | None:
        return self.collection.find_one(
            {"source": source, "market": market, "interval": interval},
            {"_id": 0},
        )


@dataclass
class ProcessCache:
    ttl_sec: float = DEFAULT_CACHE_TTL_SEC
    _data: dict[tuple[str, str, str], tuple[float, dict]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def get(self, source: str, market: str, interval: str) -> dict | None:
        key = (source, market, interval)
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            fetched, doc = entry
            if time.monotonic() - fetched >= self.ttl_sec:
                self._data.pop(key, None)
                return None
            return dict(doc)

    def put(self, source: str, market: str, interval: str, doc: Mapping[str, Any]) -> None:
        key = (source, market, interval)
        with self._lock:
            self._data[key] = (time.monotonic(), dict(doc))

    def invalidate(self, source: str, market: str, interval: str) -> None:
        key = (source, market, interval)
        with self._lock:
            self._data.pop(key, None)
            alias = sibling_market(source, market)
            if alias:
                self._data.pop((source, alias, interval), None)


_default_store: SnapshotStore | None = None
_default_cache: ProcessCache | None = None


def default_store() -> SnapshotStore:
    global _default_store
    if _default_store is None:
        try:
            from pymongo import MongoClient

            client = MongoClient(env("MONGO_URI"), serverSelectionTimeoutMS=5000)
            collection = client[env("DB_NAME")][
                env("ANALYSIS_LATEST_COLLECTION_NAME", default=LATEST_COLLECTION)
            ]
            store_ = SnapshotStore(collection)
            store_.ensure_indexes()
        except Exception as exc:
            raise SourceError(f"cannot reach the snapshot store: {exc}") from exc
        _default_store = store_
    return _default_store


def default_cache() -> ProcessCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = ProcessCache(
            ttl_sec=float(env("SNAPSHOT_CACHE_TTL_SEC", default=DEFAULT_CACHE_TTL_SEC))
        )
    return _default_cache


def read_snapshot(
    source: str,
    market: str,
    interval: str,
    *,
    store: SnapshotStore | None = None,
    cache: ProcessCache | None = None,
    now: datetime | None = None,
) -> dict | None:
    """Latest stored snapshot. Never computes. None if nothing has been written."""
    interval = normalize_interval(interval)
    cache = cache if cache is not None else default_cache()
    store = store if store is not None else default_store()

    cached = cache.get(source, market, interval)
    if cached is not None:
        return public_payload(cached, now=now)

    doc = store.get(source, market, interval)
    if doc is None:
        alias = sibling_market(source, market)
        if alias:
            doc = store.get(source, alias, interval)
    if doc is None:
        return None
    cache.put(source, market, interval, doc)
    return public_payload(doc, now=now)


def capture_one(
    source: str,
    config_market: str,
    interval: str,
    *,
    snapshot_store: SnapshotStore,
    cache: ProcessCache | None = None,
    now: datetime | None = None,
    skip_if_current: bool = True,
    run=pipeline.run,
    reader=read_ohlc,
) -> dict[str, Any]:
    """Compute and persist one (source, market, interval). Idempotent on asof."""
    interval = normalize_interval(interval)
    now = now or datetime.now(timezone.utc)
    candle = ohlc_market(source, config_market)

    asof = last_closed_asof(candle, source, interval, now=now, reader=reader)
    existing = snapshot_store.get(source, config_market, interval)
    if (
        skip_if_current
        and existing is not None
        and _as_utc(existing["asof"]) == asof
    ):
        return {
            "status": "skipped",
            "source": source,
            "market": config_market,
            "interval": interval,
            "asof": _iso(asof),
        }

    snapshot = run(
        candle,
        source,
        interval,
        indicators=SNAPSHOT_INDICATORS,
        analyses=SNAPSHOT_ANALYSES,
        save=True,
        drop_forming=True,
        latest_bar_only=True,
    )

    doc = {
        "source": source,
        "market": config_market,
        "ohlc_market": candle,
        "interval": interval,
        "asof": asof,
        "computed_at": _as_utc(now),
        "indicators": _flatten_latest((snapshot.get("indicators") or {}).get("latest") or {}),
        "analyses": _compact_analyses(snapshot.get("analyses") or {}),
        "dropped_forming_bar": bool(snapshot.get("dropped_forming_bar")),
    }
    snapshot_store.put(doc)
    if cache is not None:
        cache.invalidate(source, config_market, interval)
        cache.put(source, config_market, interval, doc)
    return {
        "status": "ok",
        "source": source,
        "market": config_market,
        "interval": interval,
        "asof": _iso(asof),
    }


def capture_enabled(
    *,
    client: ConfigManagerClient,
    snapshot_store: SnapshotStore,
    sources: list[str],
    cache: ProcessCache | None = None,
    now: datetime | None = None,
    run=pipeline.run,
    reader=read_ohlc,
) -> dict[str, Any]:
    """One pass over every enabled market. Never raises for a single failure."""
    recorded: list[dict] = []
    skipped: list[dict] = []
    failures: dict[str, str] = {}

    for source in sources:
        try:
            markets = client.list_markets(source)
        except Exception as exc:
            failures[source] = f"{type(exc).__name__}: {exc}"
            logger.warning("snapshot list_markets %s failed: %s", source, exc)
            continue
        source_attempted = 0
        source_failed = 0
        for mc in markets:
            if not mc.is_enabled:
                continue
            for interval in intervals_for(mc.market):
                source_attempted += 1
                key = f"{source}/{mc.market}/{interval}"
                try:
                    result = capture_one(
                        source,
                        mc.market,
                        interval,
                        snapshot_store=snapshot_store,
                        cache=cache,
                        now=now,
                        run=run,
                        reader=reader,
                    )
                except Exception as exc:
                    failures[key] = f"{type(exc).__name__}: {exc}"
                    source_failed += 1
                    logger.warning("snapshot %s failed: %s", key, exc)
                    continue
                if result.get("status") == "skipped":
                    skipped.append(result)
                else:
                    recorded.append(result)

        if source_attempted and source_failed == source_attempted:
            failures[f"{source}/*"] = (
                f"ALL {source_attempted} snapshot(s) failed"
            )
            logger.error(
                "snapshot source %s is DARK: all %d market/interval "
                "combination(s) failed -- consumers will see no TA for this "
                "venue at all, which reads downstream as no risk",
                source,
                source_attempted,
            )

    return {
        "recorded": recorded,
        "skipped": skipped,
        "failures": failures,
        "ok": len(recorded),
        "n_skipped": len(skipped),
        "n_failed": len(failures),
    }


class SnapshotScheduler:
    """Wake, snapshot due bars, sleep. A gap is information; no backfill."""

    def __init__(
        self,
        job: Callable[[], dict],
        poll_sec: float = DEFAULT_POLL_SEC,
        initial_delay_sec: float = 20.0,
    ):
        self.job = job
        self.poll_sec = float(poll_sec)
        self.initial_delay_sec = float(initial_delay_sec)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_run: datetime | None = None
        self.last_result: dict | None = None
        self.last_error: str | None = None
        self.runs = 0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop, name="snapshot-scheduler", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        if self._stop.wait(self.initial_delay_sec):
            return
        while not self._stop.is_set():
            self.run_once()
            self._stop.wait(self.poll_sec)

    def run_once(self) -> dict | None:
        try:
            self.last_result = self.job()
            self.last_error = None
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("snapshot pass failed: %s", self.last_error)
        self.last_run = datetime.now(timezone.utc)
        self.runs += 1
        return self.last_result

    def status(self) -> dict:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "poll_sec": self.poll_sec,
            "runs": self.runs,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "last_error": self.last_error,
            "last_result": self.last_result,
        }


_scheduler: SnapshotScheduler | None = None


def snapshot_disabled() -> bool:
    return os.environ.get("SNAPSHOT_DISABLE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def configured_sources() -> list[str]:
    raw = env("SNAPSHOT_SOURCES", default=DEFAULT_SOURCES)
    return [part.strip() for part in str(raw).split(",") if part.strip()]


def default_scheduler() -> SnapshotScheduler | None:
    """Env-configured writer, started once per process. None when disabled."""
    global _scheduler
    if snapshot_disabled():
        return None
    if _scheduler is None:
        def job() -> dict:
            return capture_enabled(
                client=ConfigManagerClient(),
                snapshot_store=default_store(),
                sources=configured_sources(),
                cache=default_cache(),
            )

        _scheduler = SnapshotScheduler(
            job,
            poll_sec=float(env("SNAPSHOT_POLL_SEC", default=DEFAULT_POLL_SEC)),
        )
        _scheduler.start()
    return _scheduler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ta.snapshot",
        description="Run one always-on snapshot pass (no HTTP).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="one pass over enabled markets, then exit (default)",
    )
    args = parser.parse_args(argv)
    del args
    try:
        result = capture_enabled(
            client=ConfigManagerClient(),
            snapshot_store=default_store(),
            sources=configured_sources(),
            cache=default_cache(),
        )
    except TAError as exc:
        json.dump({"error": str(exc), "type": type(exc).__name__}, sys.stderr, indent=2)
        sys.stderr.write("\n")
        return 1
    json.dump(result, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0 if not result["failures"] else 2


if __name__ == "__main__":
    sys.exit(main())
