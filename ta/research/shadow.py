"""Shadow mode: record what the tilt *would* have set, and change nothing.

This is the step that turns a backtest claim into evidence. Every hour it
computes the tilt, works out the `target_base` that would follow from it,
reads what the live config actually says, and stores the pair. Nothing is
proposed and nothing is written to config-manager.

Why this cannot be skipped: the backtest estimate (0.343 bp/bar, 95% CI
[0.05, 0.63]) comes from candles resampled after the fact. Shadow mode
samples the signal the way production would — on a trailing window, at a
fixed cadence, with whatever data had actually arrived by then. If those two
disagree, the difference is the part of the backtest that was an artifact of
hindsight, and it is much cheaper to discover here than after a live write.

Scoring uses the same `signal_value.evaluate` as the offline work, on the
recorded tilts and the mids recorded beside them. Two guards make that
honest:

- Rows are scored against the *next recorded row*, and only when the two are
  one cadence apart. A service restart leaves a hole, and a forward return
  measured across it would span hours instead of one — the same
  variable-horizon trap the `consecutive_only` control exists for.
- The tilt stored is the one computed at the time. It is never recomputed
  from newer candles, which would quietly leak hindsight back in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

import ta
from ..analysis.tilt import DEFAULT_CAP, implied_target_base
from ..analysis.tilt import run as run_tilt
from ..errors import SourceError, TAError
from ..ohlc import read_ohlc
from ..signal_value import evaluate, forward_returns

COLLECTION = "tilt_shadow"

LOOKBACK_BARS = 2000

REQUIRED_DAYS = 28


@dataclass(frozen=True)
class ShadowStore:
    collection: Any

    def ensure_indexes(self) -> None:
        self.collection.create_index(
            [("market", 1), ("source", 1), ("bar_time", 1)],
            unique=True,
            name="ta_shadow_key",
        )

    def put(self, row: Mapping[str, Any]) -> None:
        key = {
            "market": row["market"], "source": row["source"], "bar_time": row["bar_time"]
        }
        self.collection.replace_one(key, dict(row), upsert=True)

    def rows(self, market: str, source: str, limit: int = 5000) -> list[dict]:
        cursor = (
            self.collection.find({"market": market, "source": source})
            .sort("bar_time", 1)
            .limit(int(limit))
        )
        return list(cursor)

    def markets(self) -> list[tuple[str, str]]:
        seen = set()
        for doc in self.collection.find({}, {"market": 1, "source": 1}):
            seen.add((doc.get("market"), doc.get("source")))
        return sorted(m for m in seen if all(m))

    def count(self) -> int:
        return self.collection.count_documents({})


def observe(
    market: str,
    source: str,
    *,
    config_market: str | None = None,
    interval: str = "1H",
    cap: float = DEFAULT_CAP,
    config_client=None,
    store: ShadowStore | None = None,
    now: datetime | None = None,
) -> dict:
    """Take one observation. Returns the row; stores it when a store is given.

    The bar the tilt is stamped on is the last *complete* candle:
    `read_ohlc` plus `drop_incomplete_bar`, so a still-forming bar can never
    be recorded as if it were final.

    `config_market` exists because the two systems disagree about names. The
    candles market-data-collector writes for nobitex are `USDTIRT` (Toman);
    the config the market maker actually trades is `USDTRLS` (Rial). Same
    book, two names, and looking up the wrong one silently yields no config
    at all rather than an error. Ramzinex uses `USDTIRR` on both sides.
    """
    frame = read_ohlc(market, source, interval, limit=LOOKBACK_BARS)
    if frame is None:
        raise SourceError(f"no candles for {market}/{source} {interval}")
    frame, info = ta.normalize_ohlcv(frame)
    frame, dropped = ta.drop_incomplete_bar(frame, info.interval_sec, now=now)
    if frame.empty:
        raise SourceError("no complete candle to observe")

    verdict = run_tilt(frame, cap=cap)
    tilt = verdict.scalars.get("tilt")

    config_market = config_market or market
    live: dict[str, Any] = {}
    if config_client is not None:
        try:
            market_config = config_client.get_market(source, config_market)
            if market_config is not None:
                live = dict(market_config.config)
                live["_version"] = market_config.version
        except TAError:
            pass

    min_base, max_base = live.get("min_base"), live.get("max_base")
    implied = (
        implied_target_base(tilt, min_base, max_base, cap)
        if tilt is not None and min_base is not None
        else None
    )

    row = {
        "market": market,
        "source": source,
        "config_market": config_market,
        "interval": interval,
        "bar_time": frame.index[-1].to_pydatetime(),
        "observed_at": (now or datetime.now(timezone.utc)),
        "tilt": tilt,
        "cap": cap,
        "ma_spread_bps": verdict.scalars.get("ma_spread_bps"),
        "mid": float(frame["close"].iloc[-1]),
        "implied_target_base": implied,
        "live_target_base": live.get("target_base"),
        "live_min_base": min_base,
        "live_max_base": max_base,
        "live_config_version": live.get("_version"),
        "live_inventory": None,
        "dropped_forming_bar": bool(dropped),
        "warnings": list(verdict.warnings),
    }
    if store is not None:
        store.put(row)
    return row


def observe_all(watchlist: Iterable[Sequence[str]], **kwargs) -> dict[str, Any]:
    """Observe every watched market, surviving individual failures.

    Entries are `(market, source)` or `(market, source, config_market)`.
    """
    recorded, failures = [], {}
    for entry in watchlist:
        market, source = entry[0], entry[1]
        config_market = entry[2] if len(entry) > 2 else None
        try:
            row = observe(market, source, config_market=config_market, **kwargs)
            recorded.append({
                "market": market, "source": source,
                "bar_time": row["bar_time"], "tilt": row["tilt"],
                "implied_target_base": row["implied_target_base"],
                "live_target_base": row["live_target_base"],
            })
        except Exception as exc:
            failures[f"{market}/{source}"] = f"{type(exc).__name__}: {exc}"
    return {"recorded": recorded, "failures": failures}


def score(rows: Sequence[Mapping[str, Any]], *, cadence_sec: float = 3600.0) -> dict:
    """Score recorded tilts against the returns that followed them.

    Returns the live-sampled estimate to hold against the backtest's
    0.343 bp/bar, plus the coverage needed to judge whether it is time to
    look at all.
    """
    usable = [
        r for r in rows
        if r.get("tilt") is not None and r.get("mid") and r.get("bar_time")
    ]
    if len(usable) < 3:
        return {"rows": len(rows), "scored": 0, "verdict": "not enough observations"}

    usable = sorted(usable, key=lambda r: r["bar_time"])
    times = pd.DatetimeIndex([pd.Timestamp(r["bar_time"]) for r in usable])
    if times.tz is None:
        times = times.tz_localize("UTC")
    tilts = np.array([float(r["tilt"]) for r in usable])
    mids = np.array([float(r["mid"]) for r in usable])

    fwd = forward_returns(mids, 1)
    gaps = np.diff(times.as_unit("ns").asi8) / 1e9
    adjacent = np.concatenate([np.abs(gaps - cadence_sec) <= 0.25 * cadence_sec, [False]])
    scored_fwd = np.where(adjacent, fwd, np.nan)

    span_days = (times[-1] - times[0]).total_seconds() / 86400.0
    result = evaluate(tilts, scored_fwd, bar_seconds=cadence_sec)

    out = {
        "rows": len(rows),
        "usable": len(usable),
        "scored": int(np.isfinite(scored_fwd).sum()),
        "dropped_gap_pairs": int(len(usable) - 1 - adjacent.sum()),
        "span_days": round(span_days, 2),
        "first": times[0].isoformat(),
        "last": times[-1].isoformat(),
        "ready": span_days >= REQUIRED_DAYS,
        "required_days": REQUIRED_DAYS,
        "backtest_excess_bps": 0.343,
        "backtest_ci_bps": [0.053, 0.633],
    }
    if result is None:
        out["verdict"] = (
            f"only {out['scored']} scored pairs; need ~100 before the estimate "
            f"means anything"
        )
        return out

    out.update({
        "live_excess_bps": result.excess_bps,
        "live_ic": result.ic,
        "live_t": result.ic_t,
        "live_turnover": result.turnover,
        "live_hit_rate": result.hit_rate,
    })
    if not out["ready"]:
        out["verdict"] = (
            f"{span_days:.1f} of {REQUIRED_DAYS} days recorded — keep collecting; "
            f"the estimate so far is {result.excess_bps:+.3f} bp/bar"
        )
    elif result.excess_bps < out["backtest_ci_bps"][0]:
        out["verdict"] = (
            f"live {result.excess_bps:+.3f} bp/bar is BELOW the backtest's lower "
            f"bound {out['backtest_ci_bps'][0]}: the proposal should be dropped"
        )
    else:
        out["verdict"] = (
            f"live {result.excess_bps:+.3f} bp/bar is within the backtest interval; "
            f"step 10 may proceed"
        )
    return out


_store: ShadowStore | None = None


def default_store() -> ShadowStore:
    global _store
    if _store is None:
        from decouple import config as env

        try:
            from pymongo import MongoClient

            client = MongoClient(env("MONGO_URI"), serverSelectionTimeoutMS=5000)
            db = client[env("RESEARCH_DB_NAME", default="ta_research")]
            store = ShadowStore(db[COLLECTION])
            store.ensure_indexes()
        except Exception as exc:
            raise SourceError(f"cannot reach the shadow store: {exc}") from exc
        _store = store
    return _store


def watchlist_from_env() -> list[tuple[str, str, str]]:
    """`SHADOW_MARKETS="<market>:<source>[:<config_market>], ..."`.

    The third field covers the naming split between the candle store and the
    trading config, e.g. `USDTIRT:nobitex:USDTRLS`. Omit it when both agree,
    as on ramzinex.
    """
    from decouple import config as env

    out = []
    for chunk in str(env("SHADOW_MARKETS", default="")).split(","):
        parts = [p.strip() for p in chunk.split(":") if p.strip()]
        if len(parts) >= 2:
            market, source = parts[0], parts[1]
            out.append((market, source, parts[2] if len(parts) > 2 else market))
    return out
