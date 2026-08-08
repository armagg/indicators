"""Support and resistance: static swing pivots plus dynamic channel levels.

Static levels are confirmed fractal pivots on high/low, clustered when they
sit within a few basis points of each other, then classified relative to the
last close. Dynamic levels reuse Donchian / Bollinger / Keltner / VWAP from
the indicator library — this module does not reimplement band math.

Insight only: no overlay, no ConfigDelta, no proposal. The desk question is
"where is price relative to structure", not "which MM knob to turn".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..engine import compute_series
from .types import AnalysisResult

_MAX_EVENTS = 20


def _bps(a: float, b: float) -> float:
    """Distance from `b` to `a` in basis points of `b`."""
    return (a - b) / b * 1e4


def _find_swing_pivots(
    high: np.ndarray,
    low: np.ndarray,
    index: pd.DatetimeIndex,
    left: int,
    right: int,
) -> list[dict]:
    """Williams-style fractal pivots. Confirmed only once `right` bars exist
    after the candidate, so the last `right` bars never emit a pivot."""
    n = len(high)
    pivots: list[dict] = []
    if n < left + right + 1:
        return pivots

    for i in range(left, n - right):
        h_window = high[i - left : i + right + 1]
        l_window = low[i - left : i + right + 1]
        # Leftmost peak/trough in the window — plateaus still register once.
        if high[i] == h_window.max() and int(np.argmax(h_window)) == left:
            pivots.append(
                {
                    "time": index[i],
                    "kind": "resistance",
                    "price": float(high[i]),
                    "bar": i,
                }
            )
        if low[i] == l_window.min() and int(np.argmin(l_window)) == left:
            pivots.append(
                {
                    "time": index[i],
                    "kind": "support",
                    "price": float(low[i]),
                    "bar": i,
                }
            )
    return pivots


def _cluster_pivots(
    pivots: list[dict],
    cluster_bps: float,
    last_bar: int,
) -> list[dict]:
    """Merge pivots whose prices sit within `cluster_bps` of the cluster mean.

    Kind is provisional here (pivot origin); final support/resistance is
    assigned from last close after clustering.
    """
    if not pivots:
        return []

    ordered = sorted(pivots, key=lambda p: p["price"])
    clusters: list[dict] = []
    group = [ordered[0]]

    def flush(members: list[dict]) -> None:
        prices = [m["price"] for m in members]
        center = float(np.mean(prices))
        last = max(members, key=lambda m: m["bar"])
        age_bars = last_bar - last["bar"]
        clusters.append(
            {
                "price": center,
                "touches": len(members),
                "age_bars": age_bars,
                "last_time": last["time"],
            }
        )

    for pivot in ordered[1:]:
        center = float(np.mean([m["price"] for m in group]))
        if abs(_bps(pivot["price"], center)) <= cluster_bps:
            group.append(pivot)
        else:
            flush(group)
            group = [pivot]
    flush(group)
    return clusters


def _score_level(touches: int, age_bars: int, dist_bps: float) -> float:
    """Higher is stronger: more touches, fresher, closer to price.

    Age decays over ~100 bars; distance softens past ~50 bps. Touches dominate
    so a level that has held three times beats a lonely recent wick.
    """
    age_factor = 1.0 / (1.0 + age_bars / 100.0)
    dist_factor = 1.0 / (1.0 + abs(dist_bps) / 50.0)
    return float(touches) * age_factor * dist_factor


def _classify_and_rank(
    clusters: list[dict],
    close: float,
    max_levels: int,
) -> tuple[list[dict], list[dict]]:
    supports: list[dict] = []
    resistances: list[dict] = []
    for cluster in clusters:
        price = cluster["price"]
        if price < close:
            kind = "support"
        elif price > close:
            kind = "resistance"
        else:
            continue
        dist = _bps(price, close)
        level = {
            "time": cluster["last_time"].isoformat(),
            "event": "static_level",
            "kind": kind,
            "price": price,
            "touches": cluster["touches"],
            "age_bars": cluster["age_bars"],
            "dist_bps": dist,
            "strength": _score_level(cluster["touches"], cluster["age_bars"], dist),
        }
        (supports if kind == "support" else resistances).append(level)

    supports.sort(key=lambda L: L["strength"], reverse=True)
    resistances.sort(key=lambda L: L["strength"], reverse=True)
    return supports[:max_levels], resistances[:max_levels]


def _last(series: pd.Series | None) -> float | None:
    if series is None:
        return None
    valid = series.dropna()
    return float(valid.iloc[-1]) if not valid.empty else None


def run(
    frame: pd.DataFrame,
    swing_left: int = 2,
    swing_right: int = 2,
    max_levels: int = 5,
    cluster_bps: float = 15.0,
    donchian_window: int = 20,
    bb_window: int = 20,
    bb_std: float = 2.0,
    kc_window: int = 20,
    kc_atr: int = 10,
    kc_multiplier: float = 2.0,
    vwap_anchor: str = "day",
) -> AnalysisResult:
    """Static swing S/R plus dynamic channel levels for one OHLCV frame."""
    warnings: list[str] = []
    swing_left = int(swing_left)
    swing_right = int(swing_right)
    max_levels = int(max_levels)
    cluster_bps = float(cluster_bps)
    bb_std = float(bb_std)
    kc_multiplier = float(kc_multiplier)

    if swing_left < 1 or swing_right < 1:
        raise ValueError("swing_left and swing_right must be >= 1")
    if max_levels < 1:
        raise ValueError("max_levels must be >= 1")
    if cluster_bps < 0:
        raise ValueError("cluster_bps must be >= 0")

    if len(frame) == 0:
        return AnalysisResult(
            name="levels",
            params={
                "swing_left": swing_left,
                "swing_right": swing_right,
                "max_levels": max_levels,
                "cluster_bps": cluster_bps,
                "donchian_window": donchian_window,
                "bb_window": bb_window,
                "bb_std": bb_std,
                "kc_window": kc_window,
                "kc_atr": kc_atr,
                "kc_multiplier": kc_multiplier,
                "vwap_anchor": vwap_anchor,
            },
            asof=None,
            warnings=["empty frame"],
        )

    close = float(frame["close"].iloc[-1])
    last_bar = len(frame) - 1
    has_hl = {"high", "low"} <= set(frame.columns)

    supports: list[dict] = []
    resistances: list[dict] = []
    events: list[dict] = []

    if not has_hl:
        warnings.append("no high/low columns: static pivots and channel levels skipped")
    else:
        high = frame["high"].to_numpy(dtype="float64")
        low = frame["low"].to_numpy(dtype="float64")
        pivots = _find_swing_pivots(high, low, frame.index, swing_left, swing_right)
        if not pivots:
            warnings.append(
                f"no confirmed swing pivots with left={swing_left} right={swing_right}"
            )
        else:
            clusters = _cluster_pivots(pivots, cluster_bps, last_bar)
            supports, resistances = _classify_and_rank(clusters, close, max_levels)
            # Events: strongest levels first, then chronological among the tail.
            ranked = sorted(
                supports + resistances,
                key=lambda L: L["strength"],
                reverse=True,
            )
            events = ranked[:_MAX_EVENTS]

    # Nearest = closest in price: highest support below close, lowest resistance above.
    nearest_support = max((s["price"] for s in supports), default=None)
    nearest_resistance = min((r["price"] for r in resistances), default=None)

    dist_support_bps = (
        _bps(nearest_support, close) if nearest_support is not None else None
    )
    dist_resistance_bps = (
        _bps(nearest_resistance, close) if nearest_resistance is not None else None
    )

    # Dynamic levels via registered indicators.
    dynamic_support = None
    dynamic_resistance = None
    series: dict[str, pd.Series] = {}
    bb_upper = bb_lower = kc_upper = kc_lower = vwap = None
    dc_upper = dc_lower = None

    if has_hl:
        requests = [
            f"donchian:window={donchian_window}",
            f"bbands:window={bb_window},std_dev={bb_std}",
            f"keltner:window={kc_window},atr_period={kc_atr},multiplier={kc_multiplier}",
        ]
        if "volume" in frame.columns:
            requests.append(f"vwap:anchor={vwap_anchor}")
        else:
            warnings.append("no volume column: VWAP skipped")

        computed = compute_series(frame, requests)
        dc_upper = computed[f"dc_upper_{donchian_window}"]
        dc_lower = computed[f"dc_lower_{donchian_window}"]
        bb_upper = computed[f"bb_upper_{bb_window}_{bb_std}"]
        bb_lower = computed[f"bb_lower_{bb_window}_{bb_std}"]
        kc_upper = computed[f"kc_upper_{kc_window}_{kc_atr}_{kc_multiplier}"]
        kc_lower = computed[f"kc_lower_{kc_window}_{kc_atr}_{kc_multiplier}"]

        # Prior-bar Donchian: the live channel includes this bar's own high/low,
        # so close is always inside it (see donchian indicator notes). Dynamic
        # S/R and in_range use the confirmed previous channel instead.
        prior_dc_lower = dc_lower.shift(1)
        prior_dc_upper = dc_upper.shift(1)
        dynamic_support = _last(prior_dc_lower)
        dynamic_resistance = _last(prior_dc_upper)

        series = {
            "dc_upper": prior_dc_upper,
            "dc_lower": prior_dc_lower,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "kc_upper": kc_upper,
            "kc_lower": kc_lower,
        }
        vwap_col = f"vwap_{vwap_anchor}"
        if vwap_col in computed.columns:
            vwap = computed[vwap_col]
            series["vwap"] = vwap

    in_range = None
    if dynamic_support is not None and dynamic_resistance is not None:
        in_range = dynamic_support <= close <= dynamic_resistance

    if len(frame) < swing_left + swing_right + 1:
        warnings.append(
            f"frame shorter than swing window ({swing_left}+{swing_right}+1); "
            f"static levels may be empty"
        )

    params = {
        "swing_left": swing_left,
        "swing_right": swing_right,
        "max_levels": max_levels,
        "cluster_bps": cluster_bps,
        "donchian_window": donchian_window,
        "bb_window": bb_window,
        "bb_std": bb_std,
        "kc_window": kc_window,
        "kc_atr": kc_atr,
        "kc_multiplier": kc_multiplier,
        "vwap_anchor": vwap_anchor,
    }

    scalars = {
        "last_close": close,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "dist_support_bps": dist_support_bps,
        "dist_resistance_bps": dist_resistance_bps,
        "support_count": len(supports),
        "resistance_count": len(resistances),
        "dynamic_support": dynamic_support,
        "dynamic_resistance": dynamic_resistance,
        "donchian_lower": dynamic_support,
        "donchian_upper": dynamic_resistance,
        "bb_lower": _last(bb_lower),
        "bb_upper": _last(bb_upper),
        "kc_lower": _last(kc_lower),
        "kc_upper": _last(kc_upper),
        "vwap": _last(vwap),
        "in_range": in_range,
    }

    return AnalysisResult(
        name="levels",
        params=params,
        asof=frame.index[-1],
        scalars=scalars,
        series=series,
        events=events,
        warnings=warnings,
    )
