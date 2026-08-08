"""Analyses on frames with *constructed* regimes: the verdicts must flip
where we built them to flip."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.analysis import list_analyses, run_analysis
from ta.analysis.levels import run as run_levels
from ta.analysis.regime import run as run_regime
from ta.analysis.trend import run as run_trend
from ta.analysis.volatility import run as run_volatility
from ta.errors import ParamError


def build_frame(closes: np.ndarray, spread: float = 0.001) -> pd.DataFrame:
    """OHLCV frame around a given close path, with modest H/L envelopes."""
    n = len(closes)
    index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    open_ = np.concatenate(([closes[0]], closes[:-1]))
    high = np.maximum(open_, closes) * (1 + spread)
    low = np.minimum(open_, closes) * (1 - spread)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": closes,
         "volume": np.full(n, 10.0)},
        index=index,
    )


def ramp(n: int = 300, step: float = 0.004, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100.0 * np.exp(np.cumsum(rng.normal(step, 0.001, n)))


def wobble(n: int = 300, seed: int = 4) -> np.ndarray:
    """Direction-free chop: tiny alternating moves around a flat level."""
    rng = np.random.default_rng(seed)
    return 100.0 + 0.05 * np.sin(np.arange(n) * 0.7) + rng.normal(0, 0.02, n)


# ----------------------------------------------------------------- trend


def test_trend_up_ramp_scores_up():
    result = run_trend(build_frame(ramp()))
    assert result.scalars["direction"] == "up"
    assert result.scalars["trend_score"] > 0.3
    assert result.scalars["adx_bucket"] in ("trending", "strong")
    assert result.asof is not None


def test_trend_down_ramp_scores_down():
    result = run_trend(build_frame(ramp(step=-0.004)))
    assert result.scalars["direction"] == "down"
    assert result.scalars["trend_score"] < -0.3


def test_trend_chop_scores_flat():
    result = run_trend(build_frame(wobble()))
    assert result.scalars["direction"] == "flat"
    assert abs(result.scalars["trend_score"]) < 0.15


# ------------------------------------------------------------- volatility


def test_volatility_spike_lands_in_top_percentile():
    rng = np.random.default_rng(11)
    calm = rng.normal(0, 0.001, 250)
    wild = rng.normal(0, 0.02, 50)
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([calm, wild])))
    result = run_volatility(build_frame(closes))

    assert result.scalars["vol_percentile"] > 0.9
    assert result.scalars["vol_regime"] in ("elevated", "extreme")
    assert result.scalars["realized_vol_cc"] > 0
    assert result.scalars["realized_vol_parkinson"] > 0
    assert result.scalars["natr"] > 0


def test_volatility_calm_tail_lands_low():
    rng = np.random.default_rng(12)
    wild = rng.normal(0, 0.02, 250)
    calm = rng.normal(0, 0.001, 100)
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([wild, calm])))
    result = run_volatility(build_frame(closes))
    assert result.scalars["vol_percentile"] < 0.3
    assert result.scalars["vol_regime"] == "calm"


def test_volatility_short_frame_warns():
    closes = 100.0 + np.arange(10, dtype=float)
    result = run_volatility(build_frame(closes), window=20)
    assert result.scalars["vol_percentile"] is None
    assert any("too short" in w for w in result.warnings)


# ----------------------------------------------------------------- regime


def squeeze_then_pop(quiet: int = 120, pop: int = 6) -> np.ndarray:
    """Closes pinned dead flat (BB collapses inside the ATR channel), then a
    sharp upward expansion."""
    flat = np.full(quiet, 100.0)
    burst = 100.0 * np.exp(np.cumsum(np.full(pop, 0.01)))
    return np.concatenate([flat, burst])


def test_regime_detects_squeeze_while_quiet():
    closes = squeeze_then_pop(pop=0)
    result = run_regime(build_frame(closes, spread=0.004))
    assert result.scalars["squeeze_on"] is True
    assert result.scalars["bars_in_squeeze"] > 10
    assert result.scalars["released_now"] is False


def test_regime_detects_release_and_direction():
    # Frame ends exactly at the first expansion bars: the squeeze must have
    # just released, upward.
    for extra in range(1, 5):
        closes = squeeze_then_pop(pop=extra)
        result = run_regime(build_frame(closes, spread=0.004))
        if result.scalars["released_now"]:
            assert result.scalars["release_direction"] == "up"
            assert result.events[-1]["event"] == "squeeze_release"
            assert result.events[-1]["direction"] == "up"
            return
    pytest.fail("squeeze never released within the expansion bars")


# ------------------------------------------------------------- dispatch


# ----------------------------------------------------------------- levels


def swing_channel_frame(
    *,
    floor: float = 100.0,
    ceiling: float = 110.0,
    cycles: int = 4,
    bars_per_leg: int = 8,
) -> pd.DataFrame:
    """Constructed path that repeatedly tags `floor` and `ceiling`.

    Each cycle walks floor → ceiling → floor so swing_left=swing_right=2
    confirms pivots at both edges. Ends mid-channel so both sides exist.
    Floor/ceiling are visited once per turn (no flat plateaus).
    """
    closes: list[float] = [floor]
    for _ in range(cycles):
        up = np.linspace(floor, ceiling, bars_per_leg)[1:]
        down = np.linspace(ceiling, floor, bars_per_leg)[1:]
        closes.extend(float(x) for x in up)
        closes.extend(float(x) for x in down)
    # Park the last close in the middle of the channel (replace final floor).
    mid = (floor + ceiling) / 2.0
    closes[-1] = mid

    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    n = len(closes)
    index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    open_ = np.concatenate(([closes[0]], closes[:-1]))
    return pd.DataFrame(
        {
            "open": open_,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n, 10.0),
        },
        index=index,
    )


def test_levels_finds_static_support_and_resistance():
    frame = swing_channel_frame()
    result = run_levels(frame, swing_left=2, swing_right=2, cluster_bps=30)

    assert result.name == "levels"
    assert result.scalars["last_close"] == pytest.approx(105.0)
    assert result.scalars["nearest_support"] is not None
    assert result.scalars["nearest_resistance"] is not None
    assert result.scalars["nearest_support"] < result.scalars["last_close"]
    assert result.scalars["nearest_resistance"] > result.scalars["last_close"]
    assert result.scalars["nearest_support"] == pytest.approx(100.0, abs=0.5)
    assert result.scalars["nearest_resistance"] == pytest.approx(110.0, abs=0.5)
    assert result.scalars["dist_support_bps"] < 0
    assert result.scalars["dist_resistance_bps"] > 0
    assert result.scalars["support_count"] >= 1
    assert result.scalars["resistance_count"] >= 1
    assert any(e["kind"] == "support" for e in result.events)
    assert any(e["kind"] == "resistance" for e in result.events)


def test_levels_dynamic_channel_matches_donchian():
    frame = swing_channel_frame(cycles=6)
    result = run_levels(frame, donchian_window=20)

    assert result.scalars["dynamic_support"] is not None
    assert result.scalars["dynamic_resistance"] is not None
    assert result.scalars["dynamic_support"] < result.scalars["last_close"]
    assert result.scalars["dynamic_resistance"] > result.scalars["last_close"]
    assert result.scalars["in_range"] is True
    assert result.scalars["donchian_lower"] == result.scalars["dynamic_support"]
    assert result.scalars["donchian_upper"] == result.scalars["dynamic_resistance"]
    assert result.scalars["bb_lower"] is not None
    assert result.scalars["kc_upper"] is not None
    assert result.scalars["vwap"] is not None
    assert "dc_upper" in result.series and "vwap" in result.series


def test_levels_breakout_is_out_of_range():
    """Prior-bar Donchian: after a quiet channel, a sharp break puts close
    above the confirmed upper and in_range flips false."""
    quiet = np.full(80, 100.0)
    burst = np.linspace(100.0, 108.0, 5)
    closes = np.concatenate([quiet, burst])
    frame = build_frame(closes, spread=0.002)
    result = run_levels(frame, donchian_window=20, swing_left=2, swing_right=2)

    assert result.scalars["last_close"] > 105.0
    assert result.scalars["dynamic_resistance"] is not None
    assert result.scalars["last_close"] > result.scalars["dynamic_resistance"]
    assert result.scalars["in_range"] is False


def test_levels_short_frame_warns():
    frame = build_frame(np.full(5, 100.0))
    result = run_levels(frame, swing_left=2, swing_right=2, donchian_window=20)
    assert result.scalars["nearest_support"] is None
    assert result.scalars["nearest_resistance"] is None
    assert any("swing" in w or "pivot" in w or "short" in w for w in result.warnings)


def test_registry_dispatch_and_errors(ohlc):
    assert list_analyses() == ["levels", "regime", "tilt", "trend", "volatility"]

    result = run_analysis("volatility", ohlc, {"window": "30"})
    assert result.params["window"] == 30  # loose string coerced to int

    levels = run_analysis("levels", ohlc, {"swing_left": "2"})
    assert levels.name == "levels"
    assert levels.params["swing_left"] == 2

    with pytest.raises(ParamError, match="unknown analysis"):
        run_analysis("vibes", ohlc)
    with pytest.raises(ParamError, match="windwo"):
        run_analysis("volatility", ohlc, {"windwo": 30})


def test_as_dict_is_json_ready(ohlc):
    import json

    result = run_analysis("trend", ohlc)
    payload = result.as_dict()
    json.dumps(payload)  # must not raise on numpy types
    assert "series" not in payload
    with_series = result.as_dict(include_series=True)
    assert "adx" in with_series["series"]

    levels = run_analysis("levels", ohlc)
    json.dumps(levels.as_dict())
    assert "series" not in levels.as_dict()
    assert "dc_lower" in levels.as_dict(include_series=True)["series"]
