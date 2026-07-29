"""Overlays: exact deltas from frozen verdicts, allowlist enforcement,
composition rules, and grid-swept invariants."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.analysis.types import AnalysisResult
from ta.errors import ParamError
from ta.signals import (
    ConfigDelta,
    OverlayContext,
    combine,
    evaluate,
    squeeze_brace,
    trend_tilt,
    vol_widen,
)

ASOF = pd.Timestamp("2026-07-30 12:00", tz="UTC")


def make_ctx(**scalars_by_analysis) -> OverlayContext:
    analyses = {
        name: AnalysisResult(name=name, params={}, asof=ASOF, scalars=scalars)
        for name, scalars in scalars_by_analysis.items()
    }
    return OverlayContext(analyses=analyses, asof=ASOF)


# ------------------------------------------------------------ ConfigDelta


def test_delta_rejects_unknown_fields_at_construction():
    with pytest.raises(TypeError):
        ConfigDelta(max_buy_price=1.0)  # not on the allowlist -> impossible


def test_delta_validates_bounds():
    with pytest.raises(ParamError):
        ConfigDelta(spread_multiplier=0.0)
    with pytest.raises(ParamError):
        ConfigDelta(spread_multiplier=5.1)  # config-manager caps at 5
    with pytest.raises(ParamError):
        ConfigDelta(premium_skew_factor=1.5)
    with pytest.raises(ParamError):
        ConfigDelta(structural_premium=0.06)  # > 500bp is a bug, not a signal
    with pytest.raises(ParamError):
        ConfigDelta(vol_spread_sensitivity=-1.0)
    with pytest.raises(ParamError, match="never re-enable"):
        ConfigDelta(disable_buy=False)


def test_apply_to_composition_rules():
    baseline = {
        "spread_multiplier": 1.2,
        "vol_spread_sensitivity": 5.0,
        "disable_buy": True,  # set by the risk plane
        "structural_premium": -0.001,
        "premium_skew_factor": 0.2,
        "min_spread": 0.004,  # untouched passenger field
    }
    delta = ConfigDelta(
        spread_multiplier=1.5,
        vol_spread_sensitivity=2.0,
        structural_premium=0.0008,
        premium_skew_factor=0.5,
    )
    out = delta.apply_to(baseline)

    assert out["spread_multiplier"] == pytest.approx(1.8)  # composes, not replaces
    assert out["vol_spread_sensitivity"] == 5.0  # max(baseline, delta)
    assert out["disable_buy"] is True  # never un-disabled
    assert out["structural_premium"] == 0.0008  # premium pair overrides
    assert out["premium_skew_factor"] == 0.5
    assert out["min_spread"] == 0.004
    assert baseline["spread_multiplier"] == 1.2  # input untouched

    # Composition respects the model's hard cap.
    capped = ConfigDelta(spread_multiplier=4.9).apply_to({"spread_multiplier": 2.0})
    assert capped["spread_multiplier"] == 5.0


# -------------------------------------------------------------- vol_widen


def test_vol_widen_ramp():
    assert vol_widen(make_ctx(volatility={"vol_percentile": 0.5})).is_empty
    assert vol_widen(make_ctx(volatility={"vol_percentile": 0.75})).is_empty  # 1.0x

    mid = vol_widen(make_ctx(volatility={"vol_percentile": 0.875}))
    assert mid.spread_multiplier == pytest.approx(1.75)  # halfway to cap 2.5

    top = vol_widen(make_ctx(volatility={"vol_percentile": 1.0}))
    assert top.spread_multiplier == pytest.approx(2.5)
    assert "vol_widen" in top.reason

    assert vol_widen(make_ctx()).is_empty  # no volatility analysis at all


# ------------------------------------------------------------- trend_tilt


def test_trend_tilt_exact_values():
    assert trend_tilt(make_ctx(trend={"trend_score": 0.3})).is_empty  # under min 0.4

    up = trend_tilt(make_ctx(trend={"trend_score": 0.7}))
    # (0.7-0.4)/(1-0.4) * 15bp = 7.5bp
    assert up.structural_premium == pytest.approx(7.5e-4)
    assert up.premium_skew_factor == 0.5
    assert "+7.5bp" in up.reason

    down = trend_tilt(make_ctx(trend={"trend_score": -1.0}))
    assert down.structural_premium == pytest.approx(-15e-4)

    # The pair always ships together: premium without gate is inert live.
    assert up.premium_skew_factor is not None


# ---------------------------------------------------------- squeeze_brace


def test_squeeze_brace_fires_only_on_release():
    quiet = make_ctx(regime={"released_now": False, "release_direction": None})
    assert squeeze_brace(quiet).is_empty

    released = make_ctx(regime={"released_now": True, "release_direction": "up"})
    plain = squeeze_brace(released)
    assert plain.spread_multiplier == 1.5
    assert plain.disable_sell is None  # opt-in only

    braced = squeeze_brace(released, disable_adverse_side=True)
    assert braced.disable_sell is True and braced.disable_buy is None

    down = make_ctx(regime={"released_now": True, "release_direction": "down"})
    assert squeeze_brace(down, disable_adverse_side=True).disable_buy is True


# ---------------------------------------------------------------- combine


def test_combine_semantics():
    a = ConfigDelta(spread_multiplier=1.5, reason="a")
    b = ConfigDelta(spread_multiplier=1.4, disable_sell=True, reason="b")
    c = ConfigDelta(structural_premium=0.0005, premium_skew_factor=0.4, reason="c")
    d = ConfigDelta(structural_premium=-0.001, premium_skew_factor=0.6, reason="d")

    merged = combine(a, b, c, d)
    assert merged.spread_multiplier == pytest.approx(2.1)  # 1.5 * 1.4
    assert merged.disable_sell is True
    # Stronger |premium| wins whole, pair kept coherent -- not summed.
    assert merged.structural_premium == -0.001
    assert merged.premium_skew_factor == 0.6
    assert merged.reason == "a; b; c; d"

    assert combine().is_empty
    assert combine(ConfigDelta(), ConfigDelta()).is_empty


def test_evaluate_stack_from_string():
    ctx = make_ctx(
        volatility={"vol_percentile": 1.0},
        trend={"trend_score": 1.0},
        regime={"released_now": False},
    )
    delta = evaluate(ctx, "vol_widen:threshold=0.8,cap=3, trend_tilt, squeeze_brace")
    assert delta.spread_multiplier == pytest.approx(3.0)
    assert delta.structural_premium == pytest.approx(15e-4)

    with pytest.raises(ParamError, match="unknown overlay"):
        evaluate(ctx, "yolo_mode")
    with pytest.raises(ParamError, match="treshold"):
        evaluate(ctx, "vol_widen:treshold=0.9")


# ------------------------------------------------------------- invariants


ALLOWLIST = {
    "spread_multiplier",
    "premium_skew_factor",
    "structural_premium",
    "disable_buy",
    "disable_sell",
    "vol_spread_sensitivity",
}


def test_grid_swept_invariants():
    """Default-configured overlays never narrow, never over-tilt, never leave
    the allowlist -- across the whole input space."""
    for percentile in np.linspace(0.0, 1.0, 101):
        for score in np.linspace(-1.0, 1.0, 41):
            for released in (False, True):
                ctx = make_ctx(
                    volatility={"vol_percentile": float(percentile)},
                    trend={"trend_score": float(score)},
                    regime={"released_now": released, "release_direction": "up"},
                )
                delta = evaluate(ctx, "vol_widen, trend_tilt, squeeze_brace")
                fields = delta.fields()
                assert set(fields) <= ALLOWLIST
                if delta.spread_multiplier is not None:
                    assert 1.0 < delta.spread_multiplier <= 5.0
                if delta.structural_premium is not None:
                    assert abs(delta.structural_premium) <= 15e-4
                    assert delta.premium_skew_factor is not None
                assert delta.disable_buy is not False
                assert delta.disable_sell is not False
