"""Overlays: exact deltas from frozen verdicts, allowlist enforcement,
composition rules, cross-field invariants, and grid-swept safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.analysis.types import AnalysisResult
from ta.errors import ParamError
from ta.signals import (
    ABSOLUTE_FIELDS,
    ConfigDelta,
    OverlayContext,
    band_widen,
    combine,
    evaluate,
    inventory_tilt,
    size_down,
    squeeze_brace,
    trend_tilt,
    vol_widen,
)

ASOF = pd.Timestamp("2026-07-30 12:00", tz="UTC")

#: A plausible baseline for a mid-size IRT market.
BASELINE = {
    "price_min_spread": 0.004,
    "price_max_spread": 0.012,
    "min_base": 10.0,
    "max_base": 100.0,
    "maker_fee": 0.0015,
    "spread_multiplier": 1.0,
    "is_enable": True,
    "number_of_order_each_sides": 3,
}


def make_ctx(config=None, **scalars_by_analysis) -> OverlayContext:
    analyses = {
        name: AnalysisResult(name=name, params={}, asof=ASOF, scalars=scalars)
        for name, scalars in scalars_by_analysis.items()
    }
    return OverlayContext(analyses=analyses, asof=ASOF, config=config)


# ------------------------------------------------------------ ConfigDelta


def test_delta_rejects_unknown_fields_at_construction():
    with pytest.raises(TypeError):
        ConfigDelta(bypass_buy_volume=1.0)  # real MM field, not allowlisted


def test_delta_validates_scaling_bounds():
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
    with pytest.raises(ParamError, match="never resume"):
        ConfigDelta(disable_buy=False)


def test_delta_validates_absolute_bounds():
    with pytest.raises(ParamError, match="fraction of price"):
        ConfigDelta(price_max_spread=2.5)  # 250%: a units mistake
    with pytest.raises(ParamError):
        ConfigDelta(price_min_spread=0.0)
    with pytest.raises(ParamError):
        ConfigDelta(min_base=-1.0)
    with pytest.raises(ParamError):
        ConfigDelta(max_base=0.0)
    with pytest.raises(ParamError):
        ConfigDelta(target_base=-1.0)
    with pytest.raises(ParamError):
        ConfigDelta(number_of_order_each_sides=0)

    # is_enable accepts both directions -- unlike the per-side pauses, turning
    # a market back on is a legitimate (approval-gated) proposal.
    assert ConfigDelta(is_enable=True).is_enable is True
    assert ConfigDelta(is_enable=False).is_enable is False
    assert ConfigDelta(target_base=55.0).target_base == 55.0


def test_high_impact_flagging():
    assert ConfigDelta(is_enable=True).is_high_impact
    assert ConfigDelta(max_base=200.0).is_high_impact
    assert ConfigDelta(number_of_order_each_sides=5).is_high_impact
    # De-risking changes are not high impact.
    assert not ConfigDelta(is_enable=False).is_high_impact
    assert not ConfigDelta(spread_multiplier=2.0).is_high_impact
    assert not ConfigDelta(disable_buy=True).is_high_impact


def test_apply_to_composition_rules():
    baseline = {
        **BASELINE,
        "spread_multiplier": 1.2,
        "vol_spread_sensitivity": 5.0,
        "disable_buy": True,  # set by the risk plane
        "structural_premium": -0.001,
        "premium_skew_factor": 0.2,
        "tolerant": 0.0001,  # untouched passenger field
    }
    delta = ConfigDelta(
        spread_multiplier=1.5,
        vol_spread_sensitivity=2.0,
        structural_premium=0.0008,
        premium_skew_factor=0.5,
        price_max_spread=0.02,
    )
    out = delta.apply_to(baseline)

    assert out["spread_multiplier"] == pytest.approx(1.8)  # composes, not replaces
    assert out["vol_spread_sensitivity"] == 5.0  # max(baseline, delta)
    assert out["disable_buy"] is True  # never resumed
    assert out["structural_premium"] == 0.0008  # premium pair overrides
    assert out["premium_skew_factor"] == 0.5
    assert out["price_max_spread"] == 0.02  # absolute: set outright
    assert out["price_min_spread"] == 0.004  # untouched
    assert out["tolerant"] == 0.0001
    assert baseline["spread_multiplier"] == 1.2  # input untouched

    capped = ConfigDelta(spread_multiplier=4.9).apply_to({"spread_multiplier": 2.0})
    assert capped["spread_multiplier"] == 5.0


def test_apply_to_rejects_inverted_bands():
    with pytest.raises(ParamError, match="inverts the spread band"):
        ConfigDelta(price_min_spread=0.05).apply_to(BASELINE)  # min above max
    with pytest.raises(ParamError, match="inverts the inventory band"):
        ConfigDelta(max_base=5.0).apply_to(BASELINE)  # max below min_base

    # A baseline that is already inverted is not ours to reject when the
    # delta touches neither end.
    broken = {**BASELINE, "price_min_spread": 0.9, "price_max_spread": 0.001}
    assert ConfigDelta(spread_multiplier=2.0).apply_to(broken)["spread_multiplier"] == 2.0


def test_apply_to_rejects_quoting_inside_the_fee_floor():
    # Break-even is ONE maker fee (0.0015 here), not two: a round trip is
    # gross 2s against 2*maker_fee of fees. Quoting 10bp is a no-op live.
    with pytest.raises(ParamError, match="break-even half-spread"):
        ConfigDelta(price_min_spread=0.001).apply_to(BASELINE)

    # 20bp sits above break-even and must be allowed -- flooring at 2x here
    # would forbid exactly the band where nobitex's expected value peaks.
    assert ConfigDelta(price_min_spread=0.002).apply_to(BASELINE)["price_min_spread"] == 0.002
    assert ConfigDelta(price_min_spread=0.004).apply_to(BASELINE)["price_min_spread"] == 0.004


def test_apply_to_rejects_target_base_outside_band_or_cap():
    # Midpoint 55, half-band 45, 30% cap -> [41.5, 68.5].
    assert ConfigDelta(target_base=68.5).apply_to(BASELINE)["target_base"] == 68.5
    assert ConfigDelta(target_base=41.5).apply_to(BASELINE)["target_base"] == 41.5

    with pytest.raises(ParamError, match="outside the inventory band"):
        ConfigDelta(target_base=5.0).apply_to(BASELINE)
    with pytest.raises(ParamError, match="outside the inventory band"):
        ConfigDelta(target_base=150.0).apply_to(BASELINE)
    with pytest.raises(ParamError, match="half-band from the midpoint"):
        ConfigDelta(target_base=70.0).apply_to(BASELINE)  # inside band, past cap
    with pytest.raises(ParamError, match="needs min_base and max_base"):
        ConfigDelta(target_base=55.0).apply_to({"maker_fee": 0.001})


# -------------------------------------------------------------- vol_widen


def test_vol_widen_ramp():
    assert vol_widen(make_ctx(volatility={"vol_percentile": 0.5})).is_empty
    assert vol_widen(make_ctx(volatility={"vol_percentile": 0.75})).is_empty  # 1.0x

    mid = vol_widen(make_ctx(volatility={"vol_percentile": 0.875}))
    assert mid.spread_multiplier == pytest.approx(1.75)  # halfway to cap 2.5

    top = vol_widen(make_ctx(volatility={"vol_percentile": 1.0}))
    assert top.spread_multiplier == pytest.approx(2.5)
    assert vol_widen(make_ctx()).is_empty  # no volatility analysis at all


# ------------------------------------------------------------- band_widen


def test_band_widen_moves_only_the_far_edge_by_default():
    ctx = make_ctx(config=BASELINE, volatility={"vol_percentile": 1.0})
    delta = band_widen(ctx)

    # Full ramp, max_bump 2.0: 0.012 -> 0.024. Near edge untouched.
    assert delta.price_max_spread == pytest.approx(0.024)
    assert delta.price_min_spread is None
    assert set(delta.fields()) == {"price_max_spread"}

    applied = delta.apply_to(BASELINE)
    assert applied["price_min_spread"] == 0.004  # queue position preserved


def test_band_widen_can_move_both_edges():
    ctx = make_ctx(config=BASELINE, volatility={"vol_percentile": 1.0})
    delta = band_widen(ctx, min_bump=1.5)
    assert delta.price_max_spread == pytest.approx(0.024)
    assert delta.price_min_spread == pytest.approx(0.006)
    delta.apply_to(BASELINE)  # still a valid band


def test_band_widen_needs_config_and_respects_threshold():
    assert band_widen(make_ctx(config=BASELINE, volatility={"vol_percentile": 0.5})).is_empty

    with pytest.raises(ParamError, match="needs the current config"):
        band_widen(make_ctx(volatility={"vol_percentile": 1.0}))

    with pytest.raises(ParamError, match="never narrow"):
        band_widen(
            make_ctx(config=BASELINE, volatility={"vol_percentile": 1.0}), max_bump=0.8
        )


# -------------------------------------------------------------- size_down


def test_size_down_shrinks_max_base():
    ctx = make_ctx(config=BASELINE, volatility={"vol_percentile": 1.0})
    delta = size_down(ctx)
    assert delta.max_base == pytest.approx(50.0)  # floor_ratio 0.5 at full ramp
    assert delta.is_high_impact  # touching size always warrants a look

    half = size_down(make_ctx(config=BASELINE, volatility={"vol_percentile": 0.95}))
    assert half.max_base == pytest.approx(75.0)

    assert size_down(make_ctx(config=BASELINE, volatility={"vol_percentile": 0.5})).is_empty


def test_size_down_never_inverts_the_inventory_band():
    tight = {**BASELINE, "min_base": 90.0, "max_base": 100.0}
    delta = size_down(make_ctx(config=tight, volatility={"vol_percentile": 1.0}))
    assert delta.max_base == pytest.approx(90.0)  # clamped at min_base, not 50
    delta.apply_to(tight)  # must not raise


# ---------------------------------------------------------- squeeze_brace


def test_squeeze_brace_fires_only_on_release():
    quiet = make_ctx(regime={"released_now": False, "release_direction": None})
    assert squeeze_brace(quiet).is_empty

    released = make_ctx(regime={"released_now": True, "release_direction": "up"})
    plain = squeeze_brace(released)
    assert plain.spread_multiplier == 1.5
    assert plain.disable_sell is None  # opt-in only

    braced = squeeze_brace(released, pause_adverse_side=True)
    assert braced.disable_sell is True and braced.disable_buy is None

    down = make_ctx(regime={"released_now": True, "release_direction": "down"})
    assert squeeze_brace(down, pause_adverse_side=True).disable_buy is True


# ------------------------------------------------------------- trend_tilt


def test_trend_tilt_exact_values():
    assert trend_tilt(make_ctx(trend={"trend_score": 0.3})).is_empty  # under min 0.4

    up = trend_tilt(make_ctx(trend={"trend_score": 0.7}))
    # (0.7-0.4)/(1-0.4) * 15bp = 7.5bp
    assert up.structural_premium == pytest.approx(7.5e-4)
    assert up.premium_skew_factor == 0.5  # pair always ships together
    assert "+7.5bp" in up.reason

    down = trend_tilt(make_ctx(trend={"trend_score": -1.0}))
    assert down.structural_premium == pytest.approx(-15e-4)


# --------------------------------------------------------- inventory_tilt


def test_inventory_tilt_maps_signal_to_capped_target():
    # Midpoint 55, half-band 45: full +tilt at cap 0.30 -> 55 + 0.3*45 = 68.5
    full = inventory_tilt(make_ctx(config=BASELINE, tilt={"tilt": 1.0}))
    assert full.target_base == pytest.approx(68.5)
    assert "tilt=+1.000" in full.reason
    full.apply_to(BASELINE)

    half = inventory_tilt(make_ctx(config=BASELINE, tilt={"tilt": 0.5}))
    assert half.target_base == pytest.approx(55.0 + 0.5 * 0.30 * 45.0)

    down = inventory_tilt(make_ctx(config=BASELINE, tilt={"tilt": -1.0}))
    assert down.target_base == pytest.approx(41.5)

    # Zero tilt reproduces the midpoint; with no live target that is a no-op.
    assert inventory_tilt(make_ctx(config=BASELINE, tilt={"tilt": 0.0})).is_empty
    assert inventory_tilt(make_ctx(config=BASELINE)).is_empty  # no tilt analysis


def test_inventory_tilt_respects_tighter_cap_and_skips_noop():
    tight = inventory_tilt(
        make_ctx(config=BASELINE, tilt={"tilt": 1.0}), cap=0.15
    )
    assert tight.target_base == pytest.approx(55.0 + 0.15 * 45.0)

    already = {**BASELINE, "target_base": 68.5}
    assert inventory_tilt(make_ctx(config=already, tilt={"tilt": 1.0})).is_empty

    with pytest.raises(ParamError, match="allowlist ceiling"):
        inventory_tilt(make_ctx(config=BASELINE, tilt={"tilt": 1.0}), cap=0.5)
    with pytest.raises(ParamError, match="needs the current config"):
        inventory_tilt(make_ctx(tilt={"tilt": 1.0}))


def test_inventory_tilt_grid_never_breaks_band_or_cap():
    """On any input the overlay may never propose past the band or the cap."""
    for tilt in np.linspace(-1.0, 1.0, 41):
        for cap in (0.05, 0.15, 0.30):
            for lo, hi in ((10.0, 100.0), (10.0, 450.0), (10.0, 1100.0)):
                cfg = {**BASELINE, "min_base": lo, "max_base": hi}
                delta = inventory_tilt(
                    make_ctx(config=cfg, tilt={"tilt": float(tilt)}), cap=cap
                )
                if delta.is_empty:
                    continue
                midpoint = (lo + hi) / 2.0
                half = (hi - lo) / 2.0
                assert lo <= delta.target_base <= hi
                assert abs(delta.target_base - midpoint) <= cap * half + 1e-9
                delta.apply_to(cfg)


# ---------------------------------------------------------------- combine


def test_combine_stacks_scaling_knobs():
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


def test_combine_refuses_conflicting_absolutes():
    with pytest.raises(ParamError, match="conflicting proposals for max_base"):
        combine(ConfigDelta(max_base=50.0), ConfigDelta(max_base=80.0))

    # Agreeing proposals are fine, and pass through.
    agreed = combine(ConfigDelta(max_base=50.0), ConfigDelta(max_base=50.0, reason="x"))
    assert agreed.max_base == 50.0


def test_combine_carries_absolutes_alongside_scaling():
    merged = combine(
        ConfigDelta(spread_multiplier=2.0, reason="wide"),
        ConfigDelta(price_max_spread=0.03, reason="band"),
        ConfigDelta(max_base=40.0, reason="small"),
    )
    assert merged.spread_multiplier == 2.0
    assert merged.price_max_spread == 0.03
    assert merged.max_base == 40.0


def test_evaluate_stack_from_string():
    ctx = make_ctx(
        config=BASELINE,
        volatility={"vol_percentile": 1.0},
        trend={"trend_score": 1.0},
        regime={"released_now": False},
    )
    delta = evaluate(ctx, "vol_widen:threshold=0.8,cap=3, band_widen, trend_tilt")
    assert delta.spread_multiplier == pytest.approx(3.0)
    assert delta.price_max_spread == pytest.approx(0.024)
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
    "price_min_spread",
    "price_max_spread",
    "min_base",
    "max_base",
    "target_base",
    "is_enable",
    "number_of_order_each_sides",
}


def test_absolute_fields_are_a_subset_of_the_allowlist():
    assert ABSOLUTE_FIELDS <= ALLOWLIST


def test_grid_swept_invariants():
    """Default-configured overlays never narrow, never over-tilt, never leave
    the allowlist, and always compose into a valid config."""
    for percentile in np.linspace(0.0, 1.0, 51):
        for score in np.linspace(-1.0, 1.0, 21):
            for tilt in (-1.0, 0.0, 0.5, 1.0):
                for released in (False, True):
                    ctx = make_ctx(
                        config=BASELINE,
                        volatility={"vol_percentile": float(percentile)},
                        trend={"trend_score": float(score)},
                        tilt={"tilt": float(tilt)},
                        regime={"released_now": released, "release_direction": "up"},
                    )
                    delta = evaluate(
                        ctx,
                        "vol_widen, band_widen, size_down, trend_tilt, "
                        "inventory_tilt, squeeze_brace",
                    )
                    touched = delta.fields()
                    assert set(touched) <= ALLOWLIST

                    if delta.spread_multiplier is not None:
                        assert 1.0 < delta.spread_multiplier <= 5.0
                    if delta.structural_premium is not None:
                        assert abs(delta.structural_premium) <= 15e-4
                        assert delta.premium_skew_factor is not None
                    if delta.price_max_spread is not None:
                        # Widen-only, and never past twice the starting band.
                        assert BASELINE["price_max_spread"] < delta.price_max_spread <= 0.024
                    if delta.max_base is not None:
                        assert BASELINE["min_base"] <= delta.max_base < BASELINE["max_base"]
                    if delta.target_base is not None:
                        midpoint = (
                            BASELINE["min_base"] + BASELINE["max_base"]
                        ) / 2.0
                        half = (BASELINE["max_base"] - BASELINE["min_base"]) / 2.0
                        assert BASELINE["min_base"] <= delta.target_base <= BASELINE["max_base"]
                        assert abs(delta.target_base - midpoint) <= 0.30 * half + 1e-9
                    assert delta.disable_buy is not False
                    assert delta.disable_sell is not False
                    # No automatic overlay may ever flip a market on or off.
                    assert delta.is_enable is None

                    try:
                        delta.apply_to(BASELINE)
                    except ParamError as exc:
                        # size_down shrinks the band; inventory_tilt's target
                        # was computed against the *pre*-shrink band. The
                        # composed config can then fail the cap / in-band
                        # check -- that refusal is the safety property.
                        assert (
                            delta.target_base is not None and delta.max_base is not None
                        ), f"unexpected apply_to failure: {exc}"
                        assert "target_base" in str(exc)
