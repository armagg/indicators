"""Backtest: simulator math against hand-computed ladders, analytic fill and
fee cases, causality, and the sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.backtest import (
    DISCLAIMER,
    DIVERGENCES,
    Fill,
    build_ladder,
    compare,
    effective_spreads,
    grid,
    inventory_q,
    level_step,
    markouts,
    order_totals,
    premium_reference,
    render,
    run,
    skew_reference,
    spread_floor,
    spread_scale,
    sweep,
)
from ta.errors import FrameError, ParamError

#: maker_fee 10bp, band 40-120bp, three levels a side, band midpoint 55.
CONFIG = {
    "price_min_spread": 0.004,
    "price_max_spread": 0.012,
    "number_of_order_each_sides": 3,
    "min_base": 10.0,
    "max_base": 100.0,
    "maker_fee": 0.001,
    "spread_multiplier": 1.0,
}


def flat_frame(price=100.0, periods=200, freq="1h", spread=0.0):
    """A market that prints the same close every bar, with a fixed range."""
    index = pd.date_range("2026-01-01", periods=periods, freq=freq, tz="UTC")
    return pd.DataFrame(
        {
            "open": price,
            "high": price * (1 + spread),
            "low": price * (1 - spread),
            "close": price,
            "volume": 1.0,
        },
        index=index,
    )


# -------------------------------------------------------------- simulator


def test_spread_scale_composes_multiplier_and_volatility():
    assert spread_scale(CONFIG) == 1.0
    assert spread_scale({**CONFIG, "spread_multiplier": 2.0}) == 2.0
    assert spread_scale(
        {**CONFIG, "spread_multiplier": 2.0, "vol_spread_sensitivity": 4.0}, volatility=0.5
    ) == pytest.approx(6.0)  # 2 * (1 + 4*0.5)
    # Degenerate values fall back to identity rather than inverting the grid.
    assert spread_scale({**CONFIG, "spread_multiplier": 0}) == 1.0
    assert spread_scale({**CONFIG, "vol_spread_sensitivity": 5.0}, volatility=0.0) == 1.0


def test_spread_floor_is_one_maker_fee_not_two():
    """The 2x version is a known, costly bug -- see market_maker._spread_floor."""
    assert spread_floor({"maker_fee": 0.000961}) == 0.000961
    # Taker rate is the fallback only when maker is unset.
    assert spread_floor({"market_take_fee": 0.002}) == 0.002
    assert spread_floor({"maker_fee": 0.001, "market_take_fee": 0.002}) == 0.001

    # A 15bp min spread on a 9.61bp fee must survive: that is inside the band
    # where nobitex's expected value peaks, and a 2x floor would forbid it.
    min_s, _ = effective_spreads({"price_min_spread": 0.0015, "price_max_spread": 0.003,
                                  "maker_fee": 0.000961})
    assert min_s == pytest.approx(0.0015)


def test_effective_spreads_floor_and_ordering():
    # Tightening below break-even is clamped up to the fee.
    min_s, max_s = effective_spreads(CONFIG, scale=0.1)
    assert min_s == pytest.approx(0.001) and max_s == pytest.approx(0.0012)

    # A collapsed band keeps max >= min.
    min_s, max_s = effective_spreads({**CONFIG, "price_max_spread": 0.0001})
    assert max_s == min_s == pytest.approx(0.004)

    # Scaling moves both edges together.
    assert effective_spreads(CONFIG, scale=2.0) == pytest.approx((0.008, 0.024))


def test_spread_asymmetry_is_mean_preserving_and_clamped():
    config = {**CONFIG, "spread_asymmetry": 0.001}
    buy_min, _ = effective_spreads(config, side="buy")
    sell_min, _ = effective_spreads(config, side="sell")
    assert buy_min == pytest.approx(0.005)  # quoted further out
    assert sell_min == pytest.approx(0.003)  # quoted closer in
    assert (buy_min + sell_min) / 2 == pytest.approx(CONFIG["price_min_spread"])

    # A shift larger than the room above the floor is clamped, not allowed
    # to push a side under break-even.
    wide = {**CONFIG, "spread_asymmetry": 0.05}
    _, _ = effective_spreads(wide, side="sell")
    assert effective_spreads(wide, side="sell")[0] >= spread_floor(wide)


def test_level_step_divides_the_band():
    assert level_step(CONFIG, 3) == pytest.approx((0.012 - 0.004) / 2)
    assert level_step(CONFIG, 1) == 0.0


def test_ladder_prices_are_hand_computable():
    """Reference 100, band 40-120bp, 3 levels: bids at 40/80/120bp below."""
    ladder = build_ladder(CONFIG, reference=100.0, base_balance=55.0)

    assert [round(q.price, 6) for q in ladder.bids] == [99.6, 99.2, 98.8]
    assert [round(q.price, 6) for q in ladder.asks] == [100.4, 100.8, 101.2]
    assert [q.level for q in ladder.bids] == [0, 1, 2]

    # base 55 is the band midpoint: buy_total = 100-55 = 45, sell = 55-10 = 45,
    # split equally across 3 levels.
    assert all(q.size == pytest.approx(15.0) for q in ladder)


def test_inventory_skew_moves_the_grid_against_the_position():
    assert inventory_q(CONFIG, 55.0) == pytest.approx(0.0)  # midpoint = target
    assert inventory_q(CONFIG, 100.0) == pytest.approx(1.0)
    assert inventory_q(CONFIG, 10.0) == pytest.approx(-1.0)

    config = {**CONFIG, "inventory_skew_factor": 0.01}
    long_ref = skew_reference(config, 100.0, base_balance=100.0)
    short_ref = skew_reference(config, 100.0, base_balance=10.0)
    assert long_ref < 100.0 < short_ref  # long quotes lower, short quotes higher
    assert long_ref == pytest.approx(100.0 * (1 - 0.01 * 1.0 / 2))

    # Half-spread parameterisation takes precedence over the legacy factor.
    hs = {**CONFIG, "inventory_skew_factor": 0.01, "inventory_skew_halfspreads": 1.0}
    assert skew_reference(hs, 100.0, 100.0) == pytest.approx(100.0 * (1 - 1.0 * 1.0 * 0.004))


def test_premium_skew_is_inert_without_its_gate():
    assert premium_reference({"structural_premium": -0.002}, 100.0) == 100.0
    assert premium_reference(
        {"structural_premium": -0.002, "premium_skew_factor": 0.5}, 100.0
    ) == pytest.approx(100.0 * (1 - 0.001))


def test_order_totals_follow_the_inventory_band():
    assert order_totals(CONFIG, 55.0) == (45.0, 45.0)
    assert order_totals(CONFIG, 10.0) == (90.0, 0.0)  # at the floor: nothing to sell
    # Above max_base there is no room to buy; live falls back to a thin probe.
    buy, sell = order_totals({**CONFIG, "base_order_min": 1.0}, 120.0)
    assert buy == 3.0 and sell == 110.0


def test_disabled_sides_and_exchange_minimums_suppress_quotes():
    assert build_ladder({**CONFIG, "disable_buy": True}, 100.0, 55.0).bids == ()
    assert build_ladder({**CONFIG, "disable_sell": True}, 100.0, 55.0).asks == ()
    # Each level would be 15 base; a 20-base minimum kills them all.
    assert build_ladder({**CONFIG, "base_order_min": 20.0}, 100.0, 55.0).is_empty
    # quote_order_min applies to notional.
    assert build_ladder({**CONFIG, "quote_order_min": 1e9}, 100.0, 55.0).is_empty
    assert build_ladder(CONFIG, reference=0.0, base_balance=55.0).is_empty


# ------------------------------------------------------- analytic run cases


def test_a_flat_market_never_reaches_the_quotes():
    """Price pinned at 100 with no range: a 40bp band is never touched."""
    result = run(flat_frame(), CONFIG, warmup_bars=10)
    assert result.metrics.fills == 0
    assert result.metrics.pnl == 0.0
    assert result.metrics.buy_uptime == 1.0 and result.metrics.sell_uptime == 1.0


def test_a_bar_range_that_spans_the_band_fills_every_level():
    """Range of +/-1.5% covers all three levels (40/80/120bp) on both sides."""
    frame = flat_frame(spread=0.015, periods=60)
    result = run(frame, CONFIG, warmup_bars=10, initial_base=55.0)

    bars = result.metrics.bars
    # Both sides fill on every bar; sizes rebalance as inventory moves, but
    # the count is exactly 3 levels x 2 sides x bars.
    assert result.metrics.fills == bars * 6
    assert result.metrics.buy_fills == result.metrics.sell_fills == bars * 3


def test_symmetric_fills_earn_the_spread_minus_fees():
    """One bar, one level a side, hand-computable P&L.

    Buy at 99.6 and sell at 100.4 for size s: gross 0.8*s. Fees are
    10bp of each notional: (99.6 + 100.4) * s * 0.001 = 0.2*s.
    Net = 0.6*s, and inventory ends flat.
    """
    config = {**CONFIG, "number_of_order_each_sides": 1}
    frame = flat_frame(spread=0.015, periods=12)
    result = run(frame, config, warmup_bars=10, initial_base=55.0)

    assert result.metrics.bars == 2
    size = result.fills[0].size
    expected = 2 * (0.8 * size - (99.6 + 100.4) * size * 0.001)
    assert result.metrics.pnl == pytest.approx(expected, rel=1e-9)
    assert result.metrics.final_base == pytest.approx(55.0)  # flat, as designed
    assert result.metrics.fees_paid > 0


def test_quoting_at_break_even_nets_exactly_zero():
    """The floor's definition, made executable: at s == maker_fee, a round
    trip's gross spread exactly equals its fees."""
    config = {
        **CONFIG,
        "number_of_order_each_sides": 1,
        "price_min_spread": 0.001,  # == maker_fee
        "price_max_spread": 0.001,
    }
    result = run(flat_frame(spread=0.015, periods=12), config,
                 warmup_bars=10, initial_base=55.0)

    assert result.metrics.fills == 4  # 2 bars x 2 sides
    assert result.metrics.pnl == pytest.approx(0.0, abs=1e-9)


def test_quoting_inside_the_floor_is_clamped_up_to_it():
    """A config that tries to quote tighter than break-even is silently
    floored by the maker -- so it cannot manufacture a loss, or a profit."""
    inside = {**CONFIG, "number_of_order_each_sides": 1,
              "price_min_spread": 0.0001, "price_max_spread": 0.0001}
    at_floor = {**inside, "price_min_spread": 0.001, "price_max_spread": 0.001}
    frame = flat_frame(spread=0.015, periods=12)

    assert run(frame, inside, warmup_bars=10).metrics.pnl == pytest.approx(
        run(frame, at_floor, warmup_bars=10).metrics.pnl
    )


def test_fee_free_run_earns_the_full_spread():
    config = {**CONFIG, "number_of_order_each_sides": 1}
    result = run(flat_frame(spread=0.015, periods=12), config,
                 warmup_bars=10, initial_base=55.0, fee_rate=0.0)
    size = result.fills[0].size
    assert result.metrics.pnl == pytest.approx(2 * 0.8 * size)
    assert result.metrics.fees_paid == 0.0
    # A fee-free run must say so: on these venues the fee exceeds the touch
    # spread, so removing it changes the answer entirely.
    assert any("fee-free by request" in w for w in result.warnings)

    no_fee_config = {k: v for k, v in config.items() if k != "maker_fee"}
    silent = run(flat_frame(spread=0.015, periods=12), no_fee_config, warmup_bars=10)
    assert any("carries no maker_fee" in w for w in silent.warnings)


# ------------------------------------------------------------- causality


def test_quotes_use_the_previous_close_not_the_current_one():
    """A step change must not be quoted around until the bar after it."""
    frame = flat_frame(periods=40, spread=0.0)
    frame.iloc[20:, :] = frame.iloc[20:, :] * 2.0  # price doubles at bar 20

    result = run(frame, {**CONFIG, "number_of_order_each_sides": 1},
                 warmup_bars=10, initial_base=55.0)

    # The doubling bar has low == high == 200 while quotes still sit around
    # 100, so the ask (100.4) is crossed -- a sell, not a buy.
    at_jump = [f for f in result.fills if f.time == frame.index[20]]
    assert [f.side for f in at_jump] == ["sell"]
    assert at_jump[0].price == pytest.approx(100.4)


def test_short_frames_are_refused_rather_than_silently_trivial():
    with pytest.raises(FrameError, match="need more than"):
        run(flat_frame(periods=20), CONFIG, warmup_bars=50)


def test_sparse_and_broken_frames_warn():
    frame = flat_frame(periods=200)
    keep = np.random.default_rng(3).random(len(frame)) > 0.4
    keep[0] = keep[-1] = True
    result = run(frame[keep], CONFIG, warmup_bars=10)
    assert any("missing" in w for w in result.warnings)

    broken = flat_frame(periods=60)
    broken.iloc[5, broken.columns.get_loc("high")] = 1.0  # high < low
    assert any("high < low" in w for w in run(broken, CONFIG, warmup_bars=10).warnings)


# --------------------------------------------------------------- markout


def test_markout_signs_by_side():
    index = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=index)

    buy = Fill(index[0], "buy", 0, 100.0, 1.0, 0.0)
    sell = Fill(index[0], "sell", 0, 100.0, 1.0, 0.0)

    # Price rose 1%: good for the buy, bad for the sell.
    out = markouts([buy], prices, horizons=(3600,))
    assert out["markout_3600s_bps"] == pytest.approx(100.0)
    out = markouts([sell], prices, horizons=(3600,))
    assert out["markout_3600s_bps"] == pytest.approx(-100.0)


def test_markout_skips_fills_too_close_to_the_end():
    index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
    prices = pd.Series([100.0, 101.0, 102.0], index=index)
    late = Fill(index[-1], "buy", 0, 100.0, 1.0, 0.0)
    assert markouts([late], prices, horizons=(3600,))["markout_3600s_bps"] is None


# ----------------------------------------------------------------- sweep


def test_grid_rejects_knobs_that_are_not_sweepable():
    with pytest.raises(ParamError, match="cannot sweep"):
        grid(tolerant=[0.1, 0.2])

    points = grid(price_max_spread=[0.012, 0.02], max_base=[50.0, 100.0])
    assert len(points) == 4
    assert {"price_max_spread", "max_base"} == set(points[0])


def test_sweep_ranks_by_pnl_and_survives_bad_corners():
    frame = flat_frame(spread=0.015, periods=60)
    points = grid(price_max_spread=[0.008, 0.012, 0.02])
    rows = sweep(frame, {**CONFIG, "number_of_order_each_sides": 1}, points,
                 warmup_bars=10, initial_base=55.0)

    assert len(rows) == 3
    assert [r.pnl for r in rows] == sorted((r.pnl for r in rows), reverse=True)
    # With one level a side, only price_min_spread sets the quote, so all
    # three are equal -- the point is that the sweep ran and ranked them.
    assert all(not r.error for r in rows)


def test_sweep_row_carries_errors_instead_of_aborting():
    frame = flat_frame(spread=0.015, periods=60)
    rows = sweep(frame, CONFIG, [{"max_base": 1.0}, {"max_base": 100.0}],
                 warmup_bars=10)
    assert len(rows) == 2
    # Failed rows sort last.
    assert rows[-1].error or rows[-1].pnl <= rows[0].pnl


def test_render_includes_the_disclaimer():
    frame = flat_frame(spread=0.015, periods=60)
    rows = sweep(frame, CONFIG, grid(max_base=[50.0, 100.0]), warmup_bars=10)
    out = render(rows)

    assert "PNL" in out and "MO600" in out and "FILLS" in out
    assert "UPPER BOUND" in out
    assert "Read the ordering, not the levels." in out
    assert render([]) == "no sweep results."


# ------------------------------------------------------------- artifacts


def test_every_artifact_carries_the_caveat_and_divergences():
    result = run(flat_frame(spread=0.015, periods=60), CONFIG, warmup_bars=10)
    payload = result.as_dict()

    assert payload["disclaimer"] == DISCLAIMER
    assert "UPPER BOUND" in payload["disclaimer"]
    assert list(payload["divergences"]) == list(DIVERGENCES)
    assert any("bypass-wall" in d for d in payload["divergences"])
    assert "fills" not in payload
    assert "fills" in result.as_dict(include_fills=True)


def test_compare_runs_variants_over_the_same_bars():
    frame = flat_frame(spread=0.015, periods=60)
    out = compare(frame, CONFIG, {"wide": {**CONFIG, "price_max_spread": 0.03}},
                  warmup_bars=10)
    assert set(out) == {"baseline", "wide"}
    assert out["baseline"].metrics.bars == out["wide"].metrics.bars


def test_overlays_produce_a_timeline():
    rng = np.random.default_rng(11)
    periods = 300
    index = pd.date_range("2026-01-01", periods=periods, freq="1h", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, periods)))
    frame = pd.DataFrame(
        {"open": close, "high": close * 1.02, "low": close * 0.98,
         "close": close, "volume": 1.0},
        index=index,
    )
    result = run(frame, CONFIG, overlays="vol_widen", warmup_bars=100,
                 overlay_every=50)

    assert result.overlay_timeline  # vol crosses the percentile threshold
    entry = result.overlay_timeline[0]
    assert "time" in entry and "spread_multiplier" in entry["delta"]


# --------------------------------------------- out-of-band recovery (proposal)


RECOVERY_CONFIG = {**CONFIG, "underprice_multiplier": 1.5, "overprice_multiplier": 0.5}


def test_effective_band_applies_the_regime_multipliers():
    """The ordering the live maker gets wrong: the regime moves the bound
    before anything asks whether inventory is outside it."""
    from ta.backtest import effective_band

    assert effective_band(RECOVERY_CONFIG) == (10.0, 100.0)
    # UNDERPRICE is deliberately buying up to 1.5x max: the ceiling rises.
    assert effective_band(RECOVERY_CONFIG, "underprice") == (10.0, 150.0)
    # OVERPRICED is selling down to 0.5x min: the floor drops.
    assert effective_band(RECOVERY_CONFIG, "overprice") == (5.0, 100.0)


def test_regime_prevents_a_recovery_that_would_fight_it():
    """Holding 130 with max 100 looks over-full — until you notice the
    UNDERPRICE branch is intentionally buying up to 150."""
    from ta.backtest import band_breach, effective_band

    assert band_breach(130.0, *effective_band(RECOVERY_CONFIG))[0] == "sell"
    assert band_breach(130.0, *effective_band(RECOVERY_CONFIG, "underprice"))[0] is None


def test_band_breach_does_not_saturate_at_the_edge():
    """Unlike inventory_q, which clamps at +/-1 and so stops responding
    exactly where urgency should start."""
    from ta.backtest import band_breach

    assert band_breach(55.0, 10.0, 100.0) == (None, 0.0)   # inside
    assert band_breach(10.0, 10.0, 100.0) == (None, 0.0)   # exactly at the edge
    # span = 0.25 * 90 = 22.5 below 10.
    assert band_breach(-1.25, 10.0, 100.0)[1] == pytest.approx(0.5)
    assert band_breach(-12.5, 10.0, 100.0)[1] == pytest.approx(1.0)
    assert band_breach(-999.0, 10.0, 100.0)[1] == 1.0      # capped, not unbounded

    side, amount = band_breach(122.5, 10.0, 100.0)
    assert side == "sell" and amount == pytest.approx(1.0)

    # inventory_q, for contrast, is already saturated at both points.
    assert inventory_q(CONFIG, 5.0) == inventory_q(CONFIG, -500.0) == -1.0


def test_recovery_tightens_toward_the_fee_floor_and_provably_no_further():
    from ta.backtest import recovery_spreads, spread_floor

    fee = spread_floor(CONFIG)
    base_min, _ = effective_spreads(CONFIG, side="buy")

    assert recovery_spreads(CONFIG, "buy", 0.0)[0] == pytest.approx(base_min)
    half = recovery_spreads(CONFIG, "buy", 0.5)[0]
    assert fee < half < base_min

    for breach in (1.0, 2.0, 100.0):
        recovered_min, recovered_max = recovery_spreads(CONFIG, "buy", breach)
        assert recovered_min == pytest.approx(fee)
        assert recovered_max >= recovered_min       # ladder never inverts
        assert recovered_min >= fee                 # never through the fee


def test_recovery_makes_the_recovering_side_more_aggressive_only():
    """Below the floor the bids climb toward the reference; the asks are
    left exactly where they were."""
    starved = {**CONFIG, "min_base": 40.0, "max_base": 100.0}
    plain = build_ladder(starved, 100.0, 20.0)
    recovered = build_ladder(starved, 100.0, 20.0, recovery=True)

    assert recovered.bids[0].price > plain.bids[0].price   # more aggressive bid
    assert [q.price for q in recovered.asks] == [q.price for q in plain.asks]
    # And never through break-even: fee is 10bp, so the bid stays <= 99.9.
    assert recovered.bids[0].price <= 100.0 * (1 - spread_floor(starved)) + 1e-9


def test_recovery_is_symmetric_above_the_ceiling():
    stuffed = {**CONFIG, "min_base": 10.0, "max_base": 60.0}
    plain = build_ladder(stuffed, 100.0, 95.0)
    recovered = build_ladder(stuffed, 100.0, 95.0, recovery=True)

    assert recovered.asks[0].price < plain.asks[0].price   # more aggressive ask
    assert recovered.asks[0].price >= 100.0 * (1 + spread_floor(stuffed)) - 1e-9
    assert [q.price for q in recovered.bids] == [q.price for q in plain.bids]


def test_recovery_is_inert_inside_the_band():
    for base in (20.0, 55.0, 90.0):
        plain = build_ladder(CONFIG, 100.0, base)
        recovered = build_ladder(CONFIG, 100.0, base, recovery=True)
        assert [q.price for q in recovered] == [q.price for q in plain]


def test_recovery_never_crosses_the_book_at_any_breach():
    """The property that makes this safe to propose: however starved or
    stuffed, the best bid stays below the best ask."""
    for min_base, max_base in ((40.0, 100.0), (10.0, 60.0), (10.0, 450.0)):
        config = {**CONFIG, "min_base": min_base, "max_base": max_base}
        for base in np.linspace(-200.0, 800.0, 60):
            ladder = build_ladder(config, 100.0, float(base), recovery=True)
            if ladder.bids and ladder.asks:
                assert max(q.price for q in ladder.bids) < min(
                    q.price for q in ladder.asks
                )
