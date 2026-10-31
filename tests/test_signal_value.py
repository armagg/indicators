"""The drift-vs-covariance decomposition, on constructed cases where the
right answer is known by hand."""

from __future__ import annotations

import numpy as np
import pytest

from ta.signal_value import (
    SignalValue,
    consecutive_mask,
    evaluate,
    forward_returns,
    normalize_signal,
    permutation_pvalue,
)


def test_forward_returns_are_stamped_on_the_bar_that_precedes_them():
    close = np.array([100.0, 110.0, 121.0])
    fwd = forward_returns(close)

    assert fwd[0] == pytest.approx(np.log(1.1))
    assert fwd[1] == pytest.approx(np.log(1.1))
    assert np.isnan(fwd[-1])  # no future for the last bar


def test_a_useless_signal_on_a_trending_market_scores_zero_excess():
    """The whole point. A constant-ish signal in a market that only rises
    shows a big naive number and no excess -- because a constant inventory
    would have earned exactly the same."""
    rng = np.random.default_rng(1)
    n = 5000
    ret = rng.normal(0.0005, 0.003, n)  # strong upward drift
    signal = np.ones(n)  # says "hold more" always: no information

    value = evaluate(signal, ret)

    assert value.naive_bps == pytest.approx(value.drift_bps, rel=1e-9)
    assert value.excess_bps == pytest.approx(0.0, abs=1e-9)
    assert abs(value.ic) < 1e-9


def test_a_real_signal_scores_excess_even_without_drift():
    rng = np.random.default_rng(2)
    n = 5000
    signal = rng.normal(0, 1, n)
    ret = 0.001 * signal + rng.normal(0, 0.002, n)  # genuinely predictive

    value = evaluate(signal, ret)

    assert value.excess_bps > 0
    assert value.ic > 0.3
    assert value.ic_t > 5
    assert value.hit_rate > 0.6


def test_drift_inflates_the_naive_number_but_not_the_excess():
    """The real-world failure mode, reproduced.

    A rule that sits long most of the time has a positive mean signal, so on
    a trending market its naive score picks up `mean(s) * mean(r)` -- pure
    drift it did not earn. The excess is untouched by that.
    """
    rng = np.random.default_rng(3)
    n = 5000
    signal = 0.5 + rng.normal(0, 1, n)  # mostly-long, like an MA rule
    base = 0.0005 * signal + rng.normal(0, 0.002, n)

    plain = evaluate(signal, base)
    trending = evaluate(signal, base + 0.002)  # rial devaluation, in effect

    assert trending.naive_bps > plain.naive_bps + 5  # inflated by the drift
    assert trending.drift_bps > plain.drift_bps + 5
    assert trending.excess_bps == pytest.approx(plain.excess_bps, rel=1e-6)


def test_an_inverted_signal_scores_negative():
    rng = np.random.default_rng(4)
    n = 3000
    signal = rng.normal(0, 1, n)
    ret = -0.001 * signal + rng.normal(0, 0.002, n)

    value = evaluate(signal, ret)
    assert value.excess_bps < 0 and value.ic < 0


def test_consecutive_mask_finds_the_regular_pairs():
    import pandas as pd

    index = pd.date_range("2026-01-01", periods=10, freq="1h", tz="UTC")
    holey = index.delete([4, 5])  # a two-hour hole after bar 3

    mask = consecutive_mask(holey)
    # The bar before the hole has no immediate successor.
    assert mask[2] and not mask[3] and mask[4]
    assert not mask[-1]  # nothing follows the last bar

    assert not consecutive_mask(index[:1]).any()


def test_consecutive_mask_removes_variable_horizon_pairs():
    """The confound in miniature: a signal that only fires before gaps looks
    predictive when the gap lets the price run further."""
    import pandas as pd

    index = pd.date_range("2026-01-01", periods=12, freq="1h", tz="UTC").delete([5, 6])
    mask = consecutive_mask(index)
    assert mask.sum() < len(index) - 1


def test_normalize_maps_to_a_bounded_tilt():
    values = np.concatenate([np.arange(-50, 50, dtype=float), [10_000.0]])
    tilt = normalize_signal(values)

    assert np.nanmin(tilt) >= -1.0 and np.nanmax(tilt) <= 1.0
    # A single outlier must not set the scale for everything else.
    assert abs(tilt[-1]) == pytest.approx(1.0)
    assert np.percentile(np.abs(tilt[:-1]), 90) < 1.0

    assert np.all(normalize_signal(np.full(10, 5.0)) == 0.0)  # no dispersion
    assert np.isnan(normalize_signal(np.full(5, np.nan))).all()


@pytest.mark.parametrize("scale", [1e-9, 1.0, 1e9])
def test_normalization_ignores_roundoff_but_keeps_real_small_variation(scale):
    values = np.array([np.nan, 1.0, 1.0 + 1e-14, 1.0 - 1e-14]) * scale
    result = normalize_signal(values)
    assert np.isnan(result[0])
    assert np.all(result[1:] == 0.0)

    varying = np.array([1.0, 1.001, 0.999, 1.002]) * scale
    result = normalize_signal(varying)
    assert result.min() < 0 < result.max()


def test_evaluate_drops_warmup_and_refuses_thin_overlap():
    signal = np.concatenate([np.full(200, np.nan), np.ones(500)])
    ret = np.full(700, 0.001)
    assert evaluate(signal, ret).bars == 500

    assert evaluate(np.ones(50), np.full(50, 0.001)) is None
    with pytest.raises(ValueError, match="differ in length"):
        evaluate(np.ones(10), np.ones(11))


def test_permutation_null_is_unmoved_by_drift():
    """Why `excess` is the right statistic to permutation-test: however hard
    the market trends, the null for a worthless signal stays put. A naive
    P&L null would slide upward with the drift until noise looked
    significant."""
    rng = np.random.default_rng(5)
    n = 2000
    noise = rng.normal(0, 0.003, n)
    signals = [rng.normal(0, 1, n) for _ in range(20)]  # all worthless

    flat = permutation_pvalue(0.0, signals, noise, trials=30, seed=1)
    trending = permutation_pvalue(0.0, signals, noise + 0.001, trials=30, seed=1)

    assert flat["trials"] == trending["trials"] == 30
    # Best-of-20 worthless signals is positive by luck -- that is precisely
    # what a grid search buys you, and why the null is not zero.
    assert flat["null_mean_bps"] > 0
    assert flat["null_p95_bps"] > flat["null_mean_bps"]
    # Adding a large drift leaves the null essentially where it was.
    assert trending["null_mean_bps"] == pytest.approx(flat["null_mean_bps"], rel=0.02)


def test_permutation_pvalue_flags_a_result_inside_the_null():
    rng = np.random.default_rng(6)
    n = 2000
    ret = rng.normal(0.0005, 0.003, n)
    signals = [rng.normal(0, 1, n) for _ in range(10)]

    weak = permutation_pvalue(0.01, signals, ret, trials=30)
    strong = permutation_pvalue(50.0, signals, ret, trials=30)

    assert weak["p_value"] > 0.3   # indistinguishable from noise
    assert strong["p_value"] == 0.0
