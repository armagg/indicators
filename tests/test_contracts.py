"""One contract test, run against every registered indicator.

This is what the registry buys us: any indicator added later is automatically
held to the same rules -- pure, correctly-named columns, aligned index, no
infinities, and a warmup declaration that leaves usable output on a
500-bar frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ta
from ta.registry import all_specs

SPECS = all_specs()


def test_registry_is_populated():
    assert len(SPECS) >= 30, "importing ta should register the full library"


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_indicator_contract(spec, ohlc):
    params = spec.defaults()
    before = ohlc.copy(deep=True)

    out = spec.fn(ohlc, **params)

    # Purity: the input frame is untouched.
    pd.testing.assert_frame_equal(ohlc, before)

    assert isinstance(out, pd.DataFrame)
    assert tuple(out.columns) == spec.output_names(params)
    assert out.index.equals(ohlc.index)

    values = out.to_numpy(dtype="float64")
    assert not np.isinf(values).any(), "inf must never escape an indicator"

    # The declared warmup must leave something usable on a 500-bar frame,
    # and every declared output column must actually produce values.
    assert spec.warmup_bars(params) < len(ohlc)
    assert out.notna().any().all()


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_indicator_through_engine(spec, ohlc):
    """Every indicator also runs through `ta.compute` with default params."""
    result = ta.compute(
        ohlc, spec.name, allow_noncausal=not spec.causal, warmup="mask"
    )
    for column in spec.output_names(spec.defaults()):
        assert column in result.frame.columns
