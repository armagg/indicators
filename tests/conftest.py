"""Shared fixtures: deterministic synthetic OHLCV frames.

Seeded RNG, so every failure is reproducible. The main fixture is a plausible
geometric random walk with proper high/low envelopes -- not real market data,
which is the point: tests must not depend on Mongo.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlc(n: int = 500, seed: int = 7, freq: str = "1h") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")

    steps = rng.normal(0.0002, 0.01, n)
    close = 100.0 * np.exp(np.cumsum(steps))
    open_ = np.concatenate(([100.0], close[:-1]))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, n)))
    volume = rng.lognormal(2.0, 1.0, n)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def ohlc() -> pd.DataFrame:
    return make_ohlc()


@pytest.fixture
def tiny() -> pd.DataFrame:
    """Five bars with hand-checkable numbers."""
    index = pd.date_range("2026-01-01", periods=5, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [10.0, 11.0, 12.0, 11.0, 12.0],
            "high": [11.5, 12.5, 12.5, 12.0, 13.0],
            "low": [9.5, 10.5, 11.0, 10.5, 11.5],
            "close": [11.0, 12.0, 11.0, 12.0, 13.0],
            "volume": [1.0, 2.0, 3.0, 4.0, 5.0],
        },
        index=index,
    )
