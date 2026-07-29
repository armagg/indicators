"""Frame normalization, health reporting, and shared numeric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ta.errors import FrameError
from ta.frame import (
    drop_incomplete_bar,
    normalize_ohlcv,
    rolling_std,
    safe_divide,
    true_range,
    wilder_ema,
)


def test_normalize_from_mongo_style_docs():
    docs = [
        {
            "time": "2026-01-01T10:00:00",
            "open": "100",
            "high": 105,
            "low": 95,
            "close": 102,
            "volume": 7,
            "market_name": "USDTIRR",
            "source": "nobitex",
            "interval": "1H",
        },
        {
            "time": "2026-01-01T11:00:00",
            "open": 102,
            "high": 108,
            "low": 100,
            "close": 107,
            "volume": 3,
            "market_name": "USDTIRR",
            "source": "nobitex",
            "interval": "1H",
        },
    ]
    df, info = normalize_ohlcv(docs)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.tz is not None
    assert df["open"].dtype == "float64"
    assert info.rows == 2 and info.has_volume


def test_normalize_renames_collector_columns():
    docs = [
        {"time": "2026-01-01T10:00:00", "price_first": 1, "price_max": 2, "price_min": 0.5, "price_last": 1.5, "amount_sum": 9},
        {"time": "2026-01-01T11:00:00", "price_first": 1.5, "price_max": 3, "price_min": 1, "price_last": 2, "amount_sum": 4},
    ]
    df, _ = normalize_ohlcv(docs, require=("open", "close"))
    assert {"open", "high", "low", "close", "volume"} <= set(df.columns)


def test_normalize_drops_duplicates_and_requires_columns():
    docs = [
        {"time": "2026-01-01T10:00:00", "close": 1.0},
        {"time": "2026-01-01T10:00:00", "close": 2.0},  # duplicate timestamp
        {"time": "2026-01-01T11:00:00", "close": 3.0},
    ]
    df, _ = normalize_ohlcv(docs)
    assert len(df) == 2
    assert df["close"].iloc[0] == 2.0  # keep="last"

    with pytest.raises(FrameError, match="high"):
        normalize_ohlcv(docs, require=("high",))
    with pytest.raises(FrameError):
        normalize_ohlcv([])


def test_gap_ratio_reports_sparse_series():
    # Few enough holes that the median delta still infers the 1h grid.
    times = pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC")
    holey = times.delete([2, 3, 5, 7, 9])  # 19 of 24 bars remain
    df = pd.DataFrame({"close": np.arange(len(holey), dtype=float)}, index=holey)
    _, info = normalize_ohlcv(df)
    assert info.interval_sec == 3600.0
    assert info.expected_rows == 24
    assert info.missing_bars == 5
    assert info.is_sparse  # 5/24 > 0.2


def test_drop_incomplete_bar():
    times = pd.date_range("2026-01-01 00:00", periods=3, freq="1h", tz="UTC")
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=times)

    # 02:30 -- the 02:00 bar is still forming.
    out, dropped = drop_incomplete_bar(df, 3600, now=pd.Timestamp("2026-01-01 02:30", tz="UTC"))
    assert dropped and len(out) == 2
    # 03:00 -- the 02:00 bar just completed.
    out, dropped = drop_incomplete_bar(df, 3600, now=pd.Timestamp("2026-01-01 03:00", tz="UTC"))
    assert not dropped and len(out) == 3


def test_true_range_first_bar_is_high_minus_low(tiny):
    tr = true_range(tiny)
    assert tr.iloc[0] == pytest.approx(tiny["high"].iloc[0] - tiny["low"].iloc[0])
    # Bar 2: h-l = 1.5, |h - prev_close| = 0.5, |l - prev_close| = 1.0
    assert tr.iloc[2] == pytest.approx(1.5)


def test_wilder_ema_is_not_span_ema():
    series = pd.Series(np.linspace(1, 50, 50))
    wilder = wilder_ema(series, 14)
    span = series.ewm(span=14, adjust=False).mean()
    assert not np.allclose(wilder.dropna(), span[wilder.notna()])


def test_safe_divide_maps_zero_denominator_to_nan():
    numerator = pd.Series([1.0, 0.0, 2.0])
    denominator = pd.Series([0.0, 0.0, 4.0])
    out = safe_divide(numerator, denominator)
    assert out.isna().tolist() == [True, True, False]
    assert out.iloc[2] == 0.5


def test_rolling_std_is_population():
    series = pd.Series([1.0, 2.0, 3.0, 4.0])
    got = rolling_std(series, 4).iloc[-1]
    assert got == pytest.approx(np.std([1, 2, 3, 4]))  # ddof=0
    assert got != pytest.approx(np.std([1, 2, 3, 4], ddof=1))
