"""Exact-math tests for the core indicators (the five the legacy service had,
plus ATR and Bollinger from the expansion batch).

Each is checked against an independent reference: either a hand-computed tiny
case or a from-first-principles loop that does not share code with the
implementation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ta.indicators.momentum import (
    _rsi_series,
    macd_indicator,
    rsi_indicator,
    stoch_rsi_indicator,
)
from ta.indicators.trend import ema_indicator, sma_indicator
from ta.indicators.volatility import atr_indicator, bbands_indicator


def series(values, freq="1h"):
    index = pd.date_range("2026-01-01", periods=len(values), freq=freq, tz="UTC")
    return pd.DataFrame({"close": [float(v) for v in values]}, index=index)


# ---------------------------------------------------------------------- SMA


def test_sma_hand_case():
    out = sma_indicator(series([1, 2, 3, 4, 5]), window=3)["sma_3"]
    assert out.iloc[:2].isna().all()
    assert out.iloc[2:].tolist() == [2.0, 3.0, 4.0]


def test_sma_matches_rolling_mean(ohlc):
    out = sma_indicator(ohlc, window=20)["sma_20"]
    expected = ohlc["close"].rolling(20).mean()
    pd.testing.assert_series_equal(out, expected, check_names=False)


# ---------------------------------------------------------------------- EMA


def test_ema_matches_reference_recursion(ohlc):
    window = 10
    out = ema_indicator(ohlc, window=window)[f"ema_{window}"]

    alpha = 2.0 / (window + 1.0)
    closes = ohlc["close"].to_numpy()
    value = closes[0]
    reference = [value]
    for x in closes[1:]:
        value = value + alpha * (x - value)
        reference.append(value)

    # min_periods masks the first window-1 bars; past that the recursion
    # (seeded on the first close) must match exactly.
    assert out.iloc[: window - 1].isna().all()
    np.testing.assert_allclose(
        out.iloc[window - 1 :].to_numpy(), reference[window - 1 :], rtol=1e-12
    )


# ---------------------------------------------------------------------- RSI


def ref_wilder_average(values: np.ndarray, period: int) -> np.ndarray:
    """Independent Wilder average: seeded on the first observation, masked
    until `period` observations, NaN inputs skipped -- the ewm semantics the
    implementation relies on, rebuilt by hand."""
    alpha = 1.0 / period
    out = np.full(len(values), np.nan)
    average = None
    seen = 0
    for i, x in enumerate(values):
        if np.isnan(x):
            continue
        seen += 1
        average = x if average is None else average + alpha * (x - average)
        if seen >= period:
            out[i] = average
    return out


def test_rsi_matches_reference_loop(ohlc):
    period = 14
    out = rsi_indicator(ohlc, period=period)[f"rsi_{period}"].to_numpy()

    delta = ohlc["close"].diff().to_numpy()
    gains = np.where(np.isnan(delta), np.nan, np.clip(delta, 0.0, None))
    losses = np.where(np.isnan(delta), np.nan, np.clip(-delta, 0.0, None))
    avg_gain = ref_wilder_average(gains, period)
    avg_loss = ref_wilder_average(losses, period)

    with np.errstate(divide="ignore", invalid="ignore"):
        reference = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    reference = np.where(
        np.isfinite(avg_gain) & (avg_loss == 0.0) & (avg_gain > 0.0), 100.0, reference
    )

    np.testing.assert_allclose(out, reference, rtol=1e-10, equal_nan=True)


def test_rsi_extremes():
    monotone_up = series(np.arange(1, 60, dtype=float))
    up = rsi_indicator(monotone_up, period=14)["rsi_14"]
    assert (up.dropna() == 100.0).all()

    monotone_down = series(np.arange(60, 1, -1, dtype=float))
    down = rsi_indicator(monotone_down, period=14)["rsi_14"]
    assert (down.dropna() == 0.0).all()

    flat = series([5.0] * 60)
    assert (rsi_indicator(flat, period=14)["rsi_14"].dropna() == 50.0).all()


# ---------------------------------------------------------------------- MACD


def test_macd_matches_ewm_composition(ohlc):
    out = macd_indicator(ohlc, fast=12, slow=26, signal=9)

    close = ohlc["close"]
    line = (
        close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    )
    # The implementation masks the line's min_periods head, so its signal EMA
    # seeds on bar slow-1, not bar 0. Mirror that.
    masked_line = line.copy()
    masked_line.iloc[:25] = np.nan
    signal = masked_line.ewm(span=9, adjust=False).mean()

    # Outputs are masked by min_periods on the component EMAs; compare where defined.
    got_line = out["macd_12_26_9"]
    defined = got_line.notna()
    assert defined.sum() > 400
    np.testing.assert_allclose(got_line[defined], line[defined], rtol=1e-12)
    got_hist = out["macd_hist_12_26_9"]
    hist_defined = got_hist.notna()
    np.testing.assert_allclose(
        got_hist[hist_defined],
        (line - signal)[hist_defined],
        rtol=1e-9,
        atol=1e-12,
    )


# ---------------------------------------------------------------- Stoch RSI


def test_stoch_rsi_composition_and_bounds(ohlc):
    period, smooth_k, smooth_d = 14, 3, 3
    out = stoch_rsi_indicator(ohlc, period=period, smooth_k=smooth_k, smooth_d=smooth_d)
    k = out[f"stoch_rsi_k_{period}_{smooth_k}"]
    d = out[f"stoch_rsi_d_{period}_{smooth_k}_{smooth_d}"]

    assert k.dropna().between(0.0, 100.0).all()
    assert d.dropna().between(0.0, 100.0).all()

    rsi = _rsi_series(ohlc["close"], period)
    lowest = rsi.rolling(period, min_periods=period).min()
    highest = rsi.rolling(period, min_periods=period).max()
    span = (highest - lowest).replace(0.0, np.nan)
    raw = 100.0 * (rsi - lowest) / span
    expected_k = raw.rolling(smooth_k, min_periods=smooth_k).mean()

    pd.testing.assert_series_equal(k, expected_k, check_names=False)
    pd.testing.assert_series_equal(
        d, k.rolling(smooth_d, min_periods=smooth_d).mean(), check_names=False
    )


# ---------------------------------------------------------------------- ATR


def test_atr_matches_reference(ohlc):
    period = 14
    out = atr_indicator(ohlc, period=period)[f"atr_{period}"].to_numpy()

    high = ohlc["high"].to_numpy()
    low = ohlc["low"].to_numpy()
    close = ohlc["close"].to_numpy()
    tr = np.empty(len(ohlc))
    tr[0] = high[0] - low[0]  # Wilder: first bar has no previous close
    for i in range(1, len(ohlc)):
        tr[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )

    np.testing.assert_allclose(
        out, ref_wilder_average(tr, period), rtol=1e-10, equal_nan=True
    )


# ----------------------------------------------------------------- Bollinger


def test_bbands_geometry(ohlc):
    out = bbands_indicator(ohlc, window=20, std_dev=2.0)

    mid = out["bb_mid_20"]
    upper = out["bb_upper_20_2.0"]
    lower = out["bb_lower_20_2.0"]

    pd.testing.assert_series_equal(mid, ohlc["close"].rolling(20).mean(), check_names=False)

    # Population deviation (ddof=0), not pandas' sample default.
    deviation = ohlc["close"].rolling(20).std(ddof=0)
    np.testing.assert_allclose(
        (upper - lower).dropna(), (4.0 * deviation).dropna(), rtol=1e-12
    )

    width = out["bb_width_20_2.0"]
    np.testing.assert_allclose(
        width.dropna(), ((upper - lower) / mid).dropna(), rtol=1e-12
    )
