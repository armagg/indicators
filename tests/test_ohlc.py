"""OHLC reader helpers -- the pure parts. Live Mongo access is not tested."""

from __future__ import annotations

import pandas as pd
import pytest

from ta.errors import ParamError
from ta.ohlc import docs_to_frame, normalize_interval, query_time


def test_normalize_interval_is_forgiving_but_strict():
    assert normalize_interval("1h") == "1H"
    assert normalize_interval("15MIN") == "15Min"
    assert normalize_interval(" 1D ") == "1D"
    with pytest.raises(ParamError, match="valid:"):
        normalize_interval("2H")


def test_query_time_naive_means_tehran_wall_time():
    # Naive input is taken as Tehran wall time and used verbatim.
    assert query_time("2026-07-30 14:05:00") == "2026-07-30T14:05:00"


def test_query_time_aware_is_converted_to_tehran():
    # 10:35 UTC == 14:05 Tehran (+3:30).
    assert query_time(pd.Timestamp("2026-07-30 10:35:00", tz="UTC")) == "2026-07-30T14:05:00"


def test_docs_to_frame_relabels_tehran_strings_as_utc():
    docs = [
        {"time": "2026-07-30T14:05:00", "close": 2.0},
        {"time": "2026-07-30T13:05:00", "close": 1.0},  # out of order on purpose
    ]
    df = docs_to_frame(docs)

    assert df.index.tz is not None
    assert list(df["close"]) == [1.0, 2.0]  # sorted oldest first
    # 14:05 Tehran == 10:35 UTC: the label moves, the instant does not.
    assert df.index[-1] == pd.Timestamp("2026-07-30 10:35:00", tz="UTC")
