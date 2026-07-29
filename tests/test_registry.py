"""Registry: parameter coercion, lookup, and the catalogue."""

from __future__ import annotations

import pytest

import ta
from ta.errors import ParamError, UnknownIndicator
from ta.registry import (
    BoolParam,
    CATEGORIES,
    ChoiceParam,
    FloatParam,
    IntParam,
    all_specs,
    catalogue,
    get_spec,
)


def test_int_param_coercion():
    param = IntParam("period", 14, minimum=2, maximum=100)
    assert param.coerce("21") == 21
    assert param.coerce("14.0") == 14  # query strings arrive as text
    with pytest.raises(ParamError):
        param.coerce("abc")
    with pytest.raises(ParamError):
        param.coerce(1)  # below minimum
    with pytest.raises(ParamError):
        param.coerce(101)  # above maximum


def test_float_choice_bool_params():
    assert FloatParam("x", 2.0, minimum=0.1).coerce("2.5") == 2.5
    with pytest.raises(ParamError):
        FloatParam("x", 2.0, minimum=0.1).coerce("0")

    choice = ChoiceParam("anchor", "day", choices=("day", "week"))
    assert choice.coerce("WEEK") == "week"
    with pytest.raises(ParamError):
        choice.coerce("year")

    flag = BoolParam("save", True)
    assert flag.coerce("no") is False and flag.coerce("1") is True
    with pytest.raises(ParamError):
        flag.coerce("maybe")


def test_resolve_params_fills_defaults_and_rejects_unknown():
    spec = get_spec("macd")
    resolved = spec.resolve_params({"fast": "8"})
    assert resolved == {"fast": 8, "slow": 26, "signal": 9}

    with pytest.raises(ParamError, match="unknown parameter"):
        spec.resolve_params({"fats": 8})


def test_lookup_aliases_and_unknown():
    assert get_spec("bollinger").name == "bbands"
    assert get_spec("stochastic_rsi").name == "stoch_rsi"
    assert get_spec("RSI").name == "rsi"  # case-insensitive
    with pytest.raises(UnknownIndicator, match="known:"):
        get_spec("vibes")


def test_catalogue_shape():
    cat = catalogue()
    assert cat["count"] == len(all_specs())
    assert set(cat["categories"]) <= set(CATEGORIES)
    listed = [spec for specs in cat["categories"].values() for spec in specs]
    assert len(listed) == cat["count"]
    sample = listed[0]
    for key in ("name", "params", "outputs", "requires", "causal", "warmup_default"):
        assert key in sample
    assert cat["aliases"]["bollinger"] == "bbands"


def test_output_names_are_unique_across_library():
    """No two indicators may claim the same concrete column at defaults --
    otherwise a batch request silently collapses them."""
    seen = {}
    for spec in all_specs():
        for column in spec.output_names(spec.defaults()):
            assert column not in seen, f"{spec.name} and {seen[column]} both emit {column}"
            seen[column] = spec.name
