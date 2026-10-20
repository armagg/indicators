"""Overlay mapper: analysis verdicts -> allowlisted MM parameter deltas; pure functions, no writes."""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields
from datetime import datetime
from typing import Any, Mapping

from ..analysis.tilt import DEFAULT_CAP as DEFAULT_TILT_CAP
from ..analysis.tilt import implied_target_base
from ..analysis.types import AnalysisResult, call_with_loose_params
from ..errors import ParamError

_SPREAD_MULTIPLIER_CAP = 5.0

_MAX_STRUCTURAL_PREMIUM = 0.05

_MAX_SPREAD_FRACTION = 0.5

ABSOLUTE_FIELDS = frozenset(
    {
        "price_min_spread",
        "price_max_spread",
        "min_base",
        "max_base",
        "target_base",
        "is_enable",
        "number_of_order_each_sides",
    }
)

HIGH_IMPACT_FIELDS = frozenset({"is_enable", "max_base", "number_of_order_each_sides"})


@dataclass(frozen=True)
class ConfigDelta:
    """A restricted, validated set of proposed MM parameter changes."""

    spread_multiplier: float | None = None
    premium_skew_factor: float | None = None
    structural_premium: float | None = None
    disable_buy: bool | None = None
    disable_sell: bool | None = None
    vol_spread_sensitivity: float | None = None
    price_min_spread: float | None = None
    price_max_spread: float | None = None
    min_base: float | None = None
    max_base: float | None = None
    target_base: float | None = None
    is_enable: bool | None = None
    number_of_order_each_sides: int | None = None
    reason: str = ""

    def __post_init__(self):
        if self.spread_multiplier is not None and not (
            0.0 < self.spread_multiplier <= _SPREAD_MULTIPLIER_CAP
        ):
            raise ParamError(
                f"spread_multiplier must be in (0, {_SPREAD_MULTIPLIER_CAP}], "
                f"got {self.spread_multiplier}"
            )
        if self.premium_skew_factor is not None and not (
            0.0 <= self.premium_skew_factor <= 1.0
        ):
            raise ParamError(
                f"premium_skew_factor must be in [0, 1], got {self.premium_skew_factor}"
            )
        if self.structural_premium is not None and not (
            abs(self.structural_premium) <= _MAX_STRUCTURAL_PREMIUM
        ):
            raise ParamError(
                f"|structural_premium| must be <= {_MAX_STRUCTURAL_PREMIUM} "
                f"(that is {_MAX_STRUCTURAL_PREMIUM * 1e4:.0f}bp), got {self.structural_premium}"
            )
        for side in ("disable_buy", "disable_sell"):
            if getattr(self, side) is False:
                raise ParamError(
                    f"{side}=False is not allowed: an overlay may pause a side, "
                    f"never resume one the risk plane turned off"
                )
        if self.vol_spread_sensitivity is not None and self.vol_spread_sensitivity < 0:
            raise ParamError(
                f"vol_spread_sensitivity must be >= 0, got {self.vol_spread_sensitivity}"
            )

        for name in ("price_min_spread", "price_max_spread"):
            value = getattr(self, name)
            if value is not None and not (0.0 < value <= _MAX_SPREAD_FRACTION):
                raise ParamError(
                    f"{name} is a fraction of price and must be in "
                    f"(0, {_MAX_SPREAD_FRACTION}]; got {value}. "
                    f"(0.002 is 20bp, not 0.2.)"
                )
        if self.min_base is not None and self.min_base < 0:
            raise ParamError(f"min_base must be >= 0, got {self.min_base}")
        if self.max_base is not None and self.max_base <= 0:
            raise ParamError(f"max_base must be > 0, got {self.max_base}")
        if self.target_base is not None and self.target_base < 0:
            raise ParamError(f"target_base must be >= 0, got {self.target_base}")
        if self.number_of_order_each_sides is not None and not (
            1 <= self.number_of_order_each_sides <= 50
        ):
            raise ParamError(
                f"number_of_order_each_sides must be in [1, 50], "
                f"got {self.number_of_order_each_sides}"
            )


    def fields(self) -> dict[str, Any]:
        """The config fields this delta actually sets (reason excluded)."""
        return {
            f.name: getattr(self, f.name)
            for f in dataclass_fields(self)
            if f.name != "reason" and getattr(self, f.name) is not None
        }

    @property
    def is_empty(self) -> bool:
        return not self.fields()

    @property
    def is_high_impact(self) -> bool:
        """True when this delta commits more capital or resumes quoting."""
        touched = self.fields()
        if touched.get("is_enable") is True:
            return True
        return any(
            name in HIGH_IMPACT_FIELDS and name != "is_enable" for name in touched
        )

    def as_dict(self) -> dict:
        out = self.fields()
        out["reason"] = self.reason
        return out


    def apply_to(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Compose this delta onto a baseline config and validate the result."""
        out = dict(config)

        if self.spread_multiplier is not None:
            base = float(out.get("spread_multiplier") or 1.0)
            out["spread_multiplier"] = min(
                _SPREAD_MULTIPLIER_CAP, base * self.spread_multiplier
            )
        if self.vol_spread_sensitivity is not None:
            base = float(out.get("vol_spread_sensitivity") or 0.0)
            out["vol_spread_sensitivity"] = max(base, self.vol_spread_sensitivity)
        for side in ("disable_buy", "disable_sell"):
            if getattr(self, side):
                out[side] = True
        if self.structural_premium is not None:
            out["structural_premium"] = self.structural_premium
        if self.premium_skew_factor is not None:
            out["premium_skew_factor"] = self.premium_skew_factor

        for name in ABSOLUTE_FIELDS:
            value = getattr(self, name)
            if value is not None:
                out[name] = value

        _validate_result(out, self)
        return out


def _validate_result(config: Mapping[str, Any], delta: ConfigDelta) -> None:
    """Cross-field checks against a *composed* config."""
    min_spread = config.get("price_min_spread")
    max_spread = config.get("price_max_spread")
    if (
        min_spread is not None
        and max_spread is not None
        and float(min_spread) > float(max_spread)
        and (delta.price_min_spread is not None or delta.price_max_spread is not None)
    ):
        raise ParamError(
            f"proposal inverts the spread band: price_min_spread={min_spread} "
            f"> price_max_spread={max_spread}. The near edge cannot sit beyond "
            f"the far edge."
        )

    min_base = config.get("min_base")
    max_base = config.get("max_base")
    if (
        min_base is not None
        and max_base is not None
        and float(min_base) > float(max_base)
        and (delta.min_base is not None or delta.max_base is not None)
    ):
        raise ParamError(
            f"proposal inverts the inventory band: min_base={min_base} > "
            f"max_base={max_base}"
        )

    if delta.price_min_spread is not None:
        maker_fee = config.get("maker_fee") or config.get("market_take_fee") or 0.0
        floor = float(maker_fee)
        if floor and float(delta.price_min_spread) < floor:
            raise ParamError(
                f"price_min_spread={delta.price_min_spread} is below the "
                f"break-even half-spread of {floor} (one maker fee); the maker "
                f"would clamp it and the proposal would be a no-op"
            )

    if delta.target_base is not None:
        min_base = config.get("min_base")
        max_base = config.get("max_base")
        if min_base is None or max_base is None:
            raise ParamError(
                "target_base needs min_base and max_base on the composed config"
            )
        lo, hi = float(min_base), float(max_base)
        target = float(delta.target_base)
        if not (lo <= target <= hi):
            raise ParamError(
                f"target_base={target} sits outside the inventory band "
                f"[{lo}, {hi}]; the skew would saturate and quietly disable"
            )
        midpoint = (lo + hi) / 2.0
        half_band = (hi - lo) / 2.0
        max_move = DEFAULT_TILT_CAP * half_band
        if abs(target - midpoint) > max_move + 1e-9:
            raise ParamError(
                f"target_base={target} is more than {DEFAULT_TILT_CAP:.0%} of "
                f"half-band from the midpoint {midpoint} (allowed "
                f"[{midpoint - max_move}, {midpoint + max_move}]); a wrong "
                f"signal must not fully stretch inventory"
            )


EMPTY = ConfigDelta()


def combine(*deltas: ConfigDelta) -> ConfigDelta:
    """Fold several overlay outputs into one delta."""
    active = [d for d in deltas if not d.is_empty]
    if not active:
        return EMPTY

    multiplier = None
    for d in active:
        if d.spread_multiplier is not None:
            multiplier = (multiplier or 1.0) * d.spread_multiplier
    if multiplier is not None:
        multiplier = min(multiplier, _SPREAD_MULTIPLIER_CAP)

    sensitivities = [
        d.vol_spread_sensitivity for d in active if d.vol_spread_sensitivity is not None
    ]
    premium_holder = max(
        (d for d in active if d.structural_premium is not None),
        key=lambda d: abs(d.structural_premium),
        default=None,
    )

    absolutes: dict[str, Any] = {}
    for name in ABSOLUTE_FIELDS:
        proposed = {
            getattr(d, name) for d in active if getattr(d, name) is not None
        }
        if len(proposed) > 1:
            raise ParamError(
                f"conflicting proposals for {name}: {sorted(proposed, key=str)}. "
                f"Absolute values cannot be merged -- resolve which one you mean."
            )
        if proposed:
            absolutes[name] = proposed.pop()

    return ConfigDelta(
        spread_multiplier=multiplier,
        vol_spread_sensitivity=max(sensitivities) if sensitivities else None,
        disable_buy=True if any(d.disable_buy for d in active) else None,
        disable_sell=True if any(d.disable_sell for d in active) else None,
        structural_premium=premium_holder.structural_premium if premium_holder else None,
        premium_skew_factor=premium_holder.premium_skew_factor if premium_holder else None,
        reason="; ".join(d.reason for d in active if d.reason),
        **absolutes,
    )


@dataclass(frozen=True)
class OverlayContext:
    """What an overlay may see: analysis verdicts, the clock, and the config it is proposing against."""

    analyses: Mapping[str, AnalysisResult]
    asof: datetime | None = None
    config: Mapping[str, Any] | None = None

    def get(self, name: str) -> AnalysisResult | None:
        """Find a result by analysis name, tolerating parameterised keys."""
        result = self.analyses.get(name)
        if result is not None:
            return result
        for key, value in self.analyses.items():
            if key.split(":", 1)[0] == name:
                return value
        return None

    def scalar(self, analysis: str, key: str, default=None):
        result = self.get(analysis)
        if result is None:
            return default
        value = result.scalars.get(key, default)
        return default if value is None else value

    def config_value(self, key: str, default=None):
        if not self.config:
            return default
        value = self.config.get(key, default)
        return default if value is None else value


def vol_widen(
    ctx: OverlayContext, *, threshold: float = 0.75, cap: float = 2.5
) -> ConfigDelta:
    """Widen both edges of the grid when volatility is high *for this market*."""
    if not 0.0 <= threshold < 1.0:
        raise ParamError(f"threshold must be in [0, 1), got {threshold}")
    if cap < 1.0:
        raise ParamError(f"cap must be >= 1.0 (overlays widen, never narrow), got {cap}")

    percentile = ctx.scalar("volatility", "vol_percentile")
    if percentile is None or percentile < threshold:
        return EMPTY

    multiplier = 1.0 + (min(percentile, 1.0) - threshold) / (1.0 - threshold) * (cap - 1.0)
    if multiplier <= 1.0 + 1e-9:
        return EMPTY
    return ConfigDelta(
        spread_multiplier=round(multiplier, 4),
        reason=f"vol_widen: vol_percentile={percentile:.2f} >= {threshold:.2f}",
    )


def band_widen(
    ctx: OverlayContext,
    *,
    threshold: float = 0.75,
    max_bump: float = 2.0,
    min_bump: float = 1.0,
) -> ConfigDelta:
    """Push the *far* edge of the grid out on volatility, holding the near edge."""
    if not 0.0 <= threshold < 1.0:
        raise ParamError(f"threshold must be in [0, 1), got {threshold}")
    if max_bump < 1.0 or min_bump < 1.0:
        raise ParamError(
            f"bumps must be >= 1.0 (proposals widen, never narrow); "
            f"got min_bump={min_bump}, max_bump={max_bump}"
        )

    percentile = ctx.scalar("volatility", "vol_percentile")
    if percentile is None or percentile < threshold:
        return EMPTY

    current_max = ctx.config_value("price_max_spread")
    current_min = ctx.config_value("price_min_spread")
    if current_max is None:
        raise ParamError(
            "band_widen needs the current config: pass OverlayContext(config=...) "
            "with at least price_max_spread"
        )

    ramp = (min(percentile, 1.0) - threshold) / (1.0 - threshold)
    changes: dict[str, float] = {}

    scaled_max = float(current_max) * (1.0 + ramp * (max_bump - 1.0))
    if scaled_max > float(current_max) * (1.0 + 1e-9):
        changes["price_max_spread"] = round(scaled_max, 8)

    if min_bump > 1.0 and current_min is not None:
        scaled_min = float(current_min) * (1.0 + ramp * (min_bump - 1.0))
        if scaled_min > float(current_min) * (1.0 + 1e-9):
            changes["price_min_spread"] = round(scaled_min, 8)

    if not changes:
        return EMPTY
    edges = " and ".join(sorted(changes))
    return ConfigDelta(
        **changes,
        reason=f"band_widen: vol_percentile={percentile:.2f} >= {threshold:.2f}, raised {edges}",
    )


def size_down(
    ctx: OverlayContext, *, threshold: float = 0.90, floor_ratio: float = 0.5
) -> ConfigDelta:
    """Shrink the size shown in a market when volatility goes extreme."""
    if not 0.0 <= threshold < 1.0:
        raise ParamError(f"threshold must be in [0, 1), got {threshold}")
    if not 0.0 < floor_ratio <= 1.0:
        raise ParamError(f"floor_ratio must be in (0, 1], got {floor_ratio}")

    percentile = ctx.scalar("volatility", "vol_percentile")
    if percentile is None or percentile < threshold:
        return EMPTY

    current = ctx.config_value("max_base")
    if current is None:
        raise ParamError(
            "size_down needs the current config: pass OverlayContext(config=...) "
            "with at least max_base"
        )

    ramp = (min(percentile, 1.0) - threshold) / (1.0 - threshold)
    scaled = float(current) * (1.0 - ramp * (1.0 - floor_ratio))
    if scaled >= float(current) * (1.0 - 1e-9):
        return EMPTY

    min_base = ctx.config_value("min_base")
    if min_base is not None and scaled < float(min_base):
        scaled = float(min_base)
        if scaled >= float(current) * (1.0 - 1e-9):
            return EMPTY

    return ConfigDelta(
        max_base=round(scaled, 8),
        reason=(
            f"size_down: vol_percentile={percentile:.2f} >= {threshold:.2f}, "
            f"max_base {current} -> {round(scaled, 8)}"
        ),
    )


def trend_tilt(
    ctx: OverlayContext,
    *,
    min_score: float = 0.4,
    max_tilt_bps: float = 15.0,
    skew_gate: float = 0.5,
) -> ConfigDelta:
    """Tilt the quoting reference a few bp in the direction of a real trend."""
    if not 0.0 <= min_score < 1.0:
        raise ParamError(f"min_score must be in [0, 1), got {min_score}")
    if not 0.0 < max_tilt_bps <= _MAX_STRUCTURAL_PREMIUM * 1e4:
        raise ParamError(
            f"max_tilt_bps must be in (0, {_MAX_STRUCTURAL_PREMIUM * 1e4:.0f}], got {max_tilt_bps}"
        )
    if not 0.0 <= skew_gate <= 1.0:
        raise ParamError(f"skew_gate must be in [0, 1], got {skew_gate}")

    score = ctx.scalar("trend", "trend_score")
    if score is None or abs(score) < min_score:
        return EMPTY

    magnitude = (min(abs(score), 1.0) - min_score) / (1.0 - min_score)
    tilt_bps = magnitude * max_tilt_bps
    if tilt_bps <= 0.0:
        return EMPTY
    signed_bps = (1 if score > 0 else -1) * tilt_bps
    return ConfigDelta(
        structural_premium=round(signed_bps / 1e4, 8),
        premium_skew_factor=skew_gate,
        reason=f"trend_tilt: trend_score={score:+.2f}, tilt={signed_bps:+.1f}bp",
    )


def inventory_tilt(ctx: OverlayContext, *, cap: float = DEFAULT_TILT_CAP) -> ConfigDelta:
    """Move `target_base` with the approved SMA tilt, capped from the midpoint."""
    if not 0.0 < cap <= DEFAULT_TILT_CAP:
        raise ParamError(
            f"cap must be in (0, {DEFAULT_TILT_CAP}] (allowlist ceiling), got {cap}"
        )

    tilt = ctx.scalar("tilt", "tilt")
    if tilt is None:
        return EMPTY

    min_base = ctx.config_value("min_base")
    max_base = ctx.config_value("max_base")
    if min_base is None or max_base is None:
        raise ParamError(
            "inventory_tilt needs the current config: pass OverlayContext(config=...) "
            "with min_base and max_base"
        )

    target = implied_target_base(
        float(tilt), float(min_base), float(max_base), cap=cap
    )
    if target is None:
        return EMPTY

    current = ctx.config_value("target_base")
    if current is None:
        midpoint = (float(min_base) + float(max_base)) / 2.0
        if abs(target - midpoint) <= 1e-9:
            return EMPTY
    elif abs(float(current) - target) <= 1e-9:
        return EMPTY

    return ConfigDelta(
        target_base=round(target, 8),
        reason=(
            f"inventory_tilt: tilt={float(tilt):+.3f}, "
            f"target_base={round(target, 8)} (cap={cap})"
        ),
    )


def squeeze_brace(
    ctx: OverlayContext, *, widen: float = 1.5, pause_adverse_side: bool = False
) -> ConfigDelta:
    """Brace for expansion on the bar a volatility squeeze releases."""
    if widen < 1.0:
        raise ParamError(f"widen must be >= 1.0, got {widen}")

    if not ctx.scalar("regime", "released_now", default=False):
        return EMPTY

    direction = ctx.scalar("regime", "release_direction")
    delta_kwargs: dict[str, Any] = {
        "spread_multiplier": float(widen),
        "reason": f"squeeze_brace: release {direction or 'unknown'}",
    }
    if pause_adverse_side and direction == "up":
        delta_kwargs["disable_sell"] = True
    elif pause_adverse_side and direction == "down":
        delta_kwargs["disable_buy"] = True
    return ConfigDelta(**delta_kwargs)


OVERLAYS = {
    "vol_widen": vol_widen,
    "band_widen": band_widen,
    "size_down": size_down,
    "trend_tilt": trend_tilt,
    "inventory_tilt": inventory_tilt,
    "squeeze_brace": squeeze_brace,
}


def evaluate(ctx: OverlayContext, specs) -> ConfigDelta:
    """Run a set of overlays and combine their deltas."""
    from ..engine import parse_requests

    deltas = []
    for request in parse_requests(specs):
        fn = OVERLAYS.get(request.name)
        if fn is None:
            raise ParamError(
                f"unknown overlay {request.name!r}; known: {', '.join(sorted(OVERLAYS))}"
            )
        deltas.append(
            call_with_loose_params(fn, request.params, ctx, label=request.name)
        )
    return combine(*deltas)
