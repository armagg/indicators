"""Overlay mapper: analysis verdicts -> allowlisted MM parameter deltas.

TA feeds **overlays, not entries** (ARCHITECTURE.md). Everything here is a
pure function from `OverlayContext` (analysis results) to `ConfigDelta`; no
Mongo, no config-manager, no side effects. The backtest runner applies deltas
to a frozen baseline config; live publishing is Mode B and forbidden until
approved.

Field semantics mirror `config-manager/models.py` / `market_maker`'s
ConfigSchema exactly (checked 2026-07-30 -- do not invent fields):

- `spread_multiplier` (0, 5]: two-sided widen/tighten of the resting grid.
  The live MM floors the resulting spread at 2x maker fee, but overlays stay
  widen-only (>= 1.0) anyway: on both USDT venues the maker fee already
  exceeds the touch spread, so narrowing buys nothing and risks quoting into
  the round-trip cost. In a delta this field is a *factor composed onto* the
  baseline's own multiplier, not a replacement.
- `structural_premium` (signed fraction, e.g. -0.0021 = -21bp) shifts the
  quoting reference to `ref * (1 + premium_skew_factor * structural_premium)`.
  Inert unless `premium_skew_factor` in [0, 1] is also set -- the tilt
  overlay therefore always emits the pair together.
- `disable_buy` / `disable_sell`: one-sided kill switches. An overlay may
  *set* one, never clear one -- the live flags belong to the risk control
  plane, and a research overlay that re-enables a manually killed side would
  be a hazard. Enforced at construction and in `apply_to`.
- `vol_spread_sensitivity` (>= 0): fast vol response; composed as max().
"""

from __future__ import annotations

from dataclasses import dataclass, fields as dataclass_fields, replace
from datetime import datetime
from typing import Any, Mapping

from ..analysis.types import AnalysisResult, call_with_loose_params
from ..errors import ParamError

#: Hard cap from config-manager's model (`Field(1.0, gt=0, le=5)`).
_SPREAD_MULTIPLIER_CAP = 5.0

#: Sanity ceiling for a research tilt: 500bp. Calibrated structural premia
#: run tens of bp; anything near this bound is a bug, not a signal.
_MAX_STRUCTURAL_PREMIUM = 0.05


@dataclass(frozen=True)
class ConfigDelta:
    """A restricted, validated set of MM parameter changes.

    Only the ARCHITECTURE allowlist exists as fields, so an overlay cannot
    emit anything else even by accident (unknown kwargs are a TypeError at
    construction). `None` means "leave the baseline alone". `reason` is audit
    metadata for the overlay timeline, not a config field.
    """

    spread_multiplier: float | None = None
    premium_skew_factor: float | None = None
    structural_premium: float | None = None
    disable_buy: bool | None = None
    disable_sell: bool | None = None
    vol_spread_sensitivity: float | None = None
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
                    f"{side}=False is not allowed: an overlay may disable a side, "
                    f"never re-enable one the risk plane turned off"
                )
        if self.vol_spread_sensitivity is not None and self.vol_spread_sensitivity < 0:
            raise ParamError(
                f"vol_spread_sensitivity must be >= 0, got {self.vol_spread_sensitivity}"
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

    def as_dict(self) -> dict:
        out = self.fields()
        out["reason"] = self.reason
        return out

    def apply_to(self, config: Mapping[str, Any]) -> dict[str, Any]:
        """Compose this delta onto a baseline config dict.

        Composition rules per field (see module docstring for why):
        multiplier multiplies (capped), sensitivity takes the max, disables
        can only turn on, premium pair overrides.
        """
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
        return out


EMPTY = ConfigDelta()


def combine(*deltas: ConfigDelta) -> ConfigDelta:
    """Fold several overlay outputs into one delta.

    Widening composes multiplicatively (two independent reasons to widen are
    both real), sensitivity takes the max, disables OR together. For the
    premium pair, the delta with the larger |structural_premium| wins whole
    -- premia from different signals must not be summed into a tilt nobody
    proposed. Reasons concatenate for the audit trail.
    """
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

    return ConfigDelta(
        spread_multiplier=multiplier,
        vol_spread_sensitivity=max(sensitivities) if sensitivities else None,
        disable_buy=True if any(d.disable_buy for d in active) else None,
        disable_sell=True if any(d.disable_sell for d in active) else None,
        structural_premium=premium_holder.structural_premium if premium_holder else None,
        premium_skew_factor=premium_holder.premium_skew_factor if premium_holder else None,
        reason="; ".join(d.reason for d in active if d.reason),
    )


@dataclass(frozen=True)
class OverlayContext:
    """What an overlay is allowed to see: analysis results and the clock."""

    analyses: Mapping[str, AnalysisResult]
    asof: datetime | None = None

    def get(self, name: str) -> AnalysisResult | None:
        """Find a result by analysis name, tolerating parameterised keys
        (`trend:slow=100`) the pipeline produces."""
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


# --------------------------------------------------------------------------
# The overlays
# --------------------------------------------------------------------------


def vol_widen(
    ctx: OverlayContext, *, threshold: float = 0.75, cap: float = 2.5
) -> ConfigDelta:
    """Widen spreads when volatility is high *for this market*.

    Linear ramp on `vol_percentile`: 1.0x at `threshold`, `cap`x at 1.0.
    Below the threshold: no delta. Uses the percentile, not an absolute vol
    number, so the same overlay works on USDTIRT and BTCIRT without
    per-market tuning.
    """
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


def trend_tilt(
    ctx: OverlayContext,
    *,
    min_score: float = 0.4,
    max_tilt_bps: float = 15.0,
    skew_gate: float = 0.5,
) -> ConfigDelta:
    """Tilt the quoting reference a few bp in the direction of a real trend.

    Emits the (`structural_premium`, `premium_skew_factor`) pair together --
    the premium is inert without the gate. The premium ramps linearly from 0
    at `min_score` to `max_tilt_bps` at |trend_score| = 1, signed with the
    trend, so the ladder centres slightly ahead of where the market is going
    instead of symmetrically around where it was.

    Note for replay: this *overrides* any calibrated structural_premium in
    the baseline for the bars it fires on. That is intended -- the research
    question is whether the TA tilt would have helped -- but it means results
    are not additive with the live calibrator's premium.
    """
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


def squeeze_brace(
    ctx: OverlayContext, *, widen: float = 1.5, disable_adverse_side: bool = False
) -> ConfigDelta:
    """Brace for expansion on the bar a volatility squeeze releases.

    The release bar is where a passive quoter is most likely to be run over:
    compression has just resolved into a directional expansion. Widen for
    that bar; optionally stop quoting the side the expansion eats (an upward
    release lifts asks -- disable_sell -- and mirror for downward).
    """
    if widen < 1.0:
        raise ParamError(f"widen must be >= 1.0, got {widen}")

    if not ctx.scalar("regime", "released_now", default=False):
        return EMPTY

    direction = ctx.scalar("regime", "release_direction")
    delta_kwargs: dict[str, Any] = {
        "spread_multiplier": float(widen),
        "reason": f"squeeze_brace: release {direction or 'unknown'}",
    }
    if disable_adverse_side and direction == "up":
        delta_kwargs["disable_sell"] = True
    elif disable_adverse_side and direction == "down":
        delta_kwargs["disable_buy"] = True
    return ConfigDelta(**delta_kwargs)


OVERLAYS = {
    "vol_widen": vol_widen,
    "trend_tilt": trend_tilt,
    "squeeze_brace": squeeze_brace,
}


def evaluate(ctx: OverlayContext, specs) -> ConfigDelta:
    """Run a set of overlays and combine their deltas.

    `specs` accepts the engine's request shapes, so an overlay stack can live
    in one string: `"vol_widen:threshold=0.8,cap=3, trend_tilt"`.
    """
    from ..engine import parse_requests  # local import: engine must not import signals

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
