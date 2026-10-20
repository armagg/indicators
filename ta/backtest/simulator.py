"""A pure mirror of `market_maker`'s quoting math, with divergences named rather than smoothed over."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

DIVERGENCES = (
    "no bypass-wall / penny-jump (candles carry no order book): live quotes "
    "at least as aggressively as this simulator",
    "no martingale or size skew: order sizes are split equally across levels",
    "no tick rounding, price jitter, adverse cooldown, or special "
    "OVERPRICED/UNDERPRICE branches",
    "no latency, rate limits, or cancel/replace races",
    "inventory cannot leave the band here, because order sizes are derived "
    "from it -- so the out-of-band recovery (build_ladder(recovery=True)) "
    "never fires endogenously and this simulator cannot price it",
)


def _num(config: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    """Config values arrive from Mongo and may be None or strings."""
    try:
        value = config.get(key)
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Quote:
    side: str
    level: int
    price: float
    size: float


@dataclass(frozen=True)
class Ladder:
    bids: tuple[Quote, ...] = ()
    asks: tuple[Quote, ...] = ()

    def __iter__(self):
        return iter(self.bids + self.asks)

    @property
    def is_empty(self) -> bool:
        return not self.bids and not self.asks


def spread_scale(config: Mapping[str, Any], volatility: float = 0.0) -> float:
    """`spread_multiplier x (1 + vol_spread_sensitivity x volatility)`."""
    multiplier = _num(config, "spread_multiplier", 1.0)
    if multiplier <= 0:
        multiplier = 1.0
    vol_part = 1.0
    sensitivity = _num(config, "vol_spread_sensitivity")
    if sensitivity > 0 and volatility > 0:
        vol_part = 1.0 + sensitivity * volatility
        if vol_part <= 0:
            vol_part = 1.0
    return multiplier * vol_part


def spread_floor(config: Mapping[str, Any]) -> float:
    """Break-even half-spread: one maker fee, falling back to the taker rate."""
    maker = _num(config, "maker_fee")
    return maker if maker > 0 else _num(config, "market_take_fee")


def effective_spreads(
    config: Mapping[str, Any], scale: float = 1.0, side: str | None = None
) -> tuple[float, float]:
    """`(min_spread, max_spread)` after scale, per-side asymmetry, and floor."""
    fee = spread_floor(config)
    base_min = _num(config, "price_min_spread") * scale
    base_max = _num(config, "price_max_spread") * scale

    asymmetry = _num(config, "spread_asymmetry") * scale
    if asymmetry and side in ("buy", "sell"):
        if base_min > fee:
            room = base_min - fee
            asymmetry = max(-room, min(room, asymmetry))
        else:
            asymmetry = 0.0
        signed = asymmetry if side == "buy" else -asymmetry
        base_min += signed
        base_max += signed

    min_s, max_s = base_min, base_max
    if fee > 0:
        min_s = max(min_s, fee)
        max_s = max(max_s, fee)
    if max_s < min_s:
        max_s = min_s
    return min_s, max_s


def level_step(config: Mapping[str, Any], num_orders: int, scale: float = 1.0) -> float:
    """Spacing between ladder levels: the band divided by the gaps in it."""
    if num_orders > 1:
        min_s, max_s = effective_spreads(config, scale)
        return (max_s - min_s) / (num_orders - 1)
    return 0.0


DEFAULT_RECOVERY_SPAN = 0.25


def effective_band(
    config: Mapping[str, Any], regime: str | None = None
) -> tuple[float, float]:
    """Inventory bounds after the market-state multipliers."""
    min_base = _num(config, "min_base")
    max_base = _num(config, "max_base")
    if regime == "underprice":
        max_base *= _num(config, "underprice_multiplier", 1.0) or 1.0
    elif regime == "overprice":
        min_base *= _num(config, "overprice_multiplier", 1.0) or 1.0
    if max_base < min_base:
        min_base, max_base = max_base, min_base
    return min_base, max_base


def band_breach(
    base_balance: float,
    effective_min: float,
    effective_max: float,
    span_fraction: float = DEFAULT_RECOVERY_SPAN,
) -> tuple[str | None, float]:
    """How far outside the effective band we are: `(recovering_side, 0..1)`."""
    width = max(effective_max - effective_min, 0.0)
    span = span_fraction * width
    if span <= 0:
        return None, 0.0
    if base_balance < effective_min:
        return "buy", min(1.0, (effective_min - base_balance) / span)
    if base_balance > effective_max:
        return "sell", min(1.0, (base_balance - effective_max) / span)
    return None, 0.0


def recovery_spreads(
    config: Mapping[str, Any],
    side: str,
    breach: float,
    scale: float = 1.0,
) -> tuple[float, float]:
    """The recovering side's band, tightened toward the fee floor."""
    min_s, max_s = effective_spreads(config, scale, side=side)
    floor = spread_floor(config)
    if breach <= 0 or floor <= 0:
        return min_s, max_s
    weight = min(1.0, max(0.0, breach))
    new_min = min_s + (floor - min_s) * weight
    new_max = max_s + (floor - max_s) * weight
    return new_min, max(new_max, new_min)


def inventory_q(config: Mapping[str, Any], base_balance: float) -> float:
    """Normalized inventory error in [-1, 1]: 0 at target, +1 at `max_base`."""
    max_base = _num(config, "max_base")
    min_base = _num(config, "min_base")
    target = config.get("target_base")
    target = (max_base + min_base) / 2.0 if target is None else float(target)

    if base_balance >= target:
        span = max_base - target
        if span > 0:
            q = (base_balance - target) / span
        else:
            q = 1.0 if base_balance > target else 0.0
    else:
        span = target - min_base
        q = -((target - base_balance) / span) if span > 0 else -1.0
    return max(-1.0, min(1.0, q))


def skew_reference(
    config: Mapping[str, Any], reference: float, base_balance: float, scale: float = 1.0
) -> float:
    """Shift the grid against inventory so the book mean-reverts the position."""
    q = inventory_q(config, base_balance)
    halfspreads = _num(config, "inventory_skew_halfspreads")
    if halfspreads > 0:
        half_spread, _ = effective_spreads(config, scale)
        return reference * (1 - halfspreads * q * half_spread)
    factor = _num(config, "inventory_skew_factor")
    if factor <= 0:
        return reference
    return reference * (1 - factor * q / 2.0)


def premium_reference(config: Mapping[str, Any], reference: float) -> float:
    """Basis correction, inert unless both the premium and its gate are set."""
    factor = _num(config, "premium_skew_factor")
    premium = _num(config, "structural_premium")
    if factor <= 0 or premium == 0:
        return reference
    return reference * (1 + factor * premium)


def order_totals(
    config: Mapping[str, Any], base_balance: float
) -> tuple[float, float]:
    """`(buy_total, sell_total)` in base units, from the inventory band."""
    min_base = _num(config, "min_base")
    max_base = _num(config, "max_base")
    num_orders = max(1, int(_num(config, "number_of_order_each_sides", 1)))

    sell_total = max(0.0, base_balance - min_base)
    if base_balance < max_base:
        buy_total = max_base - base_balance
    else:
        buy_total = _num(config, "base_order_min") * num_orders
    return buy_total, sell_total


def build_ladder(
    config: Mapping[str, Any],
    reference: float,
    base_balance: float,
    *,
    volatility: float = 0.0,
    regime: str | None = None,
    recovery: bool = False,
    recovery_span: float = DEFAULT_RECOVERY_SPAN,
) -> Ladder:
    """The full two-sided quote for one instant."""
    if reference <= 0:
        return Ladder()

    breach_side, breach = (None, 0.0)
    if recovery:
        effective_min, effective_max = effective_band(config, regime)
        breach_side, breach = band_breach(
            base_balance, effective_min, effective_max, recovery_span
        )

    scale = spread_scale(config, volatility)
    num_orders = max(1, int(_num(config, "number_of_order_each_sides", 1)))
    step = level_step(config, num_orders, scale)

    corrected = premium_reference(config, reference)
    skewed = skew_reference(config, corrected, base_balance, scale)

    buy_total, sell_total = order_totals(config, base_balance)
    buy_size = buy_total / num_orders if buy_total > 0 else 0.0
    sell_size = sell_total / num_orders if sell_total > 0 else 0.0

    base_order_min = _num(config, "base_order_min")
    quote_order_min = _num(config, "quote_order_min")

    def side_band(side: str) -> tuple[float, float, float]:
        """`(min_spread, level_step)` for one side, recovery applied."""
        if breach_side == side and breach > 0:
            min_s, max_s = recovery_spreads(config, side, breach, scale)
            side_step = (max_s - min_s) / (num_orders - 1) if num_orders > 1 else 0.0
            return min_s, side_step, max_s
        min_s, _ = effective_spreads(config, scale, side=side)
        return min_s, step, 0.0

    bids: list[Quote] = []
    if not config.get("disable_buy") and buy_size > 0:
        min_s, side_step, _ = side_band("buy")
        for level in range(num_orders):
            price = skewed * (1 - (min_s + level * side_step))
            if _tradeable(price, buy_size, base_order_min, quote_order_min):
                bids.append(Quote("buy", level, price, buy_size))

    asks: list[Quote] = []
    if not config.get("disable_sell") and sell_size > 0:
        min_s, side_step, _ = side_band("sell")
        for level in range(num_orders):
            price = skewed * (1 + (min_s + level * side_step))
            if _tradeable(price, sell_size, base_order_min, quote_order_min):
                asks.append(Quote("sell", level, price, sell_size))

    return Ladder(tuple(bids), tuple(asks))


def _tradeable(
    price: float, size: float, base_order_min: float, quote_order_min: float
) -> bool:
    """Exchange minimums. A level below either one would simply be rejected."""
    if price <= 0 or size <= 0:
        return False
    if base_order_min > 0 and size < base_order_min:
        return False
    if quote_order_min > 0 and price * size < quote_order_min:
        return False
    return True
