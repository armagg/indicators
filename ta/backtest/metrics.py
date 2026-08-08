"""Scoring a simulated run.

PnL is mark-to-market and fee-aware: realized cash plus inventory valued at
the final close, minus every fee paid. Inventory is *not* assumed to be
liquidated at the end -- a strategy that ends the window holding a large
position has not earned that position's paper value, so `inventory_value` is
reported separately from `realized_cash` rather than being quietly folded in.

The markout deserves its own note. Scoring a fill at the moment it happens
always flatters a market maker: you bought below the reference, so it looks
instantly profitable. What matters is where the price is some time *later* --
if it kept falling, that fill was adverse selection wearing a spread's
clothing. This desk's established honest horizon is ~600 seconds, which is
why `MARKOUT_HORIZONS_SEC` leads with it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

#: Seconds after a fill at which to measure it. 600s is the desk's default;
#: the shorter ones show how fast any edge decays.
MARKOUT_HORIZONS_SEC = (60, 300, 600, 1800)


@dataclass(frozen=True)
class Fill:
    """One simulated execution."""

    time: pd.Timestamp
    side: str
    level: int
    price: float
    size: float
    fee: float

    @property
    def signed_base(self) -> float:
        return self.size if self.side == "buy" else -self.size

    @property
    def cash(self) -> float:
        """Cash delta: negative when buying, positive when selling, fee always paid."""
        gross = -self.price * self.size if self.side == "buy" else self.price * self.size
        return gross - self.fee

    def as_dict(self) -> dict:
        return {
            "time": self.time.isoformat(),
            "side": self.side,
            "level": self.level,
            "price": self.price,
            "size": self.size,
            "fee": self.fee,
        }


def markouts(
    fills: Sequence[Fill], prices: pd.Series, horizons=MARKOUT_HORIZONS_SEC
) -> dict[str, float | None]:
    """Mean markout in basis points, per horizon and per side.

    Positive means the fill was on the right side of where price went: a buy
    followed by a rise, or a sell followed by a fall. Fills too close to the
    end of the window to reach a horizon are skipped rather than clamped to
    the last bar, which would bias every long horizon toward zero.
    """
    out: dict[str, float | None] = {}
    if not fills or prices.empty:
        return {f"markout_{h}s_bps": None for h in horizons}

    index = prices.index
    for horizon in horizons:
        offset = pd.Timedelta(seconds=horizon)
        per_side: dict[str, list[float]] = {"buy": [], "sell": [], "all": []}
        for fill in fills:
            target = fill.time + offset
            if target > index[-1]:
                continue
            future = float(prices.iloc[index.searchsorted(target, side="left")])
            direction = 1.0 if fill.side == "buy" else -1.0
            bps = direction * (future - fill.price) / fill.price * 1e4
            per_side[fill.side].append(bps)
            per_side["all"].append(bps)
        out[f"markout_{horizon}s_bps"] = (
            float(np.mean(per_side["all"])) if per_side["all"] else None
        )
        for side in ("buy", "sell"):
            out[f"markout_{horizon}s_{side}_bps"] = (
                float(np.mean(per_side[side])) if per_side[side] else None
            )
    return out


@dataclass
class RunMetrics:
    bars: int = 0
    fills: int = 0
    buy_fills: int = 0
    sell_fills: int = 0
    volume_base: float = 0.0
    fees_paid: float = 0.0
    realized_cash: float = 0.0
    final_base: float = 0.0
    initial_base: float = 0.0
    inventory_value: float = 0.0
    pnl: float = 0.0
    pnl_bps_of_volume: float | None = None
    #: Fraction of bars on which the side had at least one live quote.
    buy_uptime: float = 0.0
    sell_uptime: float = 0.0
    max_base: float = 0.0
    min_base_held: float = 0.0
    markout: dict[str, float | None] = field(default_factory=dict)

    def as_dict(self) -> dict:
        out = {k: v for k, v in self.__dict__.items() if k != "markout"}
        out.update(self.markout)
        return out


def summarize(
    fills: Sequence[Fill],
    inventory_path: Sequence[float],
    prices: pd.Series,
    *,
    initial_base: float,
    bars: int,
    buy_quoted_bars: int,
    sell_quoted_bars: int,
) -> RunMetrics:
    """Fold a run into its scorecard."""
    realized_cash = sum(f.cash for f in fills)
    fees = sum(f.fee for f in fills)
    volume = sum(f.size for f in fills)
    final_base = initial_base + sum(f.signed_base for f in fills)
    final_price = float(prices.iloc[-1]) if len(prices) else 0.0

    # Value the *change* in inventory, not the whole stack: the starting
    # position was not earned by this run and would otherwise swamp the P&L
    # of the quoting itself.
    inventory_value = (final_base - initial_base) * final_price

    metrics = RunMetrics(
        bars=bars,
        fills=len(fills),
        buy_fills=sum(1 for f in fills if f.side == "buy"),
        sell_fills=sum(1 for f in fills if f.side == "sell"),
        volume_base=volume,
        fees_paid=fees,
        realized_cash=realized_cash,
        initial_base=initial_base,
        final_base=final_base,
        inventory_value=inventory_value,
        pnl=realized_cash + inventory_value,
        buy_uptime=(buy_quoted_bars / bars) if bars else 0.0,
        sell_uptime=(sell_quoted_bars / bars) if bars else 0.0,
        max_base=max(inventory_path) if inventory_path else initial_base,
        min_base_held=min(inventory_path) if inventory_path else initial_base,
        markout=markouts(fills, prices),
    )
    if volume > 0 and final_price > 0:
        metrics.pnl_bps_of_volume = metrics.pnl / (volume * final_price) * 1e4
    return metrics
