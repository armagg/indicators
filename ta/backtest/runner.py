"""Walk candles, quote, fill, score. Comparisons are valid; absolute PnL is an upper bound."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd

from ..analysis import run_analysis
from ..errors import FrameError, ParamError, TAError
from ..frame import normalize_ohlcv
from ..signals import ConfigDelta, OverlayContext, evaluate
from .metrics import Fill, RunMetrics, summarize
from .simulator import DIVERGENCES, Ladder, build_ladder, spread_floor

DISCLAIMER = (
    "Candle-based fills ignore queue position and assume the whole level "
    "would have been taken, so absolute PnL is an UPPER BOUND, not a "
    "forecast. Valid for comparing configurations over the same bars; not "
    "valid as a profitability claim."
)

_OVERLAY_ANALYSES = ("volatility", "trend", "regime")


@dataclass
class BacktestResult:
    metrics: RunMetrics
    fills: list[Fill]
    inventory: pd.Series
    config: dict[str, Any]
    overlay_timeline: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER
    divergences: tuple[str, ...] = DIVERGENCES

    def as_dict(self, *, include_fills: bool = False) -> dict:
        out = {
            "disclaimer": self.disclaimer,
            "divergences": list(self.divergences),
            "metrics": self.metrics.as_dict(),
            "config": self.config,
            "warnings": list(self.warnings),
            "overlay_timeline": self.overlay_timeline,
        }
        if include_fills:
            out["fills"] = [f.as_dict() for f in self.fills]
        return out


def run(
    data,
    config: Mapping[str, Any],
    *,
    overlays: str | Sequence | None = None,
    overlay_every: int = 24,
    initial_base: float | None = None,
    fee_rate: float | None = None,
    warmup_bars: int = 50,
) -> BacktestResult:
    """Replay one config over one OHLCV frame."""
    frame, info = normalize_ohlcv(data, require=("open", "high", "low", "close"))
    if len(frame) < warmup_bars + 2:
        raise FrameError(
            f"need more than {warmup_bars + 2} bars to backtest; frame has {len(frame)}"
        )

    config = dict(config)
    warnings: list[str] = []
    if info.is_sparse:
        warnings.append(
            f"{info.gap_ratio:.0%} of bars are missing: fills are only checked on "
            f"bars that exist, so quote uptime is measured against a series that "
            f"already excludes the market's quiet stretches."
        )
    if info.inconsistent_bars:
        warnings.append(
            f"{info.inconsistent_bars} bar(s) have high < low or open/close outside "
            f"the range; fills on those bars are unreliable."
        )

    fee = spread_floor(config) if fee_rate is None else float(fee_rate)
    if fee <= 0:
        warnings.append(
            "running fee-free "
            + (
                "by request (fee_rate=0)"
                if fee_rate is not None
                else "because the config carries no maker_fee or market_take_fee"
            )
            + ": on these venues the fee exceeds the touch spread, so a "
            "fee-free result says nothing about whether the config makes money."
        )

    min_base = float(config.get("min_base") or 0.0)
    max_base = float(config.get("max_base") or 0.0)
    base = (
        (min_base + max_base) / 2.0 if initial_base is None else float(initial_base)
    )
    start_base = base

    fills: list[Fill] = []
    inventory: list[float] = []
    timeline: list[dict] = []
    buy_quoted = sell_quoted = 0

    delta = ConfigDelta()
    active = config
    closes = frame["close"]

    for position in range(warmup_bars, len(frame)):
        bar = frame.iloc[position]
        stamp = frame.index[position]
        reference = float(closes.iloc[position - 1])

        if overlays and (position - warmup_bars) % max(1, overlay_every) == 0:
            delta, overlay_warnings = _overlay_delta(
                frame.iloc[:position], config, overlays
            )
            warnings.extend(overlay_warnings)
            try:
                active = delta.apply_to(config) if not delta.is_empty else config
            except TAError as exc:
                warnings.append(f"overlay produced an unusable config at {stamp}: {exc}")
                active = config
                delta = ConfigDelta()
            if not delta.is_empty:
                timeline.append(
                    {"time": stamp.isoformat(), "delta": delta.as_dict()}
                )

        ladder = build_ladder(active, reference, base)
        if ladder.bids:
            buy_quoted += 1
        if ladder.asks:
            sell_quoted += 1

        for fill in _fills_for_bar(ladder, bar, stamp, fee):
            fills.append(fill)
            base += fill.signed_base

        inventory.append(base)

    scored = frame.iloc[warmup_bars:]
    metrics = summarize(
        fills,
        inventory,
        scored["close"],
        initial_base=start_base,
        bars=len(scored),
        buy_quoted_bars=buy_quoted,
        sell_quoted_bars=sell_quoted,
    )
    return BacktestResult(
        metrics=metrics,
        fills=fills,
        inventory=pd.Series(inventory, index=scored.index, name="base"),
        config=config,
        overlay_timeline=timeline,
        warnings=warnings,
    )


def _fills_for_bar(ladder: Ladder, bar, stamp, fee: float) -> list[Fill]:
    """Which resting quotes the bar traded through: bid on low, ask on high."""
    low, high = float(bar["low"]), float(bar["high"])
    out: list[Fill] = []
    for quote in ladder.bids:
        if low <= quote.price:
            out.append(_fill(quote, stamp, fee))
    for quote in ladder.asks:
        if high >= quote.price:
            out.append(_fill(quote, stamp, fee))
    return out


def _fill(quote, stamp, fee: float) -> Fill:
    return Fill(
        time=stamp,
        side=quote.side,
        level=quote.level,
        price=quote.price,
        size=quote.size,
        fee=quote.price * quote.size * fee,
    )


def _overlay_delta(
    history: pd.DataFrame, config: Mapping[str, Any], overlays
) -> tuple[ConfigDelta, list[str]]:
    """Run the analyses over history and map them to a delta, causally."""
    warnings: list[str] = []
    results = {}
    for name in _OVERLAY_ANALYSES:
        try:
            results[name] = run_analysis(name, history)
        except TAError as exc:
            warnings.append(f"analysis {name} unavailable: {exc}")
    if not results:
        return ConfigDelta(), warnings

    context = OverlayContext(
        analyses=results, asof=history.index[-1], config=config
    )
    try:
        return evaluate(context, overlays), warnings
    except TAError as exc:
        warnings.append(f"overlay evaluation failed: {exc}")
        return ConfigDelta(), warnings


def compare(
    data, baseline: Mapping[str, Any], variants: Mapping[str, Mapping[str, Any]], **kwargs
) -> dict[str, BacktestResult]:
    """Run a baseline and named variants over the same bars."""
    out = {"baseline": run(data, baseline, **kwargs)}
    for name, config in variants.items():
        out[name] = run(data, config, **kwargs)
    return out
