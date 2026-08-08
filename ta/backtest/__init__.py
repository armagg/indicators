"""Candle-based market-maker backtest.

Self-contained: it needs only the OHLC this project already reads. No
ClickHouse, no `market_maker` import.

    from ta.backtest import run, sweep, grid, render

    result = run(candles, config)
    print(result.metrics.pnl, result.disclaimer)

**The standing caveat**, repeated here because it governs every number this
package produces: candle fills ignore queue position and assume the whole
level would have been taken, so absolute PnL is an upper bound. Use it to
rank configurations against each other over the same bars, never as a
profitability forecast. See `runner.DISCLAIMER` and `simulator.DIVERGENCES`.
"""

from .metrics import MARKOUT_HORIZONS_SEC, Fill, RunMetrics, markouts, summarize  # noqa: F401
from .runner import DISCLAIMER, BacktestResult, compare, run  # noqa: F401
from .simulator import (  # noqa: F401
    DEFAULT_RECOVERY_SPAN,
    DIVERGENCES,
    band_breach,
    effective_band,
    recovery_spreads,
    Ladder,
    Quote,
    build_ladder,
    effective_spreads,
    inventory_q,
    level_step,
    order_totals,
    premium_reference,
    skew_reference,
    spread_floor,
    spread_scale,
)
from .sweep import SWEEPABLE, SweepRow, grid, render, sweep  # noqa: F401
