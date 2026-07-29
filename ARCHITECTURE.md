# Indicators — Architecture

> Source of truth for design decisions in this project. Nothing here changes
> without explicit approval. Every coding decision in `Indicators/` must be
> consistent with this document.

## Purpose

Research-only technical analysis toolkit on Iranian-market OHLC (and later
MM book history). Compute indicators and other analysis types, persist
results, and evaluate how TA-derived **parameter overlays** would have
changed market-maker behaviour in replay.

**Chosen scope (locked):**
- **Mode A — research toolkit.** Library + thin HTTP/CLI. Writes analytics
  stores. Does **not** drive live `market_maker` or config-manager.
- **Backtest — MM-aware.** Replay historical fair mid + books against a
  quoting simulator. Classic OHLC long/short signal backtests are out of
  scope unless added later by explicit ARCHITECTURE change.

**Explicit non-goals (for now):**
- Live RSI/MACD entry signals into the MM loop.
- Replacing `crypto_price` fair value with TA mid.
- Kafka producers/consumers inside this project.

## Current state (baseline)

Small Flask service: `GET /indicator` → read Mongo `market_making.ohlc` →
hand-rolled SMA/EMA/RSI/MACD/StochRSI → insert into
`market_making.indicators_data`. No registry, no tests, no `source` filter
on OHLC, no analyzer abstraction, no backtester, not on any trading path.

## Design principles

1. **Library first, HTTP second.** Core is importable pure functions /
   classes. Flask (or CLI) is a thin adapter. Prefer `python -m …` jobs for
   batch compute over stuffing everything into request handlers.
2. **Registry of analysis types.** Adding an indicator or analysis must be
   registering a module + schema, not editing a giant if/elif forever.
3. **OHLC is the primary TA input.** Candles come from
   `market-data-collector` → Mongo `market_making.ohlc`. Always filter by
   `market_name`, `interval`, and **`source`** (baseline bug: `source` is
   ignored today).
4. **MM-aware backtest is a simulator, not a replay of `MarketMaker`.**
   Do not import and run the live agent loop. Extract / mirror the **pure
   quoting math** (ladder prices, spread scale, inventory skew, fee floor)
   into a side-effect-free simulator fed by historical series.
5. **TA feeds overlays, not entries.** In backtests, indicators produce
   parameter deltas (e.g. widen `spread_multiplier`, tilt
   `premium_skew_factor`, soft `disable_buy`/`disable_sell`) applied on
   top of a baseline MM config — the same mental model as offline
   calibrators in `crypto_price/python/analysis`, without writing live
   config.
6. **No new MD docs beyond ARCHITECTURE / PLAN / RUNNING.** Code and these
   two files carry design; avoid README sprawl unless asked.

## Layout (as built — import root is `ta`, decided during step 1)

The import root became `ta` rather than a restructured `indicators` package:
the legacy flat modules were replaced wholesale, so keeping their import root
bought no compatibility and cost a name collision between the package and its
own `indicators/` subpackage. Top-level folder stays `Indicators`.

```
Indicators/
  ARCHITECTURE.md / PLAN.md / RUNNING.md
  app.py                      # thin HTTP adapter over ta (catalogue + compute)
  requirements.txt
  conftest.py                 # pytest path hook
  ta/
    errors.py                 # TAError tree: caller-error vs upstream-error
    frame.py                  # OHLCV normalization, FrameInfo health, TR/Wilder helpers
    registry.py               # indicator registry: specs, params, catalogue
    engine.py                 # batch compute, warmup policies, causality guard
    ohlc.py                   # Mongo OHLC reader (source-aware, Tehran→UTC)
    store.py                  # idempotent indicators_data upserts (natural key)
    indicators/               # registered library, grouped by category
      _common.py              # shared MA/warmup/numeric building blocks
      trend.py  momentum.py  volatility.py  volume.py
    analysis/                 # verdict-level analyses: volatility, trend, regime
      types.py                # AnalysisResult + loose-param plumbing
    signals/
      overlays.py             # ConfigDelta (allowlisted) + vol_widen/trend_tilt/squeeze_brace
    pipeline.py               # one fetch → indicators + analyses → JSON; python -m ta.pipeline
    backtest/                 # (later) simulator.py, runner.py, metrics.py, readers
  tests/
```

## Data contracts

### OHLC (Mongo `market_making.ohlc`)

Required query keys: `market_name`, `source`, `interval`.
Columns used: `time`, `open`, `high`, `low`, `close`, `volume` (plus
extras ignored by TA). Intervals match MDC: `1Min`, `5Min`, `15Min`,
`1H`, `4H`, `1D`. Times are Tehran-naive as produced by MDC.

Default lookback: configurable (baseline 500 is too short for serious
indicator + backtest work; backtest paths must accept explicit
`start`/`end`).

### Indicator / analysis results (Mongo `market_making.indicators_data`)

Writes must be **idempotent** (replace by natural key), not blind
`insert_many` (baseline bug: duplicates on every request).

Natural key: `(symbol, source, interval, analysis, param_hash, time)`.
Document carries the analysis payload columns + metadata tags.

### MM-aware backtest inputs

| Series | Source | Role |
|--------|--------|------|
| Fair mid (+ optional vol/jump) | ClickHouse `crypto_analytics.fair_value_snapshots` (preferred) or reconstructed from Redis exports | reference price path |
| Venue book (net of own quotes when available) | ClickHouse `pnl.mm_books` | resting competition / fill probability |
| Fills (optional validation) | ClickHouse `pnl.fills` | compare sim fills vs reality, not required to run |
| Baseline MM knobs | frozen `ConfigSchema`-shaped dict in the backtest job | starting params |

If `pnl.mm_books` coverage is thin for a market/period, the simulator
falls back to a **degenerate book** (only fair mid ± fee/spread) and
marks the run as `book_quality=degraded`. Do not invent dense books.

### Outputs of a backtest run

Local artifacts (parquet/JSON under a configurable `BACKTEST_OUT_DIR`) +
optional Mongo summary doc. Metrics at minimum: simulated PnL (fee-aware),
inventory path, quote uptime per side, fill count, markout vs mid at
horizons if mid series exists. Overlay timeline stored for audit.

## Analyzer model

```
AnalysisSpec(name, params) → registry.lookup → Analysis.run(ohlc_df) → AnalysisResult
```

- **Indicators** are one family of `Analysis` (series-valued columns).
- Other families (trend, volatility, levels) return structured results
  that may include series, scalars, or event lists.
- `pipeline.run(specs, market, source, interval, range)` batches multiple
  analyses on one OHLC pull.

HTTP `GET /indicator` stays as a compatibility shim that builds one
`AnalysisSpec` and returns JSON; batch/backtest go through CLI/modules.

## Overlay model (TA → MM params)

Overlays are pure functions:

```
OverlayContext(analysis_results, wall_clock) → ConfigDelta
```

`ConfigDelta` only touches fields that exist on MM `ConfigSchema` (or an
explicit allowlist in this ARCHITECTURE). Initial allowlist:

- `spread_multiplier`
- `premium_skew_factor` / `structural_premium` (research tilt only)
- `disable_buy` / `disable_sell`
- `vol_spread_sensitivity` (optional)

No overlay invents new live MM fields. Live publish to config-manager is
**forbidden** in this project until Mode B is approved and this file is
updated.

## MM-aware simulator (honest boundaries)

**In scope:**
- Given mid(t), inventory state, config(+delta), produce bid/ask ladder
  prices/sizes using the same formulas as `market_maker` pricing/sizing
  (fee floor, spread scale, inventory skew, premium skew). Prefer a
  **copied/extracted pure module** owned here or a shared pure lib later —
  do not call exchange/Kafka/Redis from the simulator.
- Match simulated orders against historical book top-of-book / depth with
  simple rules (touch → fill probability or deterministic fill when
  market trades through). Start deterministic; stochastic later if needed.
- Walk time on the book snapshot clock (or 1s/1Min grid), not on every
  OHLC bar only.

**Out of scope (v1):**
- Full exchange latency, rate limits, partial cancel races.
- Exact reproduction of bypass-wall / penny-jump logic (may stub; document
  divergence).
- Replaying SPECIAL OVERPRICED/UNDERPRICE branches (optional later flag).

**Honesty rule:** if the simulator diverges from live MM math, tests must
compare extracted helpers against golden values from `market_maker` unit
expectations, or the divergence is listed in the run report.

## Integration map (research only)

```
market-data-collector → Mongo ohlc ──► Indicators analyses
crypto_price CH fair_value_snapshots ─┐
pnl-tracker CH mm_books / fills ──────┴► MM-aware backtest runner
market_maker ConfigSchema (shape only) ─► baseline + overlay allowlist

Live market_maker / config-manager / Kafka  ←── NOT connected
```

Future Mode B (not approved): periodic job writes allowlisted deltas to
config-manager, same as calibrators. Requires ARCHITECTURE change.

## Stack

- Python 3.13+ (match current Dockerfile; avoid 3.14-only breakage).
- pandas / numpy for series math. No `pandas_ta` / ta-lib unless approved.
- pymongo for OHLC + results. clickhouse-connect (or existing stack client)
  for backtest series.
- Flask kept as optional HTTP surface; pytest required for core math,
  registry, overlay, and simulator.

## Quality bar

- Idempotent storage.
- `source` always required for multi-venue markets.
- Tests for each registered indicator against a fixed OHLC fixture.
- Simulator tests with synthetic mid + book paths (no live DB required).
