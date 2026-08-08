# Indicators — Architecture

> Source of truth for design decisions in this project. Nothing here changes
> without explicit approval. Every coding decision in `Indicators/` must be
> consistent with this document.

## Purpose

An **insight engine** for the desk's Iranian markets. It reads market data,
forms a view on each market from technical analysis, stores that view, and
**proposes** market-maker parameter changes. Proposals reach the live system
only after a human approves them, and only through `config-manager`.

**Chosen scope (revised 2026-07-30, supersedes the original "Mode A"):**

- **Insight, not autonomy.** The project's output is a view a human can act
  on: indicator values, analysis verdicts, and concrete proposed parameter
  changes with the reasoning attached. It never decides alone.
- **Data access is in scope.** Reading Mongo (OHLC, trades) and ClickHouse
  is normal operation, not "integration". Reading is unrestricted.
- **Config writes are in scope, gated by approval.** The project may write
  allowlisted fields to `config-manager` — but every write descends from a
  proposal a human explicitly approved. There is no path from an analysis
  straight to a live config.
- **Its own store.** The project persists its own analytics, proposals, and
  decision history to Mongo (ClickHouse later if volume demands it).
- **Backtest.** Evaluate proposed parameters against history before they are
  approved. Candle-based first (self-contained); book-replay later, when
  absolute PnL claims are needed.

**Explicit non-goals:**
- Autonomous config changes of any kind. Approval is not bypassable, not
  even for "safe" fields.
- Automated cross-market ranking that decides which markets to run.
  Market selection is a human decision the project *informs* — see
  "Market tiering" below.
- Live RSI/MACD entry signals into the MM loop; TA feeds parameters, never
  entries.
- Replacing `crypto_price` fair value with a TA mid.
- Importing or running `market_maker`'s runtime code, or Kafka anywhere.

## Market tiering

Markets are not equal and the default posture is not symmetric:

- **Major markets** are enabled and are the ones the desk actively quotes.
- **Everything else is disabled by default.** Smaller markets carry more
  risk — thinner books, wider gaps, sparser candles, less trustworthy
  volume — so the burden of proof is on turning one *on*, not off.

The project maintains the tier list and surfaces per-exchange views of what
is enabled, what is disabled, and what the data says about each. Flipping
`is_enable` in either direction is a proposal requiring approval; no overlay
sets it automatically.

## Permission model

Every change follows the same path, with no shortcuts:

```
analysis → proposal (ConfigDelta + reason + evidence)
        → stored, status=pending
        → human approves or rejects   ← the gate
        → apply: re-read active config, compose, PUT to config-manager
        → record outcome + resulting version
```

Rules:

1. A proposal names only allowlisted fields (below) and carries the
   reasoning that produced it.
2. Nothing is applied without a recorded approval naming who approved it.
3. `config-manager`'s `PUT /config/market/{source}/{market_name}` takes a
   **whole** `ConfigSchema` and creates a new version, so applying is a
   read-modify-write. The active version is re-read immediately before the
   PUT and compared against the version the proposal was based on; if it
   moved, the apply **fails and returns for re-approval** rather than
   clobbering whatever the calibrators wrote in between.
   Note the read and write shapes differ: `GET` returns a wrapper
   `{source, market_name, version, timestamp, config: {...}}` while `PUT`
   takes the bare inner `config`. `ta/configclient.py` owns that unwrapping;
   nothing else should reason about it.
4. Every apply records the base version and the resulting version, so any
   change this project made can be traced and rolled back.

## Overlay allowlist

Fields the project may propose. Anything not listed is out of bounds, and
`ConfigDelta` cannot even represent it. Two groups, because they compose
differently:

**Scaling (relative, stackable)**

| Field | Meaning |
|---|---|
| `spread_multiplier` | Two-sided scale on the grid; moves both band edges together |
| `vol_spread_sensitivity` | Fast volatility response |
| `structural_premium` + `premium_skew_factor` | Reference tilt; always proposed as a pair, the premium is inert without the gate |
| `disable_buy` / `disable_sell` | Per-side pause. May be set, never cleared — the live flags belong to the risk control plane |

**Absolute (set outright, conflicts refused rather than merged)**

| Field | Meaning |
|---|---|
| `price_min_spread` / `price_max_spread` | The raw band. Needed because `spread_multiplier` cannot move one edge alone |
| `min_base` / `max_base` | Inventory band; `max_base` also sets each cycle's buy/sell totals, so it is the real "size shown" knob |
| `is_enable` | Per-market switch. Human-proposed only |
| `number_of_order_each_sides` | Ladder depth per side |
| `target_base` | **Approved 2026-07-31, implemented in PLAN step 10.** The inventory level the skew treats as neutral. Carries the directional tilt; capped so a proposal may not move it more than a set fraction (default 30%) of the band from its midpoint, and never outside `[min_base, max_base]` |

Invariants enforced when a delta is composed onto a config: `price_min_spread
<= price_max_spread`, `min_base <= max_base`, and no proposal may set
`price_min_spread` below the break-even half-spread of **one** `maker_fee` —
the maker would clamp it, making the change a silent no-op. Note it is one
fee, not two: a round trip is gross `2s` against `2 x maker_fee` of fees, so
break-even is `s == maker_fee`. `market_maker._spread_floor` records that
using `2x` here was a real bug which forbade the band where nobitex's
expected value peaks.

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
4. **Backtest is a simulator, not a replay of `MarketMaker`.** Do not
   import or run the live agent loop. Mirror the **pure quoting math**
   (ladder prices, spread scale, inventory skew, fee floor) into a
   side-effect-free simulator fed by historical series.
5. **TA feeds parameters, not entries.** Indicators produce parameter
   deltas (widen the band, tilt the reference, size down, pause a side) on
   top of a baseline MM config — the same mental model as the offline
   calibrators in `crypto_price/python/analysis`.
6. **Every proposal carries its evidence.** A stored proposal must contain
   the analysis output that justified it, not just the resulting numbers.
   An approver who cannot see why is being asked to rubber-stamp.
7. **Reads are free, writes are gated.** Any module may read any data
   source. Exactly one module may write to config-manager, and only for an
   approved proposal.
8. **No new MD docs beyond ARCHITECTURE / PLAN / RUNNING.** Code and these
   files carry design; avoid README sprawl unless asked.

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
      overlays.py             # ConfigDelta (allowlisted) + the overlay functions
    proposals/                # the approval gate
      models.py               # Proposal, Status, derived id, evidence
      lifecycle.py            # pure state machine; APPLIED only via APPROVED
      store.py                # ta_proposals + append-only ta_decisions
      __main__.py             # python -m ta.proposals list|show|approve|reject|apply
      apply.py                # the ONLY module that may write config
    configclient.py           # config-manager client (stdlib urllib, injectable transport)
    tiers.py                  # ta_market_tiers: major/other, unknown = other
    view.py                   # per-exchange view; python -m ta.view
    pipeline.py               # one fetch → indicators + analyses → JSON; python -m ta.pipeline
    backtest/
      simulator.py            # pure mirror of the live quoting math (copied, never imported)
      runner.py               # walk bars, fill, score; carries the upper-bound caveat
      metrics.py              # fee-aware PnL, uptime, inventory, markout
      sweep.py                # ranked grid over allowlisted knobs
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
OverlayContext(analysis_results, wall_clock, current_config) → ConfigDelta
```

`current_config` is part of the context because absolute fields cannot be
proposed blind — a new `price_max_spread` is meaningless without the old
one. Overlays never read it from anywhere themselves; it is passed in.

The allowlist is in "Overlay allowlist" above. `ConfigDelta` is the only
representation of a change in this project, and it is structurally
incapable of naming a field outside that list.

Composition, when several overlays fire at once: scaling fields stack
(multipliers multiply and are capped, sensitivities take the max, pauses
OR together, the strongest premium wins whole so tilts are never summed).
Absolute fields must agree — two overlays proposing different `max_base`
is a real disagreement and **raises** rather than silently picking one.

`is_enable` is deliberately not driven by any automatic overlay. Turning a
market on or off is a human decision, routed through the proposal flow.

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

## Integration map

```
READ (unrestricted)
  market-data-collector → Mongo ohlc / trades ──► analyses
  crypto_price CH fair_value_snapshots ─┐
  pnl-tracker CH mm_books / fills ──────┴──────► backtest
  config-manager GET /config/market/{source}/ ─► current state + tier view

OWN STORE
  Mongo: indicators_data, ta_proposals, ta_decisions, ta_market_tiers
                                                (CH later if volume needs it)

WRITE (single gated path)
  proposal → human approval → config-manager PUT /config/market/{src}/{mkt}

NEVER
  market_maker runtime code · Kafka · exchange APIs · autonomous writes
```

Only `ta/proposals/apply.py` may perform the write. Nothing else in the
package imports an HTTP client for config-manager.

## Stack

- Python 3.13+ (match current Dockerfile; avoid 3.14-only breakage).
- pandas / numpy for series math. No `pandas_ta` / ta-lib unless approved.
- pymongo for candles, results, proposals. The config-manager client uses
  stdlib `urllib` behind an injectable transport — it is a handful of JSON
  calls, and this project has twice been bitten by third-party wheels not
  existing for its Python version. clickhouse-connect for backtest series,
  later.
- Flask kept as the HTTP surface; pytest required for core math, registry,
  overlays, proposal lifecycle, and simulator.

## Quality bar

- Idempotent storage.
- `source` always required for multi-venue markets.
- Tests for each registered indicator against a fixed OHLC fixture.
- Simulator tests with synthetic mid + book paths (no live DB required).
- **The approval gate is tested as a safety property**, not just a
  happy path: an unapproved proposal must never produce a write, and a
  stale-version apply must fail rather than clobber.
