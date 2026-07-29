# Indicators — Plan

Working agreement for every agent picking this up:

1. **One step per request.** Finish the current step, get it accepted, stop.
   Do not start the next step in the same session unless the user says so.
2. **Read `ARCHITECTURE.md` first.** It is the source of truth; changing it
   needs explicit approval. Mode A (research toolkit) + MM-aware backtest are
   locked. TA feeds **overlays, not entries**. No live config-manager writes.
3. **Run `python -m pytest tests/ -q` before touching anything** (150 tests
   green as of 2026-07-30) and again before declaring a step done. Every new
   module ships with tests that need no Mongo/ClickHouse.
4. **No new markdown files.** Design lives in ARCHITECTURE.md, status here,
   operational notes in RUNNING.md.
5. Style: pure functions over classes-with-state, rich "why" docstrings
   (match `ta/engine.py`), no new third-party TA/backtest dependencies.

## Done

- [x] **0. Architecture lock** — ARCHITECTURE.md + this plan.
- [x] **1. Core package** — `ta/`: registry (`registry.py`), batch engine
      with warmup policies + causality guard (`engine.py`), OHLCV
      normalization + frame-health reporting (`frame.py`), error tree
      (`errors.py`), source-aware Tehran→UTC Mongo reader (`ohlc.py`),
      idempotent store on natural key (`store.py`), Flask adapter (`app.py`:
      `/indicators` catalogue, `/indicator` compute, `/health`). Legacy
      `indicators/` package deleted. Import root is `ta` (ARCHITECTURE
      updated accordingly).
- [x] **2. Indicator library** — 49 indicators registered across
      trend/momentum/volatility/volume (`ta/indicators/`), all under one
      parametrised contract test plus exact-math references for the core
      seven. Includes everything step 2 asked for (ATR, Bollinger, VWAP,
      volume SMA, ADX) and more.
- [x] **3. Analysis layer + pipeline** — `ta/analysis/` (`types.py`
      AnalysisResult + loose-param plumbing; `volatility.py` with cc +
      Parkinson estimators, NATR, `vol_percentile` and regime label;
      `trend.py` with volatility-scaled direction votes × ADX strength →
      `trend_score` in [-1,+1]; `regime.py` TTM-style Bollinger-inside-
      Keltner squeeze with release events). `ta/pipeline.py`: one fetch →
      indicators + analyses → JSON snapshot, forming-bar dropped by
      default; CLI `python -m ta.pipeline`; `GET /analysis` endpoint.
- [x] **4. Overlay mapper** — `ta/signals/overlays.py`: `ConfigDelta`
      frozen dataclass (allowlist-only fields, bounds from config-manager's
      model, disables can be set but never cleared), per-field composition
      in `apply_to` (multiplier composes ×, sensitivity max, premium pair
      overrides), `combine`, `OverlayContext`, and three overlays —
      `vol_widen` (percentile ramp), `trend_tilt` (signed bp premium +
      gate, emitted as a pair), `squeeze_brace` (release-bar widen +
      optional adverse-side disable). Grid-swept invariant tests. No
      writes anywhere.

## Next steps — precise specs

- [ ] **5. Backtest data readers** (`ta/backtest/readers.py`)

  *What:* ClickHouse readers for the replay inputs.

  - `read_fair_mid(market, source, start, end)` ←
    `crypto_analytics.fair_value_snapshots` (preferred fair path).
  - `read_books(market, source, start, end)` ← `pnl.mm_books`;
    `read_fills(...)` ← `pnl.fills` (optional, validation only).
  - Use `clickhouse-connect`; env `CLICKHOUSE_URL` (+user/pass) documented
    in RUNNING.md only. Dev access to prod CH is via SSH through the
    `germany-2` host (ask the user for a tunnel; never hardcode hosts).
  - Coverage report per range: snapshot count, median gap, largest hole →
    `book_quality: ok|degraded`. Thin coverage triggers the degenerate-book
    fallback downstream — never synthesise dense books.

  *Tests:* pure parts only — query builders, row→frame conversion, coverage
  classifier on synthetic rows. No live CH in tests.

- [ ] **6. MM quote simulator** (`ta/backtest/simulator.py`)

  *What:* side-effect-free mirror of `market_maker`'s quoting math: fee
  floor, spread scale, inventory skew, premium skew, ladder prices/sizes.
  Signature shape: `quotes(mid, inventory, config: dict) → Ladder`.

  - **Copy the math into this repo; import nothing from `market_maker`.**
    Its `agent.py` has import-time side effects (including a hardcoded
    SOCKS proxy) and a live loop we must not drag in.
  - Golden tests: hand-computed expected ladders for a handful of
    (mid, inventory, config) cases, cross-checked against `market_maker`'s
    own unit-test expectations where they exist. Any divergence from live
    behaviour gets listed in the run report, not papered over
    (ARCHITECTURE "honesty rule").
  - Out of scope v1: bypass-wall/penny-jump, SPECIAL branches, latency.

- [ ] **7. Backtest runner + metrics** (`ta/backtest/runner.py`, `metrics.py`)

  *What:* walk the book/snapshot clock (not OHLC bars), at each step apply
  baseline config + overlay deltas (recomputed on bar close only — no
  intra-bar lookahead), quote via the simulator, match fills
  deterministically v1 (fill when the market trades through the quote;
  document the rule), track inventory.

  - Metrics: fee-aware PnL, inventory path, quote uptime per side, fill
    count, and **markout vs mid at horizons including ~600s** — the desk's
    established honest horizon for scoring fills.
  - Artifacts: parquet/JSON under `BACKTEST_OUT_DIR` + the full overlay
    timeline for audit. Optional Mongo summary doc.
  - End-to-end test on a synthetic mid + book path with a known outcome
    (e.g. constant spread, no overlay → analytically computable PnL).

- [ ] **8. One real-market dry run**

  Pick one `(source, market)` with decent `mm_books` coverage (check
  coverage first with the step-5 report; ask the user for the CH tunnel).
  Run baseline vs one simple vol overlay over the same window. Deliverable
  is a short honesty report in the run artifacts: data coverage, simulator
  divergences, whether numbers are trustworthy — **not** a P&L claim.
  Adjust ARCHITECTURE only if gaps force it (needs approval).

## Options after step 8 (each needs explicit user approval)

- **Mode B — live overlay publishing:** periodic job writing allowlisted
  deltas to config-manager like the existing calibrators. Requires
  ARCHITECTURE change + review of kill-switches. The natural next step if
  step 8 looks honest.
- **Parameter sweeps:** grid over overlay params reusing the runner;
  registry already exposes param schemas for enumeration.
- **Classic OHLC signal backtest:** long/short entries on indicator
  signals. Explicitly out of scope today; smallest lift once the runner
  exists, but least aligned with the desk (fees > touch spread makes taker
  strategies structurally hard here).
- **Scheduled batch compute:** cron job running `ta.pipeline` for the
  watched markets, so `indicators_data` stays warm for dashboards.
- **Simulator parity upgrades:** bypass-wall/penny-jump, stochastic fills,
  SPECIAL branch replay — only if step 8 shows deterministic v1 is the
  binding error source.

## Pitfalls (read before coding)

- OHLC `time` in Mongo is a **Tehran-naive ISO string** (UTC+3:30, fixed);
  `ta/ohlc.py` converts to UTC — never bypass it. CH tables are UTC.
- `source` is always required; same market_name exists on several venues.
- Candle series are **sparse** (bars with no trades don't exist);
  `FrameInfo.gap_ratio` says whether windows mean what they claim.
- `volume` is sampled (nobitex public feed caps 20 trades/poll) — ratios
  and z-scores only, never absolute volume.
- Warmup: recursive indicators (EMA/RSI/ATR/ADX) are non-NaN long before
  they are correct; the engine masks by default — backtests should `trim`.
- Never import `market_maker` runtime modules; copy pure math (step 6).
- Maker fee > touch spread on both USDT venues: overlays widen, never
  narrow below fee floor.
