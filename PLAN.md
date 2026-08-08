# Indicators — Plan

Working agreement for every agent picking this up:

1. **One step per request.** Finish the current step, get it accepted, stop.
   Do not start the next step in the same session unless the user says so.
2. **Read `ARCHITECTURE.md` first.** It is the source of truth; changing it
   needs explicit approval. This is an **insight engine**: it reads freely,
   stores its own analytics, and proposes config changes that a human must
   approve before anything is written. TA feeds **parameters, not entries**.
3. **Run `python -m pytest tests/ -q` before touching anything** (289 tests
   green as of 2026-07-30) and again before declaring a step done. Every new
   module ships with tests that need no Mongo/ClickHouse/config-manager.
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
      in `apply_to`, `combine`, `OverlayContext`, and three overlays —
      `vol_widen`, `trend_tilt`, `squeeze_brace`. Grid-swept invariants.
- [x] **4b. Allowlist expansion + scope change** (2026-07-30, approved) —
      ARCHITECTURE rewritten from "research toolkit, no config writes" to
      **insight engine with approval-gated config writes**. Allowlist
      gained the absolute fields `price_min_spread`, `price_max_spread`,
      `min_base`, `max_base`, `is_enable`, `number_of_order_each_sides`;
      `ConfigDelta` now splits scaling vs absolute composition, refuses
      conflicting absolutes, validates the composed result (band inversion,
      inventory inversion, quoting inside the round-trip fee floor), and
      flags `is_high_impact`. `OverlayContext` carries the current config.
      New overlays `band_widen` (far edge only — the thing
      `spread_multiplier` structurally cannot do) and `size_down`
      (`max_base`). Automatic overlays may never touch `is_enable`.

- [x] **5. Proposal store + approval gate** (`ta/proposals/`) — `models.py`
      (`Proposal` with derived id so a recomputed recommendation is the same
      card; `delta_hash` ignores prose so rewording does not fork it;
      evidence required, either analysis results or a written reason),
      `lifecycle.py` (pure state machine, `approve` demands an approver,
      `check_version` marks a moved config stale, `FAILED` terminal on
      purpose), `store.py` (`ta_proposals` + append-only `ta_decisions`;
      re-proposal rules — pending/approved/applied suppress, **rejected
      never resurrects**, stale/failed are replaced), CLI
      `python -m ta.proposals list|show|approve|reject`. 25 tests including
      the exhaustive transition table and the safety property that APPLIED
      is reachable only through a recorded approval. No network writes.

- [x] **6. Config-manager client + gated apply** — `ta/configclient.py`
      (stdlib `urllib` behind an injectable transport, no new dependency;
      owns the wrapper/inner unwrapping and strips server-owned
      `version`/`timestamp`; `put_market` is the single write) and
      `ta/proposals/apply.py` (refuses anything not approved *and* anything
      approved without an approver; re-reads live config; stale check blocks
      the write; composes then **verifies only allowlisted fields changed**
      — the guard against a whole-document PUT silently resetting a field to
      its schema default; records base and result versions; transport
      failure lands `failed`, never half-written; an already-satisfied
      proposal closes without writing). CLI `apply` is **dry run by
      default**, `--write` requires `--by`. 19 tests on a fake transport.

  *Remaining for acceptance:* a dry-run diff against the **real**
  config-manager on a market you are willing to touch. Set
  `CONFIG_MANAGER_URL` and run `python -m ta.proposals apply <id>` (no
  `--write`) — it reads and prints, and cannot write.

- [x] **7. Exchange view + tier list** — `ta/tiers.py` (own Mongo store
      `ta_market_tiers`; `major`/`other`, unknown defaults to `other` so an
      unclassified market is never assumed safe; setting a tier records who)
      and `ta/view.py` (per-market row joining tier + live config + candle
      health; risk as **named flags** — `thin-history`, `sparse`, `stale`,
      `volatile`, `erratic-volume` — not a score, since a score invites
      treating rank as a decision; `classify` covers all six tier×config
      combinations and rows sort disagreements first with a plain-English
      footer). CLI: `python -m ta.view --source X`, `--json`, `--no-stats`,
      `tier set|list|clear`, `propose enable|disable` which drafts a
      proposal that still lands `pending`. 28 tests, no live services.

- [x] **8. Candle-based backtest** (`ta/backtest/`) — `simulator.py` mirrors
      the live quoting math (spread scale, **one**-maker-fee floor, per-side
      asymmetry with its clamp, level step, inventory skew in both
      parameterisations, premium correction, band-derived totals split
      equally); divergences from live are named in `DIVERGENCES` rather than
      hidden. `runner.py` walks bars quoting off the *previous* close (no
      lookahead), fills a bid when `low <= price`, recomputes overlays every
      N bars, and stamps `DISCLAIMER` on every artifact. `metrics.py`:
      fee-aware PnL, per-side uptime, inventory path, markout at
      60/300/600/1800s. `sweep.py`: grid over allowlisted knobs, ranked, with
      fills/uptime/markout shown alongside PnL so a wide-and-rare or
      tight-and-adverse config cannot masquerade as a winner. 29 tests
      including hand-computed ladders and the break-even identity.

  **While mirroring the math I found a bug in step 4's code**: the overlay
  validator floored `price_min_spread` at **2×** `maker_fee`.
  `market_maker._spread_floor` documents that as a real, costly mistake —
  break-even is one fee (gross `2s` vs `2×maker_fee`), and the 2× version
  forbade the 15–18bp band where nobitex's expected value peaks. Fixed in
  `ta/signals/overlays.py`, ARCHITECTURE, and the tests.

## Proposal: the inventory tilt (2026-07-31) — **APPROVED 2026-07-31**

> Approved by the user on 2026-07-31. The allowlist change (`target_base`,
> capped) is authorised and **implemented in step 10** (2026-08-06).
> Step 9 (shadow mode) writes nothing anywhere and is done.


### What was found

Brute forces across 6 markets, 2 venues, 3 signal families and 4 timeframes
(all in `ta_research.research_runs`, reproducible via the `usdt-cross-venue`
and `actionable-1h` scenarios) produced one result worth acting on:

| market | venue | tf | signal | excess | t | p | turnover |
|---|---|---|---|---|---|---|---|
| USDTIRR | ramzinex | 1H | `sma_20_30` | 0.374 bp/bar | 2.55 | 0.000 | 0.034 |
| USDTIRT | nobitex | 1H | `sma_20_30` | 0.343 bp/bar | 2.32 | 0.010 | 0.034 |

The same parameters come top on **two independent venues** quoting the same
asset. BTC and ETH are null at every horizon in both quote currencies, so
this is specific to the rial-quoted USDT books, not a general effect.

`excess` is the covariance term from `ta/signal_value.py`: the return earned
by *varying* inventory, over and above what the same average inventory would
have earned anyway. It is not a strategy P&L and the rial's devaluation
drift is already excluded from it.

A stronger-looking signal exists — `volz_dir` (volume-confirmed momentum),
p=0.000 on both venues — and is **not** proposed here. Its turnover is 0.38
per bar against `sma_20_30`'s 0.034, which is the difference between an
edge of ~0.43bp and ~10bp per unit of inventory moved. The statistically
better signal is the economically unusable one.

### How this would change what the market maker does

It does not change the spread, the ladder, or the number of orders. It moves
**one number**: `target_base`, the inventory level the skew logic treats as
neutral (`market_maker.py:1180`).

```
tilt > 0  ->  raise target_base  ->  inventory_q goes negative
          ->  apply_inventory_skew shifts the grid reference UP
          ->  bids sit relatively closer, asks further
          ->  passive fills accumulate base
```

Three properties make this the right shape for this desk:

1. **Nothing is ever crossed.** The tilt is expressed entirely in where the
   resting grid sits. The MM is quoting both sides regardless; this changes
   which side fills first. That is why an edge far too small to pay a taker
   round trip is still worth having.
2. **It is self-limiting.** `inventory_q` saturates at the band edges, so a
   tilt can never walk inventory outside `[min_base, max_base]`. The
   existing risk limits keep working untouched.
3. **`market_maker` needs no code change at all.** `target_base` is already
   in `ConfigSchema` (config-manager `models.py:247`) and already read by
   the skew. This project would set it periodically through the existing
   approved-proposal path, exactly as the crypto_price calibrators set
   `spread_multiplier` today.

### What it is worth, honestly

At 0.343 bp/bar over 8760 hourly bars, the point estimate is **~30% per year
on the notional the target is allowed to swing**.

**Measured against the real bands (step 9, 2026-07-31), that is a small
number in absolute terms.** The live configs are:

| market | band (USDT) | half-band | swung at cap 0.30 | ~30%/yr |
|---|---|---|---|---|
| nobitex `USDTRLS` v109 | 10 – 450 | 220 | 66 USDT | **~20 USDT/yr** |
| ramzinex `USDTIRR` v126 | 10 – 1100 | 545 | 164 USDT | **~49 USDT/yr** |

So roughly **70 USDT/year across both books**, and the 95% interval puts it
between ~11 and ~127 USDT/year. The percentage is attractive; the absolute
is not, because the bands are currently small.

This does not kill the proposal but it reframes it. The P&L scales linearly
with band size, so the tilt is worth having mainly as something that is
already correct and tested when inventory limits grow — not as a source of
money at today's size. Weigh that against the operational risk before step
11, and do not spend scarce review attention here that a fee-tier or
spread-calibration change would repay better.

The confidence interval is wide and must be quoted with the number. At
t=2.32 the 95% interval on the nobitex estimate runs **[4.7%, 55%] per
year**. Positive across the interval, but a fifth of the point estimate is
entirely consistent with the data. Anyone who repeats the "30%" without the
interval is overselling it.

Two further honesty notes. MA has now been tested both with and without the
`consecutive_only` gap control, so the p-value alone is not the evidence —
the cross-venue replication at identical parameters is. And the estimate
comes from ~2 years spanning one devaluation regime.

### Why `target_base` and not `structural_premium`

`structural_premium` is already on the allowlist and would produce a
directional lean today, with no ARCHITECTURE change. It is still the wrong
lever:

- It shifts the reference price outright and is **not self-limiting** — it
  keeps leaning however stretched inventory becomes, which is precisely the
  behaviour the inventory skew exists to prevent.
- `market_maker.apply_premium_skew` documents it as a bias **correction**,
  not a discretionary lean. Using it for a directional view would conflate
  two signals the design deliberately separates, and would corrupt the
  calibrator that owns it.

The tilt belongs on the inventory axis because it is a statement about how
much to hold, not about where the market's true mid is.

### What must change (this is the approval being requested)

1. Add **`target_base`** to the overlay allowlist in ARCHITECTURE and to
   `ConfigDelta`, as an absolute field, validated to sit within
   `[min_base, max_base]` of the composed config.
2. Cap the tilt: a new overlay must not move `target_base` more than a
   configured fraction of the band from its midpoint (proposed default
   **±30%**, not the full range), so a wrong signal cannot fully stretch
   inventory.
3. Everything else stays: proposals, human approval, stale-version checks,
   dry-run default.

### Risks, and what covers each

| risk | cover |
|---|---|
| Signal decays or was overfit | Shadow mode first (step 9); measure before writing |
| Tilt raises inventory variance | Cap at ±30% of band; the skew still mean-reverts to the new target |
| Wrong-footed by a jump | Band edges bound the loss; `disable_buy/sell` and the risk plane are untouched |
| Fights the calibrators | Stale-version check already refuses to clobber; `target_base` is a field no calibrator writes |
| Silent drift into a large position | Step 12 measurement compares realized inventory path against the intended one |

## Proposal: out-of-band inventory recovery (2026-07-31) — NEEDS APPROVAL

Unlike the tilt proposal above, this is a **change to `market_maker`'s own
skew logic**, not a config field. This project cannot route it through the
approval flow (that flow writes config, not code). What is offered here is a
reference implementation, its safety properties as tests, and an honest
account of what the evidence can and cannot support.

### What is wrong today (verified in the code)

**1. The skew saturates exactly where urgency should begin.**
`_inventory_q` ends with `max(-1.0, min(1.0, q))` (`market_maker.py:1194`),
so once inventory passes a band edge the normalized error pins at ±1 and the
price skew stops responding. Being 1 unit below `min_base` and being 500
below produce an identical quote. Sizing does react — `sell_total` goes to
zero below the floor — but *prices* do not, so the book stops asking for the
inventory it needs at precisely the moment it needs it most.

**2. The regime multipliers never reach the skew.**
`underprice_multiplier` and `overprice_multiplier` are read only inside the
special amount branches (`market_maker.py:1299`, `:1305`). `_inventory_q`
always uses the static `[min_base, max_base]`. So while the UNDERPRICE
branch is deliberately buying up to `max_base × underprice_multiplier`, the
skew reads that same position as over-full and leans against it. The two
mechanisms work against each other, and the ordering is the fix: **resolve
the effective band for the current regime first, then ask whether inventory
is outside it.**

### Proposed behaviour

Reference implementation lives in `ta/backtest/simulator.py`, behind
`build_ladder(recovery=True)` — off by default, since it is a proposal:

- `effective_band(config, regime)` — applies the regime multiplier **first**.
- `band_breach(base, eff_min, eff_max, span)` — how far outside, normalized
  by a fraction of band width (default 0.25) and capped at 1. Complements
  `inventory_q` rather than replacing it: the existing skew leans inside the
  band, this adds urgency outside it.
- `recovery_spreads(config, side, breach)` — interpolates the recovering
  side's spreads from normal toward `spread_floor(config)` as breach goes
  0 → 1. The worsening side is untouched.

**The fee guarantee is structural, not a check.** `effective_spreads` has
already floored its output at one maker fee, so interpolating *toward* that
floor can reach break-even and provably not pass it. There is no branch that
could be forgotten. Measured on a 10bp fee, band 40–100, reference 100:

| inventory | breach | bid today | bid proposed | gain |
|---|---|---|---|---|
| 40.0 (at floor) | 0.00 | 99.6000 | 99.6000 | 0 bp |
| 34.0 | 0.40 | 99.6000 | 99.7200 | 12 bp |
| 28.0 | 0.80 | 99.6000 | 99.8400 | 24 bp |
| 20.0 | 1.00 | 99.6000 | 99.9000 | 30 bp |
| −50.0 | 1.00 | 99.6000 | 99.9000 | 30 bp (capped at the fee) |

Tests cover: symmetry above the ceiling, inertness inside the band, the
regime ordering, and that the best bid never crosses the best ask at any
inventory from −200 to +800.

### What the evidence does NOT show

**The candle backtest cannot price this, and I will not pretend otherwise.**
In the simulator inventory can never leave the band, because order sizes are
derived from it (`sell_total = max(0, base − min_base)`). Running the two
variants over a trending market gives *byte-identical* P&L, because the
recovery never fires. That is now listed in `DIVERGENCES`.

Out-of-band states arise live from mechanisms the simulator does not model:
the UNDERPRICE branch deliberately buying past `max_base`; the thin buy
probe kept above the ceiling; partial fills and cancel/replace races; and
balance changes from outside the maker (transfers, other strategies,
allocation changes). Evaluating the proposal therefore needs either the
ClickHouse book replay or a live shadow measurement of how often and how far
the band is actually breached — see step 13.

### The trade-off, stated plainly

Faster recovery costs execution price. Inventory below the floor usually
means the asks have been getting hit into a rising market, so recovering
aggressively means buying higher; above the ceiling it means selling into a
fall. Today's saturation implicitly chooses "slow recovery, better price".
The proposal chooses the opposite. Which is correct depends on whether the
band is a real risk limit — if it is, being outside it is the thing to fix,
and the current behaviour is an unstated preference rather than a decision.
This is the part that wants a human view before anything ships.

## Next steps — precise specs

- [x] **9. Shadow mode for the tilt** — DONE 2026-07-31, now collecting.
      `ta/analysis/tilt.py` (registered analysis, `sma_20_30` -> tilt in
      [-1,1], plus `implied_target_base` anchored on the band midpoint so a
      zero tilt reproduces today's behaviour exactly);
      `ta/research/shadow.py` (observation, store, and a scorer that refuses
      pairs spanning a service outage); `ta/research/scheduler.py` (hourly
      daemon thread, no backfill — a gap in the record is information);
      `/research/shadow`, `/research/shadow/record`, `/research/shadow/score`.
      Running on germany-2 for `USDTIRT:nobitex:USDTRLS` and
      `USDTIRR:ramzinex`. **Revisit after 2026-08-28** (28 days).

      Two operational findings from wiring it up, both recorded in
      Pitfalls: the candle store and the trading config disagree about
      market names on nobitex, and config-manager is firewalled away from
      sibling containers.

  *Superseded spec (kept for the record):*

  *What:* compute what `target_base` *would* have been set to, hourly, and
  record it — writing nothing. This is the step that turns a backtest claim
  into evidence.

  - New analysis producing `tilt` in [-1, 1] from `sma_20_30` on the market's
    own 1H candles, reusing `ta.compute` (no new math).
  - Scheduled run (cron or the research service) storing
    `(time, market, source, tilt, implied_target_base, live_target_base,
    live_inventory, mid)` in `ta_research.tilt_shadow`.
  - **Run for at least 4 weeks** before anything is proposed. Then score the
    recorded tilts against realized forward returns with the same
    `signal_value.evaluate` used offline, and compare that live-sampled
    estimate against the backtest's 0.343 bp/bar. A material shortfall kills
    the proposal.

  *Acceptance:* a shadow table with 4+ weeks of hourly rows and a scored
  comparison against the backtest estimate.

- [x] **10. `target_base` allowlist + tilt overlay** — DONE 2026-08-06.

  `ConfigDelta.target_base` as an absolute field; `apply_to` refuses a
  target outside `[min_base, max_base]` or more than 30% of half-band from
  the midpoint. `inventory_tilt` overlay maps the `tilt` analysis through
  the same `implied_target_base` math as shadow mode (default cap 0.30;
  step 11 uses `cap=0.15`). Grid-swept: never outside the band, never past
  the cap. Stacking with `size_down` in one `evaluate` can refuse on
  compose when the shrunk band makes the tilt target illegal — that is
  intentional.

- [x] **10b. Levels analysis (static + dynamic S/R)** — DONE 2026-08-06
      (queued after step 10, implemented as an interrupt — step 10 still open).

  `ta/analysis/levels.py`: swing-pivot static supports/resistances
  (clustered, nearest-above/below, distance in bps) plus dynamic levels
  reused from existing Donchian/BB/Keltner/VWAP via `compute_series`.
  Donchian dynamic S/R uses the *prior* bar's channel so `in_range` can
  actually go false on a breakout (live Donchian always contains close).
  Registered as `levels` in `ANALYSES`. Insight-only (no overlay/proposal).
  Tests on constructed pivots and channels; no new third-party TA deps.
  Wired through pipeline / `GET /analysis` only.

- [ ] **11. First live tilt, one market, small**

  USDTIRT on nobitex only. Cap at ±15% of band for the first month — half
  the proposed default. Full flow: analysis → proposal with the shadow
  evidence attached → approval → dry-run → `--write`. Then leave it alone.

- [ ] **12. Did it work?**

  Compare the realized inventory path and its mark-to-market against a
  counterfactual constant `target_base` over the same window, using
  `pnl.fills` / `pnl.mm_books` if the ClickHouse readers exist by then.
  Deliverable is a number with an interval, not a verdict. If the live
  estimate sits below the backtest's lower bound, revert `target_base` to
  `None` and record why.

- [x] **13. Measure how often the band is actually breached** — DONE
      2026-08-01, one-off analysis (not yet wired into shadow mode; see
      follow-up below).

  *Method:* pulled 16.0 days of `pnl.balance_snapshots` (ClickHouse, via the
  germany-2 jump host) for both live USDT markets, and the full config
  version history (`market_config_versions` in Mongo) to reconstruct the
  actual `[min_base, max_base]` in force at each snapshot (bands changed
  ~15 times over the window; a single current-band snapshot would have
  been wrong for most of it). Verified against the raw balance distribution
  and a manual spot-check, not just the aggregate.

  *Result — breaches are neither rare nor shallow. The "drop it" condition
  in this step's original spec is falsified:*

  | market | time outside band | episodes | avg episode | mean depth | p90 depth | reach full recovery span (0.25) |
  |---|---|---|---|---|---|---|
  | USDTRLS/nobitex | **59.6%** (226.6h) | 130 | 104.6 min | 0.162 band-widths | 0.455 | 12.8% of breached snapshots |
  | USDTIRR/ramzinex | **65.8%** (252.0h) | 221 | 68.4 min | 0.070 band-widths | 0.107 | 5.2% of breached snapshots |

  Breach is overwhelmingly **below `min_base`**, not above `max_base`
  (46.3%/19.9% split for USDTRLS, 51.9%/14.8% for USDTIRR). Median held
  balance sits almost exactly on the floor (13.99 for USDTRLS, 9.96 for
  USDTIRR against `min_base=10` on both) — the desk operates *at* the
  floor as routine, not as an occasional excursion, so "outside the band"
  is close to the median state, not a tail event.

  *What this does and does not establish:*
  - **Does establish**: the situation `_band_recovery` targets is common
    and sustained (episodes average over an hour), so the saturation bug
    is live, not theoretical.
  - **Does NOT establish** that recovery is the right fix, or at what
    `recovery_aggressiveness`. Faster recovery trades execution price for
    inventory — see the proposal's trade-off section — and that trade at
    this frequency (an hour+ at a time, half the time) is a much bigger
    live decision than "breaches are rare, so it barely matters."
  - Given the frequency found, `min_base=10` itself may be set tighter
    than the desk's real operating range — worth a direct look independent
    of the recovery proposal.

  ***Regime correlation — done 2026-08-01, queried Loki directly.*** Market
  state carries no market identifier in the log line (`Market state: X`,
  one shared logger across 15 per-market threads); attributed each line to
  a market by matching the immediately-preceding `Budgets from held=...
  (min=.. max=..)` line against that market's *actual band at that
  timestamp* (bands changed ~15 times; matching only today's values first
  silently dropped everything before the last change — caught and fixed).
  Pulled 931K raw log lines from Loki (cursor-paginated; a single
  `query_range` call truncates silently once density exceeds its 5000-line
  cap, which a first, coarser attempt hit on 26 of 32 windows). Net
  coverage: 69–85% of balance snapshots matched to a regime within 5 min;
  the rest is mostly the first ~3 days, when USDTRLS and USDTIRR briefly
  shared identical bands and are genuinely indistinguishable by this method.

  **Finding: the regime this proposal is centered on essentially never
  occurs.** Over ~13 days of attributed data, both markets sat in `Regular`
  the overwhelming majority of the time; `OverPriced` was rare (0.02% of
  samples, USDTRLS; 0.7%, USDTIRR); **`UNDERPRICE` was observed zero times
  in either market.** The proposal's own worked example —
  `_effective_band` resolving UNDERPRICE's raised ceiling so a
  `max_base × underprice_multiplier` position doesn't read as over-full —
  describes a scenario with **no confirmed live occurrence** in this
  window. It remains correct as a fix (proven by the unit tests), just
  unconfirmed as something that has actually mattered yet.

  `OverPriced` did correlate strongly for USDTIRR: 100% of 413 matched
  snapshots in that state were breached, vs. 65.1% baseline. **This is
  confounded, not clean evidence**: `overprice_multiplier` has been `1.0`
  — a no-op — in every config version ever recorded for either market, so
  `OverPriced`'s floor-lowering has never actually moved the effective
  band in this data. The correlation more likely reflects that
  `OverPriced` mechanically *causes* low inventory (its special branch
  sells down toward the floor) than that the ordering bug's band-widening
  was exercised and observed.

  *Net effect on the proposal:* the "skew saturates at the edge" half is
  now well-supported (breach frequency, this step). The "regime ordering"
  half — the more novel, more surprising half of the finding — has not
  been observed to bite in practice, because the regime it depends on
  barely occurs, and the other regime's multiplier has never been set to
  do anything. That does not make the fix wrong; it means the second half
  of the justification is currently theoretical, not measured, and should
  be described that way.

  *Recommendation:* do not set `recovery_aggressiveness` above 0 on any
  market. Wire this measurement (both breach frequency and regime
  attribution) into shadow mode so it runs continuously instead of as a
  one-off, before asking for a decision on enabling it anywhere.

  *Follow-up (not done):* extend `ta/research/shadow.py` to pull
  `live_inventory` from `pnl.balance_snapshots` each hour (the ClickHouse
  reader from the options list, minimally scoped to this one table) so
  this report refreshes on its own instead of needing a repeat one-off.

## Options after step 12 (each needs explicit approval)

- **Widen the tilt to a second market.** Only BTCIRT showed any hint
  (`volz_dir` p=0.065); the MA tilt is null there. Do not assume the
  USDT result transfers — re-run the shadow step per market.
- **Revisit `volz_dir` as a quote-skew signal rather than an inventory
  one.** It is statistically the strongest thing found (p=0.000 on both
  venues) and was rejected only on turnover. If it were expressed as a
  small, fast lean on `spread_asymmetry` instead of on inventory, its
  turnover cost would be different — that is a genuinely open question
  this project has not tested.
- **Scheduled insight job:** cron running `ta.pipeline` + proposal
  generation for watched markets, so the queue is warm and the view fresh.
  Approval still gates every apply.
- **ClickHouse upgrade path:** `fair_value_snapshots` + `pnl.mm_books`
  readers and book-replay fills, replacing candle optimism with real queue
  behaviour. Needed before any absolute PnL claim. Access is via the
  `germany-2` host; ask for a tunnel, never hardcode hosts.
- **Trades-level view:** read the raw trades collection for microstructure
  (trade-size distribution, one-sided pressure) rather than inferring from
  candles — a better basis for enable/disable than OHLC alone.
- **Auto-approve for a narrow class:** e.g. de-risking changes only
  (widening, sizing down, `is_enable=False`). Would need its own
  ARCHITECTURE amendment and a kill switch; deliberately not assumed.
- **Simulator parity upgrades:** bypass-wall/penny-jump, SPECIAL branches —
  only if a run shows the v1 simulator is the binding error source.

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
- Never import `market_maker` runtime modules; copy pure math (step 8).
- Maker fee > touch spread on both USDT venues: proposals widen, never
  narrow below break-even (`apply_to` refuses it outright).
- The break-even half-spread is **one** `maker_fee`, not two. A round trip
  is gross `2s` against `2 × maker_fee` of fees. Using 2× here is a known
  bug that forbids the band where nobitex's expected value peaks — see
  `market_maker._spread_floor`.
- `PUT /config/market/{source}/{market_name}` replaces the **whole**
  document and bumps the version. Always read-modify-write, and always
  check the version has not moved — the crypto_price calibrators write the
  same documents on their own schedule.
- `spread_multiplier` scales both band edges together
  (`market_maker.py:956`). To move one edge, propose the raw
  `price_min_spread` / `price_max_spread`.
- `max_base` is not just a skew input: it sets each cycle's buy total and
  caps the sell total (`market_maker.py:1338`), so it is the real "size in
  this market" knob.
- No automatic overlay may set `is_enable`. Enabling or disabling a market
  is a human decision that goes through the proposal flow.
- **The candle store and the trading config use different market names.**
  market-data-collector writes nobitex candles as `USDTIRT`/`BTCIRT`
  (Toman); the configs the market maker actually trades are
  `USDTRLS`/`BTCRLS` (Rial). Ramzinex uses `USDTIRR` on both sides. Looking
  up the wrong name returns *no config* silently rather than erroring, so
  any code joining research to config must carry the alias explicitly —
  see `SHADOW_MARKETS`, which takes `market:source:config_market`.
- **config-manager is firewalled off from sibling containers.** A
  `DOCKER-USER` rule drops port 8000 from everything except claw, germany
  and germany-2's own public IP, so container-to-container fails even on
  the same docker network. Reach it via the bridge gateway
  (`docker network inspect mongo_default -f '{{(index .IPAM.Config 0).Gateway}}'`),
  which the rules do permit. Do not edit the firewall to work around this.
- **Always run signal research with the consecutive-bar control on**
  (`consecutive_only`, default in every scenario). Candle series are sparse,
  so without it a forward return spans a variable amount of wall-clock time
  and any activity-correlated signal is partly measuring the clock. It is
  not cosmetic: turning it on moved MA at 1H from p=0.075 to p=0.010 and
  BTCIRT volume from p=0.290 to p=0.065.
- Score signals with `excess` (the covariance term), never with
  `mean(signal x return)`. On a market with drift the naive term is
  dominated by average exposure, which a constant inventory earns anyway.
  This mistake was made once here and inverted the conclusion.
