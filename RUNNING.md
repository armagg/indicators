# Running Indicators

An **insight engine** (`ta/` package) with a thin Flask adapter. Reads OHLC
candles from MongoDB (`market_making.ohlc`, written by
`market-data-collector`), computes indicators and analyses, keeps its own
analytics and proposal stores, and **proposes** market-maker parameter
changes.

Proposals reach `config-manager` only after a human approves them
(`ta/proposals/`). There is no path from an analysis to a live config —
see ARCHITECTURE's permission model. Nothing here imports or runs
`market_maker`.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pytest
cp .env <adjust as needed>
```

`.env` keys: `MONGO_URI`, `DB_NAME`, `OHLC_COLLECTION_NAME`,
`INDICATORS_COLLECTION_NAME`, and `CONFIG_MANAGER_URL` (service root, e.g.
`http://localhost:8000`). Optional overrides:
`PROPOSALS_COLLECTION_NAME`, `DECISIONS_COLLECTION_NAME`,
`TIERS_COLLECTION_NAME`, `RESEARCH_DB_NAME` (default `ta_research`).

Requires MongoDB reachable at `MONGO_URI` with candles in
`OHLC_COLLECTION_NAME` (documents `{market_name, source, interval, time, open,
high, low, close, volume, ...}`; `time` is a Tehran-naive ISO string as MDC
writes it). Locally: the `trading-local-mongo` container on port 27017.

## Tests (no Mongo needed)

```bash
python -m pytest tests/ -q
```

312 tests: a contract test parametrised over all 49 registered indicators
(purity, column naming, no inf, warmup sanity), exact-math references for
SMA/EMA/RSI/MACD/StochRSI/ATR/Bollinger, engine/registry/frame behaviour,
store idempotency, analyses on constructed regimes (ramps, chop, vol
spikes, squeeze-and-pop), overlay deltas with grid-swept invariants, the
proposal lifecycle (exhaustive transition table + the approval-gate safety
property), the config-manager client and gated apply on a fake transport,
the exchange view's risk flags and disagreement logic, the simulator against
hand-computed ladders with analytic fill/fee cases, and the HTTP adapter
with the data layer stubbed.

## Run

```bash
python app.py    # http://127.0.0.1:5000   (port 5000 can collide with macOS
                  # AirPlay Receiver — disable it, or use another port)
```

### `GET /indicators`

Machine-readable catalogue of every registered indicator: parameters (with
types, defaults, bounds), output columns, required inputs, warmup, causality,
and per-indicator caveats. This is the source of truth — not this file.

### `GET /indicator`

```
# Legacy single-indicator form (loose params):
GET /indicator?symbol=USDTIRT&source=nobitex&interval=1H&indicator=rsi&period=21

# Batch form (one OHLC fetch, many indicators):
GET /indicator?symbol=USDTIRT&source=nobitex&interval=1H&indicators=rsi:period=21,macd,atr,bbands
```

Required: `symbol`, `source`, `interval`, and `indicator` or `indicators`.
**`source` is new and required** — the old service mixed candles from every
venue sharing a market name, which made the output fiction.

Options: `save` (default true; idempotent upserts, natural key
`(symbol, source, interval, analysis, param_hash, time)`), `latest=true`
(only the freshest value per column — drop the 1000-row payload),
`dropna=true`, `warmup=mask|trim|keep` (default mask; see `ta/engine.py`),
`limit`, `start`/`end` (Tehran wall time when naive).

Errors: 400 for anything the caller got wrong (unknown indicator, bad or
unknown params — typos are rejected, not silently defaulted), 404 for no
candles, 502 for Mongo unreachable.

### `GET /analysis`

```
GET /analysis?symbol=USDTIRT&source=nobitex&interval=1H&analyses=volatility,trend,regime&indicators=natr
```

Combined snapshot: frame health, indicator latest values, and analysis
verdicts (`volatility` → vol_percentile/regime, `trend` → trend_score
in [-1,+1] with direction, `regime` → squeeze state + release events).
Persists nothing unless `save=true`. Options: `limit`, `start`/`end`,
`warmup`, `series=true` (include analysis series).

### Pipeline CLI (same thing without HTTP)

```bash
python -m ta.pipeline -m USDTIRT -s nobitex -i 1H --indicators "rsi,macd" --analyses "volatility,trend,regime"
```

JSON on stdout; errors as JSON on stderr with exit 1. `--save` persists
indicator rows; `--keep-forming` keeps the still-forming last candle
(dropped by default — see the repainting note in `ta/frame.py`).

### Running on germany-2 (Docker)

Everything on that box is containerised — do not pip-install onto the host.
The image is `python:3.13-slim`, so the pinned `requirements.txt` works
there even though the host python is 3.10.

```bash
rsync -az --delete --exclude venv --exclude .git --exclude cache ./ germany-2:~/ta-research/Indicators/
```

```bash
ssh germany-2 'cd ~/ta-research/Indicators && docker build -t indicators:latest .'
```

The image carries `tests/` and `research/` as well as the app, so the suite
runs on the box it is deployed to:

```bash
ssh germany-2 'docker run --rm indicators:latest python -m pytest tests/ -q'
```

Research results live in the `ta_cache` docker volume, so they survive the
container:

```bash
ssh germany-2 'docker run --rm -v ta_cache:/app/cache indicators:latest python -m research.signal_brute --market USDTIRT --source nobitex --interval 1H --family volume'
```

### Research service (long-running)

One container stays up and takes jobs, instead of a container per task.
Started on germany-2 as:

```bash
docker run -d --name ta-research --restart unless-stopped --network mongo_default \
  -p 127.0.0.1:5005:5000 --cpus 2 --memory 2g \
  -e MONGO_URI=... -e DB_NAME=market_making -e OHLC_COLLECTION_NAME=ohlc \
  -e RESEARCH_DB_NAME=ta_research -e RESEARCH_WORKERS=2 -e FLASK_APP=app.py \
  indicators:latest flask run --host 0.0.0.0 --port 5000
```

Bound to loopback and capped at 2 CPUs — a research sweep must never be the
reason the collector falls behind.

```bash
curl -XPOST localhost:5005/research/scenarios/actionable-1h   # 24 jobs, one call
curl localhost:5005/research/jobs                             # queue + headlines
curl localhost:5005/research/jobs/<id>                        # one job, with its log
curl localhost:5005/research/results?family=volume            # every finished run
```

Also `POST /research/jobs` with a JSON job (or a list) for ad-hoc work.
Submissions return immediately with job ids; runs take minutes, so poll.

Jobs are stored in Mongo, not memory: a queue that dies with the process
loses the record of what was asked. Jobs left `running` by a previous
process are marked `interrupted` on startup rather than silently retried —
a half-finished run nobody observed should be visible as such. Identical
jobs already queued or running are deduplicated to the existing id.

### Saved scenarios

Named job batches live in `ta/research/scenarios.py`, versioned with the
code, because a comparison assembled from a shell loop typed at the time is
not reproducible six months later.

```bash
python -m research.signal_brute --list-scenarios
python -m research.signal_brute --scenario usdt-cross-venue
```

`consecutive_only` is on by default in every scenario. It is a **control,
not a knob**: sparse candles make a forward return span a variable amount of
wall-clock time, so any activity-correlated signal is otherwise partly
measuring the clock. It materially changes results — see the MA finding.

### Research cache (Mongo)

Computed series and finished runs live in their own database
(`RESEARCH_DB_NAME`, default `ta_research`) — deliberately separate from
`market_making`, so a cache wipe can never touch trading data.

- `indicator_columns` — one doc per (request, dataset). An EMA-20 is computed
  once ever and fetched thereafter.
- `research_runs` — one doc per finished job. A brute force repeated with
  identical inputs returns the stored result instead of redoing its
  permutation test.

Cache keys include a **fingerprint of the data**, not just the request:
row count, first and last timestamps, and a ~100-point subsample of the
closes. The subsample matters because `market-data-collector` upserts — a
bar in the middle can be rewritten when late trades land, leaving count and
endpoints identical. An endpoint-only key would serve stale series forever
in exactly that case. Use `--refresh` (series) or `--rerun` (whole run) to
bypass.

Measured on USDTIRT/nobitex 1H, 44-signal volume family, 200 permutations:
cold run vs **1.5s** for an identical rerun, **5.0s** with `--rerun` (38
series hits, 0 misses) — and the cached path reproduces the uncached numbers
exactly.

```bash
python -m research.signal_brute --market USDTIRT --source nobitex \
    --interval 1H --family volume --permutations 200
```

Families: `ma` (fast/slow MA spread) and `volume` (flow indicators — the
command prints a caveat about nobitex's 20-record tape cap; read it).

### Signal value: is a tilt worth taking?

`ta/signal_value.py` answers the question this desk actually has. It is not
"would this rule have made money" — on a market that trends, a rule that
happens to sit long most of the time prints a profit it did not earn. The
decomposition it reports separates the two:

```
mean(signal x return) = mean(signal) x mean(return) + Cov(signal, return)
                        \_______ drift _______/       \___ the signal ___/
```

You hold inventory continuously, so the drift term is yours whether or not
you have a signal — a constant inventory at the same average level earns it.
Only `Cov` is bought by *varying* the amount. `excess_bps` is that term, and
it is what `permutation_pvalue` tests, because its null stays centred on
zero however hard the market trends.

### Backtest and parameter sweep

Library only (no CLI yet — step 9 wires it into a recommendation):

```python
from ta.backtest import run, sweep, grid, render
from ta.ohlc import read_ohlc

candles = read_ohlc("USDTIRT", "nobitex", "1H", limit=2000)
rows = sweep(candles, live_config,
             grid(price_min_spread=[0.002, 0.004, 0.008],
                  price_max_spread=[0.012, 0.024]))
print(render(rows))
```

**Read the ordering, not the levels.** A resting bid fills when the bar's
low reaches it, which assumes you were at the front of the queue and that
the trade was big enough to take your whole size. Absolute PnL is therefore
an upper bound with no error bar. Comparing two configs over the same bars
is valid — both get the same optimism — and that is what the sweep is for.

The sweep shows fills, per-side uptime, and the 600s markout next to PnL
because PnL alone hides two failure modes: a very wide config whose total is
noise from a handful of fills, and a very tight one whose gross number is
built from adverse fills it gives straight back. Divergences from live
behaviour (no bypass-wall or penny-jump, equal size split, no tick rounding)
are listed in `ta.backtest.DIVERGENCES` and travel in every artifact.

### Exchange view (what to run, and what disagrees)

```bash
python -m ta.view --source nobitex
```

One row per market, joining three sources that normally live apart: the
**tier list** (this project's judgement), the **live config**
(config-manager), and **candle health** (Mongo). Rows where those disagree
sort to the top with an `ALERT`:

- `NO-CONFIG` — tiered major, but config-manager has nothing for it.
- `MAJOR-OFF` — tiered major and not quoting.
- `RISK-ON` — not a major market, yet live.

Risk appears as named flags (`thin-history`, `sparse`, `stale`, `volatile`,
`erratic-volume`), not a score — flags say what is wrong and leave the
decision to you. `--json` for tooling, `--no-stats` to skip the candle read,
`-i/--interval` and `--limit` to control what gets read.

The tier list defaults every unclassified market to `other`, i.e. should be
disabled:

```bash
python -m ta.view tier set nobitex USDTIRT major --by omid --note "core pair"
```

`tier list` shows it, `tier clear <source> <market>` returns a market to the
default. To act on a row, draft a proposal — which still needs approval:

```bash
python -m ta.view propose enable nobitex SHIBIRT --max-base 5
```

### Proposals CLI (the approval gate)

Every config change this project makes starts as a proposal and needs a
human decision. Approving does not write anything; applying is a separate
step behind its own flag (below).

```bash
python -m ta.proposals list --status pending
```

```bash
python -m ta.proposals show <id>
```

```bash
python -m ta.proposals approve <id> --by omid --note "vol is real"
```

`--by` is mandatory on approve/reject: an approval with nobody's name on it
is not an approval. `show` prints the proposed fields, the reason, the
analysis evidence behind it, and the full decision history. The `IMPACT`
column flags changes that commit capital or resume quoting (`is_enable=True`,
`max_base`, ladder depth) — those are the ones not to rubber-stamp.

Storage: `ta_proposals` (current state, deduplicated so a recomputed
recommendation reuses its card) and `ta_decisions` (append-only audit).
Collection names override via `PROPOSALS_COLLECTION_NAME` /
`DECISIONS_COLLECTION_NAME`.

### Applying (the only path that writes)

Needs `CONFIG_MANAGER_URL` in `.env` — the service **root**, e.g.
`http://localhost:8000`; the `/config/market/...` path is added by the
client.

```bash
python -m ta.proposals apply <id>
```

That is a **dry run**: it re-reads the live config, composes the change, and
prints the exact before/after diff. It cannot write, whatever the proposal
says.

```bash
python -m ta.proposals apply <id> --write --by omid
```

Only an *approved* proposal can be applied. Before writing, the live version
is compared against the version the proposal was reasoned against; if a
calibrator moved it in between, the proposal goes `stale` and must be
reviewed again rather than silently reverting that calibrator. Because
config-manager has no patch endpoint, the whole document is re-sent — the
apply verifies that only allowlisted fields differ, so a field lost between
read and write can never be quietly reset to its schema default.

### `GET /health`

`{"status": "ok", "indicators": 49}`

## Session log

- **2026-07-30 — `ta/` package replaces `indicators/`.** Registry-based
  library (49 indicators: trend/momentum/volatility/volume), compute engine
  with warmup honesty + causality guard, frame normalization with gap/health
  reporting, source-aware Tehran→UTC OHLC reader, idempotent store, full test
  suite. Legacy flat package deleted (fixed version backed up in session
  scratchpad; original in git history). `GET /indicator` stays compatible
  except that `source=` is now mandatory and response gains `latest`/`meta`.
- **Earlier session:** fixed the broken legacy modules (undefined vars, empty
  files, `_id` leakage, wrong collection name in `.env`); dropped
  pandas_ta/Celery/Redis; pinned pandas 2.3.3 / numpy 2.3.5 for Python 3.14.
