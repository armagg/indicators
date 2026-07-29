# Running Indicators

Research-only TA toolkit (`ta/` package) with a thin Flask adapter. Reads
OHLC candles from MongoDB (`market_making.ohlc`, written by
`market-data-collector`), computes indicators, persists results idempotently
to `market_making.indicators_data`. **Not wired into any trading decision
path** and calls no other project — standalone, shared Mongo only.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt pytest
cp .env <adjust as needed>   # MONGO_URI, DB_NAME, OHLC_COLLECTION_NAME, INDICATORS_COLLECTION_NAME
```

Requires MongoDB reachable at `MONGO_URI` with candles in
`OHLC_COLLECTION_NAME` (documents `{market_name, source, interval, time, open,
high, low, close, volume, ...}`; `time` is a Tehran-naive ISO string as MDC
writes it). Locally: the `trading-local-mongo` container on port 27017.

## Tests (no Mongo needed)

```bash
python -m pytest tests/ -q
```

177 tests: a contract test parametrised over all 49 registered indicators
(purity, column naming, no inf, warmup sanity), exact-math references for
SMA/EMA/RSI/MACD/StochRSI/ATR/Bollinger, engine/registry/frame behaviour,
store idempotency, analyses on constructed regimes (ramps, chop, vol
spikes, squeeze-and-pop), overlay deltas with grid-swept invariants, and
the HTTP adapter with the data layer stubbed.

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
