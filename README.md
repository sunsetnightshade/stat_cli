## START HERE (FOR COMPLETE BEGINNERS)

This project now runs on live WebSocket data (Twelve Data or Polygon) -> Redis Streams -> hourly Parquet.

Follow these exact steps.

### 1) Open PowerShell and go to the project folder
```powershell
cd "C:\Users\Naman Sinha\Desktop\quant_matrix_cli"
```

### 2) Create and activate virtual environment (first time only)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate
```

You should see `(.venv)` at the start of your terminal line.

### 3) Install dependencies
```powershell
py -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Make sure Redis is running
If Redis is not running, start it first (local install or Docker).

### 5) Set required environment variables
Choose one provider.

Twelve Data:
```powershell
set QM_ENABLE_LIVE_INGEST=1
set QM_LIVE_PROVIDER=twelvedata
set TWELVEDATA_API_KEY=your_key_here
set QM_REDIS_HOST=127.0.0.1
set QM_REDIS_PORT=6379
```

Polygon:
```powershell
set QM_ENABLE_LIVE_INGEST=1
set QM_LIVE_PROVIDER=polygon
set POLYGON_API_KEY=your_key_here
set QM_REDIS_HOST=127.0.0.1
set QM_REDIS_PORT=6379
```

### 6) Start live ingestion (Terminal 1)
```powershell
cd "C:\Users\Naman Sinha\Desktop\quant_matrix_cli"
.\.venv\Scripts\Activate
py main.py --ingest-live
```

### 7) Build a live analytics snapshot (Terminal 2)
```powershell
cd "C:\Users\Naman Sinha\Desktop\quant_matrix_cli"
.\.venv\Scripts\Activate
py main.py --snapshot-live --lookback-minutes 500 --z-window 60 --pca-components 3
```

### 8) Persist recent live data to Parquet (any terminal)
```powershell
py main.py --persist-live-hourly --persist-hours 1
```

If any command fails, copy the full error text and share it.

---

# Quant Matrix CLI

Quant Matrix CLI is a websocket-first market data pipeline for 30 symbols (Nifty IT + Nasdaq):
- WebSockets (Twelve Data / Polygon) ingest ticks in real time.
- Ticks are snapped to 1-minute OHLCV bars.
- Bars are stored in Redis Streams as hot data windows.
- Analytics run on rolling windows (log returns, rolling z-score, residual z-score).
- Hourly janitor appends data to Parquet for fast historical access.

## Core Data Model

### TYPE 1: Live Memory (Hot Layer)
- Source: WebSockets (Twelve Data or Polygon).
- Pulse: 1-minute OHLCV bars per symbol.
- Precision: fixed-point integer prices (`price * 10^9`).
- Storage: Redis Streams in RAM.
- Hot windows:
  - 60-minute stream for immediate access.
  - 500-minute stream for execution/rolling analytics.

### TYPE 2: Historical Library (Backup / Verification)
- Source: university Bloomberg workflow (external).
- In-repo integration point: local TYPE2 parquet root.
- Use: heartbeat failover verification + backfill when stream gaps occur.

## Required Runtime Behavior (Implemented)
- Heartbeat ping/pong watchdog.
  - Default ping interval: 10 seconds.
  - Pong timeout: 3 seconds.
- On heartbeat failure:
  - Freeze execution marker is written.
  - Gap event is logged.
  - TYPE2 parquet backfill is attempted.
  - Reconnect cycle starts.
- Redis write latency telemetry includes p95 and threshold alerts.
- Rolling analytics use vectorized NumPy operations and `sliding_window_view`.

## Main Commands

### Start live ingestion
```powershell
py main.py --ingest-live
```

### Build live snapshot from Redis
```powershell
py main.py --snapshot-live --lookback-minutes 500 --z-window 60 --pca-components 3
```

### Persist Redis hot data to Parquet
```powershell
py main.py --persist-live-hourly --persist-hours 1
```

### Run provider bake-off (Twelve Data vs Polygon)
```powershell
py main.py --bakeoff-live --bakeoff-seconds 300
```

## Important Environment Variables

- `QM_ENABLE_LIVE_INGEST=1` enable live ingest mode.
- `QM_LIVE_PROVIDER=twelvedata|polygon` choose provider.
- `TWELVEDATA_API_KEY` or `POLYGON_API_KEY` provider auth.
- `QM_REDIS_HOST`, `QM_REDIS_PORT`, `QM_REDIS_DB` Redis connection.
- `QM_REDIS_STREAM_60`, `QM_REDIS_STREAM_500` stream keys.
- `QM_REDIS_MAXLEN_60`, `QM_REDIS_MAXLEN_500` pruning lengths (XTRIM via stream maxlen).
- `QM_LIVE_HEARTBEAT_SECONDS` default 10.
- `QM_LIVE_PONG_TIMEOUT_SECONDS` default 3.
- `QM_LIVE_LATENCY_ALERT_MS` default 100.
- `QM_ENABLE_HOURLY_JANITOR` default 1.
- `QM_JANITOR_INTERVAL_SECONDS` default 3600.
- `QM_JANITOR_LOOKBACK_HOURS` default 2.
- `QM_TYPE2_PARQUET_ROOT` default `outputs/live/parquet`.
- `QM_BAR_SINK=redis|zeromq` optional sink override.
- `QM_ZMQ_ENDPOINT` ZeroMQ endpoint when using ZeroMQ sink.

## Output Artifacts

- Live snapshot CSVs: `outputs/latest/live/`
  - `live_close_1m_30xT.csv`
  - `live_log_returns_1m_30xT.csv`
  - `live_rolling_zscores_1m_30xT.csv`
  - `live_latest_zscore.csv`
  - `live_latest_residual_zscore.csv`
  - `live_pca_explained.csv`
- Failover markers/logs:
  - `outputs/live/freeze.flag`
  - `outputs/live/gaps.jsonl`
- Hourly parquet partitions:
  - `outputs/live/parquet/date=YYYY-MM-DD/hour=HH/bars.parquet`
- Provider bake-off reports:
  - `outputs/live/provider_bakeoff_*.json`

## Repository Layout

- `main.py` command-line entrypoint.
- `config.py` live ingest and runtime configuration.
- `live_ingest/` websocket providers, heartbeat, aggregation, sinks, analytics, snapshot, bake-off.
- `live_persistence/` parquet janitor.
- `pipeline.py` orchestration helpers (snapshot + persistence entry points).
- `tests/` unit tests for precision, aggregation, analytics, and TYPE2 fallback.

## Legacy Note

`yfinance` is no longer the active source path.
The old fetcher path is disabled by default and only runs if `QM_ENABLE_LEGACY_YFINANCE=1` is explicitly set.

## Run Tests
```powershell
py -m unittest discover -s tests -p "test_*.py"
```

## GitHub workflow hints
1. `git init` (if not already a repo) and commit source + README.
2. `git remote add origin https://github.com/sunsetnightshade/stat_cli.git`.
3. Push your branch and open a pull request.
4. Share the PR URL when ready.
