## Ops / Deployment notes

### Uvicorn

Recommended for MVP (single process, in-memory job queue):

```bash
model-tech serve --host 0.0.0.0 --port 8000 --workers 1
```

If you use multiple workers, they are separate processes, so:
- per-worker in-memory caches (artifacts cache, job queue) are **not shared**
- `/v1/jobs/*` status will only reflect the worker that created the job

For multi-worker production, move training to an external worker (e.g. separate process + shared queue) and keep API nodes stateless.

### Auto-train on startup

By default the API schedules **best-effort background training** for a predefined symbol set right after startup.
This avoids paying the training cost later on the first user request.

Environment variables:

- **MODEL_TECH_AUTOTRAIN_ENABLED**: `1|0` (default `1`)
- **MODEL_TECH_AUTOTRAIN_SYMBOLS**: comma-separated list (default `BTC,ETH,BNB,SOL,XRP,DOGE,LINK`)
  - Values are normalized to Binance symbols: `BTC` → `BTCUSDT`
- **MODEL_TECH_AUTOTRAIN_MAX_WORKERS**: parallel training jobs (default `2`)
- **MODEL_TECH_AUTOTRAIN_CATBOOST_THREADS**: CatBoost threads per job (default `2`)
- **MODEL_TECH_AUTOTRAIN_MODE**: `fast|quality` (default `fast`)
- **MODEL_TECH_AUTOTRAIN_N_FOLDS**: walk-forward folds (default `3`)
- **MODEL_TECH_AUTOTRAIN_MIN_BARS**: minimum cached bars to backfill/update before training (default `2200`)
- **MODEL_TECH_AUTOTRAIN_SINCE_DAYS**: initial backfill window in days if local data is missing (default `1095`)
- **MODEL_TECH_AUTOTRAIN_SKIP_EXISTING**: `1|0` (default `1`) — skip if `artifacts/models/<SYMBOL>/` already exists
- **MODEL_TECH_AUTOTRAIN_FORCE**: `1|0` (default `0`) — retrain even if artifacts exist

Notes:

- Auto-train uses the same in-memory `TrainJobQueue`; with `uvicorn --workers > 1` each worker schedules its own jobs.
- For latency-sensitive deployments, reduce workers/threads or disable auto-train and run training out-of-band.

### Artifacts layout

- Global model (fallback): `artifacts/`
  - `model.cbm`
  - `feature_schema.json`
  - `labeling.json`
  - `inference.json`
  - `metrics.json`
- Per-symbol model: `artifacts/models/<SYMBOL>/` (same filenames)

### Data layout

- OHLCV cache: `data/<SYMBOL>_4h.parquet`
- The API can backfill/update this store on request when `refresh=true`.

### Performance knobs

- CatBoost training config supports:
  - `ModelConfig.thread_count` (CPU threads)
  - `ModelConfig.task_type` (`CPU` or `GPU`)

### References

- [Uvicorn](https://www.uvicorn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [CatBoost Python docs](https://catboost.ai/en/docs/)


