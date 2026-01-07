## HTTP API (FastAPI)

Service entrypoint: `model_tech.api.app:app`

### Run

From repo root:

```bash
model-tech serve --host 0.0.0.0 --port 8000 --workers 1
```

Interactive OpenAPI:

- Swagger UI: `GET /docs`
- OpenAPI JSON: `GET /openapi.json`

### Endpoints

#### `GET /v1/health`

Response:

- `global_model_available`: `true|false` (whether `artifacts/` contains global model artifacts)

#### `GET /v1/predict`

Query params:

- `symbol` (required): e.g. `BTCUSDT`
- `refresh` (default `true`): if `true`, the service updates/backs-fills OHLCV before predicting (may take seconds)
- `train` (default `false`): if `true` and there is no per-symbol model yet, schedule a background per-symbol training job
- `since_days_default` (default `1095`): if the symbol has no local data, backfill this many days

Behavior:

- Uses per-symbol artifacts if present in `artifacts/models/<SYMBOL>/...`, otherwise falls back to the global model in `artifacts/...`.
- `train=true` schedules training only when per-symbol model is missing (MVP behavior).

Response fields:

- `symbol`, `as_of`, `signal`, `confidence`, `probs`
- `model_id_used`: `"global"` or `<SYMBOL>`
- `job_id`: present if a training job was scheduled

Examples:

```bash
curl "http://localhost:8000/v1/predict?symbol=BTCUSDT&refresh=true"
curl "http://localhost:8000/v1/predict?symbol=DOGEUSDT&refresh=true&train=true"
```

#### `GET /v1/jobs/{job_id}`

Returns the current status of a training job.

#### `GET /v1/train/status?symbol=...`

Returns the latest training job record for a symbol.

### Notes

- The current training queue is **in-memory**. Run with **`--workers 1`** if you need consistent job status.
- For more on FastAPI docs and lifespan, see the official docs: [FastAPI](https://fastapi.tiangolo.com/).
