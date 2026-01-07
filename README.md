## model-tech

Minimal project for training and serving a **3-class TA signal model** on **Binance 4h OHLCV**:

- **Train**: download history → compute indicators → label BUY/SELL/HOLD → walk-forward → CatBoost multiclass → save artifacts
- **Infer**: fetch latest N candles → compute indicators → `predict_proba` → rule `min_conf` → output signal

### Setup

Install in editable mode (recommended for local development):

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

### CLI

- Download data:

```bash
model-tech download --symbols BTCUSDT,ETHUSDT --since 2021-01-01
```

- Notes:

  - Uses **public** Binance Spot klines endpoint (no API keys).
  - Saves candles to `data/<SYMBOL>_4h.parquet`.

- Train:

```bash
model-tech train --symbols BTCUSDT,ETHUSDT --n-folds 6
```

- Notes:

  - Builds TA features using `ta` library.
  - Labels are defined by forward return over **H=6** bars (1 day): BUY if \(r > +\theta\), SELL if \(r < -\theta\), else HOLD.
  - `theta` is tuned by walk-forward CV to keep HOLD share in a reasonable band and maximize validation macro-F1.
  - Saves artifacts into `artifacts/`.

- Predict:

```bash
model-tech predict --symbol BTCUSDT
```

### Server (API)

Run FastAPI via Uvicorn:

```bash
model-tech serve --host 0.0.0.0 --port 8000 --workers 1
```

Notes:

- Use `--workers 1` for MVP: the in-memory job queue and artifact cache are per-process.
- Startup auto-train is enabled by default; env defaults are in `env.example` (see also `docs/ops.md`).

### Folders

- `data/`: cached OHLCV (parquet)
- `artifacts/`: saved model + schema + labeling/inference params
- `docs/`: documentation

### Artifacts

Training writes:

- `artifacts/model.cbm`: CatBoost model
- `artifacts/feature_schema.json`: feature list, symbol categorical feature, indicator params, lookback needed
- `artifacts/labeling.json`: horizon + chosen `theta`
- `artifacts/inference.json`: `min_conf` + decision rule
- `artifacts/metrics.json`: CV table for `theta` selection + chosen `min_conf`

See `docs/overview.md` for the detailed pipeline description.

### API: auto-train on startup

When you run `model-tech serve`, the service schedules background per-symbol training on startup by default.
See `docs/ops.md` for the full list of environment variables and tuning knobs.
