from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from model_tech.api.jobs import TrainJobQueue
from model_tech.api.schemas import HealthResponse, JobStatusResponse, PredictResponse
from model_tech.artifacts import artifacts_ready
from model_tech.config import DataConfig, LabelingConfig, ModelConfig, Paths, TuneConfig
from model_tech.data.symbols import normalize_symbol
from model_tech.data.update import ensure_symbol_ohlcv
from model_tech.infer import ArtifactsStore, predict_signal
from model_tech.train import train_symbol_pipeline

router = APIRouter(prefix="/v1")


def _state(request: Request):
    return request.app.state


def get_paths(request: Request) -> Paths:
    return _state(request).paths


def get_data_cfg(request: Request) -> DataConfig:
    return _state(request).data_cfg


def get_lab_cfg(request: Request) -> LabelingConfig:
    return _state(request).lab_cfg


def get_model_cfg(request: Request) -> ModelConfig:
    return _state(request).model_cfg


def get_tune_cfg(request: Request) -> TuneConfig:
    return _state(request).tune_cfg


def get_artifacts_store(request: Request) -> ArtifactsStore:
    return _state(request).artifacts_store


def get_job_queue(request: Request) -> TrainJobQueue:
    return _state(request).job_queue


@router.get("/health", response_model=HealthResponse)
def health(
    paths: Paths = Depends(get_paths),
) -> HealthResponse:
    ok = artifacts_ready(paths, model_id=None)
    return HealthResponse(global_model_available=bool(ok))


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def job_status(job_id: str, q: TrainJobQueue = Depends(get_job_queue)) -> JobStatusResponse:
    rec = q.get(job_id)
    if rec is None:
        raise HTTPException(status_code=404, detail="job not found")
    return JobStatusResponse(
        job_id=rec.job_id,
        symbol=rec.symbol,
        status=rec.status,
        created_at_utc=rec.created_at_utc,
        started_at_utc=rec.started_at_utc,
        finished_at_utc=rec.finished_at_utc,
        error=rec.error,
        result=rec.result,
    )


@router.get("/train/status", response_model=JobStatusResponse)
def train_status(symbol: str, q: TrainJobQueue = Depends(get_job_queue)) -> JobStatusResponse:
    # Normalize for user convenience: "BNB" -> "BNBUSDT"
    sym = normalize_symbol(symbol or "", default_quote="USDT")
    rec = q.latest_for_symbol(sym)
    if rec is None:
        raise HTTPException(status_code=404, detail="no jobs for symbol")
    return JobStatusResponse(
        job_id=rec.job_id,
        symbol=rec.symbol,
        status=rec.status,
        created_at_utc=rec.created_at_utc,
        started_at_utc=rec.started_at_utc,
        finished_at_utc=rec.finished_at_utc,
        error=rec.error,
        result=rec.result,
    )


@router.get("/predict", response_model=PredictResponse)
def predict(
    symbol: str,
    refresh: bool = True,
    train: bool = False,
    since_days_default: int = 365 * 3,
    paths: Paths = Depends(get_paths),
    data_cfg: DataConfig = Depends(get_data_cfg),
    lab_cfg: LabelingConfig = Depends(get_lab_cfg),
    model_cfg: ModelConfig = Depends(get_model_cfg),
    tune_cfg: TuneConfig = Depends(get_tune_cfg),
    artifacts_store: ArtifactsStore = Depends(get_artifacts_store),
    q: TrainJobQueue = Depends(get_job_queue),
) -> PredictResponse:
    raw = (symbol or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="symbol is required")
    sym = normalize_symbol(raw, default_quote="USDT")

    # Optionally refresh data on request (can take seconds; caller opted into this behavior).
    if refresh:
        # Need enough bars for indicators; we don't require model artifacts to fetch candles.
        try:
            ensure_symbol_ohlcv(
                sym,
                paths=paths,
                data_cfg=data_cfg,
                min_bars=int(data_cfg.lookback_bars),
                since_days_default=int(since_days_default),
            )
        except ValueError as e:
            # User error: unknown/invalid symbol, insufficient data, etc.
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"failed to refresh OHLCV: {e}")

    # Prefer per-symbol model if available; else fall back to global.
    model_id_used: str | None = sym if artifacts_ready(paths, model_id=sym) else None

    job_id = None
    if train and model_id_used is None:
        # Schedule a per-symbol training job (best-effort).
        def _job_fn() -> dict:
            # ensure sufficient history for training
            ensure_symbol_ohlcv(
                sym,
                paths=paths,
                data_cfg=data_cfg,
                min_bars=int(max(500, data_cfg.lookback_bars)),
                since_days_default=int(since_days_default),
            )
            return train_symbol_pipeline(
                sym,
                paths=paths,
                data_cfg=data_cfg,
                lab_cfg=lab_cfg,
                model_cfg=model_cfg,
                tune_cfg=tune_cfg,
                n_folds=3,
                mode="fast",
            )

        job_id = q.submit(sym, _job_fn)

    try:
        pred = predict_signal(
            symbol=sym,
            paths=paths,
            data_cfg=data_cfg,
            ensure_data=False,  # we already refreshed above (if requested)
            model_id=model_id_used,
            artifacts_store=artifacts_store,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictResponse(
        symbol=pred.symbol,
        as_of=pred.as_of,
        signal=str(pred.signal.value),
        confidence=float(pred.confidence),
        probs=pred.probs,
        model_id_used=(model_id_used or "global"),
        job_id=job_id,
    )


