from __future__ import annotations

from dataclasses import replace
from typing import Any

from model_tech.artifacts import artifacts_ready
from model_tech.config import AutoTrainConfig, DataConfig, LabelingConfig, ModelConfig, Paths, TuneConfig
from model_tech.data.symbols import normalize_symbol
from model_tech.data.update import ensure_symbol_ohlcv
from model_tech.logging import get_logger
from model_tech.train import train_symbol_pipeline

log = get_logger(__name__)


def schedule_autotrain(
    *,
    paths: Paths,
    data_cfg: DataConfig,
    lab_cfg: LabelingConfig,
    model_cfg: ModelConfig,
    tune_cfg: TuneConfig,
    autotrain_cfg: AutoTrainConfig,
    submit_job,  # TrainJobQueue.submit-compatible
) -> dict[str, str]:
    """
    Schedule per-symbol training jobs right after API startup.

    This is best-effort and never raises to the caller (startup must not fail).
    Returns mapping symbol->job_id for jobs that were actually submitted.
    """
    if not autotrain_cfg.enabled:
        log.info("Auto-train disabled (MODEL_TECH_AUTOTRAIN_ENABLED=0)")
        return {}

    submitted: dict[str, str] = {}

    # Normalize to Binance symbols (e.g. "BTC" -> "BTCUSDT").
    symbols = [normalize_symbol(s, default_quote="USDT") for s in autotrain_cfg.symbols]
    symbols = [s.strip().upper() for s in symbols if s.strip()]

    for sym in symbols:
        # Skip if already trained (unless force).
        if autotrain_cfg.skip_if_exists and (not autotrain_cfg.force_retrain) and artifacts_ready(paths, model_id=sym):
            log.info("Auto-train skip %s: artifacts already exist", sym)
            continue

        def _job_fn(sym=sym) -> dict[str, Any]:
            # Data backfill/update first (network, can take seconds).
            try:
                ensure_symbol_ohlcv(
                    sym,
                    paths=paths,
                    data_cfg=data_cfg,
                    min_bars=int(max(int(data_cfg.lookback_bars), int(autotrain_cfg.min_bars))),
                    since_days_default=int(autotrain_cfg.since_days_default),
                )
            except ValueError as e:
                # Common case: symbol does not exist on Binance (e.g. config typo or delisted token).
                # Treat as "skipped" so it doesn't look like an infra failure.
                msg = str(e)
                if "Invalid Binance symbol" in msg:
                    log.warning("Auto-train skip %s: %s", sym, msg)
                    return {"status": "skipped", "symbol": sym, "reason": msg}
                raise

            # Keep API responsive: cap CatBoost threads per job.
            job_model_cfg = replace(model_cfg, thread_count=int(autotrain_cfg.catboost_thread_count))
            return train_symbol_pipeline(
                sym,
                paths=paths,
                data_cfg=data_cfg,
                lab_cfg=lab_cfg,
                model_cfg=job_model_cfg,
                tune_cfg=tune_cfg,
                n_folds=int(autotrain_cfg.n_folds),
                mode=str(autotrain_cfg.mode),
            )

        try:
            job_id = submit_job(sym, _job_fn)
            submitted[sym] = job_id
            log.info("Auto-train scheduled: %s job_id=%s", sym, job_id)
        except Exception as e:
            # Never fail startup; just report.
            log.error("Auto-train failed to schedule %s: %s", sym, e)

    if not submitted:
        log.info("Auto-train: nothing to schedule (all skipped or disabled)")
    return submitted


