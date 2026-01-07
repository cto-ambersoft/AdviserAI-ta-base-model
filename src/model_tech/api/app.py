from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from model_tech.api.autotrain import schedule_autotrain
from model_tech.api.jobs import TrainJobQueue
from model_tech.api.routes import router
from model_tech.config import AutoTrainConfig, DataConfig, LabelingConfig, ModelConfig, TuneConfig, project_paths
from model_tech.data.store import ensure_dirs
from model_tech.infer import ArtifactsStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init shared state
    paths = project_paths()
    ensure_dirs(paths)

    app.state.paths = paths
    app.state.data_cfg = DataConfig()
    app.state.lab_cfg = LabelingConfig()
    app.state.model_cfg = ModelConfig()
    app.state.tune_cfg = TuneConfig()
    app.state.autotrain_cfg = AutoTrainConfig.from_env()
    app.state.artifacts_store = ArtifactsStore()
    app.state.job_queue = TrainJobQueue(max_workers=int(app.state.autotrain_cfg.max_workers))

    # Warm up global model cache if present (optional).
    try:
        app.state.artifacts_store.get(paths, model_id=None)
    except Exception:
        # Health endpoint will reflect missing artifacts.
        pass

    # Schedule best-effort auto-train in background (does not block startup).
    schedule_autotrain(
        paths=paths,
        data_cfg=app.state.data_cfg,
        lab_cfg=app.state.lab_cfg,
        model_cfg=app.state.model_cfg,
        tune_cfg=app.state.tune_cfg,
        autotrain_cfg=app.state.autotrain_cfg,
        submit_job=app.state.job_queue.submit,
    )

    yield


app = FastAPI(title="model-tech", version="0.1.0", lifespan=lifespan)
app.include_router(router)


