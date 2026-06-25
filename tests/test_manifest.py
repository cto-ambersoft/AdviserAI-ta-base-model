"""
Phase 1 integration — training writes a provenance manifest, embeds provenance
into the .cbm, exposes model_version, and produces deterministic data/code hashes.
"""
from __future__ import annotations

import json

from catboost import CatBoostClassifier

from model_tech.artifacts import artifact_paths
from model_tech.config import DataConfig, LabelingConfig, ModelConfig, Paths, TrainConfig, TuneConfig
from model_tech.provenance import read_provenance_metadata
from model_tech.train import train_pipeline
from tests.conftest import make_synthetic_ohlcv


def _train(tmp_path, *, seed: int = 5, threads: int = 2):
    data = tmp_path / "data"
    arts = tmp_path / "artifacts"
    data.mkdir(parents=True, exist_ok=True)
    arts.mkdir(parents=True, exist_ok=True)
    paths = Paths(root=tmp_path, data_dir_override=data, artifacts_dir_override=arts)
    make_synthetic_ohlcv(n=400, seed=seed).to_parquet(data / "BTCUSDT_4h.parquet", index=False)
    out = train_pipeline(
        symbols=["BTCUSDT"], paths=paths,
        data_cfg=DataConfig(lookback_bars=120),
        lab_cfg=LabelingConfig(hold_share_min=0.10, hold_share_max=0.90),
        tr_cfg=TrainConfig(n_folds=2, val_bars=25, test_bars=25, min_train_bars=100),
        model_cfg=ModelConfig(iterations=30, thread_count=threads),
        tune_cfg=TuneConfig(theta_candidates=3),
    )
    return paths, out


def test_manifest_written_and_embedded(tmp_path) -> None:
    paths, out = _train(tmp_path)
    ap = artifact_paths(paths, model_id=None)

    man = json.loads(ap.manifest_path.read_text())
    assert man["model_version"].startswith("v")
    assert len(man["training_data_hash"]) == 64
    assert len(man["feature_code_hash"]) == 64
    assert man["data_window"]["BTCUSDT"]["n_bars"] > 0
    assert "run_id" in man and man["symbols"] == ["BTCUSDT"]

    # exposed in the train result and inference.json
    assert out["model_version"] == man["model_version"]
    inf = json.loads(ap.inference_path.read_text())
    assert inf["model_version"] == man["model_version"]

    # embedded in the .cbm so the model file is self-describing
    m = CatBoostClassifier()
    m.load_model(str(ap.model_path))
    meta = read_provenance_metadata(m)
    assert meta["model_version"] == man["model_version"]
    assert meta["training_data_hash"] == man["training_data_hash"]
    assert meta["feature_code_hash"] == man["feature_code_hash"]


def test_data_and_code_hashes_are_deterministic(tmp_path) -> None:
    # Same data + same code (single-threaded -> reproducible theta) => same hashes
    # and same content-addressed model_version. (Acceptance criterion.)
    p1, _ = _train(tmp_path / "a", seed=5, threads=1)
    p2, _ = _train(tmp_path / "b", seed=5, threads=1)
    m1 = json.loads(artifact_paths(p1, model_id=None).manifest_path.read_text())
    m2 = json.loads(artifact_paths(p2, model_id=None).manifest_path.read_text())
    assert m1["training_data_hash"] == m2["training_data_hash"]
    assert m1["feature_code_hash"] == m2["feature_code_hash"]
    assert m1["model_version"] == m2["model_version"]


def test_predict_exposes_model_version(tmp_path) -> None:
    from model_tech.infer import ArtifactsStore, predict_signal

    paths, out = _train(tmp_path)
    pred = predict_signal(
        symbol="BTCUSDT", paths=paths, data_cfg=DataConfig(lookback_bars=120),
        model_id=None, artifacts_store=ArtifactsStore(),
    )
    assert pred.model_version == out["model_version"]
