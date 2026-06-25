"""
Real probability calibration (OvR per-class + renormalization, the same scheme
sklearn's CalibratedClassifierCV uses for multiclass).

`apply_calibration` is pure (no sklearn) and used at inference; `fit_calibration`
fits the per-class maps on a held-out (prob, label) set at training time and is
JSON-serializable so it can live next to the model as an artifact.
"""
from __future__ import annotations

import numpy as np

from model_tech.calibration import apply_calibration, fit_calibration
from model_tech.evaluation import multiclass_brier_score


def test_apply_renormalizes_and_preserves_shape() -> None:
    calib = {"method": "sigmoid", "n_classes": 3,
             "maps": [{"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}, {"a": 1.0, "b": 0.0}]}

    one = apply_calibration([0.2, 0.3, 0.5], calib)
    assert one.shape == (3,)
    assert abs(float(one.sum()) - 1.0) < 1e-9

    batch = apply_calibration([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]], calib)
    assert batch.shape == (2, 3)
    np.testing.assert_allclose(batch.sum(axis=1), 1.0)


def test_identity_map_passes_through() -> None:
    calib = {"method": "sigmoid", "n_classes": 3, "maps": [{"identity": True}] * 3}
    p = [0.2, 0.3, 0.5]
    np.testing.assert_allclose(apply_calibration(p, calib), p)


def _miscalibrated_dataset(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    true_p = rng.dirichlet([2.0, 2.0, 2.0], size=n)
    y = np.array([rng.choice(3, p=true_p[i]) for i in range(n)])
    # Overconfident observed probabilities (sharpened) -> poorly calibrated.
    obs = true_p ** 3
    obs = obs / obs.sum(axis=1, keepdims=True)
    return obs, y


def test_fit_sigmoid_does_not_worsen_brier() -> None:
    obs, y = _miscalibrated_dataset()
    calib = fit_calibration(obs, y, method="sigmoid")
    cal = apply_calibration(obs, calib)
    assert multiclass_brier_score(cal, y) <= multiclass_brier_score(obs, y) + 1e-9
    np.testing.assert_allclose(cal.sum(axis=1), 1.0)


def test_fit_isotonic_produces_valid_probs_and_serializable() -> None:
    import json

    obs, y = _miscalibrated_dataset(seed=1)
    calib = fit_calibration(obs, y, method="isotonic")
    json.dumps(calib)  # must be JSON-serializable (artifact)
    cal = apply_calibration(obs, calib)
    assert cal.shape == obs.shape
    assert np.all(cal >= -1e-9) and np.all(cal <= 1 + 1e-9)
    np.testing.assert_allclose(cal.sum(axis=1), 1.0)
    assert multiclass_brier_score(cal, y) <= multiclass_brier_score(obs, y) + 1e-9


# --- end-to-end: calibrator is persisted as an artifact and applied at inference -

def _train(tmp_path, *, calibration):
    from model_tech.config import DataConfig, LabelingConfig, ModelConfig, Paths, TrainConfig, TuneConfig
    from model_tech.train import train_pipeline
    from tests.conftest import make_synthetic_ohlcv

    data = tmp_path / "data"
    arts = tmp_path / "artifacts"
    data.mkdir()
    arts.mkdir()
    paths = Paths(root=tmp_path, data_dir_override=data, artifacts_dir_override=arts)
    make_synthetic_ohlcv(n=400, seed=5).to_parquet(data / "BTCUSDT_4h.parquet", index=False)

    out = train_pipeline(
        symbols=["BTCUSDT"], paths=paths,
        data_cfg=DataConfig(lookback_bars=120),
        lab_cfg=LabelingConfig(hold_share_min=0.10, hold_share_max=0.90),
        tr_cfg=TrainConfig(n_folds=2, val_bars=25, test_bars=25, min_train_bars=100, calibration=calibration),
        model_cfg=ModelConfig(iterations=30, thread_count=2),
        tune_cfg=TuneConfig(theta_candidates=3),
    )
    return paths, arts, out


def test_train_with_calibration_persists_and_infers(tmp_path) -> None:
    from model_tech.artifacts import artifact_paths
    from model_tech.config import DataConfig
    from model_tech.infer import ArtifactsStore, predict_signal

    import json

    paths, arts, out = _train(tmp_path, calibration="sigmoid")
    assert out["calibration_method"] == "sigmoid"
    cpath = artifact_paths(paths, model_id=None).calibration_path
    assert cpath.exists()
    saved = json.loads(cpath.read_text())
    # The artifact is the calibrator (method + per-class maps), not the Brier metric.
    assert saved["method"] == "sigmoid"
    assert len(saved["maps"]) == 3

    pred = predict_signal(
        symbol="BTCUSDT", paths=paths, data_cfg=DataConfig(lookback_bars=120),
        model_id=None, artifacts_store=ArtifactsStore(),
    )
    assert pred.signal.value in {"BUY", "SELL", "HOLD"}
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-9


def test_no_calibration_by_default(tmp_path) -> None:
    from model_tech.artifacts import artifact_paths

    paths, arts, out = _train(tmp_path, calibration=None)
    assert out["calibration_method"] is None
    assert not artifact_paths(paths, model_id=None).calibration_path.exists()
