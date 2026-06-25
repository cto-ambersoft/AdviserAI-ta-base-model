"""
S5 — ArtifactsStore must be thread-safe.

FastAPI runs sync endpoints in a threadpool, so concurrent /v1/predict calls hit
the shared ArtifactsStore at once. A check-then-set race used to let several
threads each load the model and clobber the cache. With a lock the model is
loaded once and every caller gets the same cached instance.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from catboost import CatBoostClassifier

from model_tech.artifacts import artifact_paths, write_json
from model_tech.config import Paths
from model_tech.infer import ArtifactsStore


def _write_global_artifacts(arts_dir) -> None:
    rng = np.random.default_rng(0)
    X = rng.random((60, 3))
    y = rng.integers(0, 3, 60)
    m = CatBoostClassifier(iterations=5, loss_function="MultiClass", allow_writing_files=False)
    m.fit(X, y, verbose=False)
    ap = artifact_paths(Paths(root=arts_dir, artifacts_dir_override=arts_dir), model_id=None)
    ap.model_path.parent.mkdir(parents=True, exist_ok=True)
    m.save_model(str(ap.model_path))
    write_json(ap.feature_schema_path, {
        "feature_cols": ["f0", "f1", "f2"], "cat_features": [],
        "indicator_params": {}, "lookback_needed": 50,
    })
    write_json(ap.inference_path, {"min_conf": 0.4})


def test_concurrent_get_loads_once_and_shares_instance(tmp_path) -> None:
    arts = tmp_path / "artifacts"
    arts.mkdir()
    _write_global_artifacts(arts)
    paths = Paths(root=tmp_path, artifacts_dir_override=arts)

    store = ArtifactsStore()
    with ThreadPoolExecutor(max_workers=16) as ex:
        results = list(ex.map(lambda _: store.get(paths, model_id=None), range(64)))

    first = results[0]
    # One cached LoadedArtifacts, shared by every concurrent caller.
    assert all(r is first for r in results)
    assert all(r.model is first.model for r in results)
