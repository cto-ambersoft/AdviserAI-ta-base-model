"""
I4 — fair predictions across coins.

The global model is the fallback for symbols it was never trained on. When
`symbol` is a categorical feature, predicting an unseen coin feeds an unknown
category, so cross-asset behavior is unvalidated/degraded. By default the model
is now shape-based (no `symbol` feature), so a never-seen coin is scored with
exactly the same features as the training coins.

These are small end-to-end training runs (tiny iterations / few bars) — slower
than the pure-unit tests but they exercise the real train -> persist -> infer path.
"""
from __future__ import annotations

import json

import pytest

from model_tech.config import DataConfig, LabelingConfig, ModelConfig, Paths, TrainConfig, TuneConfig
from model_tech.infer import ArtifactsStore, predict_signal
from model_tech.train import train_pipeline
from tests.conftest import make_synthetic_ohlcv

_DATA = DataConfig(lookback_bars=120)
_TUNE = TuneConfig(theta_candidates=3)
_MODEL = ModelConfig(iterations=30, thread_count=2)
_LAB = LabelingConfig(hold_share_min=0.10, hold_share_max=0.90)


def _train(tmp_path, symbols, *, use_symbol_feature: bool, seeds):
    data = tmp_path / "data"
    arts = tmp_path / "artifacts"
    data.mkdir(exist_ok=True)
    arts.mkdir(exist_ok=True)
    paths = Paths(root=tmp_path, data_dir_override=data, artifacts_dir_override=arts)
    for sym, seed in zip(symbols, seeds):
        make_synthetic_ohlcv(n=400, seed=seed).to_parquet(data / f"{sym}_4h.parquet", index=False)

    train_pipeline(
        symbols=symbols,
        paths=paths,
        data_cfg=_DATA,
        lab_cfg=_LAB,
        tr_cfg=TrainConfig(
            n_folds=2, val_bars=25, test_bars=25, min_train_bars=100,
            use_symbol_feature=use_symbol_feature,
        ),
        model_cfg=_MODEL,
        tune_cfg=_TUNE,
    )
    return paths, arts


def test_global_model_is_symbol_agnostic_by_default(tmp_path) -> None:
    paths, arts = _train(tmp_path, ["BTCUSDT", "ETHUSDT"], use_symbol_feature=False, seeds=[1, 2])

    schema = json.loads((arts / "feature_schema.json").read_text())
    assert schema["cat_features"] == []
    assert "symbol" not in schema["feature_cols"]

    # A coin the model never saw, served by the global model.
    make_synthetic_ohlcv(n=400, seed=99).to_parquet(paths.data_dir / "DOGEUSDT_4h.parquet", index=False)
    pred = predict_signal(
        symbol="DOGEUSDT",
        paths=paths,
        data_cfg=_DATA,
        model_id=None,
        artifacts_store=ArtifactsStore(),
    )
    assert pred.signal.value in {"BUY", "SELL", "HOLD"}
    assert pred.probs.keys() == {"SELL", "HOLD", "BUY"}
    assert abs(sum(pred.probs.values()) - 1.0) < 1e-6


def test_symbol_feature_is_opt_in(tmp_path) -> None:
    # Teeth for the test above: with the flag on, the old behavior returns.
    _, arts = _train(tmp_path, ["BTCUSDT", "ETHUSDT"], use_symbol_feature=True, seeds=[1, 2])
    schema = json.loads((arts / "feature_schema.json").read_text())
    assert schema["cat_features"] == ["symbol"]
