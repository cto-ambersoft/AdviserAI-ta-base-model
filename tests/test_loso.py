"""
I4 (second half) — leave-one-symbol-out (LOSO) generalization check.

Equivalent to scikit-learn's LeaveOneGroupOut with groups = symbol: train on
every coin except one, evaluate on the held-out coin. This honestly measures
whether the (symbol-agnostic) global model generalizes to a coin it never saw —
exactly the fallback scenario the service relies on.
"""
from __future__ import annotations

import json

from model_tech.config import DataConfig, LabelingConfig, ModelConfig, Paths, TrainConfig, TuneConfig
from model_tech.split import leave_one_symbol_out
from model_tech.train import train_pipeline
from tests.conftest import make_synthetic_ohlcv


def test_pairs_each_symbol_held_out_once() -> None:
    pairs = leave_one_symbol_out(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert [h for _, h in pairs] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    universe = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    for others, held in pairs:
        assert held not in others
        assert set(others) | {held} == universe
        assert len(others) == 2


def test_dedups_and_preserves_first_seen_order() -> None:
    assert [h for _, h in leave_one_symbol_out(["BTC", "ETH", "BTC"])] == ["BTC", "ETH"]


def test_requires_at_least_two_distinct_groups() -> None:
    assert leave_one_symbol_out(["BTC"]) == []
    assert leave_one_symbol_out([]) == []
    assert leave_one_symbol_out(["BTC", "BTC"]) == []


def test_train_pipeline_reports_loso_when_enabled(tmp_path) -> None:
    data = tmp_path / "data"
    arts = tmp_path / "artifacts"
    data.mkdir()
    arts.mkdir()
    paths = Paths(root=tmp_path, data_dir_override=data, artifacts_dir_override=arts)
    symbols = ["BTCUSDT", "ETHUSDT"]
    for sym, seed in zip(symbols, [1, 2]):
        make_synthetic_ohlcv(n=400, seed=seed).to_parquet(data / f"{sym}_4h.parquet", index=False)

    out = train_pipeline(
        symbols=symbols,
        paths=paths,
        data_cfg=DataConfig(lookback_bars=120),
        lab_cfg=LabelingConfig(hold_share_min=0.10, hold_share_max=0.90),
        tr_cfg=TrainConfig(n_folds=2, val_bars=25, test_bars=25, min_train_bars=100, loso_eval=True),
        model_cfg=ModelConfig(iterations=30, thread_count=2),
        tune_cfg=TuneConfig(theta_candidates=3),
    )

    loso = out["loso"]
    assert set(loso.keys()) == set(symbols)
    for held, rec in loso.items():
        assert held not in rec["trained_on"]
        assert 0.0 <= rec["macro_f1"] <= 1.0
        assert rec["n_test"] > 0

    # Persisted to metrics.json too.
    metrics = json.loads((arts / "metrics.json").read_text())
    assert set(metrics["loso"].keys()) == set(symbols)


def test_loso_absent_for_single_symbol(tmp_path) -> None:
    data = tmp_path / "data"
    arts = tmp_path / "artifacts"
    data.mkdir()
    arts.mkdir()
    paths = Paths(root=tmp_path, data_dir_override=data, artifacts_dir_override=arts)
    make_synthetic_ohlcv(n=400, seed=1).to_parquet(data / "BTCUSDT_4h.parquet", index=False)

    out = train_pipeline(
        symbols=["BTCUSDT"],
        paths=paths,
        data_cfg=DataConfig(lookback_bars=120),
        lab_cfg=LabelingConfig(hold_share_min=0.10, hold_share_max=0.90),
        tr_cfg=TrainConfig(n_folds=2, val_bars=25, test_bars=25, min_train_bars=100, loso_eval=True),
        model_cfg=ModelConfig(iterations=30, thread_count=2),
        tune_cfg=TuneConfig(theta_candidates=3),
    )
    assert out["loso"] == {}
