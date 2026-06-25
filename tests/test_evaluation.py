"""
S10 — fair, honest assessment of predictions.

- economic_metrics: what acting on signals actually earns (BUY=+r, SELL=-r,
  HOLD=0 over the label horizon), plus directional accuracy — F1 alone says
  nothing about profitability.
- baseline_metrics: buy-and-hold / always-HOLD references so a macro-F1 of ~0.4
  can be judged against doing nothing.
- multiclass_brier_score: probability quality (the min_conf threshold trusts the
  probabilities, so their calibration matters).
"""
from __future__ import annotations

import json

import numpy as np

from model_tech.config import DataConfig, LabelingConfig, ModelConfig, Paths, TrainConfig, TuneConfig
from model_tech.evaluation import baseline_metrics, economic_metrics, multiclass_brier_score
from model_tech.infer import ArtifactsStore, predict_signal
from model_tech.train import train_pipeline
from tests.conftest import make_synthetic_ohlcv


def test_economic_metrics_directional_pnl() -> None:
    # classes: 0=SELL, 1=HOLD, 2=BUY
    y_pred = [2, 0, 1, 2]
    fwd = [0.10, -0.05, 0.20, -0.10]
    m = economic_metrics(y_pred, fwd)
    # pnl = [+0.10, +0.05, 0, -0.10]
    assert abs(m["strategy_total_return"] - 0.05) < 1e-9
    assert abs(m["strategy_mean_return_per_bar"] - 0.0125) < 1e-9
    assert m["n_trades"] == 3
    assert abs(m["trade_rate"] - 0.75) < 1e-9
    assert abs(m["mean_return_per_trade"] - (0.05 / 3)) < 1e-9
    assert abs(m["directional_accuracy"] - (2 / 3)) < 1e-9


def test_economic_metrics_all_hold_is_flat() -> None:
    m = economic_metrics([1, 1, 1], [0.1, -0.2, 0.3])
    assert m["n_trades"] == 0
    assert m["strategy_total_return"] == 0.0
    assert m["directional_accuracy"] == 0.0


def test_baseline_metrics() -> None:
    y_true = [2, 0, 1, 2]
    fwd = [0.10, -0.05, 0.20, -0.10]
    b = baseline_metrics(y_true, fwd)
    assert abs(b["buy_and_hold_mean_return_per_bar"] - 0.0375) < 1e-9
    assert b["always_hold_return"] == 0.0
    # all-HOLD vs y_true: only HOLD class scores (prec 1/4, rec 1) -> f1 0.4; macro/3
    assert abs(b["always_hold_macro_f1"] - (0.4 / 3)) < 1e-6


def test_multiclass_brier_perfect_is_zero() -> None:
    prob = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert multiclass_brier_score(prob, [0, 1]) < 1e-12


def test_multiclass_brier_uniform() -> None:
    prob = [[1 / 3, 1 / 3, 1 / 3]]
    # (1/3-1)^2 + (1/3)^2 + (1/3)^2 = 6/9
    assert abs(multiclass_brier_score(prob, [0]) - (6 / 9)) < 1e-9


def test_brier_in_unit_interval_for_random() -> None:
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], size=50)
    y = rng.integers(0, 3, size=50)
    s = multiclass_brier_score(p, y)
    assert 0.0 <= s <= 2.0


def test_train_pipeline_reports_economics_baselines_calibration_and_predict_exposes_max_prob(tmp_path) -> None:
    data = tmp_path / "data"
    arts = tmp_path / "artifacts"
    data.mkdir()
    arts.mkdir()
    paths = Paths(root=tmp_path, data_dir_override=data, artifacts_dir_override=arts)
    make_synthetic_ohlcv(n=400, seed=3).to_parquet(data / "BTCUSDT_4h.parquet", index=False)

    out = train_pipeline(
        symbols=["BTCUSDT"],
        paths=paths,
        data_cfg=DataConfig(lookback_bars=120),
        lab_cfg=LabelingConfig(hold_share_min=0.10, hold_share_max=0.90),
        tr_cfg=TrainConfig(n_folds=2, val_bars=25, test_bars=25, min_train_bars=100),
        model_cfg=ModelConfig(iterations=30, thread_count=2),
        tune_cfg=TuneConfig(theta_candidates=3),
    )

    # S10: all three metric groups present in the return value...
    for key in ("economics", "baselines", "calibration"):
        assert key in out
    assert "directional_accuracy" in out["economics"]
    assert "buy_and_hold_mean_return_per_bar" in out["baselines"]
    assert 0.0 <= out["calibration"]["multiclass_brier"] <= 2.0
    # ...and persisted to metrics.json.
    metrics = json.loads((arts / "metrics.json").read_text())
    assert {"economics", "baselines", "calibration"} <= set(metrics.keys())

    # S4: prediction exposes max_prob / forced_hold consistently.
    pred = predict_signal(
        symbol="BTCUSDT", paths=paths, data_cfg=DataConfig(lookback_bars=120),
        model_id=None, artifacts_store=ArtifactsStore(),
    )
    assert abs(pred.max_prob - max(pred.probs.values())) < 1e-9
    assert pred.max_prob >= pred.confidence - 1e-12
    if pred.forced_hold:
        assert pred.signal.value == "HOLD"
