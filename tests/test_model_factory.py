"""
Model-factory contracts:

S2 — class imbalance: the model is built with `auto_class_weights="Balanced"`
     so minority BUY/SELL classes are up-weighted (better recall than a model
     that optimizes raw accuracy and drifts toward the majority HOLD class).
S1 — consistent capacity: theta is selected with the SAME iteration budget as
     the shipped model, so there is no separate reduced "tune" path. `_make_model`
     always honors `ModelConfig.iterations`.
"""
from __future__ import annotations

from model_tech.config import ModelConfig, TuneConfig
from model_tech.train import _make_model


def test_make_model_applies_balanced_class_weights_by_default() -> None:
    params = _make_model(ModelConfig(iterations=123)).get_params()
    assert params.get("auto_class_weights") == "Balanced"


def test_class_weights_can_be_disabled() -> None:
    params = _make_model(ModelConfig(auto_class_weights=None)).get_params()
    # When disabled the param must not be forced onto the model.
    assert params.get("auto_class_weights") is None


def test_make_model_honors_config_iterations() -> None:
    # S1: no downshift — the factory uses exactly the configured iteration budget.
    assert _make_model(ModelConfig(iterations=123)).get_params().get("iterations") == 123


def test_no_separate_reduced_tune_iteration_budget() -> None:
    # S1: the old `tune_iterations` downshift knob is gone; theta CV and the final
    # model share one capacity.
    assert not hasattr(TuneConfig(), "tune_iterations")
