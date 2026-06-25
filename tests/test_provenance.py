"""
Phase 1 governance — model provenance.

A) model_version + run_id, B) training_data_hash (SHA256), C) feature_code_hash.
Hashes must be deterministic (same data/code -> same hash) and discriminating
(any change -> different hash), and provenance must round-trip through the .cbm
model metadata so the model file is self-describing.
"""
from __future__ import annotations

import numpy as np

from model_tech.provenance import (
    compute_model_version,
    dataset_fingerprint,
    embed_provenance_metadata,
    feature_code_hash,
    hash_ohlcv,
    read_provenance_metadata,
)
from tests.conftest import make_synthetic_ohlcv


def test_hash_ohlcv_deterministic_and_order_independent() -> None:
    a = make_synthetic_ohlcv(n=200, seed=1)
    b = a.sample(frac=1.0, random_state=3).reset_index(drop=True)  # shuffled rows
    assert hash_ohlcv(a) == hash_ohlcv(b)  # canonicalized by open_time


def test_hash_ohlcv_changes_with_values() -> None:
    a = make_synthetic_ohlcv(n=200, seed=1)
    b = a.copy()
    b.loc[b.index[-1], "close"] = float(b["close"].iloc[-1]) + 1.0
    assert hash_ohlcv(a) != hash_ohlcv(b)


def test_dataset_fingerprint_combines_and_is_symbol_order_independent() -> None:
    btc = make_synthetic_ohlcv(n=150, seed=1)
    eth = make_synthetic_ohlcv(n=150, seed=2)
    fp1 = dataset_fingerprint({"BTCUSDT": btc, "ETHUSDT": eth})
    fp2 = dataset_fingerprint({"ETHUSDT": eth, "BTCUSDT": btc})
    assert fp1["training_data_hash"] == fp2["training_data_hash"]
    assert fp1["per_symbol"]["BTCUSDT"]["n_bars"] == 150
    assert "start" in fp1["per_symbol"]["BTCUSDT"] and "end" in fp1["per_symbol"]["BTCUSDT"]
    # changing one symbol's data changes the combined hash
    eth2 = eth.copy()
    eth2.loc[eth2.index[0], "open"] = 999.0
    fp3 = dataset_fingerprint({"BTCUSDT": btc, "ETHUSDT": eth2})
    assert fp3["training_data_hash"] != fp1["training_data_hash"]


def test_feature_code_hash_deterministic_and_param_sensitive() -> None:
    h1 = feature_code_hash({"rsi_window": 14})
    h2 = feature_code_hash({"rsi_window": 14})
    h3 = feature_code_hash({"rsi_window": 21})
    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 64  # sha256 hex


def test_model_version_deterministic_and_formatted() -> None:
    v1 = compute_model_version("dhash", "chash", {"theta": 0.005})
    v2 = compute_model_version("dhash", "chash", {"theta": 0.005})
    v3 = compute_model_version("dhash", "chash", {"theta": 0.006})
    assert v1 == v2 and v1 != v3
    assert v1.startswith("v") and len(v1) == 13  # "v" + 12 hex


def test_metadata_round_trips_through_cbm(tmp_path) -> None:
    from catboost import CatBoostClassifier

    m = CatBoostClassifier(iterations=3, loss_function="MultiClass", allow_writing_files=False)
    m.fit(np.random.default_rng(0).random((40, 3)), np.random.default_rng(1).integers(0, 3, 40), verbose=False)
    embed_provenance_metadata(m, {"model_version": "vabc123", "training_data_hash": "deadbeef"})
    p = tmp_path / "m.cbm"
    m.save_model(str(p))

    loaded = CatBoostClassifier()
    loaded.load_model(str(p))
    meta = read_provenance_metadata(loaded)
    assert meta.get("model_version") == "vabc123"
    assert meta.get("training_data_hash") == "deadbeef"
