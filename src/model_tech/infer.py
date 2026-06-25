from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from model_tech.artifacts import artifact_paths, read_json
from model_tech.calibration import apply_calibration
from model_tech.config import DataConfig, Paths
from model_tech.data.update import ensure_symbol_ohlcv
from model_tech.data.store import read_ohlcv
from model_tech.features.indicators import IndicatorParams, build_ta_features
from model_tech.labeling import y_to_signal
from model_tech.types import Prediction


@dataclass(frozen=True)
class LoadedArtifacts:
    model: CatBoostClassifier
    feature_cols: list[str]
    cat_features: list[str]
    min_conf: float
    indicator_params: IndicatorParams
    lookback_needed: int
    calibration: dict | None = None
    model_version: str = ""


class ArtifactsStore:
    """
    Small in-process cache of loaded artifacts to avoid disk IO on every request.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, Optional[str]], tuple[LoadedArtifacts, float]] = {}
        # FastAPI serves sync endpoints from a threadpool; guard the cache so the
        # model is loaded once under concurrent /v1/predict instead of racing.
        self._lock = Lock()

    def get(self, paths: Paths, *, model_id: str | None = None, reload: bool = False) -> LoadedArtifacts:
        key = (str(paths.artifacts_dir), model_id)
        ap = artifact_paths(paths, model_id=model_id)

        # Use latest mtime across relevant files as cache key.
        mt = _latest_mtime([ap.model_path, ap.feature_schema_path, ap.inference_path, ap.calibration_path])

        with self._lock:
            cached = self._cache.get(key)
            if cached is not None and (not reload) and cached[1] >= mt:
                return cached[0]

            schema = read_json(ap.feature_schema_path)
            infer = read_json(ap.inference_path)
            calibration = read_json(ap.calibration_path) if ap.calibration_path.exists() else None

            model = CatBoostClassifier()
            model.load_model(str(ap.model_path))
            arts = LoadedArtifacts(
                model=model,
                feature_cols=list(schema["feature_cols"]),
                cat_features=list(schema.get("cat_features", ["symbol"])),
                min_conf=float(infer["min_conf"]),
                indicator_params=IndicatorParams(**schema.get("indicator_params", {})),
                lookback_needed=int(schema.get("lookback_needed", 300)),
                calibration=calibration,
                model_version=str(infer.get("model_version", "")),
            )
            self._cache[key] = (arts, mt)
            return arts


def _latest_mtime(paths: list[Path]) -> float:
    mt = 0.0
    for p in paths:
        try:
            mt = max(mt, p.stat().st_mtime)
        except FileNotFoundError:
            # Let the caller raise later when trying to open JSON/model files
            continue
    return mt


_GLOBAL_ARTIFACTS = ArtifactsStore()


def decide_signal(prob, min_conf: float) -> tuple[Signal, float, float, bool]:
    """
    Apply the decision rule to class probabilities [P(SELL), P(HOLD), P(BUY)].

    Rule: emit argmax unless its probability is below `min_conf`, in which case
    fall back to HOLD.

    Returns ``(signal, confidence, max_prob, forced_hold)`` where:
      - confidence is P(emitted class) — what we actually output;
      - max_prob is the raw top probability before the rule (so a forced HOLD's
        low confidence is explainable);
      - forced_hold is True only when the model's argmax was not HOLD but the
        rule downgraded it to HOLD.
    """
    prob = np.asarray(prob, dtype=float)
    y_hat = int(np.argmax(prob))
    max_prob = float(prob[y_hat])
    forced_hold = bool(max_prob < float(min_conf) and y_hat != 1)
    if max_prob < float(min_conf):
        y_hat = 1  # HOLD
    sig = y_to_signal(y_hat)
    probs = {"SELL": float(prob[0]), "HOLD": float(prob[1]), "BUY": float(prob[2])}
    confidence = float(probs[sig.value])
    return sig, confidence, max_prob, forced_hold


def predict_signal(
    symbol: str,
    paths: Paths,
    data_cfg: DataConfig | None = None,
    ind_params: IndicatorParams | None = None,
    min_conf: float | None = None,
    *,
    ensure_data: bool = False,
    since_days_default: int = 365 * 3,
    model_id: str | None = None,
    artifacts_store: ArtifactsStore | None = None,
) -> Prediction:
    """
    Predict BUY/SELL/HOLD for a symbol using the latest cached OHLCV in data/.

    Note: this function does not auto-download missing history. Run `model-tech download ...` first.
    """
    if data_cfg is None:
        data_cfg = DataConfig()

    symbol = symbol.strip().upper()
    if ensure_data:
        # ensure enough bars for indicator computation
        arts_for_lb = None
        try:
            arts_for_lb = (artifacts_store or _GLOBAL_ARTIFACTS).get(paths, model_id=model_id)
        except Exception:
            # If artifacts missing, we'll error later; still can fetch data.
            arts_for_lb = None
        min_bars = int(max(int(data_cfg.lookback_bars), int(getattr(arts_for_lb, "lookback_needed", 300))))
        ensure_symbol_ohlcv(
            symbol,
            paths=paths,
            data_cfg=data_cfg,
            min_bars=min_bars,
            since_days_default=int(since_days_default),
        )

    ohlcv = read_ohlcv(paths, symbol)
    if ohlcv.empty:
        raise ValueError(f"No OHLCV for {symbol}. Run `model-tech download --symbols {symbol} --since 2021-01-01` first.")

    store = artifacts_store or _GLOBAL_ARTIFACTS
    arts = store.get(paths, model_id=model_id)
    mc = float(min_conf) if min_conf is not None else float(arts.min_conf)
    ind_params = ind_params or arts.indicator_params

    lookback = max(int(data_cfg.lookback_bars), int(arts.lookback_needed))
    tail = ohlcv.sort_values("open_time").tail(lookback).reset_index(drop=True)
    feat = build_ta_features(ohlcv=tail, symbol=symbol, params=ind_params)
    if feat.empty:
        raise ValueError("Failed to build features (empty).")

    # Use the last row only; drop rows with NaNs.
    row = feat.dropna().tail(1)
    if row.empty:
        raise ValueError("Not enough history to compute indicators; increase history (download earlier) or adjust lookback.")

    X = row[arts.feature_cols + arts.cat_features]
    prob = arts.model.predict_proba(Pool(X, cat_features=arts.cat_features))[0]
    prob = np.asarray(prob, dtype=float)
    # Apply the calibrator (if any) before the decision rule, so min_conf — tuned
    # on calibrated probabilities at training time — thresholds the same scale.
    if arts.calibration is not None:
        prob = apply_calibration(prob, arts.calibration)

    sig, conf, max_prob, forced_hold = decide_signal(prob, mc)
    probs = {"SELL": float(prob[0]), "HOLD": float(prob[1]), "BUY": float(prob[2])}

    as_of = pd.to_datetime(row["open_time"].iloc[0], utc=True).isoformat()
    return Prediction(
        symbol=symbol,
        as_of=as_of,
        signal=sig,
        confidence=conf,
        probs=probs,
        max_prob=max_prob,
        forced_hold=forced_hold,
        model_version=arts.model_version,
    )


