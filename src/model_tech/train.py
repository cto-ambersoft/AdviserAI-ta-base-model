from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score

from model_tech.artifacts import artifact_paths, write_json
from model_tech.config import DataConfig, LabelingConfig, ModelConfig, Paths, TrainConfig, TuneConfig
from model_tech.data.store import read_ohlcv
from model_tech.features.indicators import IndicatorParams, build_ta_features, infer_lookback_bars
from model_tech.labeling import compute_forward_return
from model_tech.logging import get_logger
from model_tech.split import walk_forward_splits

log = get_logger(__name__)


def train_pipeline(
    symbols: list[str],
    paths: Paths,
    data_cfg: DataConfig,
    lab_cfg: LabelingConfig,
    tr_cfg: TrainConfig,
    model_cfg: ModelConfig | None = None,
    ind_params: IndicatorParams | None = None,
    tune_cfg: TuneConfig | None = None,
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end training:
      - read cached parquet OHLCV
      - build TA features
      - tune theta via walk-forward (macro-F1 on val)
      - tune min_conf (optional, cheap) using most recent val window
      - train final model on all labeled data
      - save artifacts into artifacts/
    """
    if model_cfg is None:
        model_cfg = ModelConfig()
    if ind_params is None:
        ind_params = IndicatorParams()
    if tune_cfg is None:
        tune_cfg = TuneConfig()

    symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols:
        raise ValueError("No symbols provided")

    lookback_needed = max(int(data_cfg.lookback_bars), infer_lookback_bars(ind_params))
    df = _build_dataset(symbols=symbols, paths=paths, horizon_bars=lab_cfg.horizon_bars, lookback_needed=lookback_needed, ind_params=ind_params)
    if df.empty:
        raise ValueError("No training data. Run `model-tech download ...` first.")

    feature_cols = _feature_columns(df)
    cat_features = ["symbol"]

    # Time axis is unique open_time bars; folds are in bars, not rows.
    unique_times = np.array(sorted(df["open_time"].unique()))
    folds = walk_forward_splits(
        n_samples=len(unique_times),
        n_folds=tr_cfg.n_folds,
        val_size=tr_cfg.val_bars,
        test_size=tr_cfg.test_bars,
        min_train_size=tr_cfg.min_train_bars,
        gap=lab_cfg.horizon_bars,
    )
    if not folds:
        raise ValueError(
            f"Not enough data for walk-forward. Need >= {tr_cfg.min_train_bars + lab_cfg.horizon_bars + tr_cfg.val_bars + tr_cfg.test_bars} bars, got {len(unique_times)}."
        )

    theta_candidates = _theta_candidates(
        fwd_return=df["fwd_return"].to_numpy(),
        theta_min=float(lab_cfg.theta_min),
        theta_max=float(lab_cfg.theta_max),
        hold_share_min=float(lab_cfg.hold_share_min),
        hold_share_max=float(lab_cfg.hold_share_max),
        n_candidates=int(tune_cfg.theta_candidates),
        multipliers=tuple(float(x) for x in tune_cfg.theta_multipliers),
        mode=str(getattr(tr_cfg, "mode", "quality")),
    )
    best = None
    cv_rows: list[dict[str, Any]] = []

    # Lighter model during theta search (CV) to reduce runtime.
    tune_iterations = int(min(int(model_cfg.iterations), int(tune_cfg.tune_iterations)))
    for theta in theta_candidates:
        y = _labels_from_fwd_return(df["fwd_return"].to_numpy(), theta=float(theta))
        hold_share = float(np.mean(y == 1))
        if not (lab_cfg.hold_share_min <= hold_share <= lab_cfg.hold_share_max):
            continue

        fold_scores = []
        for fold in folds:
            train_mask, val_mask = _time_masks(df["open_time"].to_numpy(), unique_times, fold.train_idx, fold.val_idx)
            X_train = df.loc[train_mask, feature_cols + cat_features]
            y_train = y[train_mask]
            X_val = df.loc[val_mask, feature_cols + cat_features]
            y_val = y[val_mask]

            model = _make_model(model_cfg, iterations_override=tune_iterations)
            model.fit(
                Pool(X_train, y_train, cat_features=cat_features),
                eval_set=Pool(X_val, y_val, cat_features=cat_features),
                use_best_model=True,
                verbose=False,
            )
            pred = model.predict(Pool(X_val, cat_features=cat_features)).astype(int).reshape(-1)
            f1 = float(f1_score(y_val, pred, average="macro"))
            fold_scores.append(f1)

        if not fold_scores:
            continue

        mean_f1 = float(np.mean(fold_scores))
        cv_rows.append({"theta": float(theta), "hold_share": hold_share, "val_macro_f1": mean_f1, "folds": fold_scores})
        if best is None or mean_f1 > best["val_macro_f1"]:
            best = {"theta": float(theta), "hold_share": hold_share, "val_macro_f1": mean_f1}

    if best is None:
        raise ValueError("No theta satisfied HOLD-share constraints; widen theta_min/theta_max or adjust hold_share bounds.")

    best_theta = float(best["theta"])
    log.info("Chosen theta=%.5f (cv val macro-F1=%.4f, hold_share=%.3f)", best_theta, best["val_macro_f1"], best["hold_share"])

    # Tune min_conf on most recent validation window (cheap: no re-train per candidate).
    min_conf = _tune_min_conf_recent_val(
        df=df,
        unique_times=unique_times,
        theta=best_theta,
        feature_cols=feature_cols,
        cat_features=cat_features,
        model_cfg=model_cfg,
        val_bars=tr_cfg.val_bars,
        min_train_bars=tr_cfg.min_train_bars,
        hold_share_min=lab_cfg.hold_share_min,
        hold_share_max=lab_cfg.hold_share_max,
    )

    # Train final model on all labeled rows.
    y_all = _labels_from_fwd_return(df["fwd_return"].to_numpy(), theta=best_theta)
    X_all = df[feature_cols + cat_features]
    final = _make_model(model_cfg)
    final.fit(Pool(X_all, y_all, cat_features=cat_features), verbose=False)

    # Persist artifacts
    ap = artifact_paths(paths, model_id=model_id)
    ap.model_path.parent.mkdir(parents=True, exist_ok=True)
    final.save_model(str(ap.model_path))

    schema = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbols": symbols,
        "interval": data_cfg.interval,
        "feature_cols": feature_cols,
        "cat_features": cat_features,
        "indicator_params": asdict(ind_params),
        "lookback_needed": int(lookback_needed),
    }
    write_json(ap.feature_schema_path, schema)

    write_json(
        ap.labeling_path,
        {
            "horizon_bars": int(lab_cfg.horizon_bars),
            "theta": float(best_theta),
            "theta_search": {
                "theta_min": float(lab_cfg.theta_min),
                "theta_max": float(lab_cfg.theta_max),
                "hold_share_min": float(lab_cfg.hold_share_min),
                "hold_share_max": float(lab_cfg.hold_share_max),
            },
        },
    )
    write_json(
        ap.inference_path,
        {
            "min_conf": float(min_conf),
            "class_mapping": {"SELL": 0, "HOLD": 1, "BUY": 2},
            "decision_rule": "if max_prob < min_conf: HOLD else argmax",
        },
    )
    write_json(
        ap.metrics_path,
        {
            "best_theta": best,
            "cv": cv_rows,
            "min_conf": float(min_conf),
        },
    )

    # Quick honest metrics on the most recent test fold (train+val -> test)
    last_fold = folds[-1]
    train_val_idx = np.concatenate([last_fold.train_idx, last_fold.val_idx])
    train_val_mask = _time_mask(df["open_time"].to_numpy(), unique_times, train_val_idx)
    test_mask = _time_mask(df["open_time"].to_numpy(), unique_times, last_fold.test_idx)

    X_train_val = df.loc[train_val_mask, feature_cols + cat_features]
    y_train_val = y_all[train_val_mask]
    X_test = df.loc[test_mask, feature_cols + cat_features]
    y_test = y_all[test_mask]

    eval_model = _make_model(model_cfg)
    eval_model.fit(Pool(X_train_val, y_train_val, cat_features=cat_features), verbose=False)
    prob = eval_model.predict_proba(Pool(X_test, cat_features=cat_features))
    pred = np.argmax(prob, axis=1).astype(int)
    macro_f1 = float(f1_score(y_test, pred, average="macro"))
    bal_acc = float(balanced_accuracy_score(y_test, pred))

    return {
        "artifacts_dir": str(paths.artifacts_dir),
        "model_id": (str(model_id).strip().upper() if model_id else "global"),
        "symbols": symbols,
        "n_rows": int(len(df)),
        "n_bars": int(len(unique_times)),
        "theta": float(best_theta),
        "min_conf": float(min_conf),
        "test_macro_f1": macro_f1,
        "test_balanced_accuracy": bal_acc,
        "test_report": classification_report(y_test, pred, digits=4),
    }


def _build_dataset(
    symbols: list[str],
    paths: Paths,
    horizon_bars: int,
    lookback_needed: int,
    ind_params: IndicatorParams,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for sym in symbols:
        ohlcv = read_ohlcv(paths, sym)
        if ohlcv.empty:
            continue
        if len(ohlcv) < (lookback_needed + horizon_bars + 10):
            log.warning("Skipping %s: not enough bars (%d)", sym, len(ohlcv))
            continue

        feat = build_ta_features(ohlcv=ohlcv, symbol=sym, params=ind_params)
        feat["close"] = ohlcv.sort_values("open_time")["close"].to_numpy()
        feat["fwd_return"] = compute_forward_return(feat["close"], horizon_bars=horizon_bars)

        # Drop rows where indicators are not formed or forward return is missing.
        feature_cols = [c for c in feat.columns if c not in ("open_time", "symbol", "close", "fwd_return")]
        feat = feat.dropna(subset=feature_cols + ["fwd_return"]).reset_index(drop=True)

        parts.append(feat[["open_time", "symbol", "fwd_return", *feature_cols]].copy())

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("open_time").reset_index(drop=True)
    return out


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("open_time", "symbol", "fwd_return")]


def _labels_from_fwd_return(fwd: np.ndarray, theta: float) -> np.ndarray:
    y = np.full(len(fwd), 1, dtype=int)  # HOLD
    y[fwd > theta] = 2
    y[fwd < -theta] = 0
    return y


def _make_model(cfg: ModelConfig, iterations_override: int | None = None) -> CatBoostClassifier:
    iterations = int(iterations_override) if iterations_override is not None else int(cfg.iterations)
    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=cfg.learning_rate,
        depth=cfg.depth,
        l2_leaf_reg=cfg.l2_leaf_reg,
        random_seed=cfg.random_seed,
        loss_function=cfg.loss_function,
        eval_metric=cfg.eval_metric,
        thread_count=int(getattr(cfg, "thread_count", -1)),
        task_type=str(getattr(cfg, "task_type", "CPU")),
        allow_writing_files=False,
        od_type="Iter",
        od_wait=200,
        use_best_model=False,
    )


def _time_mask(open_time: np.ndarray, unique_times: np.ndarray, time_idx: np.ndarray) -> np.ndarray:
    return np.isin(open_time, unique_times[time_idx])


def _time_masks(open_time: np.ndarray, unique_times: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return _time_mask(open_time, unique_times, train_idx), _time_mask(open_time, unique_times, val_idx)


def _tune_min_conf_recent_val(
    df: pd.DataFrame,
    unique_times: np.ndarray,
    theta: float,
    feature_cols: list[str],
    cat_features: list[str],
    model_cfg: ModelConfig,
    val_bars: int,
    min_train_bars: int,
    hold_share_min: float,
    hold_share_max: float,
) -> float:
    if len(unique_times) < (min_train_bars + val_bars):
        return 0.45

    val_times = unique_times[-val_bars:]
    train_times = unique_times[: -val_bars]
    train_mask = df["open_time"].isin(train_times).to_numpy()
    val_mask = df["open_time"].isin(val_times).to_numpy()

    y = _labels_from_fwd_return(df["fwd_return"].to_numpy(), theta=theta)

    X_train = df.loc[train_mask, feature_cols + cat_features]
    y_train = y[train_mask]
    X_val = df.loc[val_mask, feature_cols + cat_features]
    y_val = y[val_mask]

    model = _make_model(model_cfg)
    model.fit(
        Pool(X_train, y_train, cat_features=cat_features),
        eval_set=Pool(X_val, y_val, cat_features=cat_features),
        use_best_model=True,
        verbose=False,
    )
    prob = model.predict_proba(Pool(X_val, cat_features=cat_features))
    argmax = np.argmax(prob, axis=1).astype(int)
    maxp = np.max(prob, axis=1)

    grid = np.linspace(0.30, 0.75, 19)
    best_conf = 0.45
    best_score = -1.0
    for mc in grid:
        pred = argmax.copy()
        pred[maxp < mc] = 1  # force HOLD
        hold_share = float(np.mean(pred == 1))
        if not (hold_share_min <= hold_share <= hold_share_max):
            continue
        score = float(f1_score(y_val, pred, average="macro"))
        if score > best_score:
            best_score = score
            best_conf = float(mc)

    if best_score < 0:
        return 0.45
    return best_conf


def _theta_candidates(
    fwd_return: np.ndarray,
    theta_min: float,
    theta_max: float,
    hold_share_min: float,
    hold_share_max: float,
    n_candidates: int,
    multipliers: tuple[float, ...],
    mode: str,
) -> list[float]:
    """
    Produce a small theta candidate set anchored by a quantile of |fwd_return|.

    Intuition:
      HOLD share is approximately P(|r| <= theta). Setting theta at the hold-share
      quantile makes HOLD share close to target by construction, without training.
    """
    n_candidates = int(max(3, n_candidates))
    theta_min = float(theta_min)
    theta_max = float(theta_max)
    if theta_min <= 0 or theta_max <= 0 or theta_max <= theta_min:
        # fallback to something sensible
        return [float(x) for x in np.linspace(max(1e-6, theta_min), max(theta_min * 2, theta_max), num=min(9, n_candidates))]

    fr = np.asarray(fwd_return, dtype=float)
    fr = fr[np.isfinite(fr)]
    if fr.size == 0:
        return [float(x) for x in np.linspace(theta_min, theta_max, num=min(9, n_candidates))]

    abs_r = np.abs(fr)

    # target HOLD share: mid of allowed band
    hold_target = float((hold_share_min + hold_share_max) * 0.5)
    hold_target = float(np.clip(hold_target, 0.05, 0.95))
    # If HOLD share is P(|r| <= theta), then theta is the hold_target-quantile of |r|.
    anchor = float(np.quantile(abs_r, hold_target))
    anchor = float(np.clip(anchor, theta_min, theta_max))

    # "fast" mode: minimal candidate set around anchor
    base = {anchor}
    for m in multipliers:
        base.add(float(np.clip(anchor * float(m), theta_min, theta_max)))

    # Fill remaining candidates with linear spacing around anchor (tight band) for quality mode.
    if str(mode).lower().strip() != "fast":
        band_lo = float(np.clip(anchor * 0.6, theta_min, theta_max))
        band_hi = float(np.clip(anchor * 1.6, theta_min, theta_max))
        for x in np.linspace(band_lo, band_hi, num=max(3, n_candidates)):
            base.add(float(np.clip(float(x), theta_min, theta_max)))

    out = sorted(base)
    # Limit size deterministically
    if len(out) > n_candidates:
        # Pick evenly across sorted set
        idx = np.linspace(0, len(out) - 1, num=n_candidates).round().astype(int)
        out = [out[i] for i in idx]
    return out


def train_symbol_pipeline(
    symbol: str,
    *,
    paths: Paths,
    data_cfg: DataConfig,
    lab_cfg: LabelingConfig,
    model_cfg: ModelConfig | None = None,
    ind_params: IndicatorParams | None = None,
    tune_cfg: TuneConfig | None = None,
    n_folds: int = 3,
    mode: str = "fast",
) -> dict[str, Any]:
    """
    Convenience wrapper for per-symbol training. Writes artifacts into:
      artifacts/models/<SYMBOL>/
    """
    sym = symbol.strip().upper()
    tr_cfg = TrainConfig(n_folds=int(n_folds), mode=str(mode))
    return train_pipeline(
        symbols=[sym],
        paths=paths,
        data_cfg=data_cfg,
        lab_cfg=lab_cfg,
        tr_cfg=tr_cfg,
        model_cfg=model_cfg,
        ind_params=ind_params,
        tune_cfg=tune_cfg,
        model_id=sym,
    )

