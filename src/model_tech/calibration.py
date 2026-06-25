"""
Multiclass probability calibration via one-vs-rest per-class maps + renormalization.

This is the scheme scikit-learn's CalibratedClassifierCV uses for multiclass:
calibrate each class independently (sigmoid/Platt or isotonic), then renormalize so
the per-class outputs sum to 1.

`fit_calibration` (training time) uses scikit-learn and returns a JSON-serializable
dict so the calibrator lives next to the model as `calibration.json`.
`apply_calibration` (inference time) is pure NumPy.

Map shapes (per class):
- sigmoid:  {"a": float, "b": float}     -> calibrated = sigmoid(a * p + b)
- isotonic: {"x": [..], "y": [..]}        -> calibrated = interp(p, x, y)
- degenerate (class absent/constant in fit data): {"identity": true} -> pass-through
"""
from __future__ import annotations

from typing import Any

import numpy as np

METHODS = ("sigmoid", "isotonic")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def apply_calibration(prob, calib: dict[str, Any]) -> np.ndarray:
    """
    Apply a fitted calibration to probabilities.

    `prob` is shape (K,) or (n, K). Returns the same shape, renormalized to sum to 1.
    """
    p = np.asarray(prob, dtype=float)
    single = p.ndim == 1
    if single:
        p = p[None, :]

    method = calib.get("method")
    maps = calib.get("maps", [])
    out = np.array(p, dtype=float)
    for k in range(p.shape[1]):
        m = maps[k] if k < len(maps) else {"identity": True}
        col = p[:, k]
        if m.get("identity"):
            out[:, k] = col
        elif method == "sigmoid":
            out[:, k] = _sigmoid(float(m["a"]) * col + float(m["b"]))
        elif method == "isotonic":
            x = np.asarray(m["x"], dtype=float)
            y = np.asarray(m["y"], dtype=float)
            out[:, k] = np.interp(col, x, y, left=float(y[0]), right=float(y[-1]))
        else:
            out[:, k] = col

    s = out.sum(axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    out = out / s
    return out[0] if single else out


def fit_calibration(prob, y_true, method: str = "sigmoid", n_classes: int | None = None) -> dict[str, Any]:
    """
    Fit per-class calibration maps on held-out (prob, label) data.

    Returns a JSON-serializable dict consumable by `apply_calibration`.
    """
    if method not in METHODS:
        raise ValueError(f"unknown calibration method: {method!r} (expected one of {METHODS})")

    p = np.asarray(prob, dtype=float)
    y = np.asarray(y_true).astype(int)
    k_classes = int(n_classes if n_classes is not None else p.shape[1])

    maps: list[dict[str, Any]] = []
    for k in range(k_classes):
        col = p[:, k]
        target = (y == k).astype(int)
        # Degenerate: a class never (or always) occurs -> nothing to learn, pass through.
        if target.sum() == 0 or target.sum() == target.shape[0]:
            maps.append({"identity": True})
            continue
        maps.append(_fit_one(col, target, method))

    return {"method": method, "n_classes": k_classes, "maps": maps}


def _fit_one(score: np.ndarray, target: np.ndarray, method: str) -> dict[str, Any]:
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression

        # Minimal regularization ~ Platt scaling: calibrated = sigmoid(a*score + b).
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(score.reshape(-1, 1), target)
        return {"a": float(lr.coef_[0][0]), "b": float(lr.intercept_[0])}

    # isotonic
    from sklearn.isotonic import IsotonicRegression

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(score, target)
    return {
        "x": [float(v) for v in np.asarray(iso.X_thresholds_, dtype=float)],
        "y": [float(v) for v in np.asarray(iso.y_thresholds_, dtype=float)],
    }
