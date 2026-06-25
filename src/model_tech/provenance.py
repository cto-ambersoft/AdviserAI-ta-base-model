"""
Model provenance: deterministic fingerprints of the training data and feature
code, a content-addressed model_version, and helpers to embed/read provenance in
the CatBoost .cbm metadata (so the model file is self-describing).

All hashes are deterministic: identical data/code/params produce identical
hashes, any change produces a different one.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import subprocess
from typing import Any

import numpy as np
import pandas as pd

_OHLCV_COLS = ("open", "high", "low", "close", "volume")


def hash_ohlcv(ohlcv: pd.DataFrame) -> str:
    """SHA256 of canonicalized OHLCV (sorted by open_time; row order irrelevant)."""
    if ohlcv.empty:
        return hashlib.sha256(b"").hexdigest()
    df = ohlcv.sort_values("open_time")
    ot = pd.to_datetime(df["open_time"], utc=True).astype("int64").to_numpy()
    vals = np.column_stack([df[c].to_numpy(dtype="float64") for c in _OHLCV_COLS])
    h = hashlib.sha256()
    h.update(ot.tobytes())
    h.update(np.ascontiguousarray(vals).tobytes())
    return h.hexdigest()


def dataset_fingerprint(per_symbol: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """
    Per-symbol OHLCV fingerprints + a combined, symbol-order-independent
    training_data_hash, plus human-readable window info (n_bars, start, end).
    """
    per: dict[str, dict[str, Any]] = {}
    for sym, df in per_symbol.items():
        d = df.sort_values("open_time")
        per[str(sym)] = {
            "sha256": hash_ohlcv(d),
            "n_bars": int(len(d)),
            "start": (str(pd.to_datetime(d["open_time"].min(), utc=True)) if len(d) else None),
            "end": (str(pd.to_datetime(d["open_time"].max(), utc=True)) if len(d) else None),
        }
    combined = "|".join(f"{s}:{per[s]['sha256']}" for s in sorted(per))
    return {
        "training_data_hash": hashlib.sha256(combined.encode()).hexdigest(),
        "per_symbol": per,
    }


def feature_code_hash(indicator_params: dict[str, Any] | None = None) -> str:
    """
    SHA256 of the feature-computation source + indicator params. Uses the function
    source (not the file) so it is stable in a container without a .git checkout.
    """
    from model_tech.features.indicators import build_ta_features

    src = inspect.getsource(build_ta_features)
    params = indicator_params or {}
    payload = src + "\n" + json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def compute_model_version(training_data_hash: str, feature_code_hash: str, extra: dict[str, Any]) -> str:
    """Content-addressed model id: 'v' + first 12 hex of sha256(data + code + extra)."""
    payload = json.dumps(
        {"data": training_data_hash, "code": feature_code_hash, "extra": extra},
        sort_keys=True,
        default=str,
    )
    return "v" + hashlib.sha256(payload.encode()).hexdigest()[:12]


def git_commit() -> str | None:
    """Best-effort git commit (env override first; .git is usually absent in the image)."""
    env = os.environ.get("MODEL_TECH_GIT_COMMIT", "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=os.path.dirname(__file__),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


# Keys embedded into the CatBoost model metadata.
PROVENANCE_KEYS = ("model_version", "run_id", "training_data_hash", "feature_code_hash", "created_at_utc")


def embed_provenance_metadata(model: Any, fields: dict[str, Any]) -> None:
    """Embed provenance into the model's metadata (values stored as strings)."""
    md = model.get_metadata()
    for k, v in fields.items():
        if v is not None:
            md[k] = str(v)


def read_provenance_metadata(model: Any, keys: tuple[str, ...] | None = None) -> dict[str, str]:
    """Read provenance back from a (loaded) model's metadata."""
    md = model.get_metadata()
    wanted = keys if keys is not None else tuple(md.keys())
    out: dict[str, str] = {}
    for k in wanted:
        if k in md:
            val = md[k]
            out[k] = val.decode() if isinstance(val, (bytes, bytearray)) else str(val)
    return out
