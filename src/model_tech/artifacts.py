from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from model_tech.config import Paths


@dataclass(frozen=True)
class ArtifactPaths:
    model_path: Path
    feature_schema_path: Path
    labeling_path: Path
    inference_path: Path
    metrics_path: Path


def artifact_paths(paths: Paths, *, model_id: str | None = None) -> ArtifactPaths:
    """
    Resolve artifact locations.

    - Global (default): artifacts/{model.cbm, feature_schema.json, ...}
    - Namespaced model: artifacts/models/<MODEL_ID>/{model.cbm, feature_schema.json, ...}

    `model_id` is typically a symbol like BTCUSDT for per-symbol models.
    """
    if model_id is None or str(model_id).strip() in {"", "global"}:
        d = paths.artifacts_dir
    else:
        d = paths.artifacts_dir / "models" / str(model_id).strip().upper()
    return ArtifactPaths(
        model_path=d / "model.cbm",
        feature_schema_path=d / "feature_schema.json",
        labeling_path=d / "labeling.json",
        inference_path=d / "inference.json",
        metrics_path=d / "metrics.json",
    )


def artifacts_ready(paths: Paths, *, model_id: str | None = None) -> bool:
    """
    True iff minimum set of inference-critical artifacts exists on disk.

    Used to decide whether to skip (re-)training.
    """
    ap = artifact_paths(paths, model_id=model_id)
    return ap.model_path.exists() and ap.feature_schema_path.exists() and ap.inference_path.exists()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _json_default(obj: Any) -> Any:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return str(obj)


