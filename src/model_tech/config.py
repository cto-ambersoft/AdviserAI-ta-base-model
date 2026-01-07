from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path
    data_dir_override: Path | None = None
    artifacts_dir_override: Path | None = None

    @property
    def data_dir(self) -> Path:
        return self.data_dir_override or (self.root / "data")

    @property
    def artifacts_dir(self) -> Path:
        return self.artifacts_dir_override or (self.root / "artifacts")


@dataclass(frozen=True)
class DataConfig:
    interval: str = "4h"  # Binance kline interval
    lookback_bars: int = 300  # must cover max indicator lookback
    symbols_default: tuple[str, ...] = (
        "BTCUSDT",
        "ETHUSDT",
        "BNBUSDT",
        "SOLUSDT",
        "XRPUSDT",
    )


@dataclass(frozen=True)
class LabelingConfig:
    horizon_bars: int = 6  # H=6 4h bars = 1 day
    hold_share_min: float = 0.35
    hold_share_max: float = 0.65
    theta_min: float = 0.003  # 0.3%
    theta_max: float = 0.02   # 2.0%


@dataclass(frozen=True)
class TrainConfig:
    n_folds: int = 6
    val_bars: int = 180   # ~30 days of 4h bars
    test_bars: int = 180  # ~30 days
    min_train_bars: int = 6 * 30 * 6  # ~6 months
    # Training mode:
    # - "quality": slower, more robust theta selection (still optimized vs v1)
    # - "fast": intended for online per-symbol retrain jobs
    mode: str = "quality"


@dataclass(frozen=True)
class TuneConfig:
    """
    Controls search cost during theta selection.
    """

    # How many theta candidates to evaluate (instead of fixed 21-point grid)
    theta_candidates: int = 9
    # Relative multipliers around the quantile anchor theta
    theta_multipliers: tuple[float, ...] = (0.70, 0.85, 1.00, 1.15, 1.30)
    # Use a lighter model during theta CV to reduce runtime
    tune_iterations: int = 400


@dataclass(frozen=True)
class OnlineTrainConfig:
    """
    Defaults for online (per-symbol) training jobs.
    """

    # If symbol has no local history, backfill this many days initially
    since_days_default: int = 365 * 3
    # Target HOLD share used for the theta quantile anchor.
    # This is used as a midpoint between hold_share_min and hold_share_max unless overridden.
    hold_share_target: float = 0.50


@dataclass(frozen=True)
class ModelConfig:
    # CatBoost default-ish; tuned lightly for robustness.
    iterations: int = 2000
    learning_rate: float = 0.05
    depth: int = 6
    l2_leaf_reg: float = 6.0
    random_seed: int = 42
    loss_function: str = "MultiClass"
    eval_metric: str = "TotalF1"
    # Performance knobs
    thread_count: int = -1  # -1 = all cores
    task_type: str = "CPU"  # "CPU" or "GPU"


def _env_bool(name: str, default: bool) -> bool:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    v = raw.strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _env_int(name: str, default: int) -> int:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw.strip())
    except ValueError:
        return int(default)


def _env_str(name: str, default: str) -> str:
    import os

    raw = os.environ.get(name)
    if raw is None:
        return str(default)
    return raw.strip()


@dataclass(frozen=True)
class AutoTrainConfig:
    """
    Auto-training at API startup (best-effort, background).

    Note: Auto-training is CPU/network intensive. Tune worker/thread counts if API latency matters.
    """

    enabled: bool = True
    # NOTE: keep defaults conservative and Binance-spot-safe. Symbols are normalized to e.g. "BTCUSDT".
    # Add extra symbols via MODEL_TECH_AUTOTRAIN_SYMBOLS.
    symbols: tuple[str, ...] = ("BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "LINK")
    max_workers: int = 2  # parallel training jobs
    mode: str = "fast"
    n_folds: int = 3
    since_days_default: int = 365 * 3
    min_bars: int = 2200
    catboost_thread_count: int = 2  # per training job
    skip_if_exists: bool = True
    force_retrain: bool = False

    @classmethod
    def from_env(cls) -> "AutoTrainConfig":
        defaults = cls()
        symbols_raw = _env_str("MODEL_TECH_AUTOTRAIN_SYMBOLS", ",".join(defaults.symbols))
        symbols = tuple(s.strip() for s in symbols_raw.split(",") if s.strip())
        return cls(
            enabled=_env_bool("MODEL_TECH_AUTOTRAIN_ENABLED", defaults.enabled),
            symbols=symbols or defaults.symbols,
            max_workers=max(1, _env_int("MODEL_TECH_AUTOTRAIN_MAX_WORKERS", defaults.max_workers)),
            mode=_env_str("MODEL_TECH_AUTOTRAIN_MODE", defaults.mode) or defaults.mode,
            n_folds=max(2, _env_int("MODEL_TECH_AUTOTRAIN_N_FOLDS", defaults.n_folds)),
            since_days_default=max(30, _env_int("MODEL_TECH_AUTOTRAIN_SINCE_DAYS", defaults.since_days_default)),
            min_bars=max(0, _env_int("MODEL_TECH_AUTOTRAIN_MIN_BARS", defaults.min_bars)),
            catboost_thread_count=max(1, _env_int("MODEL_TECH_AUTOTRAIN_CATBOOST_THREADS", defaults.catboost_thread_count)),
            skip_if_exists=_env_bool("MODEL_TECH_AUTOTRAIN_SKIP_EXISTING", defaults.skip_if_exists),
            force_retrain=_env_bool("MODEL_TECH_AUTOTRAIN_FORCE", defaults.force_retrain),
        )


def project_paths() -> Paths:
    import os

    if home := os.environ.get("MODEL_TECH_HOME"):
        return Paths(root=Path(home))

    # src/model_tech/config.py -> repo root is two parents up
    root = Path(__file__).resolve().parents[2]
    return Paths(root=root)


