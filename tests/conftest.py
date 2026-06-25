from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_synthetic_ohlcv(n: int = 700, *, seed: int = 7, start: str = "2022-01-01") -> pd.DataFrame:
    """
    Deterministic synthetic 4h OHLCV series (geometric random walk).

    Self-contained so feature tests do not depend on network or on the
    git-ignored data/ parquet cache.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.003, size=n)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.003, size=n)))
    volume = rng.uniform(100.0, 1000.0, size=n)
    open_time = pd.date_range(start=start, periods=n, freq="4h", tz="UTC")
    return pd.DataFrame(
        {
            "open_time": open_time,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    return make_synthetic_ohlcv()
