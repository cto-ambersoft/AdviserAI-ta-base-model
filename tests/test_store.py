"""I6 — coverage for the parquet OHLCV store (read/write/upsert + normalization)."""
from __future__ import annotations

import pandas as pd

from model_tech.config import Paths
from model_tech.data.store import read_ohlcv, upsert_ohlcv, write_ohlcv


def _df(times, closes) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame(
        {
            "open_time": pd.to_datetime(times, utc=True),
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * n,
        }
    )


def test_read_missing_symbol_returns_empty(tmp_path) -> None:
    paths = Paths(root=tmp_path, data_dir_override=tmp_path)
    assert read_ohlcv(paths, "BTCUSDT").empty


def test_write_read_roundtrip_is_sorted_by_time(tmp_path) -> None:
    paths = Paths(root=tmp_path, data_dir_override=tmp_path)
    write_ohlcv(paths, "BTCUSDT", _df(
        ["2022-01-01 08:00", "2022-01-01 00:00", "2022-01-01 04:00"], [3.0, 1.0, 2.0]
    ))
    out = read_ohlcv(paths, "BTCUSDT")
    assert list(out["close"]) == [1.0, 2.0, 3.0]


def test_upsert_dedups_by_open_time_keeping_latest(tmp_path) -> None:
    paths = Paths(root=tmp_path, data_dir_override=tmp_path)
    write_ohlcv(paths, "BTCUSDT", _df(["2022-01-01 00:00", "2022-01-01 04:00"], [1.0, 2.0]))
    # Re-send 04:00 with a corrected close (99) plus a brand-new 08:00 bar.
    upsert_ohlcv(paths, "BTCUSDT", _df(["2022-01-01 04:00", "2022-01-01 08:00"], [99.0, 5.0]))

    out = read_ohlcv(paths, "BTCUSDT")
    assert len(out) == 3  # 00:00, 04:00, 08:00 (no duplicate)
    row = out[out["open_time"] == pd.Timestamp("2022-01-01 04:00", tz="UTC")]
    assert float(row["close"].iloc[0]) == 99.0  # newest wins
