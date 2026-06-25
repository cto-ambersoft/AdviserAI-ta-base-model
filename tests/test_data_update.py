"""
Tests for dropping the still-forming (unclosed) candle.

Binance klines includes the current in-progress interval. Computing
indicators on (or predicting from) a partial bar makes the live signal
jitter within the interval and creates a partial-bar artifact in stored
history. `drop_unclosed_candle` removes the trailing bar while it is still
open, based on the candle's expected close time.
"""
from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

from model_tech.data.update import _interval_to_hours, drop_unclosed_candle


def _ohlcv(open_times: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(open_times)
    return pd.DataFrame(
        {
            "open_time": open_times,
            "open": [1.0] * n,
            "high": [1.0] * n,
            "low": [1.0] * n,
            "close": [1.0] * n,
            "volume": [1.0] * n,
        }
    )


def _times(n: int) -> pd.DatetimeIndex:
    # 00:00, 04:00, 08:00 ... last bar (08:00) closes at 12:00
    return pd.date_range("2022-01-01", periods=n, freq="4h", tz="UTC")


def test_interval_to_hours() -> None:
    assert _interval_to_hours("4h") == 4.0
    assert _interval_to_hours("1h") == 1.0
    assert _interval_to_hours("1d") == 24.0
    assert _interval_to_hours("15m") == 0.25
    with pytest.raises(ValueError):
        _interval_to_hours("bogus")


def test_drops_last_bar_while_still_open() -> None:
    df = _ohlcv(_times(3))  # last open = 08:00, closes 12:00
    now = pd.Timestamp("2022-01-01 10:00", tz=timezone.utc)  # 08:00 bar forming
    out = drop_unclosed_candle(df, "4h", now=now)
    assert len(out) == 2
    assert out["open_time"].max() == df["open_time"].iloc[1]  # 04:00 kept


def test_keeps_last_bar_once_closed() -> None:
    df = _ohlcv(_times(3))
    # Exactly at the close boundary the candle is considered closed.
    now = pd.Timestamp("2022-01-01 12:00", tz=timezone.utc)
    out = drop_unclosed_candle(df, "4h", now=now)
    assert len(out) == 3


def test_keeps_all_bars_when_now_far_in_future() -> None:
    df = _ohlcv(_times(3))
    now = pd.Timestamp("2022-01-02 00:00", tz=timezone.utc)
    out = drop_unclosed_candle(df, "4h", now=now)
    assert len(out) == 3


def test_empty_input_is_safe() -> None:
    empty = _ohlcv(_times(0))
    out = drop_unclosed_candle(empty, "4h", now=pd.Timestamp("2022-01-02", tz=timezone.utc))
    assert out.empty
