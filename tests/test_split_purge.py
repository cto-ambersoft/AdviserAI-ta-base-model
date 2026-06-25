"""
Purge guard against look-ahead label leakage.

Labels are the forward return over `horizon` bars, so a training bar at time
index `i` carries information about the close at `i + horizon`. When a held-out
window (validation for min_conf tuning, or test for honest metrics) starts at
index `s`, the last `horizon` training bars (`s-horizon .. s-1`) have labels that
peek INTO the held-out window. `purge_train_times` drops them — the same idea as
scikit-learn's `TimeSeriesSplit(gap=...)` ("samples to exclude from the end of
the train set before the test set").
"""
from __future__ import annotations

import numpy as np

from model_tech.split import purge_train_times


def test_drops_last_horizon_bars() -> None:
    times = np.arange(10)
    out = purge_train_times(times, 3)
    np.testing.assert_array_equal(out, np.arange(7))


def test_zero_or_negative_horizon_is_noop() -> None:
    times = np.arange(10)
    np.testing.assert_array_equal(purge_train_times(times, 0), times)
    np.testing.assert_array_equal(purge_train_times(times, -5), times)


def test_horizon_ge_size_yields_empty() -> None:
    times = np.arange(4)
    assert purge_train_times(times, 4).size == 0
    assert purge_train_times(times, 10).size == 0


def test_empty_input_is_safe() -> None:
    assert purge_train_times(np.array([], dtype=int), 6).size == 0


def test_does_not_mutate_input() -> None:
    times = np.arange(10)
    before = times.copy()
    _ = purge_train_times(times, 3)
    np.testing.assert_array_equal(times, before)


def test_no_lookahead_into_holdout_window() -> None:
    # train = unique_times[:-val_bars]; held-out window starts at unique[-val_bars]
    unique = np.arange(100)
    horizon = 6
    val_bars = 20
    train = unique[:-val_bars]                # 0..79
    holdout_start = int(unique[-val_bars])    # 80

    # Without purge the last train bar (79) would leak: 79 + 6 = 85 >= 80.
    assert train.max() + horizon >= holdout_start

    purged = purge_train_times(train, horizon)
    # After purge no training label reaches into the held-out window.
    assert purged.max() + horizon < holdout_start
