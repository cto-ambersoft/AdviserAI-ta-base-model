from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


def leave_one_symbol_out(symbols: list[str]) -> list[tuple[list[str], str]]:
    """
    Leave-one-symbol-out folds (sklearn ``LeaveOneGroupOut`` with groups=symbol).

    For each distinct symbol (in first-seen order) yields ``(train_symbols,
    held_out_symbol)`` where train_symbols is every other symbol. Returns an
    empty list when there are fewer than two distinct symbols (LOSO needs at
    least one symbol to train on and one to hold out).
    """
    distinct: list[str] = []
    for s in symbols:
        if s not in distinct:
            distinct.append(s)
    if len(distinct) < 2:
        return []
    return [([o for o in distinct if o != held], held) for held in distinct]


def purge_train_times(times: np.ndarray, horizon: int) -> np.ndarray:
    """
    Drop the last `horizon` bars from a time-sorted training window.

    Labels are the forward return over `horizon` bars, so the final `horizon`
    training bars carry information about closes inside the subsequent held-out
    window (validation/test). Removing them prevents look-ahead leakage — the
    same role as scikit-learn's ``TimeSeriesSplit(gap=...)``.

    Operates positionally on the tail and never mutates the input.
    """
    times = np.asarray(times)
    if horizon <= 0 or times.size == 0:
        return times
    if horizon >= times.size:
        return times[:0]
    return times[:-horizon]


def walk_forward_splits(
    n_samples: int,
    n_folds: int,
    val_size: int,
    test_size: int,
    min_train_size: int,
    gap: int = 0,
) -> list[Fold]:
    """
    Generate contiguous walk-forward folds from oldest->newest.

    Layout per fold:
      [train ...][gap][val ...][test ...]

    The last fold ends at the last sample.
    """
    if n_samples <= 0:
        return []
    if n_folds <= 0:
        return []
    if val_size <= 0 or test_size <= 0:
        raise ValueError("val_size and test_size must be > 0")

    total_tail = n_folds * (val_size + test_size)
    if n_samples < (min_train_size + gap + val_size + test_size):
        return []

    # Place folds so that the newest fold touches the end.
    end = n_samples
    folds: list[Fold] = []
    for i in range(n_folds):
        fold_end = end - (n_folds - 1 - i) * (val_size + test_size)
        test_end = fold_end
        test_start = test_end - test_size
        val_end = test_start
        val_start = val_end - val_size
        train_end = val_start - gap
        train_start = 0

        if train_end - train_start < min_train_size:
            continue
        if val_start < 0:
            continue
        if test_start < 0:
            continue

        folds.append(
            Fold(
                fold_id=len(folds),
                train_idx=np.arange(train_start, train_end, dtype=int),
                val_idx=np.arange(val_start, val_end, dtype=int),
                test_idx=np.arange(test_start, test_end, dtype=int),
            )
        )

    return folds


