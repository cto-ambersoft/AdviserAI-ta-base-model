"""
Train/serve invariance guard for TA features.

Features are computed on the FULL history at training time
(`train._build_dataset`) but on only the last `lookback` candles at
inference time (`infer.predict_signal`, via `tail(lookback)`).

For the model to behave the same in training and in production, every
feature value on the most recent fully-formed candle MUST be identical
whether it was computed from the full history or from a trailing window.

This test would catch the OBV train/serve skew (cumulative indicator
whose absolute level depends on the window start) and any future
regression of the same shape.
"""
from __future__ import annotations

import numpy as np
import pytest

from model_tech.features.indicators import build_ta_features, infer_lookback_bars
from tests.conftest import make_synthetic_ohlcv

FEATURE_META = {"open_time", "symbol"}


def _feature_cols(df) -> list[str]:
    return [c for c in df.columns if c not in FEATURE_META]


@pytest.mark.parametrize("lookback", [300])
def test_features_full_vs_tail_match_on_last_closed_candle(lookback: int) -> None:
    df = make_synthetic_ohlcv(n=700)

    full = build_ta_features(df, "TESTUSDT")
    tail = build_ta_features(df.tail(lookback).reset_index(drop=True), "TESTUSDT")

    last_full = full.dropna().iloc[-1]
    last_tail = tail.dropna().iloc[-1]

    # Same candle on both sides — otherwise the comparison is meaningless.
    assert last_full["open_time"] == last_tail["open_time"]

    cols = _feature_cols(full)
    assert cols, "no feature columns produced"

    mismatches = {}
    for c in cols:
        a, b = float(last_full[c]), float(last_tail[c])
        # Relative + absolute tolerance: invariant features must match exactly
        # up to floating-point accumulation error.
        if abs(a - b) > 1e-6 * (1.0 + abs(a)):
            mismatches[c] = (a, b)

    assert not mismatches, (
        "Feature(s) differ between full-history and trailing-window computation "
        f"(train/serve skew): {mismatches}"
    )


def test_lookback_covers_every_indicator_window() -> None:
    # The inference lookback must be >= the longest indicator warm-up so the
    # last row is fully formed.
    df = make_synthetic_ohlcv(n=700)
    lb = infer_lookback_bars()
    tail = build_ta_features(df.tail(lb).reset_index(drop=True), "TESTUSDT")
    assert not tail.dropna().empty, (
        f"infer_lookback_bars()={lb} is too small: last rows are NaN"
    )
