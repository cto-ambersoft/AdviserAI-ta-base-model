from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from model_tech.types import Signal


@dataclass(frozen=True)
class LabelingResult:
    horizon_bars: int
    theta: float
    labeled: pd.DataFrame  # includes open_time, symbol, y, fwd_return


def compute_forward_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    """
    r_{t,H} = Close_{t+H} / Close_t - 1
    """
    future = close.shift(-horizon_bars)
    return (future / close) - 1.0


def label_buy_sell_hold(
    df: pd.DataFrame,
    horizon_bars: int,
    theta: float,
) -> LabelingResult:
    """
    Labels rows into BUY/SELL/HOLD based on future return over horizon.

    Input df must contain: open_time, symbol, close.
    Output uses integer classes for CatBoost with stable mapping:
      0=SELL, 1=HOLD, 2=BUY.
    """
    if df.empty:
        return LabelingResult(horizon_bars=horizon_bars, theta=theta, labeled=pd.DataFrame())

    out = df[["open_time", "symbol", "close"]].copy()
    out["fwd_return"] = compute_forward_return(out["close"], horizon_bars=horizon_bars)

    y = np.full(len(out), 1, dtype=int)  # HOLD by default
    y[out["fwd_return"].to_numpy() > float(theta)] = 2  # BUY
    y[out["fwd_return"].to_numpy() < -float(theta)] = 0  # SELL
    out["y"] = y

    # Last horizon rows have NaN forward return; remove from training set
    out = out.dropna(subset=["fwd_return"]).reset_index(drop=True)
    return LabelingResult(horizon_bars=horizon_bars, theta=float(theta), labeled=out)


def y_to_signal(y: int) -> Signal:
    if int(y) == 0:
        return Signal.SELL
    if int(y) == 2:
        return Signal.BUY
    return Signal.HOLD


def signal_to_y(sig: Signal | str) -> int:
    s = Signal(sig)
    if s == Signal.SELL:
        return 0
    if s == Signal.BUY:
        return 2
    return 1


