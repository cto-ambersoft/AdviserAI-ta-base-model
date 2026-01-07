from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import ChaikinMoneyFlowIndicator, OnBalanceVolumeIndicator


@dataclass(frozen=True)
class IndicatorParams:
    rsi_window: int = 14
    adx_window: int = 14
    atr_window: int = 14
    ema_window: int = 50
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_window: int = 20
    bb_dev: float = 2.0
    vol_sma_window: int = 30
    realized_vol_window: int = 48  # 8 days for 4h bars


def build_ta_features(
    ohlcv: pd.DataFrame,
    symbol: str,
    params: IndicatorParams | None = None,
) -> pd.DataFrame:
    """
    Build a leakage-safe feature table from OHLCV candles.

    Input columns required: open_time (datetime64[ns, UTC]), open/high/low/close/volume (float).
    Output columns include "open_time", "symbol" and feature columns (float).
    """
    if params is None:
        params = IndicatorParams()

    if ohlcv.empty:
        return pd.DataFrame()

    df = ohlcv.copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    df["symbol"] = str(symbol).upper()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # Returns / momentum
    log_close = np.log(close.replace(0, np.nan))
    df["r_1"] = log_close.diff(1)
    df["r_6"] = log_close.diff(6)

    # RSI
    df["rsi_14"] = RSIIndicator(close=close, window=params.rsi_window).rsi()

    # MACD histogram
    macd = MACD(
        close=close,
        window_fast=params.macd_fast,
        window_slow=params.macd_slow,
        window_sign=params.macd_signal,
    )
    # Normalize MACD histogram by close price
    df["macd_hist"] = macd.macd_diff() / close.replace(0, np.nan)

    # ADX
    df["adx_14"] = ADXIndicator(high=high, low=low, close=close, window=params.adx_window).adx()

    # Mean reversion vs EMA
    ema50 = EMAIndicator(close=close, window=params.ema_window).ema_indicator()
    df["close_ema50_rel"] = (close - ema50) / ema50.replace(0, np.nan)

    # ATR normalized
    atr = AverageTrueRange(high=high, low=low, close=close, window=params.atr_window).average_true_range()
    df["atr14_rel"] = atr / close.replace(0, np.nan)

    # Realized volatility (std of log returns)
    df["realized_vol"] = df["r_1"].rolling(params.realized_vol_window).std()

    # Bollinger bands
    bb = BollingerBands(close=close, window=params.bb_window, window_dev=params.bb_dev)
    df["bb_percentb"] = bb.bollinger_pband()
    df["bb_width"] = bb.bollinger_wband()

    # Volume
    df["log_volume"] = np.log(volume.replace(0, np.nan))
    vol_sma = volume.rolling(params.vol_sma_window).mean()
    df["vol_rel"] = volume / vol_sma.replace(0, np.nan)

    # Optional: both CMF and OBV are cheap; keep both for now (model can ignore).
    df["cmf_20"] = ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume, window=20).chaikin_money_flow()
    df["obv"] = OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

    # Keep only relevant columns
    feature_cols = [
        "r_1",
        "r_6",
        "rsi_14",
        "macd_hist",
        "adx_14",
        "close_ema50_rel",
        "atr14_rel",
        "realized_vol",
        "bb_percentb",
        "bb_width",
        "log_volume",
        "vol_rel",
        "cmf_20",
        "obv",
    ]

    out = df[["open_time", "symbol", *feature_cols]].copy()
    return out


def infer_lookback_bars(params: IndicatorParams | None = None) -> int:
    """
    Conservative lookback required to compute the last-row features.
    """
    if params is None:
        params = IndicatorParams()
    return int(
        max(
            6,
            params.ema_window,
            params.realized_vol_window + 1,
            params.vol_sma_window + 1,
            params.bb_window + 1,
            params.atr_window + 1,
            params.adx_window + 1,
            params.rsi_window + 1,
            params.macd_slow + params.macd_signal + 1,
        )
    )


