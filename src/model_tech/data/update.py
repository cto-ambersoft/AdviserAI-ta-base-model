from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
from binance.error import ClientError

from model_tech.config import DataConfig, Paths
from model_tech.data.binance_client import BinanceClient, to_ms
from model_tech.data.store import read_ohlcv, upsert_ohlcv
from model_tech.logging import get_logger

log = get_logger(__name__)


def ensure_symbol_ohlcv(
    symbol: str,
    *,
    paths: Paths,
    data_cfg: DataConfig,
    min_bars: int,
    since_days_default: int = 365 * 3,
) -> pd.DataFrame:
    """
    Ensure we have enough candles for a symbol in the local store.

    Behavior:
    - if symbol is missing: backfill since now - since_days_default
    - else: incremental update from last stored bar
    - if still not enough bars: backfill earlier (now - since_days_default) and re-check

    NOTE: This function performs network calls to Binance and may take seconds.
    """
    symbol = symbol.strip().upper()
    min_bars = int(max(0, min_bars))
    since_days_default = int(max(7, since_days_default))

    existing = read_ohlcv(paths, symbol)
    if existing.empty:
        start = (datetime.now(timezone.utc) - timedelta(days=since_days_default)).date()
        out = update_symbol_ohlcv(symbol=symbol, start_date=start, data_cfg=data_cfg, paths=paths)
    else:
        # incremental update
        out = update_symbol_ohlcv(symbol=symbol, start_date=date.today(), data_cfg=data_cfg, paths=paths)

    if min_bars <= 0:
        return out
    if len(out) >= min_bars:
        return out

    # Not enough history (common for newly listed tokens or too short backfill).
    # Try a deeper backfill.
    start = (datetime.now(timezone.utc) - timedelta(days=since_days_default)).date()
    out2 = update_symbol_ohlcv(symbol=symbol, start_date=start, data_cfg=data_cfg, paths=paths)
    if len(out2) < min_bars:
        log.warning("Still not enough bars for %s after backfill: have=%d need=%d", symbol, len(out2), min_bars)
    return out2


def update_symbol_ohlcv(symbol: str, start_date: date, data_cfg: DataConfig, paths: Paths) -> pd.DataFrame:
    """
    Backfill + incremental updates using Binance klines pagination (limit=1000).
    Stores as parquet in data/.
    """
    existing = read_ohlcv(paths, symbol)
    client = BinanceClient()

    if existing.empty:
        start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    else:
        # start after the last stored bar
        last = existing["open_time"].max().to_pydatetime()
        start_dt = last + timedelta(hours=4)

    start_ms = to_ms(start_dt)
    all_rows: list[pd.DataFrame] = []

    # Binance returns <= limit bars; paginate using startTime = last_open_time + 1
    while True:
        try:
            raw = client.fetch_klines(symbol=symbol, interval=data_cfg.interval, start_time_ms=start_ms, limit=1000)
        except ClientError as e:
            # Common user error: invalid symbol (-1121)
            raise ValueError(f"Invalid Binance symbol: {symbol}") from e
        if not raw:
            break

        df = _raw_klines_to_df(raw)
        if df.empty:
            break

        all_rows.append(df)
        # Use raw payload for pagination; openTime is raw[i][0] in milliseconds.
        last_open_ms = int(raw[-1][0])
        start_ms = last_open_ms + 1

        # stop if we received less than limit (no more data)
        if len(raw) < 1000:
            break

    if not all_rows:
        return existing

    new_df = pd.concat(all_rows, ignore_index=True)
    out = upsert_ohlcv(paths, symbol, new_df)
    _validate_no_large_gaps(out, expected_hours=4)
    return out


def _raw_klines_to_df(raw: list[list]) -> pd.DataFrame:
    cols = [
        "open_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time_ms",
        "quote_asset_volume",
        "n_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time_ms"], unit="ms", utc=True)
    out = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    return out


def _validate_no_large_gaps(df: pd.DataFrame, expected_hours: int) -> None:
    if df.empty:
        return
    ts = df["open_time"].sort_values()
    diffs = ts.diff().dropna()
    max_gap = diffs.max()
    if pd.isna(max_gap):
        return
    max_hours = max_gap / pd.Timedelta(hours=1)
    if max_hours > expected_hours * 2:
        log.warning("Detected large gap in candles: max_gap_hours=%.2f", float(max_hours))


