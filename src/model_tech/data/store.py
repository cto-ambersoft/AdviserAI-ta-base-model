from __future__ import annotations

from pathlib import Path

import pandas as pd

from model_tech.config import Paths


def ensure_dirs(paths: Paths) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    paths.artifacts_dir.mkdir(parents=True, exist_ok=True)
    (paths.artifacts_dir / "models").mkdir(parents=True, exist_ok=True)


def symbol_ohlcv_path(paths: Paths, symbol: str) -> Path:
    return paths.data_dir / f"{symbol.upper()}_4h.parquet"


def read_ohlcv(paths: Paths, symbol: str) -> pd.DataFrame:
    p = symbol_ohlcv_path(paths, symbol)
    if not p.exists():
        return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume"])
    df = pd.read_parquet(p)
    return _normalize_ohlcv_df(df)


def write_ohlcv(paths: Paths, symbol: str, df: pd.DataFrame) -> None:
    p = symbol_ohlcv_path(paths, symbol)
    df = _normalize_ohlcv_df(df)
    df.to_parquet(p, index=False)


def upsert_ohlcv(paths: Paths, symbol: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge existing and new candles by open_time; newest wins.
    """
    cur = read_ohlcv(paths, symbol)
    if cur.empty:
        out = _normalize_ohlcv_df(new_df)
        write_ohlcv(paths, symbol, out)
        return out

    new_df = _normalize_ohlcv_df(new_df)
    out = pd.concat([cur, new_df], ignore_index=True)
    out = out.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last")
    out = _normalize_ohlcv_df(out)
    write_ohlcv(paths, symbol, out)
    return out


def _normalize_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if "open_time" in out.columns:
        out["open_time"] = pd.to_datetime(out["open_time"], utc=True)

    cols = ["open", "high", "low", "close", "volume"]
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    keep = ["open_time", "open", "high", "low", "close", "volume"]
    out = out[keep].sort_values("open_time").reset_index(drop=True)
    return out


