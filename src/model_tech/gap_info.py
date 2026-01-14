from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

import pandas as pd

try:
    # tvdatafeed exposes module name tvDatafeed
    from tvDatafeed import Interval, TvDatafeed
except Exception as e:  # pragma: no cover
    Interval = None  # type: ignore[assignment]
    TvDatafeed = None  # type: ignore[assignment]
    _TV_IMPORT_ERROR = e
else:
    _TV_IMPORT_ERROR = None


class GapInfoError(RuntimeError):
    """Base error for gap-info collection."""


class GapInfoDependencyError(GapInfoError):
    """Raised when optional dependency is missing."""


class GapInfoUpstreamError(GapInfoError):
    """Raised when TradingView data cannot be fetched."""


def _require_tvdatafeed() -> None:
    if _TV_IMPORT_ERROR is not None or TvDatafeed is None:
        raise GapInfoDependencyError(
            "tvdatafeed is not available. Install it with: "
            "pip install git+https://github.com/rongardF/tvdatafeed.git"
        ) from _TV_IMPORT_ERROR


def load_cme_btc(
    *,
    interval=None,
    bars: int = 2000,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    _require_tvdatafeed()
    if interval is None:
        interval = Interval.in_4_hour

    tv = TvDatafeed()
    # Keep the original script's fallback behavior, but add a fut_contract-based option
    # which is the recommended way to fetch continuous futures in tvdatafeed.
    candidates: list[tuple[str, str, dict[str, Any]]] = [
        ("BTC", "CME", {"fut_contract": 1}),
        ("BTC1!", "CME", {}),
        ("BTC", "CME", {}),
        ("BTCUSD", "CME", {}),
    ]

    last_err: Exception | None = None
    for _attempt in range(1, max(1, retries) + 1):
        for symbol, exchange, extra in candidates:
            try:
                df = tv.get_hist(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    n_bars=int(bars),
                    **extra,
                )
                if df is None or df.empty:
                    continue
                out = df.reset_index().rename(columns={"datetime": "timestamp"})
                out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
                # Normalize to the exact columns used by the script.
                cols = ["timestamp", "open", "high", "low", "close", "volume"]
                missing = [c for c in cols if c not in out.columns]
                if missing:
                    raise GapInfoUpstreamError(f"unexpected columns from tvdatafeed: missing={missing}")
                return out[cols].copy()
            except Exception as e:
                last_err = e
                continue
        time.sleep(max(0.0, float(sleep_seconds)))

    raise GapInfoUpstreamError(f"TradingView CME BTC data not available: {last_err}")


def load_dominance(
    *,
    symbol: str,
    bars: int = 1200,
    retries: int = 3,
    sleep_seconds: float = 2.0,
) -> pd.DataFrame:
    _require_tvdatafeed()

    tv = TvDatafeed()
    last_err: Exception | None = None
    for _attempt in range(1, max(1, retries) + 1):
        try:
            df = tv.get_hist(
                symbol=symbol,
                exchange="CRYPTOCAP",
                interval=Interval.in_daily,
                n_bars=int(bars),
            )
            if df is None or df.empty:
                raise GapInfoUpstreamError("empty dataframe")

            out = df.reset_index().rename(columns={"datetime": "timestamp", "close": "value"})
            out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
            if "value" not in out.columns:
                raise GapInfoUpstreamError("unexpected dominance schema: no close/value column")
            return out[["timestamp", "value"]].copy()
        except Exception as e:
            last_err = e
            time.sleep(max(0.0, float(sleep_seconds)))

    raise GapInfoUpstreamError(f"TradingView dominance data not available: {symbol}: {last_err}")


def define_market_regime(btc_d_trend: float, usdt_d_trend: float, eps: float = 0.0005) -> str:
    """
    Formal conditions (same logic as gap_detector_v2.py):

    BTC_STRENGTH : BTC.D (UP)   + USDT.D (DOWN)
    ALTSEASON    : BTC.D (DOWN) + USDT.D (DOWN)
    STRESS       : BTC.D (UP)   + USDT.D (UP)
    CASH_OUT     : BTC.D (DOWN) + USDT.D (UP)
    NEUTRAL      : FLAT
    """

    if btc_d_trend > eps and usdt_d_trend < -eps:
        return "BTC_STRENGTH"
    if btc_d_trend < -eps and usdt_d_trend < -eps:
        return "ALTSEASON"
    if btc_d_trend > eps and usdt_d_trend > eps:
        return "STRESS"
    if btc_d_trend < -eps and usdt_d_trend > eps:
        return "CASH_OUT"
    return "NEUTRAL"


Direction = Literal["UP", "DOWN"]
GapStatus = Literal["active", "partial", "filled"]
GapType = Literal["CME_WEEKEND", "CME_SESSION"]


@dataclass
class CMEGap:
    gap_type: GapType
    direction: Direction
    initial_low: float
    initial_high: float
    created_at: datetime
    status: GapStatus = "active"
    current_low: float | None = None
    current_high: float | None = None

    def __post_init__(self) -> None:
        if self.current_low is None:
            self.current_low = float(self.initial_low)
        if self.current_high is None:
            self.current_high = float(self.initial_high)

    def update(self, bar: Any) -> None:
        if self.status == "filled":
            return

        low = float(bar["low"])
        high = float(bar["high"])

        if self.direction == "UP":
            if low <= float(self.current_low):
                self.status = "filled"
            elif low < float(self.current_high):
                self.current_low = max(float(self.current_low), low)
                self.status = "partial"
            return

        # DOWN
        if high >= float(self.current_high):
            self.status = "filled"
        elif high > float(self.current_low):
            self.current_high = min(float(self.current_high), high)
            self.status = "partial"

    def is_open(self) -> bool:
        return self.status in ("active", "partial")

    def current_range(self) -> tuple[float, float]:
        return float(self.current_low), float(self.current_high)


class CMEGapManager:
    def __init__(self, *, min_gap_size: float = 100.0):
        self.gaps: list[CMEGap] = []
        self.min_gap_size = float(min_gap_size)

    def detect_new_gap(self, prev_bar: Any, cur_bar: Any) -> None:
        gap_up = float(cur_bar["low"]) > float(prev_bar["high"])
        gap_down = float(cur_bar["high"]) < float(prev_bar["low"])

        if not (gap_up or gap_down):
            return

        if gap_up:
            low, high, direction = float(prev_bar["high"]), float(cur_bar["low"]), "UP"
        else:
            low, high, direction = float(cur_bar["high"]), float(prev_bar["low"]), "DOWN"

        if abs(high - low) < self.min_gap_size:
            return

        prev_ts = pd.to_datetime(prev_bar["timestamp"], utc=True).to_pydatetime()
        cur_ts = pd.to_datetime(cur_bar["timestamp"], utc=True).to_pydatetime()
        gap_type: GapType = (
            "CME_WEEKEND"
            if prev_ts.weekday() == 4 and cur_ts.weekday() in (6, 0)
            else "CME_SESSION"
        )

        self.gaps.append(
            CMEGap(
                gap_type=gap_type,
                direction=direction,  # type: ignore[arg-type]
                initial_low=low,
                initial_high=high,
                created_at=cur_ts,
            )
        )

    def update_with_bar(self, bar: Any) -> None:
        for g in self.gaps:
            g.update(bar)

    def open_gaps(self) -> list[CMEGap]:
        return [g for g in self.gaps if g.is_open()]


def _dt_iso(dt: Any) -> str:
    ts = pd.to_datetime(dt, utc=True)
    # Ensure RFC3339-ish output with timezone
    return ts.to_pydatetime().isoformat()


def _row_to_ohlcv(r: Any) -> dict[str, Any]:
    return {
        "timestamp": _dt_iso(r["timestamp"]),
        "open": float(r["open"]),
        "high": float(r["high"]),
        "low": float(r["low"]),
        "close": float(r["close"]),
        "volume": float(r["volume"]),
    }


def build_gap_info(
    *,
    min_gap_size: float = 100.0,
) -> dict[str, Any]:
    # Fixed defaults per request:
    # - load_cme_btc(interval=Interval.in_4_hour, bars=2000)
    # - load_dominance(symbol, bars=1200)
    price_df = load_cme_btc(interval=Interval.in_4_hour, bars=2000)
    btc_d = load_dominance(symbol="BTC.D", bars=1200).rename(columns={"value": "btc_d"})
    usdt_d = load_dominance(symbol="USDT.D", bars=1200).rename(columns={"value": "usdt_d"})

    dom = btc_d.merge(usdt_d, on="timestamp", how="inner")
    dom["btc_d_trend"] = dom["btc_d"].diff()
    dom["usdt_d_trend"] = dom["usdt_d"].diff()
    dom["market_regime"] = dom.apply(
        lambda r: define_market_regime(float(r["btc_d_trend"]), float(r["usdt_d_trend"])),
        axis=1,
    )

    gap_manager = CMEGapManager(min_gap_size=float(min_gap_size))
    for i in range(1, len(price_df)):
        prev = price_df.iloc[i - 1]
        cur = price_df.iloc[i]
        gap_manager.detect_new_gap(prev, cur)
        gap_manager.update_with_bar(cur)

    last_price = _row_to_ohlcv(price_df.iloc[-1])
    last_dom = dom.iloc[-1]
    market_regime_current = {
        "timestamp": _dt_iso(last_dom["timestamp"]),
        "btc_d": float(last_dom["btc_d"]),
        "usdt_d": float(last_dom["usdt_d"]),
        "btc_d_trend": float(last_dom["btc_d_trend"]) if pd.notna(last_dom["btc_d_trend"]) else None,
        "usdt_d_trend": float(last_dom["usdt_d_trend"]) if pd.notna(last_dom["usdt_d_trend"]) else None,
        "market_regime": str(last_dom["market_regime"]),
    }

    def _gap_to_dict(g: CMEGap) -> dict[str, Any]:
        cur_low, cur_high = g.current_range()
        return {
            "gap_type": g.gap_type,
            "direction": g.direction,
            "status": g.status,
            "created_at": g.created_at.isoformat(),
            "initial_low": float(g.initial_low),
            "initial_high": float(g.initial_high),
            "current_low": float(cur_low),
            "current_high": float(cur_high),
        }

    all_gaps = [_gap_to_dict(g) for g in gap_manager.gaps]
    open_gaps = [_gap_to_dict(g) for g in gap_manager.open_gaps()]

    payload: dict[str, Any] = {
        "meta": {
            "interval": "4h",
            "price_bars_requested": 2000,
            "price_bars_returned": int(len(price_df)),
            "dominance_bars_requested": 1200,
            "dominance_bars_returned": int(len(dom)),
            "min_gap_size": float(min_gap_size),
        },
        "market_regime_current": market_regime_current,
        "open_gaps": open_gaps,
        "all_gaps": all_gaps,
        "last_price_bar": last_price,
    }

    # Always include histories (defaults = true).
    payload["price_history"] = [_row_to_ohlcv(r) for _, r in price_df.iterrows()]
    payload["dominance_history"] = [
        {
            "timestamp": _dt_iso(r["timestamp"]),
            "btc_d": float(r["btc_d"]),
            "usdt_d": float(r["usdt_d"]),
            "btc_d_trend": float(r["btc_d_trend"]) if pd.notna(r["btc_d_trend"]) else None,
            "usdt_d_trend": float(r["usdt_d_trend"]) if pd.notna(r["usdt_d_trend"]) else None,
            "market_regime": str(r["market_regime"]),
        }
        for _, r in dom.iterrows()
    ]

    return payload

