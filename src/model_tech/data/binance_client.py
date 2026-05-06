from __future__ import annotations

import time
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional, Callable, TypeVar, ParamSpec
from functools import wraps

from binance.spot import Spot
from binance.error import ClientError

from model_tech.logging import get_logger

log = get_logger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def retry_with_backoff(
    retries: int = 5,
    backoff_in_seconds: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (ClientError, ConnectionError, TimeoutError),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to retry function calls with exponential backoff and jitter.
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            def _is_invalid_symbol_error(exc: BaseException) -> bool:
                """
                Binance Spot returns error code -1121 for invalid symbols.
                We should NOT retry those; they are deterministic user/config errors.
                """
                if not isinstance(exc, ClientError):
                    return False
                # binance-connector exposes these attributes on ClientError in most versions.
                code = getattr(exc, "error_code", None)
                if code is None:
                    code = getattr(exc, "code", None)
                if code == -1121:
                    return True
                msg = (getattr(exc, "error_message", None) or str(exc) or "").lower()
                return "invalid symbol" in msg or "-1121" in msg

            x = 0
            delay = backoff_in_seconds
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    # Deterministic errors must fail fast.
                    if _is_invalid_symbol_error(e):
                        raise

                    x += 1
                    if x > retries:
                        log.error(f"Function {func.__name__} failed after {retries} retries. Error: {e}")
                        raise

                    # Check for 429 Too Many Requests or 418 I'm a teapot (ban) specifically
                    if isinstance(e, ClientError):
                         # If we get a 418 or 429, we should respect Retry-After if present, but here we just backoff
                         pass
                    
                    sleep_time = min(delay * (2 ** (x - 1)), max_delay)
                    # Add jitter (0.5 to 1.5 multiplier)
                    sleep_time *= (0.5 + random.random())
                    
                    log.warning(
                        f"Retry {x}/{retries} for {func.__name__} due to {e}. "
                        f"Sleeping {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
        return wrapper
    return decorator


def to_ms(dt: datetime) -> int:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


@dataclass(frozen=True)
class BinanceClient:
    """
    Thin wrapper around binance-connector Spot REST client.
    Uses public endpoints only (no API keys required) for klines.
    """

    base_url: str = "https://api.binance.com"

    def __post_init__(self):
        # dataclass immutability: nothing
        pass

    def _spot(self) -> Spot:
        return Spot(base_url=self.base_url)

    @retry_with_backoff(retries=5, backoff_in_seconds=1.0)
    def fetch_klines(
        self,
        symbol: str,
        interval: str,
        start_time_ms: Optional[int] = None,
        end_time_ms: Optional[int] = None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        """
        Returns raw kline arrays from Binance.
        Each kline: [
          openTime, open, high, low, close, volume,
          closeTime, quoteAssetVolume, numberOfTrades,
          takerBuyBaseAssetVolume, takerBuyQuoteAssetVolume, ignore
        ]
        """
        params: dict[str, Any] = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        spot = self._spot()
        return spot.klines(**params)
