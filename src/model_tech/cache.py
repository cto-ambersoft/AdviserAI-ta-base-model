from __future__ import annotations

import time
from threading import Lock
from typing import Any, Callable


class TTLCache:
    """
    Tiny thread-safe time-to-live cache with single-flight computation.

    `get_or_compute` returns the cached value while fresh, otherwise computes it
    (under the lock, so concurrent callers do not stampede the expensive source),
    stores it with an expiry, and returns it. Exceptions are never cached.

    `ttl_seconds <= 0` disables caching (always recompute) — handy as a kill switch.
    The clock is injectable for deterministic tests; production uses a monotonic clock.
    """

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = float(ttl_seconds)
        self._clock = clock
        self._lock = Lock()
        self._store: dict[Any, tuple[Any, float]] = {}

    def get_or_compute(self, key: Any, compute: Callable[[], Any]) -> Any:
        if self._ttl <= 0:
            return compute()
        with self._lock:
            now = self._clock()
            hit = self._store.get(key)
            if hit is not None and now < hit[1]:
                return hit[0]
            value = compute()
            self._store[key] = (value, now + self._ttl)
            return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
