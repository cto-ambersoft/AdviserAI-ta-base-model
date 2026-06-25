"""
S6 — gap-info should be cached.

`/v1/gap-info` re-fetches ~3200 bars from TradingView (blocking, rate-limited) on
every call. A small TTL cache memoizes the payload. TTLCache is generic and clock-
injectable so the time behavior is deterministic in tests.
"""
from __future__ import annotations

import pytest

from model_tech.cache import TTLCache


def test_caches_within_ttl_then_recomputes() -> None:
    now = {"t": 1000.0}
    cache = TTLCache(ttl_seconds=60, clock=lambda: now["t"])
    calls = {"n": 0}

    def compute() -> int:
        calls["n"] += 1
        return calls["n"]

    assert cache.get_or_compute("k", compute) == 1   # cold -> compute
    assert cache.get_or_compute("k", compute) == 1   # warm -> cached
    assert calls["n"] == 1

    now["t"] = 1061.0                                  # past ttl
    assert cache.get_or_compute("k", compute) == 2     # expired -> recompute
    assert calls["n"] == 2


def test_distinct_keys_are_independent() -> None:
    cache = TTLCache(ttl_seconds=60, clock=lambda: 0.0)
    assert cache.get_or_compute("a", lambda: "A") == "A"
    assert cache.get_or_compute("b", lambda: "B") == "B"
    assert cache.get_or_compute("a", lambda: "X") == "A"  # still cached


def test_ttl_zero_disables_caching() -> None:
    cache = TTLCache(ttl_seconds=0)
    calls = {"n": 0}

    def compute() -> int:
        calls["n"] += 1
        return calls["n"]

    assert cache.get_or_compute("k", compute) == 1
    assert cache.get_or_compute("k", compute) == 2  # never cached


def test_errors_are_not_cached() -> None:
    cache = TTLCache(ttl_seconds=60, clock=lambda: 0.0)
    state = {"fail": True}

    def compute() -> int:
        if state["fail"]:
            raise ValueError("upstream down")
        return 42

    with pytest.raises(ValueError):
        cache.get_or_compute("k", compute)
    state["fail"] = False
    assert cache.get_or_compute("k", compute) == 42  # failure was not memoized
