"""
S7 — the in-memory training queue must be bounded and shut down cleanly.

Previously the job dict grew without limit (a slow memory leak for a long-running
API) and the ThreadPoolExecutor was never shut down on app shutdown. Now the
history is FIFO-capped and the queue can be shut down (rejecting new submits).
"""
from __future__ import annotations

import pytest

from model_tech.api.jobs import TrainJobQueue


def test_job_history_is_bounded_fifo() -> None:
    q = TrainJobQueue(max_workers=2, max_history=3)
    try:
        ids = [q.submit("BTCUSDT", lambda: {}) for _ in range(5)]
        assert len(q._jobs) <= 3
        assert q.get(ids[0]) is None      # oldest evicted
        assert q.get(ids[1]) is None
        assert q.get(ids[4]) is not None  # newest retained
    finally:
        q.shutdown(wait=True)


def test_shutdown_rejects_new_submits() -> None:
    q = TrainJobQueue(max_workers=1)
    q.shutdown(wait=True)
    with pytest.raises(RuntimeError):
        q.submit("BTCUSDT", lambda: {})


def test_shutdown_is_idempotent() -> None:
    q = TrainJobQueue(max_workers=1)
    q.shutdown(wait=True)
    q.shutdown(wait=True)  # must not raise
