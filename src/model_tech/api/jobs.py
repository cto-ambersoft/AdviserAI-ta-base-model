from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Callable, Optional

from concurrent.futures import Future, ThreadPoolExecutor


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    symbol: str
    status: str  # queued|running|succeeded|failed
    created_at_utc: str
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


class TrainJobQueue:
    """
    Minimal in-memory job queue backed by ThreadPoolExecutor.

    Note: With uvicorn workers>1, each worker has its own queue/state.
    """

    def __init__(self, max_workers: int = 1) -> None:
        self._exec = ThreadPoolExecutor(max_workers=int(max_workers))
        self._lock = Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._symbol_latest: dict[str, str] = {}

    def submit(self, symbol: str, fn: Callable[[], dict]) -> str:
        sym = symbol.strip().upper()
        job_id = str(uuid.uuid4())
        rec = JobRecord(job_id=job_id, symbol=sym, status="queued", created_at_utc=_utc_now())
        with self._lock:
            self._jobs[job_id] = rec
            self._symbol_latest[sym] = job_id

        def _run() -> None:
            with self._lock:
                rec.status = "running"
                rec.started_at_utc = _utc_now()
            try:
                result = fn()
                with self._lock:
                    rec.status = "succeeded"
                    rec.result = result
                    rec.finished_at_utc = _utc_now()
            except Exception as e:
                tb = traceback.format_exc()
                with self._lock:
                    rec.status = "failed"
                    rec.error = f"{type(e).__name__}: {e}\n{tb}"
                    rec.finished_at_utc = _utc_now()

        self._exec.submit(_run)
        return job_id

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def latest_for_symbol(self, symbol: str) -> Optional[JobRecord]:
        sym = symbol.strip().upper()
        with self._lock:
            jid = self._symbol_latest.get(sym)
            if jid is None:
                return None
            return self._jobs.get(jid)


