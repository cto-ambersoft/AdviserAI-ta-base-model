from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    global_model_available: bool


class PredictResponse(BaseModel):
    symbol: str
    as_of: str
    signal: str
    confidence: float
    probs: dict[str, float]

    model_id_used: str = Field(..., description="global or SYMBOL")
    job_id: Optional[str] = Field(None, description="Training job id, if scheduled")


class JobStatusResponse(BaseModel):
    job_id: str
    symbol: str
    status: str
    created_at_utc: str
    started_at_utc: Optional[str] = None
    finished_at_utc: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


