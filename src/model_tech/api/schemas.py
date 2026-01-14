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


class GapInfoMeta(BaseModel):
    interval: str
    price_bars_requested: int
    price_bars_returned: int
    dominance_bars_requested: int
    dominance_bars_returned: int
    min_gap_size: float


class GapInfoMarketRegimeCurrent(BaseModel):
    timestamp: str
    btc_d: float
    usdt_d: float
    btc_d_trend: Optional[float] = None
    usdt_d_trend: Optional[float] = None
    market_regime: str


class GapInfoGap(BaseModel):
    gap_type: str
    direction: str
    status: str
    created_at: str
    initial_low: float
    initial_high: float
    current_low: float
    current_high: float


class GapInfoOHLCVBar(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class GapInfoDominanceBar(BaseModel):
    timestamp: str
    btc_d: float
    usdt_d: float
    btc_d_trend: Optional[float] = None
    usdt_d_trend: Optional[float] = None
    market_regime: str


class GapInfoResponse(BaseModel):
    meta: GapInfoMeta
    market_regime_current: GapInfoMarketRegimeCurrent
    open_gaps: list[GapInfoGap]
    all_gaps: list[GapInfoGap]
    last_price_bar: GapInfoOHLCVBar
    price_history: Optional[list[GapInfoOHLCVBar]] = None
    dominance_history: Optional[list[GapInfoDominanceBar]] = None


