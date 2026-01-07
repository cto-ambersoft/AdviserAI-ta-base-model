from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Signal(str, Enum):
    SELL = "SELL"
    HOLD = "HOLD"
    BUY = "BUY"


@dataclass(frozen=True)
class Prediction:
    symbol: str
    as_of: str
    signal: Signal
    confidence: float
    probs: dict[str, float]


