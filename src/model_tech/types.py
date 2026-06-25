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
    confidence: float  # probability of the emitted class
    probs: dict[str, float]
    max_prob: float  # raw top probability before the min_conf rule
    forced_hold: bool  # True if argmax != HOLD but min_conf downgraded it to HOLD


