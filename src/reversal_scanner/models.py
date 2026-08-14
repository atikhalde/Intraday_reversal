from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class Instrument:
    symbol: str
    security_id: str
    yfinance_symbol: str
    isin: str = ""
    company: str = ""
    industry: str = ""


@dataclass(frozen=True, slots=True)
class Signal:
    symbol: str
    timestamp: datetime
    pattern: str
    score: int
    spring_time: datetime
    spring_low: float
    confirmation_price: float
    pivot_high: float
    immediate_failure: float
    full_invalidation: float
    target_1r: float
    target_2r: float
    data_source: str = "unknown"
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.symbol}:{self.timestamp.isoformat()}:{self.pattern}"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        payload["spring_time"] = self.spring_time.isoformat()
        return payload
