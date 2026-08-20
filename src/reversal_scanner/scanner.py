from __future__ import annotations

import json
import logging
from datetime import date as date_type
from datetime import datetime, time, timedelta
from typing import Protocol
from zoneinfo import ZoneInfo

from .data.common import drop_incomplete_bars
from .data.coordinator import MarketDataCoordinator
from .detector import detect_latest
from .models import Instrument, Signal
from .state import SignalState

LOGGER = logging.getLogger(__name__)


class Notifier(Protocol):
    def send(self, signal: Signal) -> None: ...


def market_is_scannable(now: datetime, config: dict) -> bool:
    timezone = ZoneInfo(config["market_timezone"])
    local_now = now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)
    if local_now.weekday() >= 5:
        return False
    opening = time.fromisoformat(config["market_open"])
    closing = time.fromisoformat(config["market_close"])
    first_closed = (
        datetime.combine(local_now.date(), opening, timezone)
        + timedelta(minutes=int(config["interval_minutes"]))
    ).time()
    final_scan = (
        datetime.combine(local_now.date(), closing, timezone)
        + timedelta(minutes=int(config["interval_minutes"]))
    ).time()
    return first_closed <= local_now.time() <= final_scan


class ReversalScanner:
    def __init__(
        self,
        coordinator: MarketDataCoordinator,
        config: dict,
        state: SignalState,
        notifier: Notifier | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.config = config
        self.state = state
        self.notifier = notifier

    def scan_once(
        self,
        instruments: list[Instrument],
        now: datetime,
        dry_run: bool = False,
    ) -> tuple[list[Signal], dict[str, str]]:
        fetched, errors = self.coordinator.fetch_all(instruments, now)
        scanner_cfg = self.config["scanner"]
        strategy_cfg = self.config["strategy"]
        filters = self.config["filters"]
        local_now = (
            now.replace(tzinfo=ZoneInfo(scanner_cfg["market_timezone"]))
            if now.tzinfo is None
            else now.astimezone(ZoneInfo(scanner_cfg["market_timezone"]))
        )
        candidates: list[Signal] = []
        for instrument in instruments:
            item = fetched.get(instrument.symbol)
            if item is None:
                continue
            raw_bars, source = item
            bars = drop_incomplete_bars(
                raw_bars,
                local_now,
                int(scanner_cfg["interval_minutes"]),
                scanner_cfg["market_timezone"],
            )
            if bars.empty:
                errors[instrument.symbol] = "no completed candles"
                continue
            latest_price = float(bars.iloc[-1].close)
            if not float(filters["min_price"]) <= latest_price <= float(filters["max_price"]):
                continue
            session_date = bars.index[-1].date()
            session = bars[[timestamp.date() == session_date for timestamp in bars.index]]
            typical_price = (session.high + session.low + session.close) / 3
            turnover = float((typical_price * session.volume).sum())
            if turnover < float(filters["min_session_turnover_inr"]):
                continue

            signal = detect_latest(instrument.symbol, bars, strategy_cfg, source)
            if signal is None:
                continue
            confirmation_close = signal.timestamp + timedelta(
                minutes=int(scanner_cfg["interval_minutes"])
            )
            if confirmation_close.tzinfo is None:
                confirmation_close = confirmation_close.replace(
                    tzinfo=ZoneInfo(scanner_cfg["market_timezone"])
                )
            age_seconds = (local_now - confirmation_close).total_seconds()
            if not 0 <= age_seconds <= int(scanner_cfg["signal_max_age_seconds"]):
                continue
            if self.state.contains(signal.key):
                continue
            # Reversals are rare by design: one alert per symbol per session.
            day_symbol_key = f"day:{session_date.isoformat()}:{instrument.symbol}"
            if self.state.contains(day_symbol_key):
                continue
            candidates.append(signal)

        signals = self._apply_daily_budget(candidates, local_now.date(), scanner_cfg)
        for signal in signals:
            if dry_run:
                print(json.dumps(signal.to_dict(), ensure_ascii=False))
                continue
            if self.notifier is None:
                raise RuntimeError("A notifier is required unless --dry-run is used")
            self.notifier.send(signal)
            session_date = signal.timestamp.date()
            self.state.add(signal.key, signal.timestamp)
            self.state.add(
                f"day:{session_date.isoformat()}:{signal.symbol}",
                signal.timestamp,
            )
            LOGGER.info("Alerted %s at %s", signal.symbol, signal.timestamp.isoformat())
        return signals, errors

    def _apply_daily_budget(
        self,
        candidates: list[Signal],
        day: date_type,
        scanner_cfg: dict,
    ) -> list[Signal]:
        """Keep only the strongest confirmations within the daily alert budget."""
        if not candidates:
            return []
        budget = int(scanner_cfg.get("max_signals_per_day", 3))
        already_alerted = self.state.count_on_date(day, key_prefix="day:")
        remaining = max(0, budget - already_alerted)
        if remaining == 0:
            LOGGER.info(
                "Daily alert budget exhausted (%d/%d); %d new confirmation(s) suppressed",
                already_alerted,
                budget,
                len(candidates),
            )
            return []
        strongest = sorted(candidates, key=lambda item: (-item.score, item.timestamp))
        kept = strongest[:remaining]
        if len(kept) < len(candidates):
            LOGGER.info(
                "Daily alert budget keeps %d of %d confirmation(s)",
                len(kept),
                len(candidates),
            )
        return kept
