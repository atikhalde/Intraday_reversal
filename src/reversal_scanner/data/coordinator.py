from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Protocol

import pandas as pd

from ..models import Instrument
from .common import DataFetchError

LOGGER = logging.getLogger(__name__)


class PrimaryProvider(Protocol):
    def fetch(self, instrument: Instrument, now: datetime) -> pd.DataFrame: ...


class FallbackProvider(Protocol):
    def fetch_many(self, instruments: list[Instrument]) -> dict[str, pd.DataFrame]: ...


class MarketDataCoordinator:
    """Fetch Dhan concurrently once; immediately batch failures through Yahoo."""

    def __init__(
        self,
        primary: PrimaryProvider,
        fallback: FallbackProvider,
        max_workers: int = 32,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.max_workers = max_workers

    def fetch_all(
        self,
        instruments: list[Instrument],
        now: datetime,
    ) -> tuple[dict[str, tuple[pd.DataFrame, str]], dict[str, str]]:
        results: dict[str, tuple[pd.DataFrame, str]] = {}
        errors: dict[str, str] = {}
        failed: list[Instrument] = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.primary.fetch, instrument, now): instrument
                for instrument in instruments
            }
            for future in as_completed(futures):
                instrument = futures[future]
                try:
                    frame = future.result()
                    if frame.empty:
                        raise DataFetchError("empty frame")
                    results[instrument.symbol] = (frame, "dhan")
                except Exception:
                    # No Dhan retry and no backoff. This symbol enters Yahoo's next batch.
                    failed.append(instrument)

        if failed:
            LOGGER.warning(
                "Dhan unavailable for %d/%d symbols; falling back to yfinance immediately",
                len(failed),
                len(instruments),
            )
            yahoo_results = self.fallback.fetch_many(failed)
            for instrument in failed:
                frame = yahoo_results.get(instrument.symbol)
                if frame is not None and not frame.empty:
                    results[instrument.symbol] = (frame, "yfinance")
                else:
                    errors[instrument.symbol] = "both Dhan and yfinance failed"
        return results, errors
