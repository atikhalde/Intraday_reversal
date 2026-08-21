from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
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
    """Fetch Dhan once and start Yahoo batches as failures arrive.

    Dhan and Yahoo are intentionally not retried.  A failed Dhan future is
    handed to a Yahoo batch while other Dhan futures are still in flight, so a
    slow or unavailable Dhan request never holds back the fallback for symbols
    that have already failed.
    """

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
        fallback_jobs: dict[Future[dict[str, pd.DataFrame]], list[Instrument]] = {}
        failed_count = 0

        # A single Yahoo worker retains batched downloads and avoids flooding
        # Yahoo when Dhan is broadly unavailable.  It is intentionally outside
        # the Dhan executor so its first batch can begin before slow Dhan
        # requests have completed.
        with ThreadPoolExecutor(max_workers=1) as fallback_executor:
            with ThreadPoolExecutor(max_workers=self.max_workers) as primary_executor:
                pending = {
                    primary_executor.submit(self.primary.fetch, instrument, now): instrument
                    for instrument in instruments
                }
                while pending:
                    completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                    # Coalesce every future that happened to finish at the same
                    # time into one Yahoo request, without adding a timed wait.
                    completed.update(future for future in pending if future.done())
                    failed_batch: list[Instrument] = []
                    for future in completed:
                        instrument = pending.pop(future)
                        try:
                            frame = future.result()
                            if frame.empty:
                                raise DataFetchError("empty frame")
                            results[instrument.symbol] = (frame, "dhan")
                        except Exception:
                            failed_batch.append(instrument)

                    if failed_batch:
                        failed_count += len(failed_batch)
                        LOGGER.warning(
                            "Dhan unavailable for %d symbol(s); starting yfinance fallback now",
                            len(failed_batch),
                        )
                        fallback_job = fallback_executor.submit(
                            self.fallback.fetch_many,
                            failed_batch,
                        )
                        fallback_jobs[fallback_job] = failed_batch

            for fallback_job in as_completed(fallback_jobs):
                failed_batch = fallback_jobs[fallback_job]
                try:
                    yahoo_results = fallback_job.result()
                except Exception:
                    yahoo_results = {}
                for instrument in failed_batch:
                    frame = yahoo_results.get(instrument.symbol)
                    if frame is not None and not frame.empty:
                        results[instrument.symbol] = (frame, "yfinance")
                    else:
                        errors[instrument.symbol] = "both Dhan and yfinance failed"

        if failed_count:
            LOGGER.warning(
                "Dhan unavailable for %d/%d symbols; yfinance fallback was started per "
                "completed failure batch",
                failed_count,
                len(instruments),
            )
        return results, errors
