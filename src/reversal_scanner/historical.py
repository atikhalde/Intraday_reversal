from __future__ import annotations

import logging
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from .data.dhan import DhanDataProvider
from .data.yahoo import YahooDataProvider
from .models import Instrument

LOGGER = logging.getLogger(__name__)


def session_bounds(start_date: date, end_date: date) -> tuple[datetime, datetime]:
    if end_date < start_date:
        raise ValueError("end date must not be before start date")
    timezone = ZoneInfo("Asia/Kolkata")
    return (
        datetime.combine(start_date, time(9, 15), timezone),
        datetime.combine(end_date, time(15, 35), timezone),
    )


def fetch_historical(
    instruments: list[Instrument],
    start_date: date,
    end_date: date,
    source: str,
    dhan: DhanDataProvider | None,
    yahoo: YahooDataProvider,
    max_workers: int = 32,
) -> tuple[dict[str, tuple[pd.DataFrame, str]], dict[str, str]]:
    """Fetch a past range with Dhan-first, no-retry Yahoo fallback semantics."""
    start, end = session_bounds(start_date, end_date)
    results: dict[str, tuple[pd.DataFrame, str]] = {}
    errors: dict[str, str] = {}
    # Yahoo's end argument is exclusive. Its five-minute retention is limited,
    # so old ranges can legitimately remain unavailable after Dhan fails.
    yahoo_start = start_date.isoformat()
    yahoo_end = (end_date + timedelta(days=1)).isoformat()

    def merge_yahoo_batch(
        batch: list[Instrument],
        frames: dict[str, pd.DataFrame],
        unavailable_message: str,
    ) -> None:
        for instrument in batch:
            frame = frames.get(instrument.symbol)
            if frame is not None and not frame.empty:
                results[instrument.symbol] = (frame, "yfinance")
            else:
                errors[instrument.symbol] = unavailable_message

    if source == "yfinance":
        merge_yahoo_batch(
            list(instruments),
            yahoo.fetch_range_many(instruments, start=yahoo_start, end=yahoo_end),
            "historical range unavailable from yfinance",
        )
        return results, errors
    if source != "dhan":
        raise ValueError(f"Unsupported historical source: {source}")
    if dhan is None:
        raise ValueError("Dhan client is required for source=dhan")

    fallback_jobs: dict[Future[dict[str, pd.DataFrame]], list[Instrument]] = {}
    failed_count = 0
    # Start a queued Yahoo batch as soon as Dhan reports a failure.  Keeping
    # Yahoo to one worker preserves batched requests while avoiding a large
    # burst when a Dhan token or endpoint is unavailable.
    with ThreadPoolExecutor(max_workers=1) as fallback_executor:
        with ThreadPoolExecutor(max_workers=max_workers) as primary_executor:
            pending = {
                primary_executor.submit(dhan.fetch_range, instrument, start, end): instrument
                for instrument in instruments
            }
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                completed.update(future for future in pending if future.done())
                failed_batch: list[Instrument] = []
                for future in completed:
                    instrument = pending.pop(future)
                    try:
                        frame = future.result()
                        if frame.empty:
                            raise ValueError("empty frame")
                        results[instrument.symbol] = (frame, "dhan")
                    except Exception:
                        failed_batch.append(instrument)
                if failed_batch:
                    failed_count += len(failed_batch)
                    LOGGER.warning(
                        "Historical Dhan fetch failed for %d symbol(s); starting yfinance "
                        "fallback now",
                        len(failed_batch),
                    )
                    fallback_job = fallback_executor.submit(
                        yahoo.fetch_range_many,
                        failed_batch,
                        start=yahoo_start,
                        end=yahoo_end,
                    )
                    fallback_jobs[fallback_job] = failed_batch

        for fallback_job in as_completed(fallback_jobs):
            failed_batch = fallback_jobs[fallback_job]
            try:
                yahoo_frames = fallback_job.result()
            except Exception:
                yahoo_frames = {}
            merge_yahoo_batch(
                failed_batch,
                yahoo_frames,
                "historical range unavailable from both providers",
            )

    if failed_count:
        LOGGER.warning(
            "Historical Dhan fetch failed for %d/%d symbols; yfinance fallback was started "
            "per completed failure batch",
            failed_count,
            len(instruments),
        )
    return results, errors
