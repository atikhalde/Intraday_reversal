from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    failed = list(instruments)

    if source == "dhan":
        if dhan is None:
            raise ValueError("Dhan client is required for source=dhan")
        failed = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(dhan.fetch_range, instrument, start, end): instrument
                for instrument in instruments
            }
            for future in as_completed(futures):
                instrument = futures[future]
                try:
                    frame = future.result()
                    if frame.empty:
                        raise ValueError("empty frame")
                    results[instrument.symbol] = (frame, "dhan")
                except Exception:
                    failed.append(instrument)
        if failed:
            LOGGER.warning(
                "Historical Dhan fetch failed for %d/%d symbols; using yfinance immediately",
                len(failed),
                len(instruments),
            )
    elif source != "yfinance":
        raise ValueError(f"Unsupported historical source: {source}")

    if failed:
        # Yahoo's end argument is exclusive. Its five-minute retention is limited,
        # so old ranges can legitimately remain unavailable after Dhan fails.
        yahoo_end = (end_date + timedelta(days=1)).isoformat()
        yahoo_frames = yahoo.fetch_range_many(
            failed,
            start=start_date.isoformat(),
            end=yahoo_end,
        )
        for instrument in failed:
            frame = yahoo_frames.get(instrument.symbol)
            if frame is not None and not frame.empty:
                results[instrument.symbol] = (frame, "yfinance")
            else:
                errors[instrument.symbol] = (
                    "historical range unavailable from yfinance"
                    if source == "yfinance"
                    else "historical range unavailable from both providers"
                )
    return results, errors
