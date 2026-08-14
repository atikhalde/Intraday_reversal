from __future__ import annotations

import logging
import os
import signal
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from tempfile import gettempdir

import pandas as pd
import yfinance as yf

from ..models import Instrument

LOGGER = logging.getLogger(__name__)


class YahooBatchTimeout(TimeoutError):
    """Raised when a yfinance batch does not honor its request timeout."""


@contextmanager
def _hard_batch_deadline(seconds: float) -> Iterator[None]:
    """Bound a yfinance batch on Unix; yfinance can otherwise wait indefinitely."""
    can_alarm = (
        hasattr(signal, "SIGALRM")
        and hasattr(signal, "setitimer")
        and threading.current_thread() is threading.main_thread()
    )
    if not can_alarm:
        yield
        return

    def timeout_handler(_signum, _frame) -> None:  # noqa: ANN001
        raise YahooBatchTimeout("yfinance batch exceeded its hard deadline")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


class YahooDataProvider:
    """Batch yfinance fallback used immediately after failed Dhan requests."""

    def __init__(
        self,
        timeout_seconds: float = 8.0,
        batch_size: int = 50,
        batch_timeout_seconds: float | None = None,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size
        self.batch_timeout_seconds = (
            batch_timeout_seconds
            if batch_timeout_seconds is not None
            else max(timeout_seconds * 2, 15.0)
        )
        cache_path = Path(gettempdir()) / f"reversal-scanner-yfinance-{os.getpid()}"
        cache_path.mkdir(parents=True, exist_ok=True)
        try:
            yf.set_tz_cache_location(str(cache_path))
        except Exception:  # pragma: no cover - compatibility with older yfinance releases
            LOGGER.debug("Could not set the optional yfinance timezone cache location")

    @staticmethod
    def _extract_ticker(raw: pd.DataFrame, ticker: str, only_ticker: bool) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()
        if not isinstance(raw.columns, pd.MultiIndex):
            selected = raw.copy() if only_ticker else pd.DataFrame()
        elif ticker in raw.columns.get_level_values(0):
            selected = raw[ticker].copy()
        elif ticker in raw.columns.get_level_values(-1):
            selected = raw.xs(ticker, axis=1, level=-1).copy()
        else:
            return pd.DataFrame()
        selected.columns = [str(column).lower().replace(" ", "_") for column in selected.columns]
        if "adj_close" in selected and "close" not in selected:
            selected["close"] = selected["adj_close"]
        needed = ["open", "high", "low", "close", "volume"]
        if not set(needed).issubset(selected.columns):
            return pd.DataFrame()
        selected = selected.loc[:, needed].dropna(how="any")
        index = pd.DatetimeIndex(selected.index)
        if index.tz is None:
            index = index.tz_localize("Asia/Kolkata")
        else:
            index = index.tz_convert("Asia/Kolkata")
        selected.index = index
        selected.index.name = "datetime"
        return selected.sort_index()

    def fetch_many(self, instruments: Sequence[Instrument]) -> dict[str, pd.DataFrame]:
        return self._download(instruments, period="5d")

    def fetch_range_many(
        self,
        instruments: Sequence[Instrument],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        return self._download(instruments, start=start, end=end)

    def _download(
        self,
        instruments: Sequence[Instrument],
        **date_arguments: str,
    ) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        total_batches = (len(instruments) + self.batch_size - 1) // self.batch_size
        for batch_number, offset in enumerate(range(0, len(instruments), self.batch_size), start=1):
            batch = list(instruments[offset : offset + self.batch_size])
            tickers = [instrument.yfinance_symbol for instrument in batch]
            LOGGER.info(
                "Starting yfinance batch %d/%d for %d symbol(s)",
                batch_number,
                total_batches,
                len(batch),
            )
            try:
                with _hard_batch_deadline(self.batch_timeout_seconds):
                    raw = yf.download(
                        tickers=tickers,
                        interval="5m",
                        group_by="ticker",
                        auto_adjust=False,
                        prepost=False,
                        threads=True,
                        progress=False,
                        timeout=self.timeout_seconds,
                        **date_arguments,
                    )
            except Exception as exc:  # yfinance raises backend-specific exception types
                LOGGER.warning(
                    "yfinance batch %d/%d failed its single attempt: %s",
                    batch_number,
                    total_batches,
                    exc,
                )
                continue
            loaded = 0
            for instrument in batch:
                frame = self._extract_ticker(raw, instrument.yfinance_symbol, len(batch) == 1)
                if not frame.empty:
                    results[instrument.symbol] = frame
                    loaded += 1
            LOGGER.info(
                "Completed yfinance batch %d/%d: %d/%d loaded",
                batch_number,
                total_batches,
                loaded,
                len(batch),
            )
        return results
