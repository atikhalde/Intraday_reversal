from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import yfinance as yf

from ..models import Instrument


class YahooDataProvider:
    """Batch yfinance fallback used immediately after failed Dhan requests."""

    def __init__(self, timeout_seconds: float = 8.0, batch_size: int = 50) -> None:
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size

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
        for offset in range(0, len(instruments), self.batch_size):
            batch = list(instruments[offset : offset + self.batch_size])
            tickers = [instrument.yfinance_symbol for instrument in batch]
            try:
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
            except Exception:  # yfinance raises several backend-specific exception types
                continue
            for instrument in batch:
                frame = self._extract_ticker(raw, instrument.yfinance_symbol, len(batch) == 1)
                if not frame.empty:
                    results[instrument.symbol] = frame
        return results
