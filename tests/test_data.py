from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from reversal_scanner.data.common import DataFetchError, drop_incomplete_bars
from reversal_scanner.data.coordinator import MarketDataCoordinator
from reversal_scanner.models import Instrument


def sample_frame() -> pd.DataFrame:
    index = pd.date_range("2026-08-14 09:15", periods=3, freq="5min", tz="Asia/Kolkata")
    return pd.DataFrame(
        {
            "open": [100, 101, 102],
            "high": [102, 103, 104],
            "low": [99, 100, 101],
            "close": [101, 102, 103],
            "volume": [1000, 1100, 1200],
        },
        index=index,
    )


class FailedPrimary:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def fetch(self, instrument, now):  # noqa: ANN001, ANN201
        self.calls.append(instrument.symbol)
        raise DataFetchError("fail once")


class WorkingFallback:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_many(self, instruments):  # noqa: ANN001, ANN201
        self.calls += 1
        return {instrument.symbol: sample_frame() for instrument in instruments}


def test_dhan_failure_falls_back_once_without_retry() -> None:
    primary = FailedPrimary()
    fallback = WorkingFallback()
    instruments = [
        Instrument("AAA", "1", "AAA.NS"),
        Instrument("BBB", "2", "BBB.NS"),
    ]
    coordinator = MarketDataCoordinator(primary, fallback, max_workers=2)
    results, errors = coordinator.fetch_all(
        instruments,
        datetime(2026, 8, 14, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )
    assert sorted(primary.calls) == ["AAA", "BBB"]
    assert fallback.calls == 1
    assert set(results) == {"AAA", "BBB"}
    assert all(source == "yfinance" for _, source in results.values())
    assert errors == {}


def test_incomplete_current_bar_is_removed() -> None:
    now = datetime(2026, 8, 14, 9, 27, tzinfo=ZoneInfo("Asia/Kolkata"))
    completed = drop_incomplete_bars(sample_frame(), now, 5)
    assert list(completed.index.strftime("%H:%M")) == ["09:15", "09:20"]
