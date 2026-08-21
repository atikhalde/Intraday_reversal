import threading
import time
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from reversal_scanner.data.common import DataFetchError, drop_incomplete_bars
from reversal_scanner.data.coordinator import MarketDataCoordinator
from reversal_scanner.data.yahoo import YahooDataProvider
from reversal_scanner.historical import fetch_historical
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


def test_dhan_failure_falls_back_without_retry() -> None:
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
    # Failures can be grouped differently depending on which futures complete
    # together, but each symbol gets only one Dhan attempt and one Yahoo path.
    assert 1 <= fallback.calls <= len(instruments)
    assert set(results) == {"AAA", "BBB"}
    assert all(source == "yfinance" for _, source in results.values())
    assert errors == {}


class _FailAfterSlowPrimaryStarts:
    """Make the ordering assertion deterministic across worker scheduling."""

    def __init__(self) -> None:
        self.slow_started = threading.Event()
        self.slow_finished = threading.Event()

    def fetch(self, instrument, now):  # noqa: ANN001, ANN201
        if instrument.symbol == "AAA":
            assert self.slow_started.wait(timeout=1)
            raise DataFetchError("Dhan unavailable")
        self.slow_started.set()
        time.sleep(0.15)
        self.slow_finished.set()
        return sample_frame()


class _ImmediateFallback:
    def __init__(self, primary: _FailAfterSlowPrimaryStarts) -> None:
        self.primary = primary
        self.started_before_slow_dhan_finished = False

    def fetch_many(self, instruments):  # noqa: ANN001, ANN201
        self.started_before_slow_dhan_finished = not self.primary.slow_finished.is_set()
        return {instrument.symbol: sample_frame() for instrument in instruments}


def test_live_yahoo_fallback_starts_before_another_dhan_request_finishes() -> None:
    primary = _FailAfterSlowPrimaryStarts()
    fallback = _ImmediateFallback(primary)
    instruments = [
        Instrument("AAA", "1", "AAA.NS"),
        Instrument("BBB", "2", "BBB.NS"),
    ]

    results, errors = MarketDataCoordinator(primary, fallback, max_workers=2).fetch_all(
        instruments,
        datetime(2026, 8, 14, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata")),
    )

    assert fallback.started_before_slow_dhan_finished
    assert results["AAA"][1] == "yfinance"
    assert results["BBB"][1] == "dhan"
    assert errors == {}


def test_incomplete_current_bar_is_removed() -> None:
    now = datetime(2026, 8, 14, 9, 27, tzinfo=ZoneInfo("Asia/Kolkata"))
    completed = drop_incomplete_bars(sample_frame(), now, 5)
    assert list(completed.index.strftime("%H:%M")) == ["09:15", "09:20"]


class FailedHistoricalDhan:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_range(self, instrument, start, end):  # noqa: ANN001, ANN201
        self.calls += 1
        raise DataFetchError("one failed attempt")


class WorkingHistoricalYahoo:
    def __init__(self) -> None:
        self.calls = 0

    def fetch_range_many(self, instruments, start, end):  # noqa: ANN001, ANN201
        self.calls += 1
        return {instrument.symbol: sample_frame() for instrument in instruments}


def test_historical_dhan_failure_immediately_uses_yahoo() -> None:
    dhan = FailedHistoricalDhan()
    yahoo = WorkingHistoricalYahoo()
    instrument = Instrument("AAA", "1", "AAA.NS")
    results, errors = fetch_historical(
        instruments=[instrument],
        start_date=date(2026, 8, 14),
        end_date=date(2026, 8, 14),
        source="dhan",
        dhan=dhan,
        yahoo=yahoo,
        max_workers=1,
    )
    assert dhan.calls == 1
    assert yahoo.calls == 1
    assert results["AAA"][1] == "yfinance"
    assert errors == {}


class _HistoricalFailAfterSlowDhanStarts:
    def __init__(self) -> None:
        self.slow_started = threading.Event()
        self.slow_finished = threading.Event()

    def fetch_range(self, instrument, start, end):  # noqa: ANN001, ANN201
        if instrument.symbol == "AAA":
            assert self.slow_started.wait(timeout=1)
            raise DataFetchError("Dhan unavailable")
        self.slow_started.set()
        time.sleep(0.15)
        self.slow_finished.set()
        return sample_frame()


class _ImmediateHistoricalYahoo:
    def __init__(self, dhan: _HistoricalFailAfterSlowDhanStarts) -> None:
        self.dhan = dhan
        self.started_before_slow_dhan_finished = False

    def fetch_range_many(self, instruments, start, end):  # noqa: ANN001, ANN201
        self.started_before_slow_dhan_finished = not self.dhan.slow_finished.is_set()
        return {instrument.symbol: sample_frame() for instrument in instruments}


def test_historical_yahoo_fallback_starts_before_another_dhan_request_finishes() -> None:
    dhan = _HistoricalFailAfterSlowDhanStarts()
    yahoo = _ImmediateHistoricalYahoo(dhan)
    instruments = [
        Instrument("AAA", "1", "AAA.NS"),
        Instrument("BBB", "2", "BBB.NS"),
    ]

    results, errors = fetch_historical(
        instruments=instruments,
        start_date=date(2026, 8, 14),
        end_date=date(2026, 8, 14),
        source="dhan",
        dhan=dhan,
        yahoo=yahoo,
        max_workers=2,
    )

    assert yahoo.started_before_slow_dhan_finished
    assert results["AAA"][1] == "yfinance"
    assert results["BBB"][1] == "dhan"
    assert errors == {}


def test_yahoo_hard_deadline_skips_a_stuck_batch_without_retry(monkeypatch) -> None:  # noqa: ANN001
    calls: list[list[str]] = []

    def fake_download(**kwargs):  # noqa: ANN003, ANN202
        calls.append(kwargs["tickers"])
        if kwargs["tickers"] == ["AAA.NS"]:
            time.sleep(1)
        return sample_frame()

    monkeypatch.setattr("reversal_scanner.data.yahoo.yf.download", fake_download)
    provider = YahooDataProvider(
        timeout_seconds=0.01,
        batch_size=1,
        batch_timeout_seconds=0.05,
    )
    started = time.monotonic()
    results = provider.fetch_many(
        [Instrument("AAA", "1", "AAA.NS"), Instrument("BBB", "2", "BBB.NS")]
    )
    elapsed = time.monotonic() - started

    assert calls == [["AAA.NS"], ["BBB.NS"]]
    assert set(results) == {"BBB"}
    assert elapsed < 0.5
