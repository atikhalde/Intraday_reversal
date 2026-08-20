"""Tests for the selectivity rules that keep reversals rare and high-quality.

Covers:
- no false alerts on pure chop days (simulated);
- the mid-session confirmation window;
- the reclaim gate (shallow bounces are rejected);
- the tightened hammer path (stopping volume required);
- at most one signal per symbol per session;
- the portfolio daily alert budget in backtests; and
- the live scanner's daily budget and per-symbol deduplication.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from reversal_scanner.config import load_config
from reversal_scanner.detector import detect_latest, scan_history
from reversal_scanner.models import Instrument, Signal
from reversal_scanner.reporting import EvaluatedSignal, _cap_signals_per_day, evaluate_datasets
from reversal_scanner.scanner import ReversalScanner
from reversal_scanner.simulation import build_market
from reversal_scanner.state import SignalState


def _hammer_frame(shift_minutes: int = 0) -> pd.DataFrame:
    """The canonical fast hammer -> SOS confirmation fixture."""
    index = pd.date_range("2026-08-14 09:15", periods=10, freq="5min")
    if shift_minutes:
        index = index + pd.Timedelta(minutes=shift_minutes)
    rows = [
        (100.0, 100.5, 99.5, 100.0, 1000),
        (99.8, 100.0, 98.8, 99.0, 1000),
        (99.0, 99.2, 97.8, 98.0, 1100),
        (98.0, 98.2, 96.8, 97.0, 1000),
        (97.0, 97.2, 95.8, 96.0, 900),
        (96.0, 96.2, 94.8, 95.0, 1000),
        (95.0, 95.2, 93.8, 94.0, 900),
        (94.0, 94.2, 93.0, 93.5, 1000),
        (93.0, 93.2, 91.0, 92.8, 3000),
        (92.7, 95.2, 92.6, 95.0, 4200),
    ]
    return pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])


def test_pure_chop_days_produce_no_signals() -> None:
    config = load_config()
    datasets, _specs = build_market(3, 20, seed=1, regime_weights={"chop": 1.0})
    records = evaluate_datasets(
        datasets,
        config["strategy"],
        config["filters"],
        max_signals_per_day=99,
    )
    assert records == []


def test_confirmation_window_rejects_late_session_signals() -> None:
    strategy = load_config()["strategy"]
    # Same structure, but the confirmation candle closes at 15:20 - too late
    # for meaningful follow-through.
    assert detect_latest("LATE", _hammer_frame(shift_minutes=365), strategy) is None


def test_confirmation_window_rejects_opening_noise() -> None:
    strategy = load_config()["strategy"]
    # Confirmation at 09:40 is still inside opening noise.
    assert detect_latest("EARLY", _hammer_frame(shift_minutes=-35), strategy) is None


def test_reclaim_gate_rejects_shallow_bounces() -> None:
    frame = _hammer_frame()
    strategy = load_config()["strategy"]
    assert detect_latest("EXAMPLE", frame, strategy) is not None
    strict = dict(strategy, min_reclaim_pct_of_price=5.0)
    assert detect_latest("EXAMPLE", frame, strict) is None


def test_hammer_path_requires_stopping_volume() -> None:
    frame = _hammer_frame()
    strategy = load_config()["strategy"]
    # Without the sweep's volume surge the fast path is a falling knife.
    frame.loc[frame.index[-2], "volume"] = 1000
    assert detect_latest("EXAMPLE", frame, strategy) is None


def test_scan_history_emits_at_most_one_signal_per_symbol_per_session() -> None:
    fixture = Path(__file__).parent / "fixtures" / "orchpharma_2026-06-16_5m.csv"
    frame = pd.read_csv(fixture, parse_dates=["datetime_IST"]).set_index("datetime_IST")
    three_days = pd.concat(
        [frame, frame.copy(), frame.copy()],
        keys=[0, 1, 2],
    )
    three_days.index = three_days.index.map(
        lambda pair: pair[1] + pd.Timedelta(days=pair[0])
    )
    signals = scan_history("ORCHPHARMA", three_days, load_config()["strategy"])
    assert len(signals) == 3
    session_days = {signal.timestamp.date() for signal in signals}
    assert len(session_days) == 3


def _evaluated(signal: Signal) -> EvaluatedSignal:
    return EvaluatedSignal(signal, "1R reached", True, False, False, 1.0, 0.1, 10)


def _signal(symbol: str, at: datetime, score: int) -> Signal:
    return Signal(
        symbol=symbol,
        timestamp=at,
        pattern="spring-test-SOS",
        score=score,
        spring_time=at,
        spring_low=100.0,
        confirmation_price=110.0,
        pivot_high=105.0,
        immediate_failure=105.0,
        full_invalidation=99.0,
        target_1r=121.0,
        target_2r=132.0,
    )


def test_daily_budget_keeps_the_strongest_backtest_signals() -> None:
    day = datetime(2026, 6, 16, 13, 35)
    records = [
        _evaluated(_signal("AAA", day, 80)),
        _evaluated(_signal("BBB", day, 95)),
        _evaluated(_signal("CCC", day, 85)),
        _evaluated(_signal("DDD", day, 70)),
        _evaluated(_signal("EEE", day + pd.Timedelta(days=1), 70)),
    ]
    kept = _cap_signals_per_day(records, 3)
    kept_symbols = {record.signal.symbol for record in kept}
    assert kept_symbols == {"BBB", "CCC", "AAA", "EEE"}
    assert len(kept) == 4


class _StubCoordinator:
    def __init__(self, datasets: dict[str, pd.DataFrame]) -> None:
        self.datasets = datasets

    def fetch_all(self, instruments, now):  # noqa: ANN001, ANN202
        return (
            {symbol: (frame, "stub") for symbol, frame in self.datasets.items()},
            {},
        )


class _RecordingNotifier:
    def __init__(self) -> None:
        self.sent: list[Signal] = []

    def send(self, signal: Signal) -> None:
        self.sent.append(signal)


def test_live_scanner_respects_daily_budget_and_symbol_dedup(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "orchpharma_2026-06-16_5m.csv"
    frame = pd.read_csv(fixture, parse_dates=["datetime_IST"]).set_index("datetime_IST")
    # Shift the session onto today's date so the state store's seven-day
    # pruning keeps the deduplication keys during the test.
    today = datetime.now().date()
    offset = pd.Timestamp(today) - pd.Timestamp("2026-06-16")
    frame.index = frame.index + offset
    symbols = [f"STOCK{index}" for index in range(5)]
    coordinator = _StubCoordinator({symbol: frame for symbol in symbols})
    instruments = [
        Instrument(symbol=symbol, security_id=str(index), yfinance_symbol=f"{symbol}.NS")
        for index, symbol in enumerate(symbols)
    ]
    config = load_config()
    notifier = _RecordingNotifier()
    scanner = ReversalScanner(
        coordinator=coordinator,
        config=config,
        state=SignalState(tmp_path / "state.json"),
        notifier=notifier,
    )
    now = datetime(today.year, today.month, today.day, 13, 40)

    signals, errors = scanner.scan_once(instruments, now)
    assert errors == {}
    budget = int(config["scanner"]["max_signals_per_day"])
    assert len(signals) == budget == 3
    assert len(notifier.sent) == 3

    # A second scan on the same session adds nothing: the per-symbol day keys
    # and the daily budget both suppress repeats.
    repeat, _ = scanner.scan_once(instruments, now)
    assert repeat == []
    assert len(notifier.sent) == 3
