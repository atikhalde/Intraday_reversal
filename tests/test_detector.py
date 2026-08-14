from pathlib import Path

import pandas as pd

import reversal_scanner.detector as detector
from reversal_scanner.config import load_config
from reversal_scanner.detector import detect_latest, scan_history

FIXTURE = Path(__file__).parent / "fixtures" / "orchpharma_2026-06-16_5m.csv"


def load_orchpharma() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["datetime_IST"]).set_index("datetime_IST")


def test_orchpharma_detects_exact_spring_test_sos_sequence() -> None:
    signals = scan_history("ORCHPHARMA", load_orchpharma(), load_config()["strategy"])
    expected_time = pd.Timestamp("2026-06-16 13:35")
    matching = [signal for signal in signals if signal.timestamp == expected_time]
    assert len(matching) == 1
    signal = matching[0]
    assert signal.pattern == "spring-test-SOS"
    assert signal.score == 100
    assert signal.spring_time == pd.Timestamp("2026-06-16 13:05")
    assert signal.spring_low == 900.0
    assert signal.pivot_high == 908.25
    assert signal.confirmation_price == pytest_approx(917.35, abs=0.01)
    assert signal.metrics["sos_body_ratio"] == pytest_approx(0.955, abs=0.002)
    assert signal.metrics["sos_previous_volume_ratio"] == pytest_approx(7.40, abs=0.02)


def test_scan_history_normalizes_the_full_range_only_once(monkeypatch) -> None:  # noqa: ANN001
    calls = 0
    original = detector.normalize_bars

    def counting_normalize(bars: pd.DataFrame) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return original(bars)

    monkeypatch.setattr(detector, "normalize_bars", counting_normalize)
    signals = scan_history("ORCHPHARMA", load_orchpharma(), load_config()["strategy"])

    assert calls == 1
    assert any(signal.timestamp == pd.Timestamp("2026-06-16 13:35") for signal in signals)


def test_vectorized_sos_prefilter_matches_single_candle_logic() -> None:
    frame = detector.normalize_bars(load_orchpharma())
    config = load_config()["strategy"]
    qualifying = detector._qualifying_sos_positions(frame, config)

    for index in range(1, len(frame)):
        assert bool(qualifying[index]) is detector._sos_qualifies(frame, index, config)


def test_walk_forward_keeps_trading_sessions_independent() -> None:
    first_session = load_orchpharma()
    second_session = first_session.copy()
    second_session.index += pd.Timedelta(days=1)

    signals = scan_history(
        "ORCHPHARMA",
        pd.concat([first_session, second_session]),
        load_config()["strategy"],
    )

    assert [signal.timestamp for signal in signals] == [
        pd.Timestamp("2026-06-16 13:35"),
        pd.Timestamp("2026-06-17 13:35"),
    ]


def test_no_signal_before_orchpharma_confirmation() -> None:
    frame = load_orchpharma().loc[:"2026-06-16 13:30"]
    assert detect_latest("ORCHPHARMA", frame, load_config()["strategy"]) is None


def test_no_lookahead_from_later_news_spike() -> None:
    frame = load_orchpharma()
    before_news = frame.loc[:"2026-06-16 13:35"]
    signal = detect_latest("ORCHPHARMA", before_news, load_config()["strategy"])
    assert signal is not None
    assert signal.timestamp == pd.Timestamp("2026-06-16 13:35")


def test_fast_hammer_confirmation_variant() -> None:
    index = pd.date_range("2026-08-14 09:15", periods=10, freq="5min")
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
    frame = pd.DataFrame(rows, index=index, columns=["open", "high", "low", "close", "volume"])
    signal = detect_latest("EXAMPLE", frame, load_config()["strategy"])
    assert signal is not None
    assert signal.pattern == "hammer-SOS"
    assert signal.spring_low == 91.0


def pytest_approx(value: float, **kwargs):  # noqa: ANN201
    # Keeps this fixture file import-light while still giving readable assertions.
    import pytest

    return pytest.approx(value, **kwargs)
