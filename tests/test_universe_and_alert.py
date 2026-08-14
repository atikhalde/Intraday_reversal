from datetime import datetime

from reversal_scanner.cli import _requested_symbols
from reversal_scanner.models import Signal
from reversal_scanner.telegram import TelegramNotifier
from reversal_scanner.universe import load_nifty500


def sample_signal() -> Signal:
    return Signal(
        symbol="ORCHPHARMA",
        timestamp=datetime(2026, 6, 16, 13, 35),
        pattern="spring-test-SOS",
        score=100,
        spring_time=datetime(2026, 6, 16, 13, 5),
        spring_low=900.0,
        confirmation_price=917.35,
        pivot_high=908.25,
        immediate_failure=903.5,
        full_invalidation=899.8,
        target_1r=934.9,
        target_2r=952.45,
        data_source="dhan",
        reasons=("swept sell-side liquidity", "higher-low test", "bullish displacement"),
    )


def test_bundled_universe_has_exactly_500_unique_mapped_stocks() -> None:
    instruments = load_nifty500()
    assert len(instruments) == 500
    assert len({instrument.symbol for instrument in instruments}) == 500
    assert all(instrument.security_id for instrument in instruments)
    assert all(instrument.yfinance_symbol.endswith(".NS") for instrument in instruments)
    assert "RELIANCE" in {instrument.symbol for instrument in instruments}


def test_nifty500_backtest_marker_expands_to_full_universe() -> None:
    symbols = _requested_symbols("NSE_NIFTY_500")
    assert len(symbols) == 500
    assert len(set(symbols)) == 500
    assert "RELIANCE" in symbols


def test_telegram_message_contains_actionable_structure() -> None:
    message = TelegramNotifier.format_signal(sample_signal())
    assert "ORCHPHARMA" in message
    assert "₹908.25" in message
    assert "₹899.80" in message
    assert "Data: dhan" in message


def test_telegram_sample_alert_is_clearly_marked(monkeypatch) -> None:  # noqa: ANN001
    captured: dict = {}

    class SuccessfulResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"ok": True}

    def fake_post(url, json, timeout):  # noqa: ANN001, ANN202
        captured.update(url=url, json=json, timeout=timeout)
        return SuccessfulResponse()

    monkeypatch.setattr("reversal_scanner.telegram.requests.post", fake_post)
    TelegramNotifier("test-token", "test-chat").send_test(sample_signal())
    text = captured["json"]["text"]
    assert "SAMPLE ALERT TEST — NOT A LIVE SIGNAL" in text
    assert "Historical ORCHPHARMA fixture replay" in text
    assert "CONFIRMED REVERSAL — ORCHPHARMA" in text
