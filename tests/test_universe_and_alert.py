from datetime import datetime

from reversal_scanner.models import Signal
from reversal_scanner.telegram import TelegramNotifier
from reversal_scanner.universe import load_nifty500


def test_bundled_universe_has_exactly_500_unique_mapped_stocks() -> None:
    instruments = load_nifty500()
    assert len(instruments) == 500
    assert len({instrument.symbol for instrument in instruments}) == 500
    assert all(instrument.security_id for instrument in instruments)
    assert all(instrument.yfinance_symbol.endswith(".NS") for instrument in instruments)
    assert "RELIANCE" in {instrument.symbol for instrument in instruments}


def test_telegram_message_contains_actionable_structure() -> None:
    signal = Signal(
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
    message = TelegramNotifier.format_signal(signal)
    assert "ORCHPHARMA" in message
    assert "₹908.25" in message
    assert "₹899.80" in message
    assert "Data: dhan" in message
