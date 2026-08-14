from __future__ import annotations

import html

import requests

from .models import Signal


class TelegramNotifier:
    def __init__(self, bot_token: str, chat_id: str, timeout_seconds: float = 4.0) -> None:
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._timeout_seconds = timeout_seconds

    @staticmethod
    def format_signal(signal: Signal) -> str:
        source = html.escape(signal.data_source)
        pattern = html.escape(signal.pattern)
        symbol = html.escape(signal.symbol)
        timestamp = signal.timestamp.strftime("%d %b %Y %H:%M %Z").strip()
        reasons = "\n".join(f"• {html.escape(reason)}" for reason in signal.reasons[:6])
        return (
            f"🚨 <b>CONFIRMED REVERSAL — {symbol}</b>\n"
            f"<b>{pattern}</b> | Score <b>{signal.score}/100</b>\n"
            f"🕒 {timestamp}\n"
            f"✅ Confirmation: ₹{signal.confirmation_price:.2f}\n"
            f"↗️ Broken pivot / retest: ₹{signal.pivot_high:.2f}\n"
            f"⚠️ Immediate failure: ₹{signal.immediate_failure:.2f}\n"
            f"🛑 Full invalidation: ₹{signal.full_invalidation:.2f}\n"
            f"🎯 1R / 2R: ₹{signal.target_1r:.2f} / ₹{signal.target_2r:.2f}\n"
            f"📡 Data: {source}\n\n"
            f"<b>Why it fired</b>\n{reasons}\n\n"
            "Signal only—not financial advice. Wait for your execution rules."
        )

    def send(self, signal: Signal) -> None:
        # The token is necessarily in Telegram's bot URL; never log this URL or raw exceptions.
        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": self._chat_id,
                    "text": self.format_signal(signal),
                    "parse_mode": "HTML",
                    "disable_web_page_preview": True,
                },
                timeout=self._timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not payload.get("ok"):
                raise RuntimeError("Telegram rejected the alert")
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError("Telegram alert delivery failed") from exc
