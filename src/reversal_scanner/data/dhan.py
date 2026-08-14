from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..models import Instrument
from .common import DataFetchError


class DhanDataProvider:
    """Thin no-retry client for Dhan's v2 intraday candle endpoint."""

    endpoint = "https://api.dhan.co/v2/charts/intraday"

    def __init__(
        self,
        client_id: str,
        access_token: str,
        timeout_seconds: float = 1.0,
        interval_minutes: int = 5,
        lookback_days: int = 2,
        session: requests.Session | None = None,
    ) -> None:
        self.client_id = client_id
        self.access_token = access_token
        self.timeout_seconds = timeout_seconds
        self.interval_minutes = interval_minutes
        self.lookback_days = lookback_days
        self.session = session or requests.Session()
        # Explicitly disable urllib3 retries: one Dhan failure goes directly to Yahoo.
        no_retry = Retry(total=0, connect=0, read=0, redirect=0, status=0)
        self.session.mount("https://", HTTPAdapter(max_retries=no_retry))

    def fetch(self, instrument: Instrument, now: datetime) -> pd.DataFrame:
        timezone = ZoneInfo("Asia/Kolkata")
        local_now = now.replace(tzinfo=timezone) if now.tzinfo is None else now.astimezone(timezone)
        start = (local_now - timedelta(days=self.lookback_days)).replace(
            hour=9,
            minute=15,
            second=0,
            microsecond=0,
        )
        return self.fetch_range(instrument, start, local_now)

    def fetch_range(
        self,
        instrument: Instrument,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        timezone = ZoneInfo("Asia/Kolkata")
        local_start = (
            start.replace(tzinfo=timezone) if start.tzinfo is None else start.astimezone(timezone)
        )
        local_end = end.replace(tzinfo=timezone) if end.tzinfo is None else end.astimezone(timezone)
        payload: dict[str, Any] = {
            "securityId": instrument.security_id,
            "exchangeSegment": "NSE_EQ",
            "instrument": "EQUITY",
            "interval": str(self.interval_minutes),
            "oi": False,
            "fromDate": local_start.strftime("%Y-%m-%d %H:%M:%S"),
            "toDate": local_end.strftime("%Y-%m-%d %H:%M:%S"),
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id,
            "User-Agent": "IntradayReversalScanner/0.1",
        }
        try:
            response = self.session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=(self.timeout_seconds, self.timeout_seconds),
            )
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            # Do not include headers, response bodies, or credential-bearing details.
            raise DataFetchError(f"Dhan failed for {instrument.symbol}") from exc

        try:
            timestamps = data["timestamp"]
            frame = pd.DataFrame(
                {
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["close"],
                    "volume": data["volume"],
                },
                index=pd.to_datetime(timestamps, unit="s", utc=True).tz_convert("Asia/Kolkata"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            message = f"Dhan returned malformed candles for {instrument.symbol}"
            raise DataFetchError(message) from exc
        if frame.empty:
            raise DataFetchError(f"Dhan returned no candles for {instrument.symbol}")
        frame.index.name = "datetime"
        return frame.sort_index()
