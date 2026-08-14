from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd


class DataFetchError(RuntimeError):
    """A market-data provider could not return usable candles."""


def drop_incomplete_bars(
    bars: pd.DataFrame,
    now: datetime,
    interval_minutes: int,
    timezone: str = "Asia/Kolkata",
) -> pd.DataFrame:
    """Remove an in-progress provider candle using its opening timestamp."""
    if bars.empty:
        return bars.copy()
    frame = bars.copy()
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize(timezone) if index.tz is None else index.tz_convert(timezone)
    frame.index = index
    local_now = now
    if local_now.tzinfo is None:
        local_now = local_now.replace(tzinfo=ZoneInfo(timezone))
    else:
        local_now = local_now.astimezone(ZoneInfo(timezone))
    cutoff = pd.Timestamp(local_now).floor("min") - pd.Timedelta(minutes=interval_minutes)
    return frame.loc[frame.index <= cutoff]
