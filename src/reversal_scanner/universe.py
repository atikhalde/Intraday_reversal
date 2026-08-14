from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import pandas as pd

from .models import Instrument


def load_nifty500(path: str | Path | None = None) -> list[Instrument]:
    override = path or os.getenv("NIFTY500_UNIVERSE_PATH")
    source = (
        Path(override)
        if override
        else resources.files("reversal_scanner.resources").joinpath("nifty500_universe.csv")
    )
    with source.open("rb") as handle:
        frame = pd.read_csv(handle, dtype=str).fillna("")
    required = {"symbol", "security_id", "yfinance_symbol"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Universe is missing columns: {sorted(missing)}")
    if frame["symbol"].duplicated().any():
        raise ValueError("Universe contains duplicate NSE symbols")
    return [
        Instrument(
            symbol=row.symbol,
            security_id=row.security_id,
            yfinance_symbol=row.yfinance_symbol,
            isin=getattr(row, "isin", ""),
            company=getattr(row, "company", ""),
            industry=getattr(row, "industry", ""),
        )
        for row in frame.itertuples(index=False)
    ]
