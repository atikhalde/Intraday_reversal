#!/usr/bin/env python3
"""Refresh the committed Nifty 500 -> Dhan security-ID mapping."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

NIFTY_URL = "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
DHAN_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"
ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = (
    ROOT / "data" / "nifty500_universe.csv",
    ROOT / "src" / "reversal_scanner" / "resources" / "nifty500_universe.csv",
)


def main() -> None:
    nse_headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.nseindia.com/",
    }
    nifty_response = requests.get(NIFTY_URL, headers=nse_headers, timeout=20)
    nifty_response.raise_for_status()
    dhan_response = requests.get(
        DHAN_URL,
        headers={"User-Agent": "IntradayReversalScanner/0.1"},
        timeout=45,
    )
    dhan_response.raise_for_status()
    nifty = pd.read_csv(io.BytesIO(nifty_response.content), dtype=str)
    dhan = pd.read_csv(io.BytesIO(dhan_response.content), dtype=str, low_memory=False)
    dhan = dhan[
        (dhan.SEM_EXM_EXCH_ID == "NSE")
        & (dhan.SEM_SEGMENT == "E")
        & (dhan.SEM_INSTRUMENT_NAME == "EQUITY")
    ].copy()
    dhan["_rank"] = dhan.SEM_SERIES.map({"EQ": 0, "BE": 1, "BZ": 2, "SM": 3}).fillna(9)
    dhan = dhan.sort_values("_rank").drop_duplicates("SEM_TRADING_SYMBOL")
    merged = nifty.merge(
        dhan[["SEM_TRADING_SYMBOL", "SEM_SMST_SECURITY_ID"]],
        left_on="Symbol",
        right_on="SEM_TRADING_SYMBOL",
        how="left",
    )
    missing = merged.loc[merged.SEM_SMST_SECURITY_ID.isna(), "Symbol"].tolist()
    if missing:
        raise RuntimeError(f"Missing Dhan IDs for Nifty symbols: {missing}")
    output = merged[
        ["Symbol", "SEM_SMST_SECURITY_ID", "ISIN Code", "Company Name", "Industry"]
    ].rename(
        columns={
            "Symbol": "symbol",
            "SEM_SMST_SECURITY_ID": "security_id",
            "ISIN Code": "isin",
            "Company Name": "company",
            "Industry": "industry",
        }
    )
    output["yfinance_symbol"] = output.symbol + ".NS"
    if len(output) != 500 or output.symbol.duplicated().any():
        raise RuntimeError("Expected exactly 500 unique constituents")
    for destination in OUTPUTS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(destination, index=False)
        print(f"Updated {destination} with {len(output)} constituents")


if __name__ == "__main__":
    main()
