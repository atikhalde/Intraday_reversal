from __future__ import annotations

import argparse
import logging
import time as time_module
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import load_config, required_env
from .data import DhanDataProvider, MarketDataCoordinator, YahooDataProvider
from .data.common import DataFetchError
from .detector import scan_history
from .scanner import ReversalScanner, market_is_scannable
from .state import SignalState
from .telegram import TelegramNotifier
from .universe import load_nifty500

LOGGER = logging.getLogger(__name__)


class UnavailablePrimary:
    def fetch(self, instrument, now):  # noqa: ANN001, ANN201
        raise DataFetchError("primary disabled")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Nifty 500 intraday reversal scanner")
    parser.add_argument("--config", help="Optional YAML overrides")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan the latest completed five-minute candles")
    mode = scan.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="Run one scan (default)")
    mode.add_argument("--loop", action="store_true", help="Stay alive and scan after each boundary")
    scan.add_argument("--provider", choices=["dhan", "yfinance"], default="dhan")
    scan.add_argument("--dry-run", action="store_true", help="Print signals; do not send Telegram")
    scan.add_argument("--require-market-open", action="store_true")
    scan.add_argument("--symbols", help="Comma-separated NSE symbols; default is all Nifty 500")
    scan.add_argument("--max-symbols", type=int, help="Limit universe for a smoke test")
    scan.add_argument("--universe", help="Override the bundled universe CSV")

    backtest = subparsers.add_parser("backtest", help="Walk forward through an OHLCV CSV")
    backtest.add_argument("csv", type=Path)
    backtest.add_argument("--symbol", required=True)
    backtest.add_argument("--datetime-column", default="datetime_IST")
    return parser


def _build_scanner(config: dict, provider_name: str, dry_run: bool) -> ReversalScanner:
    scanner_cfg = config["scanner"]
    provider_cfg = config["provider"]
    yahoo = YahooDataProvider(
        timeout_seconds=float(provider_cfg["yfinance_timeout_seconds"]),
        batch_size=int(provider_cfg["yfinance_batch_size"]),
    )
    if provider_name == "dhan":
        primary = DhanDataProvider(
            client_id=required_env("DHAN_CLIENT_ID"),
            access_token=required_env("DHAN_ACCESS_TOKEN"),
            timeout_seconds=float(provider_cfg["dhan_timeout_seconds"]),
            interval_minutes=int(scanner_cfg["interval_minutes"]),
            lookback_days=int(provider_cfg["lookback_days"]),
        )
    else:
        primary = UnavailablePrimary()
    coordinator = MarketDataCoordinator(
        primary=primary,
        fallback=yahoo,
        max_workers=int(scanner_cfg["max_workers"]),
    )
    notifier = None
    if not dry_run:
        notifier = TelegramNotifier(
            bot_token=required_env("TELEGRAM_BOT_TOKEN"),
            chat_id=required_env("TELEGRAM_CHAT_ID"),
        )
    return ReversalScanner(
        coordinator=coordinator,
        config=config,
        state=SignalState(scanner_cfg["state_file"]),
        notifier=notifier,
    )


def _select_instruments(args: argparse.Namespace):  # noqa: ANN202
    instruments = load_nifty500(args.universe)
    if args.symbols:
        wanted = {symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()}
        instruments = [instrument for instrument in instruments if instrument.symbol in wanted]
        found = {instrument.symbol for instrument in instruments}
        if missing := wanted - found:
            raise RuntimeError(f"Symbols are not in the bundled Nifty 500: {sorted(missing)}")
    if args.max_symbols:
        instruments = instruments[: args.max_symbols]
    if not instruments:
        raise RuntimeError("Selected universe is empty")
    return instruments


def _sleep_to_next_boundary(interval_minutes: int, delay_seconds: int) -> None:
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    minute_bucket = (now.minute // interval_minutes + 1) * interval_minutes
    next_run = now.replace(second=0, microsecond=0)
    if minute_bucket >= 60:
        next_run = (next_run + timedelta(hours=1)).replace(minute=0)
    else:
        next_run = next_run.replace(minute=minute_bucket)
    next_run += timedelta(seconds=delay_seconds)
    time_module.sleep(max((next_run - now).total_seconds(), 1.0))


def _run_scan(args: argparse.Namespace, config: dict) -> int:
    scanner_cfg = config["scanner"]
    instruments = _select_instruments(args)
    scanner = _build_scanner(config, args.provider, args.dry_run)
    LOGGER.info("Loaded %d Nifty 500 instruments", len(instruments))
    while True:
        now = datetime.now(ZoneInfo(scanner_cfg["market_timezone"]))
        if args.require_market_open and not market_is_scannable(now, scanner_cfg):
            LOGGER.info("Outside NSE scan window; no provider calls made")
        else:
            signals, errors = scanner.scan_once(instruments, now, args.dry_run)
            LOGGER.info(
                "Scan complete: %d signal(s), %d symbols unavailable",
                len(signals),
                len(errors),
            )
        if not args.loop:
            return 0
        _sleep_to_next_boundary(
            int(scanner_cfg["interval_minutes"]),
            int(scanner_cfg["loop_boundary_delay_seconds"]),
        )


def _run_backtest(args: argparse.Namespace, config: dict) -> int:
    frame = pd.read_csv(args.csv)
    if args.datetime_column not in frame:
        raise RuntimeError(f"CSV has no datetime column {args.datetime_column!r}")
    index = pd.to_datetime(frame.pop(args.datetime_column), errors="raise")
    frame.index = index
    signals = scan_history(args.symbol.upper(), frame, config["strategy"])
    if not signals:
        print("No confirmed reversal signals found.")
        return 0
    for signal in signals:
        print(
            f"{signal.timestamp} | {signal.symbol} | {signal.pattern} | "
            f"score={signal.score} | spring={signal.spring_time} @ {signal.spring_low:.2f} | "
            f"confirm={signal.confirmation_price:.2f} | pivot={signal.pivot_high:.2f} | "
            f"invalidation={signal.full_invalidation:.2f}"
        )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(args.config)
        status = _run_scan(args, config) if args.command == "scan" else _run_backtest(args, config)
    except KeyboardInterrupt:
        status = 130
    except Exception as exc:
        LOGGER.error("%s", exc)
        status = 1
    raise SystemExit(status)


if __name__ == "__main__":
    main()
