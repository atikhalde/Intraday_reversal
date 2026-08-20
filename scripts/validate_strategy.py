"""Measure reversal-strategy accuracy and signal frequency on simulated markets.

Usage:
    python scripts/validate_strategy.py [--days 30] [--symbols 24] [--seed 7]

Prints, for the configured default settings:
- total signals and signals per simulated trading day;
- precision: share of signals whose 1R target was reached before invalidation;
- outcome mix (2R / 1R / stopped / ambiguous / nothing by the close);
- a regime breakdown showing which session types produced the alerts.

The simulated sessions are deterministic for a given seed, so the harness is a
stable walk-forward measurement for tuning thresholds.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reversal_scanner.config import load_config  # noqa: E402
from reversal_scanner.reporting import evaluate_datasets  # noqa: E402
from reversal_scanner.simulation import build_market  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--symbols", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", help="Optional YAML override file")
    args = parser.parse_args()

    config = load_config(args.config)
    datasets, specs = build_market(args.days, args.symbols, seed=args.seed)
    records = evaluate_datasets(datasets, config["strategy"], config["filters"])

    regime_by_key = {
        (spec.symbol, spec.date.date()): spec.regime for spec in specs
    }

    print(f"Simulated: {args.days} day(s) x {args.symbols} symbol(s) = {len(specs)} sessions")
    print(f"Signals total: {len(records)}  (avg {len(records) / args.days:.2f} per day)")

    by_day = Counter(record.signal.timestamp.date() for record in records)
    if by_day:
        busiest = max(by_day.values())
        print(f"Signals per day: min {min(by_day.values())}, max {busiest}; "
              f"days with signals: {len(by_day)}/{args.days}")

    if not records:
        print("No signals fired.")
        return 0

    winners_1r = [r for r in records if r.reached_1r and not r.stopped]
    winners_2r = [r for r in records if r.reached_2r and not r.stopped]
    stopped = [r for r in records if r.stopped]
    print(
        f"Precision 1R (1R before invalidation): {len(winners_1r)}/{len(records)} "
        f"= {len(winners_1r) / len(records):.1%}"
    )
    print(
        f"Precision 2R: {len(winners_2r)}/{len(records)} "
        f"= {len(winners_2r) / len(records):.1%}"
    )
    print(f"Stopped/ambiguous: {len(stopped)}")

    outcomes = Counter(record.outcome for record in records)
    print("Outcome mix:")
    for outcome, count in outcomes.most_common():
        print(f"  {count:4d}  {outcome}")

    print("Signals by session regime:")
    regime_counts = Counter(
        regime_by_key.get((record.signal.symbol, record.signal.timestamp.date()), "?")
        for record in records
    )
    for regime in ("reversal", "fakeout", "chop", "trend_down", "trend_up", "?"):
        if regime in regime_counts:
            print(f"  {regime:11s} {regime_counts[regime]}")

    print("\nDetails:")
    for record in records:
        regime = regime_by_key.get(
            (record.signal.symbol, record.signal.timestamp.date()), "?"
        )
        print(
            f"  {record.signal.timestamp} {record.signal.symbol} [{regime:10s}] "
            f"{record.signal.pattern} score={record.signal.score} "
            f"entry={record.signal.confirmation_price:.2f} "
            f"stop={record.signal.full_invalidation:.2f} "
            f"mfe={record.mfe_r:.2f}R mae={record.mae_r:.2f}R -> {record.outcome}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
