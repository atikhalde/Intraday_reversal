from datetime import date
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from reversal_scanner.config import load_config
from reversal_scanner.reporting import evaluate_datasets, generate_pdf_report, write_results_csv

FIXTURE = Path(__file__).parent / "fixtures" / "orchpharma_2026-06-16_5m.csv"


def load_orchpharma() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, parse_dates=["datetime_IST"]).set_index("datetime_IST")


def test_evaluates_orchpharma_outcome_without_lookahead() -> None:
    config = load_config()
    records = evaluate_datasets(
        {"ORCHPHARMA": (load_orchpharma(), "fixture")},
        config["strategy"],
        config["filters"],
    )
    assert len(records) == 1
    record = records[0]
    assert record.signal.timestamp == pd.Timestamp("2026-06-16 13:35")
    assert record.reached_1r is True
    assert record.reached_2r is True
    assert record.stopped is False
    assert record.outcome == "2R reached"
    assert record.bars_observed == 22


def test_writes_downloadable_pdf_and_csv(tmp_path: Path) -> None:
    datasets = {"ORCHPHARMA": (load_orchpharma(), "fixture")}
    config = load_config()
    records = evaluate_datasets(datasets, config["strategy"], config["filters"])
    pdf_path = tmp_path / "backtest-report.pdf"
    csv_path = tmp_path / "backtest-results.csv"
    write_results_csv(records, csv_path)
    generate_pdf_report(
        records=records,
        output_path=pdf_path,
        start_date=date(2026, 6, 16),
        end_date=date(2026, 6, 16),
        requested_symbols=["ORCHPHARMA"],
        datasets=datasets,
        errors={},
    )
    assert pdf_path.read_bytes().startswith(b"%PDF-")
    assert pdf_path.stat().st_size > 4_000
    reader = PdfReader(pdf_path)
    assert len(reader.pages) >= 2
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert "Score 100/100" in pdf_text
    assert "Broken pivot / retest" in pdf_text
    assert "Immediate failure" in pdf_text
    assert "Full invalidation" in pdf_text
    assert "Target 1R / 2R" in pdf_text
    assert "Why it fired" in pdf_text
    assert "Outcome: 2R reached" in pdf_text
    results = pd.read_csv(csv_path)
    assert results.loc[0, "outcome"] == "2R reached"
    assert bool(results.loc[0, "reached_2r"])
    assert results.loc[0, "immediate_failure"] > 0


def test_zero_signal_run_still_creates_report_artifacts(tmp_path: Path) -> None:
    pdf_path = tmp_path / "empty-report.pdf"
    csv_path = tmp_path / "empty-results.csv"
    write_results_csv([], csv_path)
    generate_pdf_report(
        records=[],
        output_path=pdf_path,
        start_date=date(2026, 8, 1),
        end_date=date(2026, 8, 1),
        requested_symbols=["RELIANCE"],
        datasets={},
        errors={"RELIANCE": "historical range unavailable"},
    )
    assert len(PdfReader(pdf_path).pages) >= 2
    assert list(pd.read_csv(csv_path).columns) == [
        "symbol",
        "confirmation_time",
        "pattern",
        "score",
        "data_source",
        "spring_time",
        "spring_low",
        "confirmation_price",
        "pivot_high",
        "immediate_failure",
        "full_invalidation",
        "target_1r",
        "target_2r",
        "outcome",
        "reached_1r",
        "reached_2r",
        "stopped",
        "mfe_r",
        "mae_r",
        "bars_observed",
        "reasons",
    ]
