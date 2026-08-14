from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .detector import scan_history
from .models import Signal


@dataclass(frozen=True, slots=True)
class EvaluatedSignal:
    signal: Signal
    outcome: str
    reached_1r: bool
    reached_2r: bool
    stopped: bool
    mfe_r: float
    mae_r: float
    bars_observed: int

    def to_dict(self) -> dict[str, object]:
        signal = self.signal
        return {
            "symbol": signal.symbol,
            "confirmation_time": signal.timestamp.isoformat(),
            "pattern": signal.pattern,
            "score": signal.score,
            "data_source": signal.data_source,
            "spring_time": signal.spring_time.isoformat(),
            "spring_low": signal.spring_low,
            "confirmation_price": signal.confirmation_price,
            "pivot_high": signal.pivot_high,
            "immediate_failure": signal.immediate_failure,
            "full_invalidation": signal.full_invalidation,
            "target_1r": signal.target_1r,
            "target_2r": signal.target_2r,
            "outcome": self.outcome,
            "reached_1r": self.reached_1r,
            "reached_2r": self.reached_2r,
            "stopped": self.stopped,
            "mfe_r": round(self.mfe_r, 3),
            "mae_r": round(self.mae_r, 3),
            "bars_observed": self.bars_observed,
            "reasons": " | ".join(signal.reasons),
        }


def evaluate_signal(signal: Signal, bars: pd.DataFrame) -> EvaluatedSignal:
    frame = bars.sort_index()
    session_date = signal.timestamp.date()
    session_mask = pd.DatetimeIndex(frame.index).date == session_date
    future = frame[(frame.index > signal.timestamp) & session_mask]
    risk = max(signal.confirmation_price - signal.full_invalidation, 0.01)
    if future.empty:
        return EvaluatedSignal(signal, "No subsequent bars", False, False, False, 0.0, 0.0, 0)

    max_high = float(future.high.max())
    min_low = float(future.low.min())
    mfe_r = max(0.0, (max_high - signal.confirmation_price) / risk)
    mae_r = max(0.0, (signal.confirmation_price - min_low) / risk)
    reached_1r = False
    reached_2r = False
    stopped = False
    ambiguous = False

    for row in future.itertuples():
        stop_hit = float(row.low) <= signal.full_invalidation
        target_1_hit = float(row.high) >= signal.target_1r
        target_2_hit = float(row.high) >= signal.target_2r
        if stop_hit and (target_1_hit or target_2_hit):
            # Five-minute OHLC cannot reveal ordering; use the conservative path.
            stopped = True
            ambiguous = True
            break
        if stop_hit:
            stopped = True
            break
        reached_1r = reached_1r or target_1_hit
        if target_2_hit:
            reached_1r = True
            reached_2r = True
            break

    if ambiguous:
        outcome = "Ambiguous bar - stop assumed first"
    elif reached_2r:
        outcome = "2R reached"
    elif stopped and reached_1r:
        outcome = "1R reached, then full stop"
    elif stopped:
        outcome = "Full invalidation hit"
    elif reached_1r:
        outcome = "1R reached; 2R not reached"
    else:
        outcome = "Neither target nor stop by session end"
    return EvaluatedSignal(
        signal=signal,
        outcome=outcome,
        reached_1r=reached_1r,
        reached_2r=reached_2r,
        stopped=stopped,
        mfe_r=mfe_r,
        mae_r=mae_r,
        bars_observed=len(future),
    )


def _passes_historical_filters(
    signal: Signal,
    bars: pd.DataFrame,
    filters: dict | None,
) -> bool:
    if filters is None:
        return True
    if not float(filters["min_price"]) <= signal.confirmation_price <= float(filters["max_price"]):
        return False
    frame = bars.sort_index().loc[: signal.timestamp]
    session = frame[pd.DatetimeIndex(frame.index).date == signal.timestamp.date()]
    typical_price = (session.high + session.low + session.close) / 3
    turnover = float((typical_price * session.volume).sum())
    return turnover >= float(filters["min_session_turnover_inr"])


def evaluate_datasets(
    datasets: dict[str, tuple[pd.DataFrame, str]],
    strategy_config: dict,
    filters: dict | None = None,
) -> list[EvaluatedSignal]:
    evaluated: list[EvaluatedSignal] = []
    for symbol, (bars, source) in sorted(datasets.items()):
        for signal in scan_history(symbol, bars, strategy_config, source):
            if _passes_historical_filters(signal, bars, filters):
                evaluated.append(evaluate_signal(signal, bars))
    return sorted(evaluated, key=lambda item: (item.signal.timestamp, item.signal.symbol))


def write_results_csv(records: list[EvaluatedSignal], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = [
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
    frame = pd.DataFrame([record.to_dict() for record in records], columns=columns)
    frame.to_csv(destination, index=False)
    return destination


def _ascii(value: object) -> str:
    return escape(str(value).replace("₹", "INR ").replace("→", "->"))


def _page(canvas, document) -> None:  # noqa: ANN001
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(15 * mm, 9 * mm, "Intraday Reversal Scanner - research report")
    canvas.drawRightString(
        landscape(A4)[0] - 15 * mm,
        9 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


def generate_pdf_report(
    records: list[EvaluatedSignal],
    output_path: str | Path,
    start_date: date,
    end_date: date,
    requested_symbols: list[str],
    datasets: dict[str, tuple[pd.DataFrame, str]],
    errors: dict[str, str],
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(destination),
        pagesize=landscape(A4),
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=14 * mm,
        bottomMargin=16 * mm,
        title="Intraday reversal backtest",
        author="Intraday Reversal Scanner",
    )
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=8 * mm,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallBody",
            parent=styles["BodyText"],
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#334155"),
        )
    )
    story = [
        Paragraph("Past Intraday Reversal Backtest", styles["ReportTitle"]),
        Paragraph(
            f"Period: <b>{start_date.isoformat()}</b> to <b>{end_date.isoformat()}</b> | "
            f"Generated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
            styles["Heading3"],
        ),
        Spacer(1, 4 * mm),
    ]

    one_r = sum(record.reached_1r for record in records)
    two_r = sum(record.reached_2r for record in records)
    stops = sum(record.stopped for record in records)
    average_mfe = sum(record.mfe_r for record in records) / len(records) if records else 0.0
    average_mae = sum(record.mae_r for record in records) / len(records) if records else 0.0
    source_counts: dict[str, int] = {}
    for _, source in datasets.values():
        source_counts[source] = source_counts.get(source, 0) + 1

    summary_data = [
        ["Requested", "Data loaded", "Signals", "1R reached", "2R reached", "Stopped"],
        [
            str(len(requested_symbols)),
            str(len(datasets)),
            str(len(records)),
            f"{one_r} ({one_r / len(records):.1%})" if records else "0",
            f"{two_r} ({two_r / len(records):.1%})" if records else "0",
            f"{stops} ({stops / len(records):.1%})" if records else "0",
        ],
        ["Avg MFE", "Avg MAE", "Unavailable", "Dhan sets", "Yahoo/fixture", "Timeframe"],
        [
            f"{average_mfe:.2f}R",
            f"{average_mae:.2f}R",
            str(len(errors)),
            str(source_counts.get("dhan", 0)),
            str(
                source_counts.get("yfinance", 0)
                + source_counts.get("fixture", 0)
                + source_counts.get("csv", 0)
            ),
            "5 minute",
        ],
    ]
    summary = Table(summary_data, colWidths=[42 * mm] * 6)
    summary.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F766E")),
                ("BACKGROUND", (0, 2), (-1, 2), colors.HexColor("#E2E8F0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.extend([summary, Spacer(1, 6 * mm), Paragraph("Signal Results", styles["Heading2"])])

    table_rows = [
        [
            "Time",
            "Symbol",
            "Pattern",
            "Score",
            "Confirm",
            "Pivot/retest",
            "Full stop",
            "MFE",
            "MAE",
            "Outcome",
        ]
    ]
    for record in records:
        signal = record.signal
        table_rows.append(
            [
                signal.timestamp.strftime("%d %b %H:%M"),
                signal.symbol,
                signal.pattern,
                str(signal.score),
                f"{signal.confirmation_price:.2f}",
                f"{signal.pivot_high:.2f}",
                f"{signal.full_invalidation:.2f}",
                f"{record.mfe_r:.2f}R",
                f"{record.mae_r:.2f}R",
                Paragraph(_ascii(record.outcome), styles["SmallBody"]),
            ]
        )
    if len(table_rows) == 1:
        table_rows.append(["-", "-", "No confirmed signals", "-", "-", "-", "-", "-", "-", "-"])
    signal_table = Table(
        table_rows,
        repeatRows=1,
        colWidths=[
            24 * mm,
            24 * mm,
            31 * mm,
            13 * mm,
            20 * mm,
            20 * mm,
            20 * mm,
            15 * mm,
            15 * mm,
            58 * mm,
        ],
    )
    signal_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (3, 1), (8, -1), "RIGHT"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.extend([signal_table, PageBreak(), Paragraph("Setup Details", styles["Heading2"])])

    if records:
        for number, record in enumerate(records, start=1):
            signal = record.signal
            story.append(
                Paragraph(
                    f"{number}. {_ascii(signal.symbol)} - {_ascii(signal.pattern)} | "
                    f"Score {signal.score}/100 | {signal.timestamp.strftime('%Y-%m-%d %H:%M')}",
                    styles["Heading3"],
                )
            )
            levels = (
                f"Spring: {signal.spring_time.strftime('%Y-%m-%d %H:%M')} at INR "
                f"{signal.spring_low:.2f} | Confirmation: INR {signal.confirmation_price:.2f} | "
                f"Broken pivot / retest: INR {signal.pivot_high:.2f} | "
                f"Immediate failure: INR {signal.immediate_failure:.2f} | "
                f"Full invalidation: INR {signal.full_invalidation:.2f} | "
                f"Target 1R / 2R: INR {signal.target_1r:.2f} / INR {signal.target_2r:.2f}"
            )
            outcome = (
                f"Outcome: {_ascii(record.outcome)} | MFE: {record.mfe_r:.2f}R | "
                f"MAE: {record.mae_r:.2f}R | Bars observed: {record.bars_observed} | "
                f"1R reached: {'Yes' if record.reached_1r else 'No'} | "
                f"2R reached: {'Yes' if record.reached_2r else 'No'} | "
                f"Full stop hit: {'Yes' if record.stopped else 'No'} | "
                f"Data: {_ascii(signal.data_source)}"
            )
            story.append(Paragraph(levels, styles["SmallBody"]))
            story.append(Paragraph(outcome, styles["SmallBody"]))
            story.append(Paragraph("<b>Why it fired</b>", styles["SmallBody"]))
            for reason in signal.reasons:
                story.append(Paragraph(f"- {_ascii(reason)}", styles["SmallBody"]))
            story.append(Spacer(1, 3 * mm))
    else:
        story.append(
            Paragraph("No setup passed every configured confirmation rule.", styles["BodyText"])
        )

    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph("Methodology and Limitations", styles["Heading2"]),
            Paragraph(
                "Signals are generated walk-forward from completed five-minute candles. "
                "The detector requires prior decline, a sell-side-liquidity sweep, stopping "
                "volume or hammer rejection, a higher-low test or fast hammer path, and a "
                "bullish displacement close above the local pivot. Configured price and running "
                "session-turnover filters are applied at confirmation time. No future candle is "
                "used to create a signal.",
                styles["BodyText"],
            ),
            Spacer(1, 2 * mm),
            Paragraph(
                "Outcome statistics are technical excursions, not executable P&L. Entry is "
                "represented by the confirmation close. Fees, taxes, spread, slippage, "
                "partial exits, position sizing and news classification are excluded. If a "
                "five-minute candle touches a target and stop, the report conservatively "
                "assumes the stop occurred first because OHLC does not reveal order.",
                styles["BodyText"],
            ),
        ]
    )
    if errors:
        failed_text = ", ".join(
            f"{_ascii(symbol)} ({_ascii(message)})" for symbol, message in sorted(errors.items())
        )
        story.extend(
            [
                Spacer(1, 3 * mm),
                Paragraph("Unavailable Data", styles["Heading3"]),
                Paragraph(failed_text, styles["SmallBody"]),
            ]
        )

    document.build(story, onFirstPage=_page, onLaterPages=_page)
    return destination
