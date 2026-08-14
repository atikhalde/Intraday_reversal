from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .models import Signal

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(slots=True)
class Candidate:
    spring_index: int
    test_index: int | None
    no_supply_index: int | None
    pivot_high: float
    score: int
    pattern: str
    reasons: list[str]
    metrics: dict[str, float]


def normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Return sorted, numeric OHLCV bars and reject malformed candles."""
    frame = bars.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"OHLCV data is missing columns: {sorted(missing)}")
    frame = frame.loc[:, REQUIRED_COLUMNS]
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna().sort_index()
    valid = (
        (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
        & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
        & (frame["volume"] >= 0)
        & (frame["close"] > 0)
    )
    return frame.loc[valid]


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator > 0 else default


def _candle_metrics(row: pd.Series) -> dict[str, float]:
    candle_range = float(row.high - row.low)
    body = abs(float(row.close - row.open))
    lower_wick = float(min(row.open, row.close) - row.low)
    upper_wick = float(row.high - max(row.open, row.close))
    return {
        "range": candle_range,
        "body": body,
        "body_ratio": _safe_ratio(body, candle_range),
        "lower_wick": lower_wick,
        "upper_wick": upper_wick,
        "lower_wick_body_ratio": _safe_ratio(lower_wick, body, 99.0),
        "close_location": _safe_ratio(float(row.close - row.low), candle_range, 0.5),
    }


def _median_positive(series: pd.Series, default: float) -> float:
    positive = pd.to_numeric(series, errors="coerce")
    positive = positive[positive > 0]
    return float(positive.median()) if not positive.empty else default


def _find_test(
    bars: pd.DataFrame,
    spring_idx: int,
    sos_idx: int,
    spring_low: float,
    spring_volume: float,
    pivot_high: float,
    cfg: dict[str, Any],
) -> int | None:
    start = spring_idx + int(cfg["test_min_bars_after_spring"])
    if start >= sos_idx:
        return None
    recovery = max(pivot_high - spring_low, 0.0)
    max_test_low = spring_low + recovery * float(cfg["test_max_fraction_of_recovery"])
    choices: list[tuple[float, int]] = []
    for index in range(start, sos_idx):
        row = bars.iloc[index]
        metrics = _candle_metrics(row)
        higher_low = float(row.low) > spring_low
        returned_to_demand = float(row.low) <= max_test_low
        rejected = (
            metrics["close_location"] >= float(cfg["test_min_close_location"])
            and metrics["lower_wick_body_ratio"] >= float(cfg["test_min_wick_body_ratio"])
        )
        volume_contracts = float(row.volume) <= (
            spring_volume * float(cfg["test_max_spring_volume_ratio"])
        )
        if higher_low and returned_to_demand and rejected and volume_contracts:
            # Prefer the deepest valid test; that is normally the clearest LPS.
            choices.append((float(row.low), index))
    return min(choices)[1] if choices else None


def _find_no_supply(
    bars: pd.DataFrame,
    test_idx: int | None,
    sos_idx: int,
    median_range: float,
    cfg: dict[str, Any],
) -> int | None:
    if test_idx is None:
        return None
    test_volume = float(bars.iloc[test_idx].volume)
    for index in range(test_idx + 1, sos_idx):
        row = bars.iloc[index]
        candle_range = float(row.high - row.low)
        if (
            float(row.volume)
            <= test_volume * float(cfg["no_supply_max_test_volume_ratio"])
            and candle_range <= median_range * float(cfg["no_supply_max_median_range_ratio"])
        ):
            return index
    return None


def _evaluate_spring(
    bars: pd.DataFrame,
    spring_idx: int,
    sos_idx: int,
    cfg: dict[str, Any],
) -> Candidate | None:
    spring = bars.iloc[spring_idx]
    sos = bars.iloc[sos_idx]
    liquidity_lookback = int(cfg["liquidity_lookback_bars"])
    volume_lookback = int(cfg["volume_lookback_bars"])
    range_lookback = int(cfg["range_lookback_bars"])

    prior_start = max(0, spring_idx - liquidity_lookback)
    prior = bars.iloc[prior_start:spring_idx]
    if len(prior) < int(cfg["min_context_bars"]):
        return None

    prior_support = float(prior.low.min())
    sweep_limit = prior_support * (1 + float(cfg["sweep_tolerance_pct"]) / 100)
    if float(spring.low) > sweep_limit:
        return None

    context = bars.iloc[:spring_idx]
    context_high = float(context.high.max())
    decline_pct = _safe_ratio(context_high - float(spring.low), context_high) * 100
    if decline_pct < float(cfg["min_decline_pct"]):
        return None

    spring_metrics = _candle_metrics(spring)
    if spring_metrics["close_location"] < float(cfg["min_spring_close_location"]):
        return None

    volume_window = bars.iloc[max(0, spring_idx - volume_lookback):spring_idx].volume
    range_window = (
        bars.iloc[max(0, sos_idx - range_lookback):sos_idx].high
        - bars.iloc[max(0, sos_idx - range_lookback):sos_idx].low
    )
    median_volume = _median_positive(volume_window, max(float(spring.volume), 1.0))
    median_range = _median_positive(range_window, max(float(sos.high - sos.low), 0.01))
    spring_volume_ratio = _safe_ratio(float(spring.volume), median_volume)
    spring_is_hammer = (
        spring_metrics["close_location"] >= float(cfg["test_min_close_location"])
        and spring_metrics["lower_wick_body_ratio"] >= float(cfg["spring_wick_body_ratio"])
    )
    stopping_volume = spring_volume_ratio >= float(cfg["stopping_volume_ratio"])
    if not (stopping_volume or spring_is_hammer):
        return None

    between = bars.iloc[spring_idx + 1 : sos_idx]
    pivot_high = float(between.high.max()) if not between.empty else float(spring.high)
    if float(sos.close) <= pivot_high:
        return None

    sos_metrics = _candle_metrics(sos)
    sos_range_ratio = _safe_ratio(sos_metrics["range"], median_range)
    sos_volume_median = _median_positive(
        bars.iloc[max(0, sos_idx - volume_lookback):sos_idx].volume,
        max(float(sos.volume), 1.0),
    )
    sos_volume_ratio = _safe_ratio(float(sos.volume), sos_volume_median)
    previous_volume = float(bars.iloc[sos_idx - 1].volume)
    sos_previous_volume_ratio = _safe_ratio(float(sos.volume), previous_volume, 99.0)
    sos_volume_ok = (
        sos_volume_ratio >= float(cfg["sos_min_volume_ratio"])
        or sos_previous_volume_ratio >= float(cfg["sos_min_previous_volume_ratio"])
    )
    sos_ok = (
        float(sos.close) > float(sos.open)
        and sos_metrics["body_ratio"] >= float(cfg["sos_min_body_ratio"])
        and sos_range_ratio >= float(cfg["sos_min_range_ratio"])
        and sos_volume_ok
    )
    if not sos_ok:
        return None

    test_idx = _find_test(
        bars,
        spring_idx,
        sos_idx,
        float(spring.low),
        float(spring.volume),
        pivot_high,
        cfg,
    )
    direct_hammer = spring_is_hammer and sos_idx - spring_idx <= 3
    if test_idx is None and not direct_hammer:
        return None
    no_supply_idx = _find_no_supply(bars, test_idx, sos_idx, median_range, cfg)

    score = 15  # meaningful prior decline
    reasons = [f"{decline_pct:.2f}% decline into sell-side liquidity"]
    score += 15
    reasons.append(f"swept prior low ₹{prior_support:.2f} at ₹{float(spring.low):.2f}")
    score += 5
    reasons.append(f"spring close location {spring_metrics['close_location']:.0%}")
    if stopping_volume:
        score += 10
        reasons.append(f"stopping volume {spring_volume_ratio:.2f}x median")
    if spring_is_hammer:
        score += 10
        reasons.append("spring candle has bullish lower-wick rejection")
    if test_idx is not None:
        score += 20
        test = bars.iloc[test_idx]
        reasons.append(
            f"higher-low test at ₹{float(test.low):.2f} on "
            f"{_safe_ratio(float(test.volume), float(spring.volume)):.2f}x spring volume"
        )
    if no_supply_idx is not None:
        score += 10
        reasons.append("no-supply contraction formed before displacement")
    score += 10
    reasons.append(f"bullish close broke the ₹{pivot_high:.2f} local pivot")
    score += 10
    reasons.append(f"SOS body is {sos_metrics['body_ratio']:.0%} of range")
    if sos_range_ratio >= float(cfg["sos_min_range_ratio"]):
        score += 5
        reasons.append(f"SOS range expanded to {sos_range_ratio:.2f}x median")
    if sos_volume_ok:
        score += 5
        reasons.append(
            f"SOS volume {sos_volume_ratio:.2f}x median / "
            f"{sos_previous_volume_ratio:.2f}x previous bar"
        )

    pattern = "spring-test-SOS" if test_idx is not None else "hammer-SOS"
    return Candidate(
        spring_index=spring_idx,
        test_index=test_idx,
        no_supply_index=no_supply_idx,
        pivot_high=pivot_high,
        score=min(score, 100),
        pattern=pattern,
        reasons=reasons,
        metrics={
            "decline_pct": decline_pct,
            "prior_support": prior_support,
            "spring_volume_ratio": spring_volume_ratio,
            "spring_close_location": spring_metrics["close_location"],
            "sos_body_ratio": sos_metrics["body_ratio"],
            "sos_range_ratio": sos_range_ratio,
            "sos_volume_ratio": sos_volume_ratio,
            "sos_previous_volume_ratio": sos_previous_volume_ratio,
        },
    )


def _sos_qualifies(frame: pd.DataFrame, sos_idx: int, config: dict[str, Any]) -> bool:
    """Reject a non-SOS candle once before evaluating possible springs."""
    sos = frame.iloc[sos_idx]
    metrics = _candle_metrics(sos)
    range_lookback = int(config["range_lookback_bars"])
    range_window = (
        frame.iloc[max(0, sos_idx - range_lookback) : sos_idx].high
        - frame.iloc[max(0, sos_idx - range_lookback) : sos_idx].low
    )
    median_range = _median_positive(range_window, max(float(sos.high - sos.low), 0.01))
    volume_lookback = int(config["volume_lookback_bars"])
    median_volume = _median_positive(
        frame.iloc[max(0, sos_idx - volume_lookback) : sos_idx].volume,
        max(float(sos.volume), 1.0),
    )
    volume_ratio = _safe_ratio(float(sos.volume), median_volume)
    previous_volume_ratio = _safe_ratio(
        float(sos.volume),
        float(frame.iloc[sos_idx - 1].volume),
        99.0,
    )
    return (
        float(sos.close) > float(sos.open)
        and metrics["body_ratio"] >= float(config["sos_min_body_ratio"])
        and _safe_ratio(metrics["range"], median_range)
        >= float(config["sos_min_range_ratio"])
        and (
            volume_ratio >= float(config["sos_min_volume_ratio"])
            or previous_volume_ratio >= float(config["sos_min_previous_volume_ratio"])
        )
    )


def _qualifying_sos_positions(frame: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    """Precompute possible SOS positions for one session using only prior bars."""
    opens = frame["open"].to_numpy(dtype=float)
    highs = frame["high"].to_numpy(dtype=float)
    lows = frame["low"].to_numpy(dtype=float)
    closes = frame["close"].to_numpy(dtype=float)
    volumes = frame["volume"].to_numpy(dtype=float)
    ranges = highs - lows
    bodies = np.abs(closes - opens)
    body_ratios = np.divide(bodies, ranges, out=np.zeros_like(bodies), where=ranges > 0)
    qualifying = np.zeros(len(frame), dtype=bool)
    range_lookback = int(config["range_lookback_bars"])
    volume_lookback = int(config["volume_lookback_bars"])

    for index in range(1, len(frame)):
        if closes[index] <= opens[index] or body_ratios[index] < float(
            config["sos_min_body_ratio"]
        ):
            continue
        prior_ranges = ranges[max(0, index - range_lookback) : index]
        positive_ranges = prior_ranges[prior_ranges > 0]
        median_range = (
            float(np.median(positive_ranges))
            if positive_ranges.size
            else max(float(ranges[index]), 0.01)
        )
        if _safe_ratio(float(ranges[index]), median_range) < float(
            config["sos_min_range_ratio"]
        ):
            continue
        prior_volumes = volumes[max(0, index - volume_lookback) : index]
        positive_volumes = prior_volumes[prior_volumes > 0]
        median_volume = (
            float(np.median(positive_volumes))
            if positive_volumes.size
            else max(float(volumes[index]), 1.0)
        )
        volume_ratio = _safe_ratio(float(volumes[index]), median_volume)
        previous_volume_ratio = _safe_ratio(
            float(volumes[index]),
            float(volumes[index - 1]),
            99.0,
        )
        qualifying[index] = (
            volume_ratio >= float(config["sos_min_volume_ratio"])
            or previous_volume_ratio >= float(config["sos_min_previous_volume_ratio"])
        )
    return qualifying


def _detect_latest_session(
    symbol: str,
    frame: pd.DataFrame,
    config: dict[str, Any],
    data_source: str,
    *,
    check_sos: bool = True,
) -> Signal | None:
    """Evaluate the final candle of one already-normalized trading session."""
    if len(frame) < int(config["min_context_bars"]) + 2:
        return None

    sos_idx = len(frame) - 1
    if check_sos and not _sos_qualifies(frame, sos_idx, config):
        return None
    first_spring = max(
        int(config["min_context_bars"]),
        sos_idx - int(config["max_setup_bars"]),
    )
    candidates = [
        candidate
        for spring_idx in range(first_spring, sos_idx)
        if (candidate := _evaluate_spring(frame, spring_idx, sos_idx, config)) is not None
    ]
    if not candidates:
        return None
    candidate = max(candidates, key=lambda item: (item.score, item.spring_index))
    if candidate.score < int(config["min_score"]):
        return None

    spring = frame.iloc[candidate.spring_index]
    sos = frame.iloc[sos_idx]
    ranges = frame.high - frame.low
    atr = _median_positive(ranges.iloc[max(0, sos_idx - 14) : sos_idx], float(sos.high - sos.low))
    buffer_value = atr * float(config["stop_atr_buffer"])
    full_invalidation = max(0.0, float(spring.low) - buffer_value)
    immediate_failure = min(candidate.pivot_high, float(sos.low))
    risk = max(float(sos.close) - full_invalidation, atr * 0.1)

    timestamp = pd.Timestamp(frame.index[sos_idx]).to_pydatetime()
    spring_time = pd.Timestamp(frame.index[candidate.spring_index]).to_pydatetime()
    return Signal(
        symbol=symbol,
        timestamp=timestamp,
        pattern=candidate.pattern,
        score=candidate.score,
        spring_time=spring_time,
        spring_low=float(spring.low),
        confirmation_price=float(sos.close),
        pivot_high=candidate.pivot_high,
        immediate_failure=immediate_failure,
        full_invalidation=full_invalidation,
        target_1r=float(sos.close) + risk,
        target_2r=float(sos.close) + 2 * risk,
        data_source=data_source,
        reasons=tuple(candidate.reasons),
        metrics={key: float(value) for key, value in candidate.metrics.items()},
    )


def detect_latest(
    symbol: str,
    bars: pd.DataFrame,
    config: dict[str, Any],
    data_source: str = "unknown",
) -> Signal | None:
    """Detect a confirmed long reversal whose SOS is the final completed candle.

    The algorithm uses only data at or before the final candle. It accepts the
    full spring -> higher-low test -> no-supply -> SOS sequence, as well as a
    faster hammer -> SOS confirmation variant.
    """
    frame = normalize_bars(bars)
    if frame.empty:
        return None

    # A provider request can span days. Intraday structure must never leak from
    # a previous session into today's setup.
    last_session = frame.index[-1].date()
    frame = frame[[timestamp.date() == last_session for timestamp in frame.index]]
    return _detect_latest_session(symbol, frame, config, data_source)


def scan_history(
    symbol: str,
    bars: pd.DataFrame,
    config: dict[str, Any],
    data_source: str = "csv",
) -> list[Signal]:
    """Walk forward once per session without look-ahead and return every signal."""
    frame = normalize_bars(bars)
    signals: list[Signal] = []
    first_confirmation = int(config["min_context_bars"]) + 1
    session_dates = pd.DatetimeIndex(frame.index).date
    for _session_date, session in frame.groupby(session_dates, sort=False):
        qualifying_sos = _qualifying_sos_positions(session, config)
        for end in range(first_confirmation, len(session)):
            if not qualifying_sos[end]:
                continue
            signal = _detect_latest_session(
                symbol,
                session.iloc[: end + 1],
                config,
                data_source,
                check_sos=False,
            )
            if signal and (not signals or signal.key != signals[-1].key):
                signals.append(signal)
    return signals
