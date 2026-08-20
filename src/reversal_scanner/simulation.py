"""Regime-based five-minute intraday market simulator for strategy validation.

Generates realistic NSE-style trading sessions (09:15-15:30 IST, 75 five-minute
bars) across five behavioural regimes:

- ``chop``: mean-reverting noise around a flat anchor, no real direction.
- ``trend_down``: persistent intraday selling with relief pops that fail.
- ``trend_up``: persistent intraday buying.
- ``reversal``: the intended setup - decline, liquidity sweep on stopping
  volume, higher-low test on contracted volume, then displacement with real
  follow-through for the rest of the session.
- ``fakeout``: the same anatomy, but the bounce is trapped and price resumes
  making new lows afterwards.

One-minute sub-steps are used inside each five-minute bar so highs, lows and
wicks are realistic rather than synthetic open/close constructions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

SESSION_START = (9, 15)
SESSION_END = (15, 30)
BARS_PER_SESSION = 75  # 6h15m / 5m
SUBSTEPS = 5  # one-minute resolution inside each five-minute bar


@dataclass(frozen=True, slots=True)
class SessionSpec:
    symbol: str
    date: pd.Timestamp
    regime: str
    base_price: float
    seed: int


def _session_index(date: pd.Timestamp) -> pd.DatetimeIndex:
    start = date.replace(hour=SESSION_START[0], minute=SESSION_START[1], second=0)
    return pd.date_range(start, periods=BARS_PER_SESSION, freq="5min")


def _u_shape_volume_weights() -> np.ndarray:
    positions = np.linspace(-1.0, 1.0, BARS_PER_SESSION)
    return 1.0 + 1.6 * positions**2


def _aggregate_bars(
    minute_prices: np.ndarray,
    minute_volumes: np.ndarray,
    index: pd.DatetimeIndex,
) -> pd.DataFrame:
    opens, highs, lows, closes, volumes = [], [], [], [], []
    for bar in range(BARS_PER_SESSION):
        chunk = minute_prices[bar * SUBSTEPS : (bar + 1) * SUBSTEPS]
        opens.append(chunk[0])
        highs.append(chunk.max())
        lows.append(chunk.min())
        closes.append(chunk[-1])
        volumes.append(int(minute_volumes[bar * SUBSTEPS : (bar + 1) * SUBSTEPS].sum()))
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=index,
    )


def _simulate_session(spec: SessionSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    index = _session_index(spec.date)
    price = spec.base_price
    sigma = price * 0.0015  # per-minute volatility
    total_minutes = BARS_PER_SESSION * SUBSTEPS
    drift = np.zeros(total_minutes)
    minute_volumes = np.repeat(
        rng.lognormal(mean=math.log(4000), sigma=0.35, size=BARS_PER_SESSION)
        * _u_shape_volume_weights(),
        SUBSTEPS,
    )
    volume_boost = np.ones(total_minutes)

    regime = spec.regime
    if regime == "chop":
        anchor = price
        prices = [price]
        for _minute in range(total_minutes):
            reversion = 0.04 * (anchor - prices[-1])
            step = reversion + rng.normal(0.0, sigma)
            prices.append(prices[-1] + step)
        minute_prices = np.array(prices[1:])
        impact = np.abs(np.diff(minute_prices, prepend=minute_prices[0])) / sigma
        volume_boost = 1.0 + 0.8 * impact
    elif regime in {"trend_down", "trend_up"}:
        direction = -1.0 if regime == "trend_down" else 1.0
        drift[:] = direction * sigma * 0.22
        if regime == "trend_down":
            # Persistent distribution: relief pops are weak and brief, and
            # sellers keep showing up on every bounce (no volume contraction).
            for start in rng.integers(40, total_minutes - 30, size=3):
                drift[start : start + 8] = sigma * rng.uniform(0.18, 0.30)
        minute_prices = _random_walk(rng, price, drift, sigma, total_minutes)
        impact = np.abs(np.diff(minute_prices, prepend=minute_prices[0])) / sigma
        volume_boost = 1.0 + 0.8 * impact
        if regime == "trend_down":
            volume_boost[rng.choice(total_minutes, size=total_minutes // 8, replace=False)] = (
                rng.uniform(1.5, 2.2, size=total_minutes // 8)
            )
    elif regime in {"reversal", "fakeout"}:
        minute_prices, minute_volumes = _simulate_reversal_anatomy(
            rng, price, sigma, total_minutes, minute_volumes, volume_boost, regime
        )
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unknown regime {regime}")

    minute_prices = np.maximum(minute_prices, price * 0.05)
    bars = _aggregate_bars(minute_prices, minute_volumes * volume_boost, index)
    return _round_to_tick(bars)


def _random_walk(
    rng: np.random.Generator,
    start: float,
    drift: np.ndarray,
    sigma: float,
    total_minutes: int,
) -> np.ndarray:
    steps = drift + rng.normal(0.0, sigma, total_minutes)
    return start + np.cumsum(steps)


def _simulate_reversal_anatomy(
    rng: np.random.Generator,
    price: float,
    sigma: float,
    total_minutes: int,
    minute_volumes: np.ndarray,
    volume_boost: np.ndarray,
    regime: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build decline -> sweep -> response -> test -> continuation/trap."""
    decline_end = int(rng.integers(150, 220))  # 12:30-13:30 area in minutes
    total_decline = price * rng.uniform(0.025, 0.045)
    drift = np.zeros(total_minutes)
    drift[:decline_end] = -total_decline / decline_end

    # Sweep: sharp stab to a new low on a volume surge, then absorption.
    sweep_start = decline_end
    sweep_len = int(rng.integers(4, 8))
    drift[sweep_start : sweep_start + sweep_len] = -sigma * rng.uniform(0.8, 1.4)
    absorb_len = int(rng.integers(4, 8))
    drift[sweep_start + sweep_len : sweep_start + sweep_len + absorb_len] = sigma * 0.7

    # Real traps are rarely perfect: they usually leak one structural flaw -
    # a shallow bounce, a test that keeps leaking lower, or an unconvincing
    # confirmation candle. A minority are genuinely undetectable until they fail.
    flaw: str | None = None
    if regime == "fakeout":
        flaw = str(
            rng.choice(
                ["weak_confirm", "failed_test", "shallow_bounce", "none"],
                p=[0.28, 0.20, 0.16, 0.36],
            )
        )
        if flaw == "none":
            flaw = None

    bounce_start = sweep_start + sweep_len + absorb_len
    bounce_len = int(rng.integers(8, 14))
    bounce_strength = sigma * rng.uniform(0.35, 0.6)
    if flaw == "shallow_bounce":
        bounce_strength *= rng.uniform(0.3, 0.5)
    drift[bounce_start : bounce_start + bounce_len] = bounce_strength

    test_start = bounce_start + bounce_len
    test_len = int(rng.integers(10, 18))
    if flaw == "failed_test":
        # The "test" keeps leaking lower and trades back through demand.
        drift[test_start : test_start + test_len] = -sigma * rng.uniform(0.35, 0.55)
    else:
        drift[test_start : test_start + test_len] = -sigma * 0.18

    confirm_start = test_start + test_len
    confirm_strength = sigma * rng.uniform(0.28, 0.45)
    confirm_volume = rng.uniform(1.6, 2.4)
    if regime == "reversal":
        drift[confirm_start:] = confirm_strength
    else:  # fakeout trap: brief pop, then renewed distribution into the close
        trap_len = min(int(rng.integers(6, 10)), total_minutes - confirm_start - 30)
        pop_strength = sigma * rng.uniform(0.3, 0.5)
        if flaw == "weak_confirm":
            pop_strength *= rng.uniform(0.35, 0.55)
            confirm_volume = rng.uniform(0.9, 1.25)
        drift[confirm_start : confirm_start + trap_len] = pop_strength
        drift[confirm_start + trap_len :] = -sigma * rng.uniform(0.35, 0.6)

    minute_prices = _random_walk(rng, price, drift, sigma, total_minutes)

    # Volume anatomy: stopping volume on the sweep, contraction on the test,
    # expansion on the confirmation push.
    volume_boost = np.ones(total_minutes)
    volume_boost[sweep_start : sweep_start + sweep_len] = rng.uniform(2.2, 3.5)
    volume_boost[bounce_start : bounce_start + bounce_len] = rng.uniform(0.9, 1.3)
    if flaw == "failed_test":
        # Leaking tests trade on expanding, distribution-like volume.
        volume_boost[test_start : test_start + test_len] = rng.uniform(0.9, 1.4)
    else:
        volume_boost[test_start : test_start + test_len] = rng.uniform(0.35, 0.6)
    if regime == "reversal":
        volume_boost[confirm_start:] = confirm_volume
    else:
        volume_boost[confirm_start : confirm_start + 12] = confirm_volume
        volume_boost[confirm_start + 12 :] = rng.uniform(1.0, 1.5)
    return minute_prices, minute_volumes * volume_boost


def _round_to_tick(bars: pd.DataFrame) -> pd.DataFrame:
    def tick_for(level: float) -> float:
        if level < 250:
            return 0.05
        if level < 1000:
            return 0.10
        return 0.50

    rounded = bars.copy()
    for column in ("open", "high", "low", "close"):
        rounded[column] = [
            round(value / tick_for(value)) * tick_for(value) for value in rounded[column]
        ]
    # Keep OHLC internally consistent after rounding.
    rounded["high"] = rounded[["open", "high", "low", "close"]].max(axis=1)
    rounded["low"] = rounded[["open", "high", "low", "close"]].min(axis=1)
    rounded["volume"] = rounded["volume"].clip(lower=0).astype(int)
    return rounded


def build_market(
    days: int,
    symbols_per_day: int,
    start: str = "2026-06-01",
    seed: int = 7,
    regime_weights: dict[str, float] | None = None,
) -> tuple[dict[str, tuple[pd.DataFrame, str]], list[SessionSpec]]:
    """Return per-symbol datasets (all sessions concatenated) plus the specs."""
    rng = np.random.default_rng(seed)
    weights = regime_weights or {
        "chop": 0.40,
        "trend_down": 0.18,
        "trend_up": 0.17,
        "reversal": 0.16,
        "fakeout": 0.09,  # fully-formed bull traps are rare in practice
    }
    regimes = list(weights)
    probabilities = np.array([weights[regime] for regime in regimes], dtype=float)
    probabilities /= probabilities.sum()

    trading_days = pd.bdate_range(start, periods=days)
    specs: list[SessionSpec] = []
    frames: dict[str, list[pd.DataFrame]] = {}
    for day in trading_days:
        for slot in range(symbols_per_day):
            symbol = f"SIM{slot:02d}"
            regime = str(rng.choice(regimes, p=probabilities))
            spec = SessionSpec(
                symbol=symbol,
                date=day,
                regime=regime,
                base_price=float(rng.uniform(80.0, 2500.0)),
                seed=int(rng.integers(0, 2**31 - 1)),
            )
            specs.append(spec)
            frames.setdefault(symbol, []).append(_simulate_session(spec))

    datasets: dict[str, tuple[pd.DataFrame, str]] = {}
    for symbol, sessions in frames.items():
        datasets[symbol] = (pd.concat(sessions), "sim")
    return datasets, specs
