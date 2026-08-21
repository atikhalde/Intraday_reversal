# Intraday Reversal Scanner

A five-minute **Nifty 500 bullish reversal scanner** built around the same structure seen in ORCHPHARMA on 16 June 2026:

> decline → sell-side liquidity sweep/spring → demand response → higher-low test → no-supply contraction → bullish displacement/SOS → local pivot break

It also recognizes the faster **hammer/spring → full-bodied confirmation** variant. It sends a Telegram alert only after a completed confirmation candle; it does not place orders.

## What is implemented

- **Universe:** exactly 500 current Nifty 500 symbols, mapped to Dhan security IDs and Yahoo `.NS` symbols.
- **Primary candles:** Dhan v2 five-minute intraday historical API.
- **Immediate fallback:** each Dhan request gets one 250 ms attempt, with zero retries and zero retry backoff. As each failure arrives, a single-attempt batched yfinance fallback starts while remaining Dhan requests are still in flight; Yahoo uses an isolated cache and hard per-batch deadline.
- **No look-ahead:** only completed five-minute candles are analyzed.
- **Walk-forward detector:** every historical test evaluates only information available at that candle.
- **Rare by design:** confirmations are accepted only mid-session, only after a deep decline, a flushed session low, a reclaimed bounce, and strong displacement volume. At most **one alert per symbol per session** and a hard **portfolio budget of three alerts per day** (strongest setups win).
- **Higher-low stop:** the working invalidation sits just below the most recent higher-low test (the spring low anchors the hammer path), keeping 1R targets close and realistic.
- **Telegram alerts:** setup score, confirmation, broken pivot/retest level, immediate failure, full invalidation, and 1R/2R reference levels.
- **Deduplication:** local signal keys are retained for seven days.
- **Automation:** CI plus two clearly named operational workflows: live intraday scanning and past-data PDF backtesting.
- **Backtest artifacts:** walk-forward signal outcomes, MFE/MAE in R, a readable PDF report, and a detailed CSV.
- **Credential-free validation:** a deterministic regime-based market simulator (`reversal_scanner.simulation`) plus `scripts/validate_strategy.py` measure signal frequency and 1R/2R precision without any provider access.

## Pattern logic

The scanner is a structural state machine, not a candlestick-name search. Every gate exists to keep alerts rare and precise; together they reject the overwhelming majority of candidate candles.

### 1. Context

The stock must have declined by a configurable minimum percentage (default 2.5%) into the candidate low. Sub-2.5% dips on liquid Nifty 500 stocks are noise, not distribution.

### 2. Spring / failed breakdown

The candidate low must sweep or marginally pierce the lowest low in the configured liquidity lookback **and** sit within sweep tolerance of the entire session's low so far — the flush must clear the whole move, not land mid-trend. It must then show at least one of:

- stopping-volume expansion; or
- hammer-like lower-wick rejection.

A red body does **not** disqualify a spring/test candle if its location and rejection anatomy are bullish.

### 3. Test / LPS

For the full setup, the scanner seeks a retest that:

- holds above the spring low;
- returns into the lower portion of the recovery;
- rejects lower prices through its wick and close location; and
- trades on materially less volume than the spring.

A narrow, lower-volume candle after the test earns a no-supply/LPS score component but is optional.

### 4. SOS confirmation

The latest completed candle must:

- close bullish and near its high (close location ≥ 0.75 of the range);
- have a large body relative to its range;
- expand versus the recent median range;
- carry at least double the recent median volume on its own — a pop that only looks big next to a tiny test bar is not displacement;
- close above the intervening local pivot; and
- reclaim a meaningful share of the whole decline, both as a fraction of the move and in absolute price terms (default ≥ 1.2% above the spring low). Shallow bounces in a live downtrend fail this gate.

That close is treated as local CHoCH/BOS confirmation. The scanner does not claim that a later news-driven repricing was technically predictable.

### 5. Rarity controls

- Confirmations are accepted only inside the mid-session window (default 09:45–15:05 IST).
- The higher-low test must hold: no bar between the test and the displacement may trade back through the tested low.
- The fast hammer → SOS path is tightly gated: the sweep must carry stopping volume and the displacement must break the hammer's own high.
- Walk-forward history keeps only the strongest confirmation per symbol per session, and both backtests and the live scanner cap alerts at `scanner.max_signals_per_day` (default 3) per trading day, highest score first.

## ORCHPHARMA regression result

The repository includes the 16 June 2026 five-minute fixture. The walk-forward detector produces exactly the intended technical signal:

```text
2026-06-16 13:35:00 | ORCHPHARMA | spring-test-SOS | score=100 |
spring=2026-06-16 13:05:00 @ 900.00 | confirm=917.35 |
pivot=908.25 | invalidation=899.88
```

The working stop (899.88) sits just below the 13:25 higher-low test at 900.55; the later 14:55 news spike is not needed by or visible to the 13:35 decision.

## Local installation

Python 3.11 or newer is required.

```bash
git clone https://github.com/atikhalde/Intraday_reversal.git
cd Intraday_reversal
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

Set credentials in the shell or load them from your own secret manager. The scanner intentionally does not parse or commit `.env` automatically.

```bash
export DHAN_CLIENT_ID="..."
export DHAN_ACCESS_TOKEN="..."
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_CHAT_ID="..."
```

Never put these values in YAML, Python files, workflow files, commits, or Action logs.

## Run it

### One production scan

```bash
reversal-scanner scan --once --require-market-open
```

### Always-on mode — preferred for live alerts

```bash
reversal-scanner scan --loop --require-market-open
```

The loop wakes just after each five-minute boundary and analyzes the newly completed candle.

### Safe Yahoo-only smoke test

```bash
reversal-scanner scan --once --provider yfinance --dry-run --symbols RELIANCE,TCS
```

### Limit a Dhan test

```bash
reversal-scanner scan --once --dry-run --max-symbols 5
```

### Replay the ORCHPHARMA case

```bash
reversal-scanner backtest tests/fixtures/orchpharma_2026-06-16_5m.csv \
  --symbol ORCHPHARMA
```

### Generate a historical PDF report locally

Credential-free deterministic fixture run:

```bash
reversal-scanner backtest-report \
  --source fixture \
  --symbols ORCHPHARMA \
  --start 2026-06-16 \
  --end 2026-06-16 \
  --fixture tests/fixtures/orchpharma_2026-06-16_5m.csv \
  --output-pdf artifacts/backtest-report.pdf \
  --results-csv artifacts/backtest-results.csv
```

Production Dhan run for all current Nifty 500 constituents:

```bash
reversal-scanner backtest-report \
  --source dhan \
  --symbols NSE_NIFTY_500 \
  --start 2026-07-14 \
  --end 2026-08-14
```

`NSE_NIFTY_500` expands to the complete bundled 500-stock universe. A comma-separated list such as `RELIANCE,TCS,INFY` remains supported for a narrower report.

Dhan is attempted once per symbol. Any failed symbol is submitted directly to yfinance with no retry, backoff, or intentional delay. Yahoo uses a process-isolated cache and a hard deadline for each single-attempt batch so a stalled library call cannot block an entire workflow. Yahoo five-minute history has a short retention window, so Dhan is the appropriate source for older ranges.

The report evaluates only candles after each confirmation and within the same session. It records whether 1R, 2R, or full invalidation was reached, plus maximum favorable/adverse excursion. If a single five-minute candle touches both a target and stop, the report conservatively counts the stop first because OHLC data does not reveal intrabar order. These statistics exclude costs, slippage, partial exits, and position sizing.

## Telegram setup

For an existing bot:

1. Set `TELEGRAM_BOT_TOKEN` to the BotFather token.
2. Send a message to the bot or add it to the destination group.
3. Set `TELEGRAM_CHAT_ID` to the user/group ID. Group IDs commonly begin with `-100`.
4. Run a small dry run first, then run without `--dry-run` when ready.

A signal alert contains technical levels only. It is not an order and not financial advice.

## GitHub Actions

Create these repository secrets under **Settings → Secrets and variables → Actions**:

- `DHAN_CLIENT_ID`
- `DHAN_ACCESS_TOKEN`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

The repository has CI plus these **two operational workflows**, listed in Actions with numbered names:

1. **`1 - Intraday scanner`** — manual dispatch and a weekday five-minute schedule. Scheduled runs use Dhan and Telegram. For a credential-free manual smoke test choose `provider: yfinance`, `dry_run: true`, and a small `max_symbols` value. To verify Telegram end to end, set `sample_alert: true`; this replays the ORCHPHARMA fixture and sends a prominent **SAMPLE TEST — NOT A LIVE SIGNAL** alert without requiring Dhan.
2. **`2 - Past backtest PDF`** — manual historical walk-forward run. It defaults to Dhan with immediate yfinance fallback and `NSE_NIFTY_500`, while start and end dates are intentionally blank for manual selection. It always uploads `backtest-report.pdf` and `backtest-results.csv` as the `past-backtest-<run number>` artifact retained for 30 days. The PDF includes each alert's score, spring, confirmation, broken pivot/retest, immediate failure, full invalidation, 1R/2R targets, trigger reasons, provider, MFE/MAE, and same-session outcome.

For a production backtest, open **Actions → 2 - Past backtest PDF → Run workflow**, retain `dhan` and `NSE_NIFTY_500`, enter the required dates, and run it. Dhan repository secrets must be configured. Selecting `yfinance` is credential-free but its five-minute retention is provider-limited. The deterministic fixture option remains available only for `ORCHPHARMA` with dates covering `2026-06-16`.

`CI` separately runs lint, the automated suite, PDF parsing checks, and the ORCHPHARMA walk-forward regression on pushes and pull requests.

The intraday scheduled workflow starts one minute after nominal candle boundaries and exits without provider calls outside the NSE scan window.

### Important scheduling limitation

GitHub Actions cron is best-effort. It can start several minutes late, so it cannot guarantee time-sensitive intraday delivery. The scanner rejects stale confirmations after seven minutes, which avoids presenting an old signal as fresh but means a delayed Action can miss a setup. Use `--loop` on an always-on VPS/container for dependable live operation; use Actions as a convenient backup or smoke-test runner.

Dhan access-token lifetime also depends on the token type issued by your account. Keep the `DHAN_ACCESS_TOKEN` secret current.

## Configuration

Defaults are documented in [`config/default.yaml`](config/default.yaml). Do not edit strategy source code to tune thresholds. Create an override file instead:

```yaml
# config/my-settings.yaml
strategy:
  min_decline_pct: 2.2
  min_score: 70
filters:
  min_session_turnover_inr: 20000000
```

Run with:

```bash
reversal-scanner --config config/my-settings.yaml scan --loop --require-market-open
```

Main safeguards:

- minimum price: ₹20;
- maximum price: ₹10,000;
- minimum running session turnover: ₹2 crore;
- last completed candle only;
- maximum signal age: seven minutes;
- confirmations only mid-session (default 09:45–15:05 IST), never in opening noise or the final squeeze;
- one alert per symbol per session and a portfolio budget of three alerts per day;
- working stop just below the most recent higher-low test (buffered), full invalidation anchored at the spring low on the hammer path.

Thresholds should be recalibrated through out-of-sample walk-forward tests before live decision-making.

## Validation without credentials

`scripts/validate_strategy.py` replays a deterministic regime-based market (chop, trend, genuine reversal, and flawed or perfect bull traps) through the detector and reports signals-per-day, 1R/2R precision, and a regime breakdown:

```bash
python scripts/validate_strategy.py --days 30 --symbols 24 --seed 7
```

With the committed defaults this harness produces well under one alert per day on average, never more than three, with roughly four in five alerts reaching 1R before the higher-low stop — and virtually every genuine reversal structure captured. The simulator is deterministic per seed, so the harness doubles as a stable regression check when tuning thresholds.

## Refresh the Nifty 500 mapping

The committed universe avoids downloading Dhan's large scrip master every five minutes. Refresh it after an index rebalance:

```bash
python scripts/update_universe.py
pytest -q
```

The update script downloads the official NSE Nifty 500 constituent CSV and Dhan's official scrip master, requires all 500 symbols to map, and updates both source and packaged copies.

## Tests

```bash
ruff check .
pytest --cov=reversal_scanner --cov-report=term-missing
```

Tests cover:

- the exact ORCHPHARMA 13:05–13:35 spring/test/SOS sequence;
- no alert before 13:35;
- no dependency on the later news spike;
- the faster hammer-confirmation variant, including its stopping-volume requirement;
- zero false alerts across sixty simulated pure-chop sessions;
- the mid-session confirmation window and the reclaim gate;
- at most one signal per symbol per session;
- the portfolio daily budget in backtests and the live scanner's budget plus per-symbol deduplication;
- one-attempt Dhan → immediate Yahoo fallback behavior for live and historical ranges;
- incomplete-candle removal;
- PDF/CSV report generation and PDF parsing;
- post-signal 1R/2R outcome evaluation;
- 500 unique mapped constituents; and
- Telegram alert structure.

## Risk and scope

This is a research/alerting tool, not an execution engine. Historical candle quality, corporate actions, feed latency, rate limits, auction prints, news, circuits, and slippage can all change real outcomes. Validate signals independently and define position sizing and maximum-loss rules outside the scanner.
