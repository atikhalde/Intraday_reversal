# Intraday Reversal Scanner

A five-minute **Nifty 500 bullish reversal scanner** built around the same structure seen in ORCHPHARMA on 16 June 2026:

> decline → sell-side liquidity sweep/spring → demand response → higher-low test → no-supply contraction → bullish displacement/SOS → local pivot break

It also recognizes the faster **hammer/spring → full-bodied confirmation** variant. It sends a Telegram alert only after a completed confirmation candle; it does not place orders.

## What is implemented

- **Universe:** exactly 500 current Nifty 500 symbols, mapped to Dhan security IDs and Yahoo `.NS` symbols.
- **Primary candles:** Dhan v2 five-minute intraday historical API.
- **Immediate fallback:** each Dhan request has a one-second timeout, zero retries, and zero retry backoff. Failed symbols are passed straight to a batched yfinance request.
- **No look-ahead:** only completed five-minute candles are analyzed.
- **Walk-forward detector:** every historical test evaluates only information available at that candle.
- **Telegram alerts:** setup score, confirmation, broken pivot/retest level, immediate failure, full invalidation, and 1R/2R reference levels.
- **Deduplication:** local signal keys are retained for seven days.
- **Automation:** CI plus a scheduled/manual scanner GitHub Actions workflow.

## Pattern logic

The scanner is a structural state machine, not a candlestick-name search.

### 1. Context

The stock must have declined by a configurable minimum percentage into the candidate low. This prevents every isolated green candle from being called a reversal.

### 2. Spring / failed breakdown

The candidate low must sweep or marginally pierce the lowest low in the configured liquidity lookback. It must then show at least one of:

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

- close bullish;
- have a large body relative to its range;
- expand versus the recent median range;
- expand in volume versus the median or immediately preceding candle; and
- close above the intervening local pivot.

That close is treated as local CHoCH/BOS confirmation. The scanner does not claim that a later news-driven repricing was technically predictable.

## ORCHPHARMA regression result

The repository includes the 16 June 2026 five-minute fixture. The walk-forward detector produces exactly the intended technical signal:

```text
2026-06-16 13:35:00 | ORCHPHARMA | spring-test-SOS | score=100 |
spring=2026-06-16 13:05:00 @ 900.00 | confirm=917.35 |
pivot=908.25 | invalidation=899.83
```

The later 14:55 news spike is not needed by or visible to the 13:35 decision.

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

Two workflows are included:

- `CI`: lint, eight automated tests, and the ORCHPHARMA walk-forward regression.
- `Intraday scanner`: manual dispatch and a weekday five-minute schedule.

The scheduled workflow starts one minute after nominal candle boundaries and exits without provider calls outside the NSE scan window.

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
- minimum running session turnover: ₹1 crore;
- last completed candle only;
- maximum signal age: seven minutes;
- complete structural invalidation below the buffered spring low.

Thresholds should be recalibrated through out-of-sample walk-forward tests before live decision-making.

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
- the faster hammer-confirmation variant;
- one-attempt Dhan → immediate Yahoo fallback behavior;
- incomplete-candle removal;
- 500 unique mapped constituents; and
- Telegram alert structure.

## Risk and scope

This is a research/alerting tool, not an execution engine. Historical candle quality, corporate actions, feed latency, rate limits, auction prints, news, circuits, and slippage can all change real outcomes. Validate signals independently and define position sizing and maximum-loss rules outside the scanner.
