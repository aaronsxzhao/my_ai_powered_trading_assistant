# Brooks Price Action Trading Coach

A decision-support, journaling, analytics, and premarket briefing system for discretionary day traders, grounded in Al Brooks price action concepts.

⚠️ **ADVISORY ONLY**: This system does NOT auto-trade. It is read-only market data + analysis by default.

## Features

### 1. Trade Journal + Post-Trade Coach (Brooks-style)
- Manual trade entry or CSV import from brokers
- Automatic computation of:
  - R-multiple (PnL / initial risk)
  - MAE/MFE (Maximum Adverse/Favorable Excursion)
  - Hold time, slippage, win/loss, expectancy
- Brooks-style trade review:
  - Context analysis (trend vs trading range, always-in direction)
  - Setup classification (breakout pullback, 2nd entry, wedge, failed breakout, etc.)
  - Trader's equation evaluation (probability × reward vs risk)
  - Error detection (countertrend without reversal, poor scalp math, etc.)
  - Actionable coaching: what was good, what was flawed, rule for next time

### 2. Strategy Tracking + Edge Discovery
- Strategy taxonomy (with-trend, countertrend, trading range, special)
- Per-strategy statistics:
  - Count, win rate, avg R, expectancy, profit factor
  - MAE/MFE analysis, best time of day
  - Recent 20-trade performance
- Edge analysis: strengths, weaknesses, coaching focus
- Weekly "coaching focus": behaviors to stop and double down on

### 3. Premarket / Before-Session Report
- Multi-timeframe analysis (daily, 2h, 5m)
- Regime detection (trend vs range)
- Always-in direction
- Magnet map (key levels, measured moves)
- Plan A / Plan B with specific setups
- Avoid list (low-quality conditions)

### 4. Daily + Weekly Review
- EOD summary: PnL, best/worst trade, rule violations, improvement focus
- Weekly summary: strategy leaderboard, leaks, top 3 rules for next week

### 5. Extra Features
- **Always-In inference engine**: Estimates always-in direction across timeframes
- **Two legs correction detector**: Warns about expecting V-reversals too early
- **Magnet map**: Prior day H/L/C, gaps, measured moves, swing points
- **Risk controls checklist**: Max daily loss, losing streak warnings

## Installation

### Prerequisites
- Python 3.11 or higher
- pip or pipenv

### Setup

1. Clone the repository:
```bash
cd my_ai_powered_trading_assistant
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy and configure environment variables:
```bash
cp env.example .env
# Edit .env with your API keys (optional)
```

5. Initialize the database:
```bash
python -m app.main config init
```

## Usage

### CLI Commands

#### Trade Management

```bash
# Add a trade manually
brooks trade add \
  --ticker AAPL \
  --direction long \
  --entry 150.00 \
  --exit 152.00 \
  --stop 148.50 \
  --size 100 \
  --strategy second_entry_buy

# Import trades from CSV
brooks trade import trades.csv --broker generic

# Review a trade with coaching
brooks trade review 1

# List recent trades
brooks trade list --limit 20
```

#### Reports

```bash
# Generate premarket report
brooks report premarket --date 2024-01-15

# Generate premarket for specific ticker
brooks report premarket --ticker AAPL

# Generate end-of-day report
brooks report eod --date 2024-01-15

# Generate weekly report
brooks report weekly --week 2024-W03
```

#### Statistics

```bash
# View strategy leaderboard
brooks stats strategies

# Analyze your edge
brooks stats edge

# Get performance summary
brooks stats summary --days 30
```

#### Configuration

```bash
# View current config
brooks config show

# Manage favorite tickers
brooks config tickers list
brooks config tickers add NVDA
brooks config tickers remove TSLA
```

## Configuration

Edit `config.yaml` to customize:

```yaml
# Favorite tickers for premarket reports
tickers:
  - SPY
  - QQQ
  - AAPL
  - NVDA

# Risk controls
risk:
  max_daily_loss_r: 3.0
  max_losing_streak: 3
  default_risk_per_trade_pct: 1.0

# LLM enhancement (optional)
llm:
  enabled: false
  model: gpt-4o
```

## CSV Import Format

The CSV should have these columns (case-insensitive):

| Column | Required | Description |
|--------|----------|-------------|
| ticker/symbol | Yes | Stock symbol |
| entry_price/entry | Yes | Entry price |
| exit_price/exit | Yes | Exit price |
| stop_price/stop | Recommended | Stop loss price |
| direction/side | No | "long" or "short" (default: long) |
| trade_date/date | No | Trade date (default: today) |
| size/qty/shares | No | Position size (default: 1) |
| strategy | No | Strategy name |
| notes | No | Trade notes |

## Sample Output

### Premarket Report

```markdown
# Premarket Report: SPY
**Date**: 2024-01-15
**Current Price**: $475.32
**Overall Bias**: LONG (high confidence)

## Daily Chart Analysis
- **Regime**: trend_up
- **Always-In**: long
- Strong uptrend with higher highs and higher lows

## Plan A (Most Likely)
**Scenario**: Uptrend continuation
**Bias**: LONG

**Setups to look for**:
- 2nd entry buy on pullback
- Breakout pullback after new high

## ⚠️ Avoid List
- Avoid shorting without clear reversal structure
- Avoid chasing - wait for pullbacks
```

### Trade Review

```markdown
# Trade Review: AAPL (ID: 1)

## Context
- **Regime**: trend_up
- **Always-In**: long
- Strong uptrend with higher highs

## Grade: B
*Good trade with minor areas for improvement*

## Coaching
### What Was Good
- ✅ Traded with the always-in direction (long)
- ✅ Profitable trade: +1.50R

### What Was Flawed
- ❌ LEFT MONEY ON TABLE: MFE was 2.5R but only captured 1.5R

### Rule for Next Time
📌 RULE: In trending markets, trail stop below prior swing instead of fixed targets
```

## Project Structure

```
my_ai_powered_trading_assistant/
├── app/
│   ├── main.py              # CLI entry point (Typer)
│   ├── config.py            # Configuration management
│   ├── data/
│   │   ├── providers.py     # Market data providers (yfinance, etc.)
│   │   └── cache.py         # Local OHLCV caching
│   ├── features/
│   │   ├── ohlc_features.py # Technical indicators
│   │   ├── brooks_patterns.py # Brooks pattern detection
│   │   └── magnets.py       # Key level detection
│   ├── journal/
│   │   ├── models.py        # SQLAlchemy models
│   │   ├── ingest.py        # Trade import/entry
│   │   ├── analytics.py     # R-multiple, expectancy, stats
│   │   └── coach.py         # Brooks-style trade review
│   ├── reports/
│   │   ├── premarket.py     # Premarket report generator
│   │   ├── eod.py           # End-of-day report
│   │   ├── weekly.py        # Weekly summary
│   │   └── render.py        # Markdown/chart rendering
│   └── llm/
│       ├── client.py        # OpenAI-compatible client
│       └── prompts.py       # LLM prompt templates
├── tests/                   # Pytest tests
├── outputs/                 # Generated reports (by date)
├── data/                    # SQLite database + cache
├── config.yaml              # User configuration
├── requirements.txt         # Dependencies
└── README.md
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_analytics.py
```

## LLM Enhancement (Optional)

To enable LLM narrative enhancement:

1. Set your OpenAI API key in `.env`:
```
OPENAI_API_KEY=your_key_here
```

2. Enable in `config.yaml`:
```yaml
llm:
  enabled: true
  model: gpt-4o
```

The LLM will:
- Convert computed findings into Brooks-style narrative
- Never invent price data
- Always cite computed context

## Brooks Price Action Concepts

This system is built on Al Brooks price action methodology:

- **Trend vs Trading Range**: Market regime classification
- **Always-In**: The direction you should be if forced to have a position
- **2nd Entry**: Second attempt after a failed first entry
- **Breakout Pullback**: Pullback to retest breakout level
- **Wedge/3-Push**: Three pushes with diminishing momentum
- **Failed Breakout**: Price breaks level but fails to follow through
- **Trader's Equation**: Probability × Reward must exceed Risk
- **Magnets**: Price levels that attract price (prior H/L, gaps, measured moves)

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Trading involves substantial risk of loss. Past performance is not indicative of future results. Always do your own research and consider your financial situation before trading.

## License

MIT License - see LICENSE file for details.
