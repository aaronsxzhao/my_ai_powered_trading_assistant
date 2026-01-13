# Brooks Trading Coach - Examples

## Example Workflow

### Morning Routine (Before Market Open)

```bash
# 1. Generate premarket reports for all favorite tickers
brooks report premarket

# This creates:
# - outputs/2024-01-15/premarket/SPY_premarket.md
# - outputs/2024-01-15/premarket/QQQ_premarket.md
# - outputs/2024-01-15/premarket/SUMMARY.md
```

### During Market Hours (Logging Trades)

```bash
# Log a winning trade
brooks trade add \
  --ticker SPY \
  --direction long \
  --entry 475.50 \
  --exit 477.25 \
  --stop 474.50 \
  --size 50 \
  --strategy second_entry_buy \
  --notes "2nd entry after pullback to EMA, strong trend day"

# Log a losing trade
brooks trade add \
  --ticker AAPL \
  --direction short \
  --entry 185.00 \
  --exit 186.50 \
  --stop 186.00 \
  --size 100 \
  --strategy countertrend_short \
  --notes "Tried to fade the rally, stopped out"
```

### After Market Close

```bash
# 1. Review the day's trades
brooks trade list

# 2. Get coaching on specific trades
brooks trade review 1
brooks trade review 2

# 3. Generate end-of-day report
brooks report eod

# This creates:
# - outputs/2024-01-15/eod_report.md
# - outputs/2024-01-15/trades.csv
```

### Weekly Review (End of Week)

```bash
# Generate weekly report
brooks report weekly --week 2024-W03

# Check strategy performance
brooks stats strategies

# Analyze your edge
brooks stats edge
```

## Sample Premarket Report Output

```markdown
# Premarket Report: SPY
**Date**: 2024-01-15
**Current Price**: $475.32
**Overall Bias**: LONG (high confidence)

---

## Daily Chart Analysis
- **Regime**: trend_up
- **Always-In**: long
- **Confidence**: high
- Strong uptrend with higher highs and higher lows

### Key Levels (Daily)
- $480.25: Swing High (resistance)
- $472.50: Swing Low (support)
- $468.00: Prior Swing High (support)

## 2-Hour Chart Analysis
- **Regime**: trend_up
- **Always-In**: long
- Moderate uptrend, price above EMA

## 5-Minute Context (Past 3 Days)
- **Recent Regime**: trend_up
- **Strength**: strong_bull

## Magnet Map

### Resistance (above current price)
- $477.15: Prior Day High
- $480.00: Round Number (50 interval)
- $480.25: Swing High

### Support (below current price)
- $474.30: Prior Day Close
- $473.85: Prior Day Low
- $472.50: Swing Low

### Measured Move Targets
- ↑ $482.50 (leg size: $7.25)

---

## Plan A (Most Likely)
**Scenario**: Uptrend continuation
**Bias**: LONG

**Setups to look for**:
- 2nd entry buy on pullback
- Breakout pullback after new high
- Trend resumption after test of prior swing

**Entry Zones**: Pullbacks to 20 EMA or prior swing highs
**Targets**: Measured move targets, prior highs, round numbers
**Stops**: Below prior swing low or 1 ATR below entry

## Plan B (If Reversal)
**Scenario**: Trend reversal to range or downtrend
**Trigger**: Strong break below 20 EMA + bear trend bar + test that fails
**New Bias**: NEUTRAL to SHORT
**Action**: Stop looking for longs, wait for selling climax before new longs

---

## ⚠️ Avoid List
- Avoid shorting without clear reversal structure
- Avoid buying after extended move (3+ legs up without correction)
- Avoid chasing - wait for pullbacks
```

## Sample Trade Review Output

```markdown
# Trade Review: AAPL (ID: 5)

## Context
- **Regime**: trend_up
- **Always-In**: long
- Strong uptrend with higher highs and higher lows

## Setup
- **Classification**: countertrend_short
- **Quality**: POOR

## Trader's Equation
- **Risk/Reward**: Good R:R of 1.5:1 - reward justifies risk
- **Probability**: LOW probability - countertrend trade, needs strong reversal structure

## Performance
- **R-Multiple**: -1.50R
- **MAE**: 1.75R
- **MFE**: 0.25R

## Grade: D
*Below average - significant issues to address*

## Errors Detected
- ⚠️ COUNTERTREND WITHOUT CLEAR REVERSAL: Faded an uptrend without documented reversal structure (need strong bear bar, break of trendline + test)
- ⚠️ HELD LOSER TOO LONG: Lost 1.50R - should have exited at initial stop

## Coaching

### What Was Good
- ✅ Had documented entry reason
- ✅ Trade was logged - tracking is the first step to improvement

### What Was Flawed
- ❌ Losing trade: -1.50R
- ❌ COUNTERTREND WITHOUT CLEAR REVERSAL
- ❌ HELD LOSER TOO LONG

### Rule for Next Time
📌 RULE: Before taking countertrend trades, require: (1) strong reversal bar, (2) break of trendline, (3) successful test. If any missing, pass on the trade.
```

## Sample Weekly Report Output

```markdown
# Weekly Report: 2024-W03
**Period**: 2024-01-15 to 2024-01-21

## Performance Summary 🟢

| Metric | Value |
|--------|-------|
| Trading Days | 5 |
| Total Trades | 23 |
| Winners / Losers | 14 / 9 |
| Win Rate | 60.9% |
| **Total R** | **+8.75R** |
| Total PnL | $2,187.50 |
| Expectancy | +0.380R |
| Profit Factor | 1.85 |
| Avg R per Trade | +0.380R |
| Avg R per Day | +1.75R |

---

## Strategy Leaderboard 📊

| Rank | Strategy | Trades | Total R | Win% | PF |
|------|----------|--------|---------|------|-----|
| 1 | second_entry_buy | 8 | +5.25R | 75% | 3.5 |
| 2 | breakout_pullback_long | 5 | +2.50R | 60% | 2.1 |
| 3 | trend_resumption_long | 4 | +1.50R | 50% | 1.8 |
| 4 | range_fade_high | 3 | +0.50R | 67% | 1.3 |
| 5 | countertrend_short | 3 | -1.00R | 33% | 0.5 |

🏆 **Best Strategy**: second_entry_buy
⚠️ **Worst Strategy**: countertrend_short

---

## Edge Analysis

### Strengths 💪
- Overall edge is positive: +0.38R per trade
- Strong at second_entry_buy: 75% win rate, +0.66R expectancy
- Good at with-trend entries

### Weaknesses 📉
- Losing money on countertrend_short: -0.33R expectancy
- Average winner (1.1R) smaller than average loser (1.3R)

### Biggest Leaks 🕳️
- ❌ Strategy 'countertrend_short' lost 1.0R (3 trades)
- ❌ Win rate only 61% - be more selective

---

## Coaching for Next Week

### Top 3 Rules 📌
- RULE 1: Only take countertrend with clear reversal structure
- RULE 2: Trail stops in trends instead of fixed targets
- RULE 3: Only take A+ setups - skip marginal trades

### Stop Doing 🛑
- Stop trading countertrend_short until reviewed

### Double Down On ✅
- Look for more second_entry_buy setups
```

## CSV Import Format Example

Create a file `my_trades.csv`:

```csv
ticker,direction,entry_price,exit_price,stop_price,size,trade_date,strategy,notes
SPY,long,475.50,477.25,474.50,50,2024-01-15,second_entry_buy,Strong pullback to EMA
AAPL,long,185.00,186.50,183.50,100,2024-01-15,breakout_pullback_long,Break of prior high
QQQ,short,405.00,403.50,406.50,75,2024-01-15,range_fade_high,Double top at range high
SPY,long,476.00,475.00,475.00,50,2024-01-15,unclassified,Stopped out at breakeven
```

Import with:

```bash
brooks trade import my_trades.csv
```

## Strategy Names

Use these strategy names when logging trades:

### With-Trend
- `breakout_pullback_long`
- `breakout_pullback_short`
- `second_entry_buy`
- `second_entry_sell`
- `trend_resumption_long`
- `trend_resumption_short`

### Countertrend
- `failed_breakout_long`
- `failed_breakout_short`
- `wedge_reversal_long`
- `wedge_reversal_short`
- `double_bottom_long`
- `double_top_short`
- `climax_reversal_long`
- `climax_reversal_short`

### Trading Range
- `range_fade_high`
- `range_fade_low`
- `range_scalp_long`
- `range_scalp_short`

### Special
- `trend_from_open_long`
- `trend_from_open_short`
- `opening_reversal_long`
- `opening_reversal_short`
- `gap_fill_long`
- `gap_fill_short`

### Other
- `unclassified` (default if not specified)
