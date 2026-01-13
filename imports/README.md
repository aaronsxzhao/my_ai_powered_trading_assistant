# Trade Import Folder

Drop your trade CSV files here for bulk import.

## How to Use

1. **Place CSV files in this folder** (`imports/`)
2. **Run the import command**: `brooks trade bulk-import`
3. **Processed files** will be moved to `imports/processed/`

## CSV Format

Your CSV should have these columns (case-insensitive):

| Column | Required | Description |
|--------|----------|-------------|
| `ticker` or `symbol` | ✅ Yes | Stock symbol (e.g., SPY, AAPL) |
| `entry_price` or `entry` | ✅ Yes | Entry price |
| `exit_price` or `exit` | ✅ Yes | Exit price |
| `stop_price` or `stop` | Recommended | Stop loss price |
| `direction` or `side` | No | "long" or "short" (default: long) |
| `trade_date` or `date` | No | Trade date YYYY-MM-DD (default: today) |
| `size` or `qty` or `shares` | No | Position size (default: 1) |
| `strategy` | No | Strategy name (LLM will classify if empty) |
| `notes` | No | Trade notes |
| `entry_reason` | No | Reason for entry |

## Example CSV

```csv
ticker,direction,entry_price,exit_price,stop_price,size,trade_date,notes
SPY,long,475.50,478.00,474.00,100,2024-01-15,Strong pullback to EMA
AAPL,short,185.00,182.50,187.00,50,2024-01-15,Failed breakout at resistance
QQQ,long,405.00,403.50,403.00,75,2024-01-15,Stopped out quickly
NVDA,long,550.00,565.00,545.00,25,2024-01-15,Trend continuation
```

## LLM Classification

If you don't specify a `strategy` column, the system will use AI (Claude/GPT) to automatically classify each trade into Brooks-style setups like:
- `second_entry_buy`
- `breakout_pullback_long`
- `wedge_reversal_short`
- etc.

Make sure `OPENAI_API_KEY` is set in your `.env` file for this feature.

## Broker-Specific Formats

The import supports multiple broker formats:

- `--broker generic` (default)
- `--broker thinkorswim`
- `--broker tradovate`

## Tips

- You can drop multiple CSV files at once
- After import, check `imports/processed/` for your files
- Run `brooks trade list` to verify imports
- Run `brooks stats strategies` to see strategy breakdown
