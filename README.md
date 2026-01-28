# Brooks Price Action Trading Coach

An AI-powered trading journal and coaching system grounded in Al Brooks price action methodology.

> **Advisory Only** — This tool does NOT auto-trade. It provides decision-support, journaling, and analysis.

---

## Features

| Feature | Description |
|---------|-------------|
| **Trade Journal** | Log trades with automatic P&L, R-multiple, and duration tracking |
| **AI Coaching** | Brooks-style trade review with setup classification and coaching |
| **Multi-Timeframe Analysis** | Daily, 2H, and 5-minute market context |
| **Strategy Tracking** | Categorize trades, track performance, discover your edge |
| **Training Materials (RAG)** | Upload your trading books; AI finds relevant sections per trade |
| **Bulk Import** | CSV import from TradingView, Robinhood, IBKR, or generic format |
| **Multi-User** | Secure accounts with private, isolated data |

---

## Quick Start (Local Development)

```bash
# 1. Clone and setup
git clone <your-repo-url>
cd my_ai_powered_trading_assistant

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp env.example .env
# Edit .env with your LLM API key (see Configuration below)

# 5. Start the app
python -m app.main web

# 6. Open http://localhost:8000
#    Create an account and start logging trades!
```

**To stop:** Press `Ctrl+C` in the terminal.

---

## Configuration

### Required: LLM API Key

The AI coaching feature requires an LLM API. Add to your `.env`:

```bash
LLM_API_KEY=your-api-key
LLM_BASE_URL=https://your-llm-endpoint.com/v1
LLM_MODEL=claude-sonnet-4.5
```

### Optional: Market Data

```bash
# Default: yfinance (free, no key needed)
DATA_PROVIDER=yfinance

# Better: Polygon.io (faster, free tier available)
DATA_PROVIDER=polygon
POLYGON_API_KEY=your-polygon-key
```

### Optional: Futures Data

```bash
DATABENTO_API_KEY=your-databento-key
```

---

## Web Interface

### Pages

| Page | Description |
|------|-------------|
| **Dashboard** | Overview with stats, recent trades, P&L chart |
| **Trade Journal** | Full trade list with search, filter, and AI reviews |
| **Add Trade** | Manual trade entry or bulk CSV import |
| **Statistics** | Strategy performance, win rates, edge analysis |
| **Settings** | Manage tickers, prompts, strategies, training materials |

### Keyboard Shortcuts

Press `?` anywhere to see all shortcuts.

| Shortcut | Action |
|----------|--------|
| `g d` | Go to Dashboard |
| `g t` | Go to Trades |
| `g s` | Go to Statistics |
| `Ctrl/Cmd + N` | New Trade |
| `Esc` | Close modal |

---

## AI Trade Review

Click **"Generate Review"** on any trade to get:

- **Setup Classification** — Identifies Brooks-style patterns (2nd entry, breakout pullback, wedge, etc.)
- **Context Analysis** — Daily/2H/5min market regime and always-in direction
- **Entry Quality** — Signal bar analysis, entry location evaluation
- **Coaching Feedback** — What was good, areas to improve, specific rules
- **Grade** — A through F with explanation

### Training Materials (RAG)

Upload your trading books and rules in **Settings → Training Materials**:

- Supports PDF and TXT files
- AI automatically finds relevant sections for each trade
- Your rules files are prioritized over general content

---

## Bulk Import

### Supported Formats

| Format | Description |
|--------|-------------|
| **Generic CSV** | Flexible column mapping (see below) |
| **TradingView** | Auto-detected from export headers |
| **Robinhood** | Direct API integration |
| **IBKR Flex** | Auto-import via Flex Web Service |

### Generic CSV Format

```csv
ticker,direction,entry_price,exit_price,size,trade_date,sl,tp,notes
SPY,long,475.50,478.00,100,2024-01-15,474.00,480.00,Strong pullback
```

| Column | Required | Description |
|--------|----------|-------------|
| `ticker` or `symbol` | Yes | Stock symbol |
| `entry_price` | Yes | Entry price |
| `exit_price` | Yes | Exit price |
| `direction` or `side` | No | "long" or "short" (default: long) |
| `size` or `qty` | No | Position size (default: 1) |
| `trade_date` | No | Trade date (default: today) |
| `sl` or `stop_loss` | No | Stop loss level |
| `tp` or `take_profit` | No | Take profit level |
| `strategy` | No | Strategy name |
| `notes` | No | Trade notes |

---

## CLI Commands

```bash
# Add a trade
python -m app.main trade add --ticker AAPL --direction long --entry 150 --exit 152 --size 100

# List trades
python -m app.main trade list --limit 20

# Generate premarket report
python -m app.main report premarket

# View strategy stats
python -m app.main stats strategies
```

---

## Deployment (Production)

### Deploy to Render with Supabase

For production deployment with user accounts and cloud storage:

#### 1. Create Supabase Project

1. Create project at [supabase.com](https://supabase.com)
2. Get credentials from **Settings → API**:
   - `SUPABASE_URL`
   - `SUPABASE_ANON_KEY`
   - `SUPABASE_SERVICE_KEY`
3. Get database URL from **Settings → Database → Connection string**

#### 2. Run Migrations

In Supabase **SQL Editor**, run:
- `supabase/migrations/001_initial_schema.sql`
- `supabase/migrations/002_storage_policies.sql`

#### 3. Create Storage Bucket

In **Storage**, create a private bucket named `materials`

#### 4. Deploy to Render

1. Push code to GitHub
2. In Render, create **New → Web Service**
3. Connect your repo
4. Set environment variables:

```
DATABASE_URL=postgresql://postgres.PROJECT:PASSWORD@aws-0-REGION.pooler.supabase.co:6543/postgres
SUPABASE_URL=https://PROJECT.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key
LLM_API_KEY=your-llm-key
LLM_BASE_URL=https://your-llm-endpoint/v1
LLM_MODEL=claude-sonnet-4.5
APP_URL=https://your-app.onrender.com
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **"Invalid email or password"** | Create an account first (click "Create an Account") |
| **"LLM analysis unavailable"** | Check `LLM_API_KEY` and `LLM_BASE_URL` in `.env` |
| **Slow data fetching** | Switch from yfinance to Polygon for faster data |
| **Database connection failed** | For local dev, ensure Supabase keys are commented out in `.env` |

---

## Project Structure

```
my_ai_powered_trading_assistant/
├── app/
│   ├── main.py           # CLI entry point
│   ├── auth/             # Authentication
│   ├── data/             # Market data providers
│   ├── journal/          # Trade models & analytics
│   ├── llm/              # AI analysis engine
│   └── web/              # FastAPI server & templates
├── supabase/migrations/  # Database schema
├── materials/            # Training materials (local)
├── data/                 # SQLite database (local)
└── .env                  # Configuration
```

---

## Brooks Price Action Concepts

This system implements Al Brooks methodology:

- **Trend vs Trading Range** — Market regime classification
- **Always-In** — Direction you should hold if forced
- **2nd Entry** — Second attempt after failed first
- **Breakout Pullback** — Pullback to retest breakout
- **Wedge/3-Push** — Three pushes with diminishing momentum
- **Failed Breakout** — Break that fails to follow through
- **Trader's Equation** — Probability × Reward > Risk

---

## Disclaimer

This software is for educational purposes only. It is not financial advice. Trading involves substantial risk of loss. Past performance is not indicative of future results.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
