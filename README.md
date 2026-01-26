# Brooks Price Action Trading Coach

A decision-support, journaling, analytics, and premarket briefing system for discretionary day traders, grounded in Al Brooks price action concepts.

**ADVISORY ONLY**: This system does NOT auto-trade. It provides read-only market data + AI-powered analysis.

## Quick Start

```bash
# 1. Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up .env file
cp env.example .env
# Edit .env with your API keys (see below)

# 4. Start the web interface
python -m app.main web

# 5. Open http://localhost:8000 in your browser
#    - Register a new account or sign in
#    - All your trades are private to your account

# 6. To stop the application:
#    Press Ctrl+C in the terminal, or run:
pkill -f "python -m app.main web"
or
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "✅ Application stopped" || echo "No application running on port 8000"   
```

### Environment Variables (.env)

Create a `.env` file in the project root:

```bash
# LLM Configuration (REQUIRED for AI analysis)
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://your-llm-proxy.com/v1
LLM_MODEL=claude-sonnet-4.5
LLM_WORKERS=20  # Concurrent LLM calls (default: 20)

# Authentication (REQUIRED - generate a secure random string)
JWT_SECRET=your-super-secret-jwt-key-change-this
JWT_EXPIRATION_HOURS=24  # Optional, defaults to 24 hours

# Data Provider (optional - defaults to yfinance)
DATA_PROVIDER=polygon  # or yfinance
POLYGON_API_KEY=your_polygon_key_here

# Robinhood Integration (optional)
ROBINHOOD_USERNAME=your_username
ROBINHOOD_PASSWORD=your_password

# API Key for external scripts (optional)
APP_API_KEY=your-api-key-for-scripts
```

## Web Interface

Start the web UI:
```bash
python -m app.main web
# Or with custom port:
python -m app.main web --port 3000
```

> **Note**: After activating the virtual environment, `python` works. Before activation, use `python3`.

Open http://localhost:8000 in your browser.

**Stop the application:**
```bash
# Option 1: Press Ctrl+C in the terminal
# Option 2: Kill from another terminal
pkill -f "app.main web"
```

### Features

- **User Authentication** - Secure email/password login with JWT sessions
- **Multi-User Support** - Each user's trades are private and isolated
- **Dashboard** - Stats, recent trades, strategy performance
- **Trade Journal** - Add/edit trades with automatic P&L calculation
- **AI Coaching Review** - Brooks-style trade analysis with manual trigger and cancel button
- **Bulk Import** - CSV upload with timezone support (TradingView, Robinhood, generic)
- **Strategy Management** - Categorize, merge, and track strategies
- **Settings** - Customize prompts, manage tickers, upload training materials
- **Dark/Light Mode** - Toggle theme preference

## Key Features

### User Authentication & Data Privacy

The system provides secure multi-user support:

- **Account Registration** - Create an account with email and password
- **Secure Sessions** - JWT-based authentication with HTTP-only cookies
- **Data Isolation** - Each user only sees their own trades and analytics
- **Protected Routes** - All pages require login; unauthenticated users are redirected to sign in
- **API Protection** - Write operations require authentication (login or API key)

To get started:
1. Visit `http://localhost:8000`
2. Click "Register" to create an account
3. Sign in with your credentials
4. All your trades are private to your account

### AI-Powered Trade Analysis

Click "Generate Review" on any trade to get:
- **Setup Classification** - AI identifies Brooks-style setups
- **Context Analysis** - Daily/2H/5min regime and always-in direction
- **Entry Quality** - Signal bar analysis, entry location
- **Coaching** - What was good, areas to improve, rules for next time
- **Grade** - A through F with explanation

Features:
- **Manual Trigger** - AI review only runs when you click the button
- **Cancel Button** - Stop generation mid-process
- **Caching** - Reviews are cached and reused
- **No Look-Ahead Bias** - Only uses data available at entry time

### Trade Journal

- **Stop Loss (SL) & Take Profit (TP)** - Separate fields for risk management
- **Timezone Support** - Import trades in any timezone, displays in market timezone
- **Multi-leg Trade Matching** - Position Accumulator algorithm for complex trades
- **Currency Conversion** - Automatic USD conversion with historical rates
- **Duration Tracking** - Human-readable hold time display

### Training Materials (RAG)

Upload your trading books and rules - the AI uses **Retrieval-Augmented Generation** to find relevant sections for each trade:

- **Smart Retrieval** - Chunks materials and finds relevant sections per trade
- **Full Book Support** - Upload entire Al Brooks books; RAG finds relevant pages
- **Vector Search** - Uses ChromaDB + sentence-transformers for semantic matching
- **Auto-Indexing** - Materials are indexed on upload
- **Prioritizes Text** - Your `.txt` rules files are weighted higher than PDFs

### Bulk Import

Supports multiple formats:
- **TradingView Order History** - Auto-detects from CSV headers
- **Robinhood** - Direct API integration
- **Generic CSV** - Flexible column mapping

### Hong Kong Stocks (AllTick API)

For better HK stock data, we support [AllTick API](https://alltick.co):
- Real-time and historical K-line data
- 10-level order book
- Free tier available

To use AllTick for HK stocks, add to your `.env`:
```bash
ALLTICK_TOKEN=your_alltick_token
```

Get a free token at [alltick.co](https://alltick.co)

The system automatically uses AllTick for HK stocks when the token is configured, falling back to yfinance otherwise.

### International Stocks

- Hong Kong (HKEX:0700 → 0700.HK)
- China (SSE, SZSE)
- UK, Japan, and more
- Automatic fallback to yfinance for non-US stocks

## CSV Import Format

### Generic CSV
```csv
ticker,direction,entry_price,exit_price,size,trade_date,sl,tp,notes
SPY,long,475.50,478.00,100,2024-01-15,474.00,480.00,Strong pullback
AAPL,short,185.00,182.50,50,2024-01-15,187.00,180.00,Failed breakout
```

| Column | Required | Description |
|--------|----------|-------------|
| ticker/symbol | Yes | Stock symbol |
| entry_price | Yes | Entry price |
| exit_price | Yes | Exit price |
| direction/side | No | "long" or "short" (default: long) |
| size/qty/shares | No | Position size (default: 1) |
| trade_date/date | No | Trade date (default: today) |
| sl/stop_loss | No | Stop Loss level |
| tp/take_profit | No | Take Profit level |
| strategy | No | Strategy name |
| notes | No | Trade notes |

### TradingView Order History

Upload directly from TradingView's export. The system auto-detects:
- Order matching using Position Accumulator algorithm
- Timezone conversion (select your local timezone on upload)
- Multi-leg trade consolidation

## CLI Commands

### Trade Management

```bash
# Add a trade manually
python -m app.main trade add \
  --ticker AAPL \
  --direction long \
  --entry 150.00 \
  --exit 152.00 \
  --sl 148.50 \
  --tp 154.00 \
  --size 100

# List recent trades
python -m app.main trade list --limit 20
```

### Reports

```bash
# Generate premarket report
python -m app.main report premarket --date 2024-01-15

# Generate end-of-day report
python -m app.main report eod --date 2024-01-15

# Generate weekly report
python -m app.main report weekly --week 2024-W03
```

### Statistics

```bash
# View strategy leaderboard
python -m app.main stats strategies

# Analyze your edge
python -m app.main stats edge

# Get performance summary
python -m app.main stats summary --days 30
```

## Configuration

### config.yaml

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

# Data provider
data_provider: polygon  # or yfinance
```

### Settings Page

The web interface Settings page allows you to:
- **Manage Tickers** - Add/remove favorite tickers
- **AI Prompts** - Customize system and user prompts for trade analysis
- **Cache Settings** - Enable/disable review caching
- **Training Materials (RAG)** - Upload PDFs and documents; smart retrieval finds relevant sections per trade
- **Strategy Management** - Edit, merge, categorize strategies

## Project Structure

```
my_ai_powered_trading_assistant/
├── tickers.txt              # Your favorite tickers
├── imports/                 # Drop CSVs here for bulk import
├── .env                     # API keys and secrets
├── config.yaml              # Settings
├── app/
│   ├── main.py              # CLI entry point
│   ├── config.py            # Configuration management
│   ├── config_prompts.py    # LLM prompt templates
│   ├── auth/                # Authentication module
│   │   ├── service.py       # JWT, password hashing, user management
│   │   └── email.py         # Email service (optional)
│   ├── data/
│   │   ├── providers.py     # Market data (yfinance/Polygon)
│   │   ├── cache.py         # OHLCV caching
│   │   ├── currency.py      # Currency conversion
│   │   └── robinhood.py     # Robinhood integration
│   ├── features/
│   │   ├── ohlc_features.py # Technical indicators
│   │   ├── brooks_patterns.py # Pattern detection
│   │   └── magnets.py       # Key level detection
│   ├── journal/
│   │   ├── models.py        # SQLAlchemy models (User, Trade, Strategy)
│   │   ├── ingest.py        # Trade import
│   │   ├── analytics.py     # Statistics
│   │   └── coach.py         # AI trade review
│   ├── reports/
│   │   ├── premarket.py     # Premarket reports
│   │   ├── eod.py           # End-of-day report
│   │   └── weekly.py        # Weekly summary
│   ├── web/
│   │   ├── server.py        # FastAPI web server
│   │   ├── routes/          # API route handlers
│   │   │   └── auth.py      # Authentication routes
│   │   └── templates/       # HTML templates
│   └── llm/
│       ├── analyzer.py      # LLM analysis engine
│       └── prompts.py       # Prompt templates
├── outputs/                 # Generated reports
├── data/                    # SQLite database
└── materials/               # Training materials for LLM
```

## Brooks Price Action Concepts

This system implements Al Brooks price action methodology:

- **Trend vs Trading Range** - Market regime classification
- **Always-In** - Direction you should be if forced to hold
- **2nd Entry** - Second attempt after failed first entry
- **Breakout Pullback** - Pullback to retest breakout level
- **Wedge/3-Push** - Three pushes with diminishing momentum
- **Failed Breakout** - Break that fails to follow through
- **Trader's Equation** - Probability × Reward must exceed Risk
- **Magnets** - Key levels that attract price

## Troubleshooting

### Common Issues

**"Authentication required" or redirect to login**
- Ensure you're logged in - all pages require authentication
- Check that `JWT_SECRET` is set in `.env`
- Clear cookies and try logging in again

**"LLM analysis unavailable"**
- Check your `LLM_API_KEY` in `.env`
- Verify `LLM_BASE_URL` is correct

**Rate limiting on data fetch**
- The system includes automatic retry with exponential backoff
- For Polygon, ensure you have sufficient API credits
- yfinance is free but slower

**International stocks not loading**
- Use format `HKEX:0700` or `0700.HK` for Hong Kong
- System auto-converts to yfinance format

**Cached review showing old data**
- Click "Regenerate Review" to force a fresh analysis

**Can't see my trades after login**
- Trades are scoped per user - you only see trades you created
- If migrating from single-user mode, run the migration script

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. Trading involves substantial risk of loss. Past performance is not indicative of future results. Always do your own research and consider your financial situation before trading.

## License

MIT License - see LICENSE file for details.
