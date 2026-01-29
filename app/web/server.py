"""
FastAPI Web Server for AI Trading Coach.

Provides a browser-based interface for:
- Dashboard with stats
- Trade management
- CSV upload
- Ticker management
- Report generation
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    IMPORTS_DIR,
    MATERIALS_DIR,
    OUTPUTS_DIR,
    get_polygon_api_key,
    load_tickers_from_file,
)
from app.config_prompts import get_cache_settings
from app.journal.analytics import TradeAnalytics
from sqlalchemy.orm import joinedload
from app.journal.models import Strategy, Trade, TradeOutcome, get_session, init_db
from app.logging_utils import install_log_safety
from app.web.dependencies import require_login
from app.web.routes import (
    auth_router,
    imports_router,
    materials_router,
    reports_router,
    settings_router,
    strategies_router,
    system_router,
    tickers_router,
    trades_router,
)
from app.web.utils import (
    get_active_strategies_cached,
    get_template_context_with_user,
    ticker_display,
    ticker_exchange,
)

logger = logging.getLogger(__name__)
try:
    install_log_safety()
except Exception:
    # Logging should never prevent app startup.
    pass


# Initialize FastAPI app with OpenAPI tags for better documentation
tags_metadata = [
    {"name": "health", "description": "Health check and status endpoints"},
    {"name": "pages", "description": "HTML page rendering"},
    {"name": "trades", "description": "Trade management operations"},
    {"name": "reviews", "description": "AI trade review generation"},
    {"name": "strategies", "description": "Strategy management"},
    {"name": "materials", "description": "Training materials and RAG"},
    {"name": "settings", "description": "Application settings"},
    {"name": "imports", "description": "Data import operations"},
    {"name": "tickers", "description": "Favorite ticker configuration"},
    {"name": "reports", "description": "Report generation and batch analysis"},
    {"name": "auth", "description": "Authentication and user management"},
]

app = FastAPI(
    title="AI Trading Coach",
    description="Advisory system for discretionary day traders using price action methodology",
    version="0.1.0",
    openapi_tags=tags_metadata,
)

app.include_router(system_router)
app.include_router(trades_router)
app.include_router(strategies_router)
app.include_router(materials_router)
app.include_router(tickers_router)
app.include_router(imports_router)
app.include_router(reports_router)
app.include_router(settings_router)
app.include_router(auth_router)

# Mount static files directory
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
# Enable auto-reload for templates in development (instant frontend changes)
templates.env.auto_reload = True

# Register custom filters
templates.env.filters["ticker_display"] = ticker_display
templates.env.filters["ticker_exchange"] = ticker_exchange


# Initialize database on startup
@app.on_event("startup")
async def startup():
    init_db()
    IMPORTS_DIR.mkdir(exist_ok=True)

    # Recalculate trade numbers on startup (ensures chronological order)
    from app.journal.models import recalculate_trade_numbers

    count = recalculate_trade_numbers()
    if count > 0:
        logger.info(f"📊 Recalculated trade numbers for {count} trades")


@app.on_event("shutdown")
async def shutdown():
    """
    Best-effort cleanup for background executors.

    Some third-party libs (notably joblib/loky, pulled in via ML deps) can leave
    a tracked semaphore behind and emit a noisy warning on interpreter shutdown.
    This proactively shuts down any reusable loky executor if it was created.
    """
    try:
        from joblib.externals.loky import reusable_executor  # type: ignore

        executor = getattr(reusable_executor, "_executor", None)
        if executor is not None:
            try:
                executor.shutdown(wait=True)
            except TypeError:
                # Older/newer signatures differ; fall back to default shutdown.
                executor.shutdown()
            try:
                reusable_executor._executor = None  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        # Non-fatal: app can still shut down cleanly.
        pass


# Favicon - prevents 404 errors in browser
@app.get("/favicon.ico")
async def favicon():
    """Return a simple SVG favicon."""
    from fastapi.responses import Response

    # Simple chart icon as SVG favicon
    svg = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
        <rect width="32" height="32" rx="4" fill="#3B82F6"/>
        <path d="M6 22 L12 16 L18 20 L26 10" stroke="white" stroke-width="2.5" fill="none" stroke-linecap="round"/>
        <circle cx="26" cy="10" r="2" fill="#10B981"/>
    </svg>"""
    return Response(content=svg, media_type="image/svg+xml")


# ==================== HEALTH & STATUS ====================

# ==================== PAGES ====================


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, user=Depends(require_login)):
    """Main dashboard page."""
    base_context = await get_template_context_with_user(request, user)
    user_id = user.id

    def _get_dashboard_data():
        analytics = TradeAnalytics()
        session = get_session()
        try:
            from sqlalchemy import func
            
            # Get recent trades for this user (limit 10) - newest by trade_number first
            recent_trades = (
                session.query(Trade)
                .filter(Trade.user_id == user_id)
                .order_by(Trade.trade_number.desc())
                .limit(10)
                .all()
            )

            # Use efficient COUNT queries for this user's trades only
            total_trades = (
                session.query(func.count(Trade.id)).filter(Trade.user_id == user_id).scalar() or 0
            )
            winners = (
                session.query(func.count(Trade.id))
                .filter(Trade.user_id == user_id, Trade.outcome == TradeOutcome.WIN)
                .scalar()
                or 0
            )
            total_r = (
                session.query(func.sum(Trade.r_multiple)).filter(Trade.user_id == user_id).scalar() or 0
            )
            win_rate = winners / total_trades if total_trades > 0 else 0

            # Calculate total P&L in USD (converting from original currency if needed)
            all_trades = session.query(Trade).filter(Trade.user_id == user_id).all()
            total_pnl = sum(t.pnl_usd for t in all_trades if t.pnl_usd is not None)

            # Get strategy stats for this user
            strategy_stats = analytics.get_all_strategy_stats(user_id=user_id)[:5]

            # Get portfolio stats with drawdown and advanced metrics
            portfolio_stats = analytics.get_portfolio_stats(user_id=user_id)

            return {
                "recent_trades": recent_trades,
                "total_trades": total_trades,
                "total_r": total_r,
                "total_pnl": total_pnl,
                "win_rate": win_rate,
                "winners": winners,
                "strategy_stats": strategy_stats,
                "portfolio_stats": portfolio_stats,
            }
        finally:
            session.close()

    data = await asyncio.to_thread(_get_dashboard_data)
    polygon_available = get_polygon_api_key() is not None

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            **base_context,
            **data,
            "tickers": load_tickers_from_file(),
            "polygon_available": polygon_available,
        },
    )


@app.get("/trades", response_class=HTMLResponse)
async def trades_page(
    request: Request,
    page: int = 1,
    per_page: int = 50,
    ticker: Optional[str] = None,
    outcome: Optional[str] = None,
    user=Depends(require_login),
):
    """
    All trades page with pagination and filtering.

    Args:
        page: Page number (1-indexed)
        per_page: Number of trades per page (default 50, max 200)
        ticker: Filter by ticker symbol
        outcome: Filter by outcome (win, loss, breakeven)
    """
    from math import ceil

    base_context = await get_template_context_with_user(request, user)
    user_id = user.id

    # Validate pagination params
    page_num = max(1, page)
    per_page_num = min(max(10, per_page), 200)  # Between 10 and 200

    def _get_trades_data():
        session = get_session()
        try:
            # Build query with user filter and optional filters
            query = session.query(Trade).filter(Trade.user_id == user_id)

            if ticker:
                query = query.filter(Trade.ticker.ilike(f"%{ticker}%"))
            if outcome:
                outcome_lower = outcome.lower()
                if outcome_lower == "win":
                    query = query.filter(Trade.outcome == TradeOutcome.WIN)
                elif outcome_lower == "loss":
                    query = query.filter(Trade.outcome == TradeOutcome.LOSS)
                elif outcome_lower == "breakeven":
                    query = query.filter(Trade.outcome == TradeOutcome.BREAKEVEN)

            # Get total count for pagination
            total_trades = query.count()
            total_pages = ceil(total_trades / per_page_num) if total_trades > 0 else 1

            # Ensure page is within bounds
            actual_page = min(page_num, total_pages)

            # Get paginated trades with eager loading of strategy and fills relationships
            offset = (actual_page - 1) * per_page_num
            trades = query.options(
                joinedload(Trade.strategy),
                joinedload(Trade.fills)
            ).order_by(Trade.trade_number.desc()).offset(offset).limit(per_page_num).all()

            strategies = get_active_strategies_cached(session)
            
            # Expunge trades so they remain usable after session closes
            # This is needed because the template renders after the session is closed
            # Note: strategies come from cache and aren't in this session
            for trade in trades:
                session.expunge(trade)

            return {
                "trades": trades,
                "strategies": strategies,
                "page": actual_page,
                "per_page": per_page_num,
                "total_trades": total_trades,
                "total_pages": total_pages,
                "has_next": actual_page < total_pages,
                "has_prev": actual_page > 1,
            }
        finally:
            session.close()

    data = await asyncio.to_thread(_get_trades_data)

    return templates.TemplateResponse(
        request,
        "trades.html",
        {
            **base_context,
            **data,
            "filter_ticker": ticker or "",
            "filter_outcome": outcome or "",
        },
    )


@app.get("/trades/{trade_id}", response_class=HTMLResponse)
async def trade_detail(request: Request, trade_id: int, user=Depends(require_login)):
    """Trade detail page - loads instantly, review fetched via AJAX."""
    from app.materials_reader import has_materials

    base_context = await get_template_context_with_user(request, user)

    session = get_session()
    try:
        # Verify trade exists and belongs to this user
        trade = session.query(Trade).filter(Trade.id == trade_id, Trade.user_id == user.id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        # Don't block on LLM - page loads instantly, review fetched async
        return templates.TemplateResponse(
            request,
            "trade_detail.html",
            {
                **base_context,
                "trade": trade,
                "review": None,  # Will be loaded via AJAX
                "has_materials": has_materials(),
            },
        )
    finally:
        session.close()


@app.get("/add-trade", response_class=HTMLResponse)
async def add_trade_page(request: Request, user=Depends(require_login)):
    """Add trade form page (includes single trade and bulk import)."""
    base_context = await get_template_context_with_user(request, user)

    session = get_session()
    try:
        strategies = get_active_strategies_cached(session)

        # Import data for bulk import tab
        import_files = list(IMPORTS_DIR.glob("*.csv"))
        processed_dir = IMPORTS_DIR / "processed"
        processed_files = list(processed_dir.glob("*.csv")) if processed_dir.exists() else []

        return templates.TemplateResponse(
            request,
            "add_trade.html",
            {
                **base_context,
                "strategies": strategies,
                "tickers": load_tickers_from_file(),
                "imports_path": str(IMPORTS_DIR),
                "import_files": import_files,
                "processed_files": processed_files,
            },
        )
    finally:
        session.close()


@app.get("/import")
async def import_page():
    """Redirect to add-trade page with import tab."""
    return RedirectResponse(url="/add-trade?tab=import", status_code=302)


@app.get("/tickers", response_class=HTMLResponse)
async def tickers_page(request: Request, user=Depends(require_login)):
    """Ticker management page."""
    base_context = await get_template_context_with_user(request, user)
    tickers = load_tickers_from_file()
    return templates.TemplateResponse(
        request,
        "tickers.html",
        {
            **base_context,
            "tickers": tickers,
        },
    )


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request, user=Depends(require_login)):
    """Reports page."""
    base_context = await get_template_context_with_user(request, user)

    # List available reports
    report_dirs = sorted(OUTPUTS_DIR.glob("*"), reverse=True)[:20]

    reports = []
    for d in report_dirs:
        if d.is_dir():
            reports.append(
                {
                    "date": d.name,
                    "path": d,
                    "has_premarket": (d / "premarket").exists(),
                    "has_eod": (d / "eod_report.md").exists(),
                }
            )

    return templates.TemplateResponse(
        request,
        "reports.html",
        {
            **base_context,
            "reports": reports,
            "tickers": load_tickers_from_file(),
        },
    )


@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request, user=Depends(require_login)):
    """Statistics page."""
    base_context = await get_template_context_with_user(request, user)
    analytics = TradeAnalytics()

    # Get stats for this user
    strategy_stats = analytics.get_all_strategy_stats(user_id=user.id)
    edge_analysis = analytics.analyze_edge()
    portfolio_stats = analytics.get_portfolio_stats(user_id=user.id)

    # Get all trades for equity curve (filtered by user)
    trades = analytics.get_all_trades(user_id=user.id)

    # Total P&L in USD (matches Dashboard/Trades pages)
    total_pnl_usd = 0.0
    for t in trades:
        if t.pnl_usd is not None:
            total_pnl_usd += t.pnl_usd

    # Build equity curve with drawdown data
    equity_data = []
    drawdown_data = []
    cumulative = 0
    peak = 0
    for t in sorted(
        trades, key=lambda x: (x.trade_date, x.exit_time or x.entry_time or x.trade_date)
    ):
        if t.r_multiple:
            cumulative += t.r_multiple
            if cumulative > peak:
                peak = cumulative
            drawdown = peak - cumulative
            equity_data.append(
                {
                    "date": str(t.trade_date),
                    "r": round(cumulative, 2),
                    "pnl": round(t.pnl_dollars, 2) if t.pnl_dollars else 0,
                }
            )
            drawdown_data.append({"date": str(t.trade_date), "dd": round(drawdown, 2)})

    return templates.TemplateResponse(
        request,
        "stats.html",
        {
            **base_context,
            "strategy_stats": strategy_stats,
            "edge_analysis": edge_analysis,
            "portfolio_stats": portfolio_stats,
            "total_pnl_usd": round(total_pnl_usd, 2),
            "equity_data": equity_data,
            "drawdown_data": drawdown_data,
        },
    )


# ==================== API ENDPOINTS ====================


@app.get("/settings")
async def settings_page(request: Request, user=Depends(require_login)):
    """Settings page for prompts and candle counts."""
    from app.config_prompts import load_settings, SETTINGS_FILE

    from app.config_prompts import get_editable_prompt

    base_context = await get_template_context_with_user(request, user)
    current_settings = load_settings()

    # Get editable prompts (without protected JSON schemas)
    system_prompts = {
        "trade_analysis": get_editable_prompt("trade_analysis", is_user_prompt=False),
        "market_context": get_editable_prompt("market_context", is_user_prompt=False),
    }
    user_prompts = {
        "trade_analysis": get_editable_prompt("trade_analysis", is_user_prompt=True),
        "market_context": get_editable_prompt("market_context", is_user_prompt=True),
    }

    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            **base_context,
            "candles": current_settings.get("candles", {}),
            "system_prompts": system_prompts,
            "user_prompts": user_prompts,
            "cache_settings": get_cache_settings(),
            "settings_file": str(SETTINGS_FILE),
        },
    )


# ============== STRATEGIES MANAGEMENT ==============


@app.get("/strategies", response_class=HTMLResponse)
async def strategies_page(request: Request, user=Depends(require_login)):
    """Strategy management page - edit, merge, categorize strategies."""

    base_context = await get_template_context_with_user(request, user)

    session = get_session()
    try:
        strategies = session.query(Strategy).order_by(Strategy.category, Strategy.name).all()

        # Get trade counts per strategy
        strategy_stats = {}
        for strategy in strategies:
            trade_count = session.query(Trade).filter(Trade.strategy_id == strategy.id).count()
            strategy_stats[strategy.id] = trade_count

        # Get custom categories
        from app.web.routes.strategies import get_categories as get_categories_list

        categories = get_categories_list()

        return templates.TemplateResponse(
            request,
            "strategies.html",
            {
                **base_context,
                "strategies": strategies,
                "strategy_stats": strategy_stats,
                "categories": categories,
            },
        )
    finally:
        session.close()


# ==================== ROBINHOOD API ====================

# ==================== MATERIALS ====================


@app.get("/materials", response_class=HTMLResponse)
async def materials_page(request: Request, user=Depends(require_login)):
    """Training materials management page."""
    base_context = await get_template_context_with_user(request, user)
    materials = []

    if MATERIALS_DIR.exists():
        for f in sorted(MATERIALS_DIR.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                materials.append(
                    {
                        "name": f.name,
                        "path": str(f),
                        "size_kb": round(f.stat().st_size / 1024, 1),
                    }
                )

    return templates.TemplateResponse(
        request,
        "materials.html",
        {
            **base_context,
            "materials": materials,
        },
    )


# ==================== BULK ANALYSIS ====================


@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request, user=Depends(require_login)):
    """Bulk trade analysis page."""
    base_context = await get_template_context_with_user(request, user)
    return templates.TemplateResponse(
        request,
        "analysis.html",
        {
            **base_context,
        },
    )


# ==================== RUN SERVER ====================


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = True):
    """Run the web server.

    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload on code changes (default True for development)
    """
    import uvicorn
    import signal
    import sys

    # Handle Ctrl+C properly to kill the process (not just suspend)
    def signal_handler(sig, frame):
        print("\n\n👋 Shutting down server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n🚀 AI Trading Coach Web UI")
    print(f"   Open http://{host}:{port} in your browser")
    if reload:
        print("   🔄 Auto-reload enabled - changes will apply automatically")
    print("   Press Ctrl+C to stop\n")

    if reload:
        # Use reload mode - requires passing app as string
        uvicorn.run(
            "app.web.server:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=["app"],
        )
    else:
        uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the trading coach web server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, reload=not args.no_reload)
