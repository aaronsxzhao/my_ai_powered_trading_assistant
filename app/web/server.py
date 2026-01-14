"""
FastAPI Web Server for Brooks Trading Coach.

Provides a browser-based interface for:
- Dashboard with stats
- Trade management
- CSV upload
- Ticker management
- Report generation
"""

import os
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    settings, IMPORTS_DIR, OUTPUTS_DIR, PROJECT_ROOT,
    load_tickers_from_file, save_tickers_to_file, get_llm_api_key,
    get_polygon_api_key
)
from app.journal.models import init_db, Trade, Strategy, get_session, TradeDirection, TradeOutcome
from app.journal.ingest import TradeIngester
from app.journal.analytics import TradeAnalytics
from app.journal.coach import TradeCoach

# Initialize FastAPI app
app = FastAPI(
    title="Brooks Trading Coach",
    description="Advisory system for discretionary day traders",
    version="0.1.0",
)

# Templates directory
TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# Initialize database on startup
@app.on_event("startup")
async def startup():
    init_db()
    IMPORTS_DIR.mkdir(exist_ok=True)


# ==================== PAGES ====================

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    analytics = TradeAnalytics()
    
    # Get recent trades
    session = get_session()
    try:
        recent_trades = (
            session.query(Trade)
            .order_by(Trade.trade_date.desc(), Trade.id.desc())
            .limit(10)
            .all()
        )
        
        # Calculate stats
        all_trades = session.query(Trade).all()
        total_trades = len(all_trades)
        
        if all_trades:
            winners = len([t for t in all_trades if t.outcome == TradeOutcome.WIN])
            r_values = [t.r_multiple for t in all_trades if t.r_multiple]
            total_r = sum(r_values) if r_values else 0
            win_rate = winners / total_trades if total_trades > 0 else 0
        else:
            winners = 0
            total_r = 0
            win_rate = 0
        
        # Get strategy stats
        strategy_stats = analytics.get_all_strategy_stats()[:5]
        
        # Check LLM status
        llm_available = get_llm_api_key() is not None
        
        # Check data provider
        data_provider = settings.data_provider
        polygon_available = get_polygon_api_key() is not None
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "recent_trades": recent_trades,
            "total_trades": total_trades,
            "total_r": total_r,
            "win_rate": win_rate,
            "winners": winners,
            "strategy_stats": strategy_stats,
            "llm_available": llm_available,
            "tickers": load_tickers_from_file(),
            "data_provider": data_provider,
            "polygon_available": polygon_available,
        })
    finally:
        session.close()


@app.get("/trades", response_class=HTMLResponse)
async def trades_page(request: Request):
    """All trades page."""
    session = get_session()
    try:
        trades = (
            session.query(Trade)
            .order_by(Trade.trade_date.desc(), Trade.id.desc())
            .limit(100)
            .all()
        )
        
        strategies = session.query(Strategy).filter(Strategy.is_active == True).all()
        
        return templates.TemplateResponse("trades.html", {
            "request": request,
            "trades": trades,
            "strategies": strategies,
            "data_provider": settings.data_provider,
            "llm_available": get_llm_api_key() is not None,
        })
    finally:
        session.close()


@app.get("/trades/{trade_id}", response_class=HTMLResponse)
async def trade_detail(request: Request, trade_id: int):
    """Trade detail and review page."""
    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        # Get coaching review
        coach = TradeCoach()
        review = coach.review_trade(trade_id)
        
        return templates.TemplateResponse("trade_detail.html", {
            "request": request,
            "trade": trade,
            "review": review,
            "data_provider": settings.data_provider,
            "llm_available": get_llm_api_key() is not None,
        })
    finally:
        session.close()


@app.get("/add-trade", response_class=HTMLResponse)
async def add_trade_page(request: Request):
    """Add trade form page."""
    session = get_session()
    try:
        strategies = session.query(Strategy).filter(Strategy.is_active == True).all()
        return templates.TemplateResponse("add_trade.html", {
            "request": request,
            "strategies": strategies,
            "tickers": load_tickers_from_file(),
            "data_provider": settings.data_provider,
            "llm_available": get_llm_api_key() is not None,
        })
    finally:
        session.close()


@app.get("/import", response_class=HTMLResponse)
async def import_page(request: Request):
    """CSV import page."""
    # List files in imports folder
    import_files = list(IMPORTS_DIR.glob("*.csv"))
    processed_dir = IMPORTS_DIR / "processed"
    processed_files = list(processed_dir.glob("*.csv")) if processed_dir.exists() else []
    
    return templates.TemplateResponse("import.html", {
        "request": request,
        "import_files": import_files,
        "processed_files": processed_files,
        "imports_path": str(IMPORTS_DIR),
        "data_provider": settings.data_provider,
        "llm_available": get_llm_api_key() is not None,
    })


@app.get("/tickers", response_class=HTMLResponse)
async def tickers_page(request: Request):
    """Ticker management page."""
    tickers = load_tickers_from_file()
    return templates.TemplateResponse("tickers.html", {
        "request": request,
        "tickers": tickers,
        "data_provider": settings.data_provider,
        "llm_available": get_llm_api_key() is not None,
    })


@app.get("/reports", response_class=HTMLResponse)
async def reports_page(request: Request):
    """Reports page."""
    # List available reports
    report_dirs = sorted(OUTPUTS_DIR.glob("*"), reverse=True)[:20]
    
    reports = []
    for d in report_dirs:
        if d.is_dir():
            reports.append({
                "date": d.name,
                "path": d,
                "has_premarket": (d / "premarket").exists(),
                "has_eod": (d / "eod_report.md").exists(),
            })
    
    return templates.TemplateResponse("reports.html", {
        "request": request,
        "reports": reports,
        "tickers": load_tickers_from_file(),
        "data_provider": settings.data_provider,
        "llm_available": get_llm_api_key() is not None,
    })


@app.get("/stats", response_class=HTMLResponse)
async def stats_page(request: Request):
    """Statistics page."""
    analytics = TradeAnalytics()
    
    strategy_stats = analytics.get_all_strategy_stats()
    edge_analysis = analytics.analyze_edge()
    
    # Get all trades for equity curve
    trades = analytics.get_all_trades()
    equity_data = []
    cumulative = 0
    for t in sorted(trades, key=lambda x: x.trade_date):
        if t.r_multiple:
            cumulative += t.r_multiple
            equity_data.append({"date": str(t.trade_date), "r": cumulative})
    
    return templates.TemplateResponse("stats.html", {
        "request": request,
        "strategy_stats": strategy_stats,
        "edge_analysis": edge_analysis,
        "equity_data": equity_data,
        "data_provider": settings.data_provider,
        "llm_available": get_llm_api_key() is not None,
    })


# ==================== API ENDPOINTS ====================

@app.post("/api/trades")
async def create_trade(
    ticker: str = Form(...),
    direction: str = Form(...),
    entry_price: float = Form(...),
    exit_price: float = Form(...),
    stop_price: float = Form(...),
    size: float = Form(1.0),
    trade_date: str = Form(None),
    strategy: str = Form(None),
    notes: str = Form(None),
    entry_reason: str = Form(None),
):
    """Create a new trade."""
    ingester = TradeIngester()
    
    parsed_date = datetime.strptime(trade_date, "%Y-%m-%d").date() if trade_date else date.today()
    
    trade = ingester.add_trade_manual(
        ticker=ticker,
        trade_date=parsed_date,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_price=stop_price,
        size=size,
        strategy_name=strategy if strategy else None,
        notes=notes,
        entry_reason=entry_reason,
    )
    
    return RedirectResponse(url=f"/trades/{trade.id}", status_code=303)


@app.post("/api/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file to imports folder."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    file_path = IMPORTS_DIR / file.filename
    
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    return JSONResponse({"message": f"Uploaded {file.filename}", "path": str(file_path)})


@app.post("/api/import-csv")
async def import_csv(
    file: UploadFile = File(...),
    format: str = Form("generic"),
):
    """Upload and import a CSV file with specified format."""
    if not file.filename.endswith('.csv'):
        return JSONResponse({"error": "Only CSV files allowed", "imported": 0, "errors": 1})
    
    # Save file temporarily
    import tempfile
    import os
    
    try:
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Import with specified format
        ingester = TradeIngester()
        imported, errors, messages = ingester.import_csv(tmp_path, format=format)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return JSONResponse({
            "imported": imported,
            "errors": errors,
            "messages": messages[:10],
            "format": format,
        })
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "imported": 0,
            "errors": 1,
        })


@app.post("/api/bulk-import")
async def bulk_import():
    """Import all CSVs from imports folder."""
    ingester = TradeIngester()
    imported, errors, messages = ingester.bulk_import_from_folder()
    
    return JSONResponse({
        "imported": imported,
        "errors": errors,
        "messages": messages[:10],
    })


@app.post("/api/tickers")
async def update_tickers(tickers: str = Form(...)):
    """Update tickers list."""
    ticker_list = [t.strip().upper() for t in tickers.split('\n') if t.strip() and not t.strip().startswith('#')]
    save_tickers_to_file(ticker_list)
    return RedirectResponse(url="/tickers", status_code=303)


@app.post("/api/tickers/add")
async def add_ticker(ticker: str = Form(...)):
    """Add a ticker."""
    tickers = load_tickers_from_file()
    ticker = ticker.upper().strip()
    if ticker and ticker not in tickers:
        tickers.append(ticker)
        save_tickers_to_file(tickers)
    return RedirectResponse(url="/tickers", status_code=303)


@app.post("/api/tickers/remove/{ticker}")
async def remove_ticker(ticker: str):
    """Remove a ticker."""
    tickers = load_tickers_from_file()
    ticker = ticker.upper()
    if ticker in tickers:
        tickers.remove(ticker)
        save_tickers_to_file(tickers)
    return RedirectResponse(url="/tickers", status_code=303)


@app.post("/api/generate-premarket")
async def generate_premarket(ticker: str = Form(None)):
    """Generate premarket report."""
    from app.reports.premarket import PremarketReport
    
    generator = PremarketReport()
    
    if ticker:
        reports = [generator.generate_ticker_report(ticker)]
    else:
        reports = generator.generate_all_reports()
    
    output_dir = generator.save_reports(reports, date.today())
    
    return JSONResponse({
        "message": f"Generated {len(reports)} reports",
        "path": str(output_dir),
    })


@app.post("/api/generate-eod")
async def generate_eod():
    """Generate end-of-day report."""
    from app.reports.eod import EndOfDayReport
    
    generator = EndOfDayReport()
    report = generator.generate_report()
    output_path = generator.save_report(report)
    
    return JSONResponse({
        "message": "Generated EOD report",
        "path": str(output_path),
    })


@app.post("/api/reclassify")
async def reclassify_trades():
    """Reclassify unclassified trades with LLM."""
    ingester = TradeIngester()
    reclassified, failed = ingester.reclassify_all_trades()

    return JSONResponse({
        "reclassified": reclassified,
        "failed": failed,
    })


@app.post("/api/recalculate-metrics")
async def recalculate_all_metrics():
    """Recalculate R-multiple, P&L, and other metrics for all trades."""
    with get_session() as session:
        trades = session.query(Trade).all()
        updated = 0
        for trade in trades:
            trade.compute_metrics()
            updated += 1
        session.commit()
    
    return JSONResponse({
        "updated": updated,
        "message": f"Recalculated metrics for {updated} trades"
    })


@app.delete("/api/trades/{trade_id}")
async def delete_trade(trade_id: int):
    """Delete a trade."""
    ingester = TradeIngester()
    if ingester.delete_trade(trade_id):
        return JSONResponse({"message": "Trade deleted"})
    raise HTTPException(status_code=404, detail="Trade not found")


@app.delete("/api/trades")
async def delete_all_trades():
    """Delete ALL trades. Use with caution!"""
    ingester = TradeIngester()
    count = ingester.delete_all_trades()
    return JSONResponse({"message": f"Deleted {count} trades", "count": count})


@app.patch("/api/trades/{trade_id}")
async def update_trade(trade_id: int, request: Request):
    """Update trade fields (size, prices, times, currency)."""
    from datetime import datetime
    data = await request.json()
    
    with get_session() as session:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        # Update fields if provided
        if data.get("size") is not None:
            trade.size = float(data["size"])
        if data.get("entry_price") is not None:
            trade.entry_price = float(data["entry_price"])
        if data.get("exit_price") is not None:
            trade.exit_price = float(data["exit_price"])
        if data.get("stop_price") is not None:
            trade.stop_price = float(data["stop_price"])
        if data.get("currency") is not None:
            trade.currency = data["currency"]
        if data.get("currency_rate") is not None:
            trade.currency_rate = float(data["currency_rate"])
        
        # Update times if provided
        if data.get("entry_time"):
            try:
                trade.entry_time = datetime.fromisoformat(data["entry_time"])
            except ValueError:
                pass
        if data.get("exit_time"):
            try:
                trade.exit_time = datetime.fromisoformat(data["exit_time"])
                # Also update trade_date to match exit date
                trade.trade_date = trade.exit_time.date()
            except ValueError:
                pass
        
        # Recalculate metrics after update
        trade.compute_metrics()
        session.commit()
        
        return JSONResponse({
            "message": "Trade updated",
            "trade_id": trade_id,
            "r_multiple": trade.r_multiple,
            "pnl_dollars": trade.pnl_dollars,
        })


@app.get("/api/status")
async def get_status():
    """Get current system status and configuration."""
    return JSONResponse({
        "data_provider": settings.data_provider,
        "polygon_available": get_polygon_api_key() is not None,
        "llm_available": get_llm_api_key() is not None,
        "tickers": load_tickers_from_file(),
        "outputs_dir": str(OUTPUTS_DIR),
        "imports_dir": str(IMPORTS_DIR),
    })


@app.get("/api/exchange-rate")
async def get_exchange_rate_api(currency: str, date: str = None):
    """
    Get historical exchange rate for a currency.
    
    Args:
        currency: Target currency code (e.g., 'HKD')
        date: Date in YYYY-MM-DD format (defaults to today)
        
    Returns:
        Exchange rate (1 USD = X currency)
    """
    from app.data.currency import get_exchange_rate
    from datetime import date as date_type, datetime
    
    trade_date = None
    if date:
        try:
            trade_date = datetime.fromisoformat(date.replace('Z', '+00:00')).date()
        except ValueError:
            try:
                trade_date = date_type.fromisoformat(date[:10])
            except ValueError:
                pass
    
    rate = get_exchange_rate(currency, trade_date)
    
    if rate is None:
        # Return fallback rates if API fails
        fallback_rates = {
            'HKD': 7.78, 'EUR': 0.92, 'GBP': 0.79, 'JPY': 149.5,
            'CNY': 7.24, 'CAD': 1.36, 'AUD': 1.53, 'CHF': 0.88,
            'SGD': 1.34, 'KRW': 1320.0, 'TWD': 31.5,
        }
        rate = fallback_rates.get(currency, 1.0)
        return JSONResponse({
            "currency": currency,
            "rate": rate,
            "date": date or str(date_type.today()),
            "source": "fallback",
        })
    
    return JSONResponse({
        "currency": currency,
        "rate": rate,
        "date": date or str(date_type.today()),
        "source": "frankfurter",
    })


# ==================== ROBINHOOD API ====================

@app.get("/api/robinhood/status")
async def robinhood_status():
    """Check Robinhood connection status."""
    ingester = TradeIngester()
    return JSONResponse({
        "has_session": ingester.robinhood_has_session(),
    })


@app.post("/api/robinhood/import")
async def robinhood_import(request: Request):
    """Import trades from Robinhood with login."""
    try:
        data = await request.json()
        username = data.get('username')
        password = data.get('password')
        mfa_code = data.get('mfa_code')
        days_back = data.get('days_back', 30)
        
        if not username or not password:
            return JSONResponse({
                "error": "Username and password required",
                "imported": 0,
                "errors": 1,
                "needs_mfa": False,
                "needs_device_approval": False,
            })
        
        ingester = TradeIngester()
        result = ingester.import_from_robinhood(
            username=username,
            password=password,
            mfa_code=mfa_code,
            days_back=days_back,
        )
        
        return JSONResponse(result)
        
    except Exception as e:
        return JSONResponse({
            "error": str(e),
            "imported": 0,
            "errors": 1,
            "needs_mfa": False,
            "needs_device_approval": False,
        })


@app.post("/api/robinhood/import-session")
async def robinhood_import_session(request: Request):
    """Import trades from Robinhood using stored session."""
    try:
        data = await request.json()
        days_back = data.get('days_back', 30)
        
        ingester = TradeIngester()
        
        # Try to login with stored session
        if not ingester.robinhood_login_with_session():
            return JSONResponse({"error": "No stored session. Please login first.", "imported": 0, "errors": 1})
        
        from app.data.robinhood import get_robinhood_client
        client = get_robinhood_client()
        
        # Get orders
        orders = client.get_stock_orders(days_back=days_back)
        
        if not orders:
            return JSONResponse({
                "imported": 0,
                "errors": 0,
                "messages": [f"No filled orders found in the last {days_back} days."],
            })
        
        # Process orders (simplified - just count for now)
        return JSONResponse({
            "imported": len(orders),
            "errors": 0,
            "messages": [f"Found {len(orders)} orders to process."],
        })
        
    except Exception as e:
        return JSONResponse({"error": str(e), "imported": 0, "errors": 1})


@app.post("/api/robinhood/disconnect")
async def robinhood_disconnect():
    """Disconnect Robinhood (clear stored session)."""
    ingester = TradeIngester()
    ingester.robinhood_clear_session()
    return JSONResponse({"message": "Disconnected"})


# ==================== RUN SERVER ====================

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the web server."""
    import uvicorn
    print(f"\n🚀 Brooks Trading Coach Web UI")
    print(f"   Open http://{host}:{port} in your browser\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
