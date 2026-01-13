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
    load_tickers_from_file, save_tickers_to_file, get_anthropic_api_key
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
        llm_available = get_anthropic_api_key() is not None
        
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
    })


@app.get("/tickers", response_class=HTMLResponse)
async def tickers_page(request: Request):
    """Ticker management page."""
    tickers = load_tickers_from_file()
    return templates.TemplateResponse("tickers.html", {
        "request": request,
        "tickers": tickers,
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


@app.delete("/api/trades/{trade_id}")
async def delete_trade(trade_id: int):
    """Delete a trade."""
    ingester = TradeIngester()
    if ingester.delete_trade(trade_id):
        return JSONResponse({"message": "Trade deleted"})
    raise HTTPException(status_code=404, detail="Trade not found")


# ==================== RUN SERVER ====================

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the web server."""
    import uvicorn
    print(f"\n🚀 Brooks Trading Coach Web UI")
    print(f"   Open http://{host}:{port} in your browser\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
