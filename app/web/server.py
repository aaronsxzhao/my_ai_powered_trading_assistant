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
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException

logger = logging.getLogger(__name__)
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import (
    settings, IMPORTS_DIR, OUTPUTS_DIR, PROJECT_ROOT, MATERIALS_DIR,
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
            .order_by(Trade.exit_time.desc(), Trade.id.desc())
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
            .order_by(Trade.exit_time.desc(), Trade.id.desc())
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
    """Trade detail page - loads instantly, review fetched via AJAX."""
    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        # Don't block on LLM - page loads instantly, review fetched async
        return templates.TemplateResponse("trade_detail.html", {
            "request": request,
            "trade": trade,
            "review": None,  # Will be loaded via AJAX
            "data_provider": settings.data_provider,
            "llm_available": get_llm_api_key() is not None,
        })
    finally:
        session.close()


@app.patch("/api/trades/{trade_id}/notes")
async def update_trade_notes(trade_id: int, request: Request):
    """Update trade notes, personal review, and Brooks intent fields."""
    data = await request.json()

    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        # Update note fields
        if 'notes' in data:
            trade.notes = data['notes']
        if 'entry_reason' in data:
            trade.entry_reason = data['entry_reason']
        if 'mistakes' in data:
            trade.mistakes = data['mistakes']
        if 'lessons' in data:
            trade.lessons = data['lessons']
        
        # Update Brooks intent fields
        if 'trade_type' in data:
            trade.trade_type = data['trade_type']
        if 'confidence_level' in data:
            trade.confidence_level = data['confidence_level']
        if 'emotional_state' in data:
            trade.emotional_state = data['emotional_state']
        if 'followed_plan' in data:
            trade.followed_plan = data['followed_plan']
        if 'stop_reason' in data:
            trade.stop_reason = data['stop_reason']
        if 'target_reason' in data:
            trade.target_reason = data['target_reason']
        if 'invalidation_condition' in data:
            trade.invalidation_condition = data['invalidation_condition']
        
        # Clear cached review when trade intent is updated (to force re-analysis)
        trade.cached_review = None
        trade.review_generated_at = None

        session.commit()

        return JSONResponse({"success": True, "message": "Trade details saved"})
    finally:
        session.close()


@app.get("/api/trades/{trade_id}/review")
async def get_trade_review(trade_id: int, force: bool = False):
    """
    Get AI review for a trade (non-blocking async endpoint).
    
    Uses cached review if available and trade hasn't been modified.
    Set force=true to regenerate the review.
    """
    import asyncio
    import json
    from datetime import datetime
    
    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            return JSONResponse({"success": False, "error": "Trade not found"})
        
        # Check for cached review
        if not force and trade.cached_review and trade.review_generated_at:
            # Check if trade was modified after review was generated
            if trade.updated_at and trade.updated_at > trade.review_generated_at:
                logger.info(f"Trade {trade_id} was modified after last review, regenerating...")
            else:
                # Return cached review
                logger.info(f"Returning cached review for trade {trade_id} (generated at {trade.review_generated_at})")
                try:
                    cached = json.loads(trade.cached_review)
                    return JSONResponse({
                        "success": True,
                        "cached": True,
                        "generated_at": trade.review_generated_at.isoformat() if trade.review_generated_at else None,
                        "review": cached
                    })
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse cached review for trade {trade_id}")
        
        # Generate new review
        logger.info(f"Generating new AI review for trade {trade_id}...")
        
        def _get_review():
            coach = TradeCoach()
            return coach.review_trade(trade_id)
        
        # Run LLM call in thread pool to not block event loop
        review = await asyncio.to_thread(_get_review)
        
        if review:
            review_dict = {
                "grade": review.grade,
                "grade_explanation": review.grade_explanation,
                "regime": review.regime,
                "always_in": review.always_in,
                "context_description": review.context_description,
                "setup_classification": review.setup_classification,
                "setup_quality": review.setup_quality,
                "what_was_good": review.what_was_good if isinstance(review.what_was_good, list) else [review.what_was_good] if review.what_was_good else [],
                "what_was_flawed": review.what_was_flawed if isinstance(review.what_was_flawed, list) else [review.what_was_flawed] if review.what_was_flawed else [],
                "errors_detected": review.errors_detected if isinstance(review.errors_detected, list) else [review.errors_detected] if review.errors_detected else [],
                "rule_for_next_time": review.rule_for_next_time,
            }
            
            # Cache the review
            trade.cached_review = json.dumps(review_dict)
            trade.review_generated_at = datetime.utcnow()
            session.commit()
            logger.info(f"Cached new review for trade {trade_id}")
            
            return JSONResponse({
                "success": True,
                "cached": False,
                "generated_at": trade.review_generated_at.isoformat(),
                "review": review_dict
            })
        else:
            return JSONResponse({"success": False, "error": "Review not available"})
    except Exception as e:
        logger.error(f"Error getting trade review: {e}")
        return JSONResponse({"success": False, "error": str(e)})
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
    balance_file: Optional[UploadFile] = File(None),
):
    """Upload and import a CSV file (non-blocking)."""
    import asyncio
    import tempfile
    import os
    
    if not file.filename.endswith('.csv'):
        return JSONResponse({"error": "Only CSV files allowed", "imported": 0, "errors": 1})
    
    try:
        # Save main file
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Save balance file if provided
        balance_path = None
        if balance_file and balance_file.filename:
            with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as bal_tmp:
                bal_content = await balance_file.read()
                bal_tmp.write(bal_content)
                balance_path = bal_tmp.name
        
        def _do_import():
            ingester = TradeIngester()
            if format == 'tv_order_history' and balance_path:
                from pathlib import Path
                return ingester._import_tv_order_history(
                    Path(tmp_path), skip_errors=True, balance_file_path=Path(balance_path)
                )
            else:
                return ingester.import_csv(tmp_path, format=format)
        
        # Run import in thread to not block event loop
        imported, errors, messages = await asyncio.to_thread(_do_import)
        
        # Clean up temp files
        os.unlink(tmp_path)
        if balance_path:
            os.unlink(balance_path)
        
        return JSONResponse({
            "imported": imported,
            "errors": errors,
            "messages": messages[:10],
            "format": format,
            "cross_validated": balance_path is not None,
        })
        
    except Exception as e:
        logger.error(f"Import error: {e}")
        return JSONResponse({"error": str(e), "imported": 0, "errors": 1})


@app.post("/api/bulk-import")
async def bulk_import():
    """Import all CSVs from imports folder (non-blocking)."""
    import asyncio
    
    def _do_import():
        ingester = TradeIngester()
        return ingester.bulk_import_from_folder()
    
    imported, errors, messages = await asyncio.to_thread(_do_import)
    
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
    """Generate premarket report (non-blocking)."""
    import asyncio
    from app.reports.premarket import PremarketReport

    def _generate():
        generator = PremarketReport()
        if ticker:
            reports = [generator.generate_ticker_report(ticker)]
        else:
            reports = generator.generate_all_reports()
        output_dir = generator.save_reports(reports, date.today())
        return len(reports), str(output_dir)

    count, output_dir = await asyncio.to_thread(_generate)

    return JSONResponse({
        "message": f"Generated {count} reports",
        "path": output_dir,
    })


@app.post("/api/generate-eod")
async def generate_eod():
    """Generate end-of-day report (non-blocking)."""
    import asyncio
    from app.reports.eod import EndOfDayReport

    def _generate():
        generator = EndOfDayReport()
        report = generator.generate_report()
        output_path = generator.save_report(report)
        return str(output_path)

    output_path = await asyncio.to_thread(_generate)

    return JSONResponse({
        "message": "Generated EOD report",
        "path": output_path,
    })


@app.post("/api/reclassify")
async def reclassify_trades():
    """Reclassify unclassified trades with LLM (non-blocking)."""
    import asyncio
    
    def _reclassify():
        ingester = TradeIngester()
        return ingester.reclassify_all_trades()
    
    reclassified, failed = await asyncio.to_thread(_reclassify)

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


@app.post("/api/reclassify-trades")
async def reclassify_all_trades(force: bool = False):
    """Reclassify trades using LLM with 20 concurrent workers.
    
    By default, only reclassifies trades that are unclassified or have 'unclassified' strategy.
    Set force=true to reclassify all trades.
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from app.llm.analyzer import LLMAnalyzer
    from app.journal.models import Strategy
    from app.data.cache import get_cached_ohlcv
    from datetime import timedelta
    
    analyzer = LLMAnalyzer()
    
    if not analyzer.is_available:
        return JSONResponse({
            "error": "LLM not available. Check your API key in settings.",
            "classified": 0
        })
    
    session = get_session()
    try:
        trades = session.query(Trade).all()
        trade_data = []
        skipped = 0
        
        # Prepare trade data for concurrent processing
        for trade in trades:
            # Skip already classified trades unless force=true
            if not force and trade.strategy and trade.strategy.name not in ['unclassified', 'insufficient_information']:
                skipped += 1
                continue
            
            # Note: Skip OHLCV fetch during bulk reclassify to avoid rate limits
            # OHLCV context will be fetched only during individual trade review
            ohlcv_context = None
            
            trade_data.append({
                "id": trade.id,
                "ticker": trade.ticker,
                "direction": trade.direction.value,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price or trade.entry_price,
                "stop_price": trade.stop_price or trade.entry_price * 0.98,
                "entry_reason": trade.entry_reason,
                "notes": trade.notes,
                "ohlcv_context": ohlcv_context,
                "timeframe": trade.timeframe or "5m",
                "trade_type": trade.trade_type,
            })
        
        # Function to classify a single trade
        def classify_single(data):
            try:
                result = analyzer.classify_trade_setup(
                    ticker=data["ticker"],
                    direction=data["direction"],
                    entry_price=data["entry_price"],
                    exit_price=data["exit_price"],
                    stop_price=data["stop_price"],
                    entry_reason=data["entry_reason"],
                    notes=data["notes"],
                    ohlcv_context=data.get("ohlcv_context"),
                    timeframe=data.get("timeframe"),
                    trade_type=data.get("trade_type"),
                )
                return {"id": data["id"], "result": result, "error": None}
            except Exception as e:
                return {"id": data["id"], "result": None, "error": str(e)}
        
        # Run with 5 concurrent workers (reduced to avoid API rate limits)
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = await loop.run_in_executor(
                None,
                lambda: list(executor.map(classify_single, trade_data))
            )
        
        # Process results and update database
        classified = 0
        errors = 0
        
        for res in results:
            trade_id = res["id"]
            result = res["result"]
            
            if res["error"] or not result or "error" in result:
                errors += 1
                continue
            
            try:
                trade = session.query(Trade).get(trade_id)
                if not trade:
                    continue
                
                # Get strategy name - prefer primary_setup from new format
                strategy_name = result.get("primary_setup") or result.get("strategy_name", "unclassified")
                
                # Find or create strategy
                strategy = session.query(Strategy).filter(
                    Strategy.name == strategy_name
                ).first()
                
                if not strategy:
                    strategy = Strategy(
                        name=strategy_name,
                        category=result.get("setup_category") or result.get("strategy_category", "unknown"),
                        description=result.get("reasoning", ""),
                    )
                    session.add(strategy)
                    session.flush()
                
                trade.strategy_id = strategy.id
                
                # Store setup notes including missing info
                missing_info = result.get("missing_info", [])
                if missing_info:
                    trade.setup_notes = f"Missing: {', '.join(missing_info)}"
                
                # Clear cached review since strategy changed
                trade.cached_review = None
                trade.review_generated_at = None
                
                classified += 1
            except Exception as e:
                logger.error(f"Error updating trade {trade_id}: {e}")
                errors += 1
        
        session.commit()
        
        return JSONResponse({
            "skipped": skipped,
            "count": classified,
            "classified": classified,
            "errors": errors,
            "message": f"Classified {classified} trades, {errors} errors"
        })
    finally:
        session.close()


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
        if data.get("timeframe") is not None:
            trade.timeframe = data["timeframe"]
        
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


@app.get("/settings")
async def settings_page(request: Request):
    """Settings page for prompts and candle counts."""
    from app.config_prompts import load_settings, SETTINGS_FILE

    current_settings = load_settings()

    return templates.TemplateResponse("settings.html", {
        "request": request,
        "candles": current_settings.get('candles', {}),
        "system_prompts": current_settings.get('system_prompts', {}),
        "user_prompts": current_settings.get('user_prompts', {}),
        "settings_file": str(SETTINGS_FILE),
        "data_provider": settings.data_provider,
        "llm_available": get_llm_api_key() is not None,
    })


@app.post("/api/settings/candles")
async def update_candle_settings(request: Request):
    """Update candle count settings."""
    from app.config_prompts import update_candles
    
    data = await request.json()
    success = update_candles(
        daily=data.get('daily'),
        hourly=data.get('hourly'),
        five_min=data.get('5min')
    )
    
    if success:
        return JSONResponse({"message": "Candle settings saved"})
    raise HTTPException(status_code=500, detail="Failed to save settings")


@app.post("/api/settings/prompts")
async def update_prompt_settings(request: Request):
    """Update a specific prompt (system or user)."""
    from app.config_prompts import update_prompt

    data = await request.json()
    prompt_type = data.get('prompt_type')
    prompt_text = data.get('prompt_text')
    is_user_prompt = data.get('is_user_prompt', False)

    if not prompt_type or prompt_text is None:
        raise HTTPException(status_code=400, detail="Missing prompt_type or prompt_text")

    success = update_prompt(prompt_type, prompt_text, is_user_prompt=is_user_prompt)
    
    prompt_label = "User" if is_user_prompt else "System"
    if success:
        return JSONResponse({"message": f"{prompt_label} prompt '{prompt_type}' saved"})
    raise HTTPException(status_code=500, detail="Failed to save prompt")


@app.post("/api/settings/reset")
async def reset_settings():
    """Reset all settings to defaults."""
    from app.config_prompts import save_settings, DEFAULT_SETTINGS
    
    success = save_settings(DEFAULT_SETTINGS.copy())
    
    if success:
        return JSONResponse({"message": "Settings reset to defaults"})
    raise HTTPException(status_code=500, detail="Failed to reset settings")


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


# ==================== MATERIALS ====================

@app.get("/materials", response_class=HTMLResponse)
async def materials_page(request: Request):
    """Training materials management page."""
    materials = []
    
    if MATERIALS_DIR.exists():
        for f in sorted(MATERIALS_DIR.iterdir()):
            if f.is_file() and not f.name.startswith('.'):
                materials.append({
                    "name": f.name,
                    "path": str(f),
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })
    
    return templates.TemplateResponse("materials.html", {
        "request": request,
        "materials": materials,
        "llm_available": get_llm_api_key() is not None,
        "data_provider": settings.data_provider,
    })


@app.post("/api/materials")
async def upload_materials(files: list[UploadFile] = File(...)):
    """Upload training materials (PDFs, text files)."""
    uploaded = 0
    errors = 0
    
    MATERIALS_DIR.mkdir(exist_ok=True)
    
    for file in files:
        try:
            # Validate file type
            allowed_extensions = {'.pdf', '.txt', '.md', '.doc', '.docx'}
            ext = Path(file.filename).suffix.lower()
            
            if ext not in allowed_extensions:
                errors += 1
                continue
            
            # Save file
            file_path = MATERIALS_DIR / file.filename
            
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            
            uploaded += 1
            
        except Exception as e:
            logger.error(f"Error uploading {file.filename}: {e}")
            errors += 1
    
    return JSONResponse({
        "uploaded": uploaded,
        "errors": errors,
        "message": f"Uploaded {uploaded} files"
    })


@app.delete("/api/materials/{filename:path}")
async def delete_material(filename: str):
    """Delete a training material file."""
    import urllib.parse
    filename = urllib.parse.unquote(filename)
    file_path = MATERIALS_DIR / filename
    
    if file_path.exists() and file_path.is_file():
        file_path.unlink()
        return JSONResponse({"message": f"Deleted {filename}"})
    
    raise HTTPException(status_code=404, detail="File not found")


@app.delete("/api/materials")
async def delete_all_materials():
    """Delete all training materials."""
    deleted = 0
    
    if MATERIALS_DIR.exists():
        for f in MATERIALS_DIR.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                f.unlink()
                deleted += 1
    
    return JSONResponse({"deleted": deleted, "message": f"Deleted {deleted} files"})


# ==================== BULK ANALYSIS ====================

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_page(request: Request):
    """Bulk trade analysis page."""
    return templates.TemplateResponse("analysis.html", {
        "request": request,
        "llm_available": get_llm_api_key() is not None,
        "data_provider": settings.data_provider,
    })


@app.post("/api/bulk-analysis")
async def bulk_analysis(request: Request):
    """Analyze multiple trades using LLM."""
    from datetime import timedelta
    from app.llm.analyzer import LLMAnalyzer
    from app.config_prompts import get_prompt
    
    analyzer = LLMAnalyzer()
    
    if not analyzer.is_available:
        return JSONResponse({
            "error": "LLM not available. Check your API key in settings.",
            "trade_count": 0
        })
    
    data = await request.json()
    analysis_type = data.get('type', 'count')  # 'count' or 'days'
    value = data.get('value', 10)
    
    session = get_session()
    try:
        # Get trades based on selection
        if analysis_type == 'days':
            cutoff_date = date.today() - timedelta(days=value)
            trades = (
                session.query(Trade)
                .filter(Trade.trade_date >= cutoff_date)
                .order_by(Trade.exit_time.desc())
                .all()
            )
        else:  # count
            trades = (
                session.query(Trade)
                .order_by(Trade.exit_time.desc())
                .limit(value)
                .all()
            )
        
        if not trades:
            return JSONResponse({
                "error": "No trades found for the selected criteria.",
                "trade_count": 0
            })
        
        # Calculate stats
        wins = sum(1 for t in trades if t.outcome and t.outcome.value == 'win')
        losses = sum(1 for t in trades if t.outcome and t.outcome.value == 'loss')
        r_values = [t.r_multiple for t in trades if t.r_multiple]
        total_r = sum(r_values) if r_values else 0
        avg_r = total_r / len(r_values) if r_values else 0
        win_rate = (wins / len(trades) * 100) if trades else 0
        
        # Strategy breakdown
        strategy_stats = {}
        for trade in trades:
            strategy_name = trade.strategy.name if trade.strategy else 'Unclassified'
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {'count': 0, 'wins': 0, 'r_values': []}
            strategy_stats[strategy_name]['count'] += 1
            if trade.outcome and trade.outcome.value == 'win':
                strategy_stats[strategy_name]['wins'] += 1
            if trade.r_multiple:
                strategy_stats[strategy_name]['r_values'].append(trade.r_multiple)
        
        strategy_breakdown = {}
        for name, stats in strategy_stats.items():
            strategy_breakdown[name] = {
                'count': stats['count'],
                'win_rate': (stats['wins'] / stats['count'] * 100) if stats['count'] > 0 else 0,
                'total_r': sum(stats['r_values']) if stats['r_values'] else 0
            }
        
        # Build trade summary for LLM
        trade_summaries = []
        for t in trades[:50]:  # Limit to 50 trades for LLM context
            summary = (
                f"#{t.id}: {t.ticker} {t.direction.value.upper()} "
                f"Entry=${t.entry_price:.2f} Exit=${t.exit_price:.2f if t.exit_price else 0:.2f} "
                f"R={t.r_multiple:.2f if t.r_multiple else 0:.2f} "
                f"Result={t.outcome.value.upper() if t.outcome else 'UNKNOWN'} "
                f"Strategy={t.strategy.name if t.strategy else 'unclassified'} "
                f"Duration={t.duration_display}"
            )
            if t.notes:
                summary += f" Notes: {t.notes[:100]}"
            trade_summaries.append(summary)
        
        # Build LLM prompt
        system_prompt = """You are an expert Al Brooks price action trading coach analyzing a trader's recent performance.

Analyze the provided trades and identify:
1. PATTERNS: Recurring patterns in wins and losses
2. STRENGTHS: What the trader is doing well
3. WEAKNESSES: Areas that need improvement
4. RECOMMENDATIONS: Specific, actionable advice
5. STRATEGY ANALYSIS: Which strategies are working and which aren't

Be specific, use Brooks terminology, and provide actionable insights.

Respond in JSON format:
{
    "patterns": ["list of patterns identified"],
    "strengths": ["list of strengths"],
    "weaknesses": ["list of weaknesses"],
    "recommendations": ["list of specific recommendations"],
    "full_analysis": "A detailed 2-3 paragraph analysis of the trader's performance"
}"""
        
        user_prompt = f"""Analyze these {len(trades)} recent trades:

SUMMARY STATS:
- Win Rate: {win_rate:.1f}%
- Total R: {total_r:.2f}R
- Average R per trade: {avg_r:.2f}R
- Wins: {wins}, Losses: {losses}

TRADES:
{chr(10).join(trade_summaries)}

STRATEGY BREAKDOWN:
{chr(10).join(f"- {name}: {stats['count']} trades, {stats['win_rate']:.0f}% win rate, {stats['total_r']:.2f}R" for name, stats in strategy_breakdown.items())}

Provide a comprehensive analysis of this trader's performance with specific patterns, strengths, weaknesses, and recommendations."""

        # Call LLM (non-blocking)
        import json
        import asyncio
        result_text = await asyncio.to_thread(
            analyzer._call_llm, system_prompt, user_prompt, 3000
        )
        
        if not result_text:
            return JSONResponse({
                "error": "LLM analysis failed. Please try again.",
                "trade_count": len(trades)
            })
        
        # Parse LLM response
        try:
            # Try to extract JSON from response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                analysis = json.loads(result_text[json_start:json_end])
            else:
                analysis = {
                    "patterns": [],
                    "strengths": [],
                    "weaknesses": [],
                    "recommendations": [],
                    "full_analysis": result_text
                }
        except json.JSONDecodeError:
            analysis = {
                "patterns": [],
                "strengths": [],
                "weaknesses": [],
                "recommendations": [],
                "full_analysis": result_text
            }
        
        # Add strategy breakdown to analysis
        analysis['strategy_breakdown'] = strategy_breakdown
        
        return JSONResponse({
            "trade_count": len(trades),
            "stats": {
                "win_rate": win_rate,
                "total_r": total_r,
                "avg_r": avg_r,
                "wins": wins,
                "losses": losses
            },
            "analysis": analysis
        })
        
    except Exception as e:
        logger.error(f"Bulk analysis error: {e}")
        return JSONResponse({
            "error": str(e),
            "trade_count": 0
        })
    finally:
        session.close()


# ==================== RUN SERVER ====================

def run_server(host: str = "127.0.0.1", port: int = 8000):
    """Run the web server."""
    import uvicorn
    print(f"\n🚀 Brooks Trading Coach Web UI")
    print(f"   Open http://{host}:{port} in your browser\n")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
