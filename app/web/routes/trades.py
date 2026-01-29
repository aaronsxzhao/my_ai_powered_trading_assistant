"""
Trade management API routes.

Handles trade CRUD operations, chart data, and trade updates.
"""

import logging
import asyncio
import json
from datetime import date, datetime, timezone

from fastapi import APIRouter, Form, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse

from app.config_prompts import get_cache_settings
from app.journal.coach import TradeCoach
from app.journal.ingest import TradeIngester
from app.journal.models import Strategy, Trade, TradeDirection, TradeOutcome, get_session
from app.web.dependencies import (
    get_user_from_request,
    require_auth,
    require_write_auth,
    verify_trade_ownership,
)
from app.web.schemas import success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trades", tags=["trades"])


# Track cancelled review generations (trade_id -> True if cancelled)
_cancelled_reviews: dict[int, bool] = {}


@router.post("/{trade_id}/recalculate")
async def recalculate_trade_metrics(trade_id: int):
    """Recalculate metrics for a single trade."""
    
    def _do_recalculate():
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                return None
            trade.compute_metrics()
            session.commit()
            return {
                "trade_id": trade_id,
                "r_multiple": trade.r_multiple,
                "pnl_dollars": trade.pnl_dollars,
                "outcome": trade.outcome.value if trade.outcome else None,
            }
        finally:
            session.close()
    
    result = await asyncio.to_thread(_do_recalculate)
    if result is None:
        raise HTTPException(status_code=404, detail="Trade not found")
    
    return JSONResponse(success_response(data=result, message="Metrics recalculated"))


@router.patch("/{trade_id}/notes", dependencies=[require_write_auth])
async def update_trade_notes(trade_id: int, request: Request):
    """Update trade notes, personal review, and trade intent fields."""
    data = await request.json()

    session = get_session()
    try:
        trade, _ = await verify_trade_ownership(trade_id, request, session)

        # Allowed fields that can be updated via this endpoint
        allowed_fields = {
            # Notes
            "notes",
            "entry_reason",
            "exit_reason",
            "setup_type",
            "signal_reason",
            "mistakes",
            "lessons",
            "mistakes_and_lessons",
            # Trade intent
            "trade_type",
            "confidence_level",
            "emotional_state",
            "followed_plan",
            "stop_reason",
            "target_reason",
            "invalidation_condition",
            # Extended analysis
            "trend_assessment",
            "signal_reason",
            "was_signal_present",
            "strategy_alignment",
            "entry_exit_emotions",
            "entry_tp_distance",
        }

        def _do_update():
            # Update all provided fields
            for field in allowed_fields:
                if field in data:
                    setattr(trade, field, data[field])
            # Clear cached review when trade intent is updated (to force re-analysis)
            trade.cached_review = None
            trade.review_generated_at = None
            session.commit()

        await asyncio.to_thread(_do_update)

        return JSONResponse({"success": True, "message": "Trade details saved"})
    finally:
        session.close()


@router.get("/{trade_id}/review")
async def get_trade_review(trade_id: int, force: bool = False, check_only: bool = False):
    """
    Get AI review for a trade (non-blocking async endpoint).

    Uses cached review if available and trade hasn't been modified.
    Set force=true to regenerate the review.
    Set check_only=true to just check status without triggering generation.
    """
    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            return JSONResponse({"success": False, "error": "Trade not found"})

        # Check if generation already in progress
        if trade.review_in_progress:
            return JSONResponse(
                {
                    "success": True,
                    "in_progress": True,
                    "message": "Review generation in progress...",
                }
            )

        cache_settings = get_cache_settings()

        # Return cached review if available (unless force=true)
        if (
            not force
            and cache_settings.get("enable_review_cache", True)
            and trade.cached_review
            and trade.review_generated_at
        ):
            try:
                cached = json.loads(trade.cached_review)
                return JSONResponse(
                    {
                        "success": True,
                        "cached": True,
                        "generated_at": trade.review_generated_at.isoformat()
                        if trade.review_generated_at
                        else None,
                        "review": cached,
                    }
                )
            except json.JSONDecodeError:
                pass  # Fall through to regenerate

        # Check only mode - don't trigger generation
        if check_only:
            return JSONResponse(
                {
                    "success": False,
                    "error": "No cached review available",
                    "in_progress": False,
                }
            )

        # Auto-regenerate option
        if cache_settings.get("auto_regenerate", False):
            force = True

        # Mark in progress and start generation
        trade.review_in_progress = True
        _cancelled_reviews[trade_id] = False
        session.commit()

        def _get_review():
            def is_cancelled():
                return _cancelled_reviews.get(trade_id, False)

            return TradeCoach().review_trade(trade_id, cancellation_check=is_cancelled)

        # Run in thread pool to not block event loop
        review = await asyncio.to_thread(_get_review)

        # Check if cancelled
        if _cancelled_reviews.get(trade_id, False):
            trade.review_in_progress = False
            session.commit()
            _cancelled_reviews.pop(trade_id, None)
            return JSONResponse({"success": False, "error": "Review generation was cancelled"})

        if review:
            session.expire(trade)
            session.refresh(trade)

            ai_setup = review.setup_classification

            # Store original AI classification (once only)
            if ai_setup and not trade.ai_setup_classification:
                trade.ai_setup_classification = ai_setup

            # Auto-set strategy if not manually set
            if ai_setup and ai_setup.lower() not in [
                "unknown",
                "unclassified",
                "insufficient_information",
            ]:
                if not trade.strategy_id:
                    strategy = session.query(Strategy).filter(Strategy.name == ai_setup).first()
                    if not strategy:
                        strategy = Strategy(
                            name=ai_setup, description="Auto-created from AI review"
                        )
                        session.add(strategy)
                        session.flush()
                    trade.strategy_id = strategy.id

            current_strategy = trade.strategy.name if trade.strategy else ai_setup

            review_dict = {
                "grade": review.grade,
                "grade_explanation": review.grade_explanation,
                "regime": review.regime,
                "always_in": review.always_in,
                "context_description": review.context_description,
                # Use trade's strategy for consistency, fall back to AI classification
                "setup_classification": current_strategy or ai_setup or "Unknown",
                "ai_setup_classification": ai_setup,  # Keep the AI's raw classification too
                "setup_quality": review.setup_quality,
                "what_was_good": review.what_was_good or [],
                "what_was_flawed": review.what_was_flawed or [],
                "errors_detected": review.errors_detected or [],
                "rule_for_next_time": review.rule_for_next_time or [],
            }

            # Cache and clear in_progress
            trade.cached_review = json.dumps(review_dict)
            trade.review_generated_at = datetime.now(timezone.utc)
            trade.review_in_progress = False
            session.commit()

            return JSONResponse(
                {
                    "success": True,
                    "cached": False,
                    "generated_at": trade.review_generated_at.isoformat(),
                    "review": review_dict,
                }
            )

        # Clear in_progress flag on failure
        trade.review_in_progress = False
        session.commit()
        return JSONResponse({"success": False, "error": "Review not available"})
    except Exception as e:
        logger.error(f"Error getting trade review: {e}")
        # Clear in_progress flag on error
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if trade:
                trade.review_in_progress = False
                session.commit()
        except Exception:
            pass  # Best effort to clear flag
        return JSONResponse({"success": False, "error": str(e)})
    finally:
        session.close()


@router.post("/{trade_id}/review/cancel", dependencies=[require_write_auth])
async def cancel_trade_review(trade_id: int):
    """Cancel an in-progress review generation."""
    
    def _do_cancel():
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                return {"success": False, "error": "Trade not found"}
            
            _cancelled_reviews[trade_id] = True
            
            if trade.review_in_progress:
                trade.review_in_progress = False
                session.commit()
                return {"success": True, "message": "Review cancelled"}
            return {"success": True, "message": "No review in progress"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            session.close()
    
    result = await asyncio.to_thread(_do_cancel)
    return JSONResponse(result)


@router.get("/{trade_id}/check-data")
async def check_trade_data_availability(trade_id: int):
    """
    Check if local data is available for a futures trade.

    Returns:
        - has_data: True if all required data is available locally
        - is_futures: True if this is a futures ticker
        - missing_schemas: List of schemas that need data
        - missing_dates: Date range that needs data
        - ticker: The normalized ticker
    """
    from datetime import timedelta

    from app.data.providers import DatabentoProvider, normalize_ticker

    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            return JSONResponse({"success": False, "error": "Trade not found"})

        normalized_ticker, exchange = normalize_ticker(trade.ticker)

        # Only check for futures - stocks use Yahoo/Polygon which don't need local data
        if exchange != "FUTURES":
            return JSONResponse(
                {
                    "success": True,
                    "is_futures": False,
                    "has_data": True,
                    "message": "Stock data fetched from online APIs",
                }
            )

        # For futures, check local Databento data
        provider = DatabentoProvider()

        # Determine the date range we need for analysis
        # We need: daily (60 days back), 2h (30 days), 5m (5 days)
        trade_date = trade.trade_date

        # Date ranges needed for each schema
        schemas_needed = {
            "ohlcv-1d": (trade_date - timedelta(days=60), trade_date),
            "ohlcv-1h": (trade_date - timedelta(days=30), trade_date),
            "ohlcv-1m": (trade_date - timedelta(days=5), trade_date),
        }

        missing_schemas: list[str] = []
        missing_date_ranges: dict[str, dict] = {}

        for schema, (start_date, end_date) in schemas_needed.items():
            available = provider.get_available_dates(schema)

            # Check if we have data for the trade date and some context
            # For daily, we need at least 20 bars; for intraday, at least the trade date
            if schema == "ohlcv-1d":
                # Check if we have at least 20 days of data before trade
                count = sum(1 for d in available if start_date <= d <= end_date)
                if count < 20:
                    missing_schemas.append(schema)
                    missing_date_ranges[schema] = {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                        "have": count,
                        "need": 20,
                    }
            else:
                # For intraday, check if trade date is covered
                if trade_date not in available:
                    missing_schemas.append(schema)
                    missing_date_ranges[schema] = {
                        "start": start_date.isoformat(),
                        "end": end_date.isoformat(),
                        "have": len([d for d in available if start_date <= d <= end_date]),
                        "need": "trade date",
                    }

        has_data = len(missing_schemas) == 0

        return JSONResponse(
            {
                "success": True,
                "is_futures": True,
                "has_data": has_data,
                "ticker": normalized_ticker,
                "trade_date": trade_date.isoformat(),
                "missing_schemas": missing_schemas,
                "missing_details": missing_date_ranges,
                "message": "Data available locally"
                if has_data
                else f"Missing data for: {', '.join(missing_schemas)}",
            }
        )

    except Exception as e:
        logger.exception(f"Error checking data availability: {e}")
        return JSONResponse({"success": False, "error": str(e)})
    finally:
        session.close()


@router.post("/{trade_id}/download-data", dependencies=[require_write_auth])
async def download_trade_data(trade_id: int):
    """
    Download missing Databento data for a futures trade.

    Only downloads what's needed for the specific trade analysis.
    """
    from datetime import timedelta

    from app.data.providers import DatabentoProvider, get_databento_api_key, normalize_ticker

    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            return JSONResponse({"success": False, "error": "Trade not found"})

        normalized_ticker, exchange = normalize_ticker(trade.ticker)

        if exchange != "FUTURES":
            return JSONResponse({"success": True, "message": "No download needed for stocks"})

        # Check API key
        api_key = get_databento_api_key()
        if not api_key:
            return JSONResponse(
                {
                    "success": False,
                    "error": "DATABENTO_API_KEY not configured. Please add it to your .env file or download data manually from databento.com",
                }
            )

        # Get base symbol from ticker (e.g., MES=F -> MES)
        provider = DatabentoProvider()
        if normalized_ticker not in provider.FUTURES_MAP:
            return JSONResponse(
                {"success": False, "error": f"Unknown futures symbol: {normalized_ticker}"}
            )

        _, base_symbol = provider.FUTURES_MAP[normalized_ticker]
        trade_date = trade.trade_date

        # Define what we need to download
        download_tasks = [
            ("ohlcv-1d", trade_date - timedelta(days=60), trade_date),
            ("ohlcv-1h", trade_date - timedelta(days=30), trade_date),
            ("ohlcv-1m", trade_date - timedelta(days=5), trade_date),
        ]

        # Run download in thread pool
        def _download():
            # Suppress SWIG warnings before import
            import warnings

            warnings.filterwarnings("ignore", category=DeprecationWarning)
            import databento as db

            client = db.Historical(key=api_key)
            results: list[dict] = []

            for schema, start, end in download_tasks:
                try:
                    # Check what we already have
                    available = provider.get_available_dates(schema)

                    # Find missing dates (skip weekends - futures don't trade Sat/Sun)
                    missing_dates = []
                    current = start
                    while current <= end:
                        # Skip weekends (5=Saturday, 6=Sunday)
                        if current.weekday() < 5 and current not in available:
                            missing_dates.append(current)
                        current += timedelta(days=1)

                    if not missing_dates:
                        results.append(
                            {"schema": schema, "status": "skipped", "message": "Already have data"}
                        )
                        continue

                    # Download the missing range
                    download_start = min(missing_dates)
                    download_end = max(missing_dates)

                    logger.info(
                        f"Downloading {base_symbol} {schema} from {download_start} to {download_end}"
                    )

                    # Use [ROOT].FUT format for futures parent symbol
                    # This gets all contracts for the product (e.g., MES.FUT gets MESZ5, MESH6, etc.)
                    parent_symbol = f"{base_symbol}.FUT"
                    data = client.timeseries.get_range(
                        dataset="GLBX.MDP3",
                        symbols=[parent_symbol],
                        stype_in="parent",  # Get all contracts for this product
                        schema=schema,
                        start=download_start.strftime("%Y-%m-%d"),
                        end=(download_end + timedelta(days=1)).strftime("%Y-%m-%d"),
                    )

                    # Check if we got data
                    df = data.to_df()
                    if df.empty:
                        logger.warning(f"No data returned for {base_symbol} {schema}")
                        results.append(
                            {
                                "schema": schema,
                                "status": "no_data",
                                "message": f"No data available for {download_start} to {download_end}",
                            }
                        )
                        continue

                    # Save to file
                    filename = f"{base_symbol}_{schema}_{download_start.strftime('%Y-%m-%d')}_{download_end.strftime('%Y-%m-%d')}.dbn.zst"
                    output_path = provider.data_dir / filename
                    data.to_file(str(output_path))

                    size_kb = output_path.stat().st_size / 1024
                    results.append(
                        {
                            "schema": schema,
                            "status": "downloaded",
                            "file": filename,
                            "size_kb": round(size_kb, 1),
                            "dates": len(missing_dates),
                            "bars": len(df),
                        }
                    )

                except Exception as e:
                    logger.error(f"Error downloading {schema}: {e}")
                    results.append({"schema": schema, "status": "error", "error": str(e)})

            return results

        results = await asyncio.to_thread(_download)

        # Check if any downloads failed
        errors = [r for r in results if r["status"] == "error"]
        if errors:
            return JSONResponse(
                {
                    "success": False,
                    "partial": True,
                    "results": results,
                    "error": f"Some downloads failed: {errors[0]['error']}",
                }
            )

        return JSONResponse(
            {"success": True, "results": results, "message": "Data downloaded successfully"}
        )

    except Exception as e:
        logger.exception(f"Error downloading data: {e}")
        return JSONResponse({"success": False, "error": str(e)})
    finally:
        session.close()


@router.get("/{trade_id}/chart-data")
async def get_trade_chart_data(
    trade_id: int,
    timeframe: str = "5m",
    rth_only: bool = True,
    show_after_entry: bool = False,
):
    """
    Get OHLCV data for charting a trade.

    Args:
        trade_id: Trade ID
        timeframe: Candle timeframe (1m, 5m, 15m, 1h, 2h, 1d)
        rth_only: If True, filter to Regular Trading Hours only (9:30 AM - 4:00 PM)
        show_after_entry: If True, show data after entry time. If False, cut off at entry.

    Returns candlestick data up to (and optionally after) the trade's entry time.
    """
    # Run the blocking data fetch in a thread to not block the event loop.
    result = await asyncio.to_thread(
        _fetch_chart_data_sync, trade_id, timeframe, rth_only, show_after_entry
    )
    return JSONResponse(result)


def _fetch_chart_data_sync(
    trade_id: int, timeframe: str, rth_only: bool, show_after_entry: bool
) -> dict:
    """Synchronous chart data fetching - runs in a thread pool."""
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    from app.data.cache import get_cached_ohlcv
    from app.data.providers import get_provider_for_ticker

    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            return {"success": False, "error": "Trade not found"}

        # Derive market timezone from ticker (more reliable than stored field for old trades)
        from app.journal.ingest import get_market_timezone

        market_tz_str = get_market_timezone(trade.ticker)
        market_tz = ZoneInfo(market_tz_str)

        # Determine the time range to fetch
        # Trade times are stored as naive datetimes in market_timezone
        entry_time = trade.entry_time or trade.exit_time
        exit_time = trade.exit_time or trade.entry_time

        if not entry_time and not exit_time:
            # Fallback to trade_date with market-specific times
            from datetime import datetime as dt

            # Market open/close times by timezone
            market_hours = {
                "Asia/Hong_Kong": (9, 30, 16, 0),  # 9:30 AM - 4:00 PM HKT
                "Asia/Shanghai": (9, 30, 15, 0),  # 9:30 AM - 3:00 PM CST
                "Asia/Tokyo": (9, 0, 15, 0),  # 9:00 AM - 3:00 PM JST
                "Europe/London": (8, 0, 16, 30),  # 8:00 AM - 4:30 PM GMT/BST
                "America/New_York": (9, 30, 16, 0),  # 9:30 AM - 4:00 PM EST/EDT
            }
            open_h, open_m, close_h, close_m = market_hours.get(market_tz_str, (9, 30, 16, 0))
            entry_time = dt.combine(
                trade.trade_date, dt.min.time().replace(hour=open_h, minute=open_m)
            )
            exit_time = dt.combine(
                trade.trade_date, dt.min.time().replace(hour=close_h, minute=close_m)
            )

        # Make times timezone-aware in market timezone
        if entry_time.tzinfo is None:
            entry_time_aware = entry_time.replace(tzinfo=market_tz)
        else:
            entry_time_aware = entry_time

        if exit_time.tzinfo is None:
            exit_time_aware = exit_time.replace(tzinfo=market_tz)
        else:
            exit_time_aware = exit_time

        # Store original timeframe (may fall back to daily for old trades)
        original_timeframe = timeframe

        # Calculate lookback based on timeframe
        timeframe_bars = {
            "1m": 200,
            "5m": 150,
            "15m": 100,
            "1h": 80,
            "2h": 60,
            "1d": 60,
        }
        bars_to_fetch = timeframe_bars.get(timeframe, 150)

        # Calculate time delta per bar
        timeframe_deltas = {
            "1m": timedelta(minutes=1),
            "5m": timedelta(minutes=5),
            "15m": timedelta(minutes=15),
            "1h": timedelta(hours=1),
            "2h": timedelta(hours=2),
            "1d": timedelta(days=1),
        }
        bar_delta = timeframe_deltas.get(timeframe, timedelta(minutes=5))

        # Fetch data with lookback before entry
        lookback = bar_delta * bars_to_fetch
        start_time = entry_time_aware - lookback

        # End time: either at entry or include some bars after
        if show_after_entry:
            # Show a few bars after exit for context
            end_time = exit_time_aware + (bar_delta * 10)
        else:
            # Cut off at entry time - show only what trader saw before entering
            end_time = entry_time_aware + bar_delta  # Include entry bar

        # For intraday, add extra days for weekend handling
        if timeframe in ["1m", "5m", "15m", "1h", "2h"]:
            start_time = start_time - timedelta(days=5)
            if show_after_entry:
                end_time = end_time + timedelta(days=2)
        else:
            start_time = start_time - timedelta(days=30)
            if show_after_entry:
                end_time = end_time + timedelta(days=10)

        # Fetch OHLCV data
        # For futures, pass entry_price to help select correct contract month
        provider = get_provider_for_ticker(trade.ticker)
        target_price = trade.entry_price if trade.entry_price else None

        # Try cache first (but cache doesn't support target_price filtering)
        df = get_cached_ohlcv(trade.ticker, timeframe, start_time, end_time)

        if df.empty:
            # Cache miss - fetch from provider with target_price and trade_date for contract selection
            import inspect

            sig = inspect.signature(provider.get_ohlcv)
            if "target_price" in sig.parameters and "trade_date" in sig.parameters:
                df = provider.get_ohlcv(
                    trade.ticker,
                    timeframe,
                    start_time,
                    end_time,
                    target_price=target_price,
                    trade_date=trade.trade_date,
                )
            elif "target_price" in sig.parameters:
                df = provider.get_ohlcv(
                    trade.ticker, timeframe, start_time, end_time, target_price=target_price
                )
            else:
                df = provider.get_ohlcv(trade.ticker, timeframe, start_time, end_time)

        # If intraday data is empty for old trades, try falling back to daily data
        if df.empty and timeframe != "1d":
            # Check if trade is older than 30 days (Yahoo only keeps ~30 days of intraday)
            days_old = (datetime.now().date() - trade.trade_date).days
            if days_old > 30:
                logger.info(
                    f"Trade is {days_old} days old, falling back to daily data for {trade.ticker}"
                )
                # Fetch daily data with wider range
                daily_start = start_time - timedelta(days=30)
                daily_end = end_time + timedelta(days=10)
                df = get_cached_ohlcv(trade.ticker, "1d", daily_start, daily_end)
                if df.empty:
                    df = provider.get_ohlcv(trade.ticker, "1d", daily_start, daily_end)
                timeframe = "1d"  # Update timeframe for response

        if df.empty:
            # Provide more helpful error message
            days_old = (datetime.now().date() - trade.trade_date).days
            if days_old > 30:
                error_msg = (
                    f"No data available for {trade.ticker}. "
                    f"Trade is {days_old} days old - intraday data typically only available for ~30 days. "
                    f"Try viewing with Daily timeframe."
                )
            else:
                error_msg = f"No data available for {trade.ticker} ({original_timeframe})"
            return {"success": False, "error": error_msg}

        # Filter to RTH (Regular Trading Hours) if requested
        if rth_only and timeframe != "1d":
            rth_hours = {
                "America/New_York": (9, 30, 16, 0),  # 9:30 AM - 4:00 PM ET
                "Asia/Hong_Kong": (9, 30, 16, 0),  # 9:30 AM - 4:00 PM HKT
                "Asia/Shanghai": (9, 30, 15, 0),  # 9:30 AM - 3:00 PM CST
                "Asia/Tokyo": (9, 0, 15, 0),  # 9:00 AM - 3:00 PM JST
                "Europe/London": (8, 0, 16, 30),  # 8:00 AM - 4:30 PM GMT
            }
            open_h, open_m, close_h, close_m = rth_hours.get(market_tz_str, (9, 30, 16, 0))

            # Convert datetime to market timezone for filtering
            df_filtered = df.copy()
            if df_filtered["datetime"].dt.tz is not None:
                df_times = df_filtered["datetime"].dt.tz_convert(market_tz_str)
            else:
                df_times = df_filtered["datetime"]

            # Create time bounds
            rth_start_minutes = open_h * 60 + open_m
            rth_end_minutes = close_h * 60 + close_m

            # Calculate minutes since midnight for each bar
            bar_minutes = df_times.dt.hour * 60 + df_times.dt.minute

            # Filter to RTH
            rth_mask = (bar_minutes >= rth_start_minutes) & (bar_minutes < rth_end_minutes)
            df = df[rth_mask]

            if df.empty:
                return {
                    "success": False,
                    "error": "No RTH data available. Try disabling RTH filter.",
                }

        # Filter to cut off at entry time if show_after_entry is False
        if not show_after_entry:
            if df["datetime"].dt.tz is not None:
                entry_compare = entry_time_aware
            else:
                entry_compare = entry_time_aware.replace(tzinfo=None)

            df = df[df["datetime"] <= entry_compare]

            if df.empty:
                return {"success": False, "error": "No data before entry time."}

        # Convert to TradingView Lightweight Charts format
        candles = []
        for _, row in df.iterrows():
            timestamp = int(row["datetime"].timestamp())
            candles.append(
                {
                    "time": timestamp,
                    "open": round(float(row["open"]), 4),
                    "high": round(float(row["high"]), 4),
                    "low": round(float(row["low"]), 4),
                    "close": round(float(row["close"]), 4),
                    "volume": int(row["volume"]) if "volume" in row and row["volume"] else 0,
                }
            )

        candles = sorted(candles, key=lambda x: x["time"])

        markers = []

        def find_closest_candle_time(target_ts, candle_list):
            """Find the candle time closest to the target timestamp."""
            if not candle_list:
                return target_ts
            closest = min(candle_list, key=lambda c: abs(c["time"] - target_ts))
            return closest["time"]

        # Entry marker
        if trade.entry_time:
            if trade.entry_time.tzinfo is None:
                entry_aware = trade.entry_time.replace(tzinfo=market_tz)
            else:
                entry_aware = trade.entry_time
            entry_ts = int(entry_aware.timestamp())
            snapped_entry_ts = find_closest_candle_time(entry_ts, candles)

            is_long = trade.direction.value == "long"
            entry_color = "#26a69a" if is_long else "#ef5350"
            entry_label = "𝗕𝗨𝗬" if is_long else "𝗦𝗘𝗟𝗟"

            markers.append(
                {
                    "time": snapped_entry_ts,
                    "position": "belowBar" if is_long else "aboveBar",
                    "color": entry_color,
                    "shape": "arrowUp" if is_long else "arrowDown",
                    "text": f"{entry_label} ${trade.entry_price:.2f}",
                }
            )

        # Exit marker (only show when after entry is selected)
        if show_after_entry and trade.exit_time and trade.exit_price:
            if trade.exit_time.tzinfo is None:
                exit_aware = trade.exit_time.replace(tzinfo=market_tz)
            else:
                exit_aware = trade.exit_time
            exit_ts = int(exit_aware.timestamp())
            snapped_exit_ts = find_closest_candle_time(exit_ts, candles)

            is_long = trade.direction.value == "long"
            exit_color = "#ef5350" if is_long else "#26a69a"
            exit_label = "𝗦𝗘𝗟𝗟" if is_long else "𝗕𝗨𝗬"

            markers.append(
                {
                    "time": snapped_exit_ts,
                    "position": "aboveBar" if is_long else "belowBar",
                    "color": exit_color,
                    "shape": "arrowDown" if is_long else "arrowUp",
                    "text": f"{exit_label} ${trade.exit_price:.2f}",
                }
            )

        # Price lines for entry, exit, stop loss, target
        price_lines = []
        is_long = trade.direction.value == "long"

        entry_line_color = "#26a69a" if is_long else "#ef5350"
        price_lines.append(
            {
                "price": trade.entry_price,
                "color": entry_line_color,
                "lineWidth": 1,
                "lineStyle": 2,
                "title": "Entry",
            }
        )

        if show_after_entry and trade.exit_price:
            exit_line_color = "#ef5350" if is_long else "#26a69a"
            price_lines.append(
                {
                    "price": trade.exit_price,
                    "color": exit_line_color,
                    "lineWidth": 1,
                    "lineStyle": 2,
                    "title": "Exit",
                }
            )

        if trade.effective_stop_loss:
            price_lines.append(
                {
                    "price": trade.effective_stop_loss,
                    "color": "#f97316",
                    "lineWidth": 1,
                    "lineStyle": 2,
                    "title": "Stop",
                }
            )

        if trade.effective_take_profit:
            price_lines.append(
                {
                    "price": trade.effective_take_profit,
                    "color": "#3b82f6",
                    "lineWidth": 1,
                    "lineStyle": 2,
                    "title": "Target",
                }
            )

        debug_info = {
            "market_timezone": market_tz_str,
            "entry_time_raw": str(trade.entry_time) if trade.entry_time else None,
            "exit_time_raw": str(trade.exit_time) if trade.exit_time else None,
            "candle_count": len(candles),
            "first_candle_time": candles[0]["time"] if candles else None,
            "last_candle_time": candles[-1]["time"] if candles else None,
        }

        timeframe_note = None
        if timeframe != original_timeframe:
            days_old = (datetime.now().date() - trade.trade_date).days
            timeframe_note = (
                f"Intraday data not available (trade is {days_old} days old). Showing daily chart."
            )

        return {
            "success": True,
            "ticker": trade.ticker,
            "timeframe": timeframe,
            "requested_timeframe": original_timeframe,
            "timeframe_note": timeframe_note,
            "rth_only": rth_only,
            "show_after_entry": show_after_entry,
            "candles": candles,
            "markers": markers,
            "price_lines": price_lines,
            "trade_info": {
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "stop_loss": trade.effective_stop_loss,
                "take_profit": trade.effective_take_profit,
                "direction": trade.direction.value,
                "r_multiple": trade.r_multiple,
            },
            "debug": debug_info,
        }

    except Exception as e:
        logger.error(f"Error fetching chart data: {e}")
        import traceback

        return {"success": False, "error": str(e), "traceback": traceback.format_exc()}
    finally:
        session.close()


@router.get("/export")
async def export_trades(
    request: Request,
    ticker: str | None = None,
    outcome: str | None = None,
    include_reviews: bool = False,
):
    """
    Export trades to CSV format.

    Args:
        ticker: Filter by ticker symbol
        outcome: Filter by outcome (win, loss, breakeven)
        include_reviews: Include AI review summaries in export

    Returns:
        CSV file download
    """
    import csv
    import io

    # Get current user
    user = await get_user_from_request(request)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")

    session = get_session()
    try:
        # Build query with user filter
        query = session.query(Trade).filter(Trade.user_id == user.id)

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

        trades = query.order_by(Trade.trade_date.desc(), Trade.exit_time.desc()).all()

        output = io.StringIO()
        writer = csv.writer(output)

        headers = [
            "Trade #",
            "Date",
            "Ticker",
            "Direction",
            "Size",
            "Entry Price",
            "Exit Price",
            "Stop Loss",
            "Take Profit",
            "Entry Time",
            "Exit Time",
            "Duration",
            "P&L ($)",
            "R-Multiple",
            "Outcome",
            "Strategy",
            "Setup Type",
            "Notes",
            "Entry Reason",
            "Exit Reason",
        ]
        if include_reviews:
            headers.extend(["AI Grade", "AI Summary"])
        writer.writerow(headers)

        for trade in trades:
            duration_str = ""
            if trade.entry_time and trade.exit_time:
                duration = trade.exit_time - trade.entry_time
                hours, remainder = divmod(int(duration.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                if hours > 0:
                    duration_str = f"{hours}h {minutes}m"
                else:
                    duration_str = f"{minutes}m {seconds}s"

            row = [
                trade.trade_number or trade.id,
                trade.trade_date.strftime("%Y-%m-%d") if trade.trade_date else "",
                trade.ticker,
                trade.direction.value if trade.direction else "",
                trade.size,
                trade.entry_price,
                trade.exit_price,
                trade.stop_loss or "",
                trade.take_profit or "",
                trade.entry_time.strftime("%Y-%m-%d %H:%M:%S") if trade.entry_time else "",
                trade.exit_time.strftime("%Y-%m-%d %H:%M:%S") if trade.exit_time else "",
                duration_str,
                round(trade.pnl_dollars, 2) if trade.pnl_dollars else "",
                round(trade.r_multiple, 2) if trade.r_multiple else "",
                trade.outcome.value if trade.outcome else "",
                trade.strategy.name if trade.strategy else "",
                trade.setup_type or "",
                trade.notes or "",
                trade.entry_reason or "",
                trade.exit_reason or "",
            ]

            if include_reviews:
                grade = ""
                summary = ""
                if trade.cached_review:
                    try:
                        review = json.loads(trade.cached_review)
                        grade = review.get("grade", "")
                        summary_text = review.get("summary", "")
                        if summary_text:
                            summary = (
                                summary_text[:200] + "..."
                                if len(summary_text) > 200
                                else summary_text
                            )
                    except Exception:
                        pass
                row.extend([grade, summary])

            writer.writerow(row)

        output.seek(0)
        filename = f"trades_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    finally:
        session.close()


@router.post("", dependencies=[require_write_auth])
async def create_trade(
    request: Request,
    ticker: str = Form(...),
    direction: str = Form(...),
    entry_price: float = Form(...),
    exit_price: float = Form(...),
    stop_loss: float = Form(None),  # SL - optional
    take_profit: float = Form(None),  # TP - optional
    size: float = Form(1.0),
    entry_time: str | None = Form(None),
    exit_time: str | None = Form(None),
    timeframe: str = Form("5m"),
    strategy: str | None = Form(None),
    notes: str | None = Form(None),
    entry_reason: str | None = Form(None),
):
    """Create a new trade."""
    user = await get_user_from_request(request)
    user_id = user.id if user else None

    ingester = TradeIngester()

    parsed_entry_time = None
    parsed_exit_time = None

    if entry_time:
        try:
            parsed_entry_time = datetime.fromisoformat(
                entry_time.replace("Z", "+00:00").replace("T", " ").split("+")[0]
            )
        except ValueError:
            try:
                parsed_entry_time = datetime.strptime(entry_time, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                try:
                    parsed_entry_time = datetime.strptime(entry_time, "%Y-%m-%dT%H:%M")
                except ValueError:
                    parsed_entry_time = None

    if exit_time:
        try:
            parsed_exit_time = datetime.fromisoformat(
                exit_time.replace("Z", "+00:00").replace("T", " ").split("+")[0]
            )
        except ValueError:
            try:
                parsed_exit_time = datetime.strptime(exit_time, "%Y-%m-%dT%H:%M:%S")
            except ValueError:
                try:
                    parsed_exit_time = datetime.strptime(exit_time, "%Y-%m-%dT%H:%M")
                except ValueError:
                    parsed_exit_time = None

    # Derive trade_date from exit_time, entry_time, or today
    if parsed_exit_time:
        parsed_date = parsed_exit_time.date()
    elif parsed_entry_time:
        parsed_date = parsed_entry_time.date()
    else:
        parsed_date = date.today()

    trade = ingester.add_trade_manual(
        ticker=ticker,
        trade_date=parsed_date,
        direction=direction,
        entry_price=entry_price,
        exit_price=exit_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        size=size,
        timeframe=timeframe,
        entry_time=parsed_entry_time,
        exit_time=parsed_exit_time,
        strategy_name=strategy if strategy else None,
        notes=notes,
        entry_reason=entry_reason,
        user_id=user_id,
    )

    return RedirectResponse(url=f"/trades/{trade.id}", status_code=303)


@router.delete("/{trade_id}", dependencies=[require_auth])
async def delete_trade(trade_id: int):
    """Delete a trade. Requires API key if APP_API_KEY is set."""
    ingester = TradeIngester()
    if ingester.delete_trade(trade_id):
        return JSONResponse(success_response(message="Trade deleted"))
    raise HTTPException(status_code=404, detail="Trade not found")


@router.delete("", dependencies=[require_auth])
async def delete_all_trades():
    """Delete ALL trades. Use with caution! Requires API key if APP_API_KEY is set."""
    ingester = TradeIngester()
    count = ingester.delete_all_trades()
    return JSONResponse(success_response(data={"count": count}, message=f"Deleted {count} trades"))


@router.patch("/{trade_id}", dependencies=[require_write_auth])
async def update_trade(trade_id: int, request: Request):
    """Update trade fields (size, prices, times, currency)."""
    data = await request.json()

    session = get_session()
    try:
        trade, _ = await verify_trade_ownership(trade_id, request, session)
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")

        def _do_update():
            if data.get("ticker") is not None:
                trade.ticker = data["ticker"].upper()
            if data.get("direction") is not None:
                direction_str = data["direction"].lower()
                if direction_str == "long":
                    trade.direction = TradeDirection.LONG
                elif direction_str == "short":
                    trade.direction = TradeDirection.SHORT
            if data.get("size") is not None:
                trade.size = float(data["size"])
            if data.get("entry_price") is not None:
                trade.entry_price = float(data["entry_price"])
            if data.get("exit_price") is not None:
                trade.exit_price = float(data["exit_price"])

            # Stop Loss (SL) - can be set to null
            if "stop_loss" in data:
                trade.stop_loss = float(data["stop_loss"]) if data["stop_loss"] else None

            # Take Profit (TP) - can be set to null
            if "take_profit" in data:
                trade.take_profit = float(data["take_profit"]) if data["take_profit"] else None

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

            trade.compute_metrics()
            session.commit()
            return {"r_multiple": trade.r_multiple, "pnl_dollars": trade.pnl_dollars}

        result = await asyncio.to_thread(_do_update)

        return JSONResponse(
            success_response(
                data={"trade_id": trade_id, **result},
                message="Trade updated",
            )
        )
    finally:
        session.close()


@router.patch("/{trade_id}/strategy", dependencies=[require_write_auth])
async def update_trade_strategy(trade_id: int, request: Request):
    """Manually update a trade's strategy (override AI classification)."""
    data = await request.json()
    strategy_name = data.get("strategy_name")

    session = get_session()
    try:
        trade, _ = await verify_trade_ownership(trade_id, request, session)

        def _do_update():
            strategy = session.query(Strategy).filter(Strategy.name == strategy_name).first()
            if not strategy:
                new_strategy = Strategy(
                    name=strategy_name,
                    category=data.get("category", "unknown"),
                    description="Manually created",
                )
                session.add(new_strategy)
                session.flush()
                trade.strategy_id = new_strategy.id
            else:
                trade.strategy_id = strategy.id
            session.commit()
            return trade.ai_setup_classification

        original_ai = await asyncio.to_thread(_do_update)

        return JSONResponse(
            {
                "message": f"Trade strategy set to '{strategy_name}'",
                "original_ai": original_ai,
            }
        )
    finally:
        session.close()
