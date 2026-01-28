"""
Import-related API routes.

Includes:
- CSV upload/import/bulk import
- Robinhood import helpers
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from app.config import IMPORTS_DIR
from app.journal.ingest import TradeIngester
from app.web.dependencies import get_user_from_request, require_write_auth
from app.web.utils import (
    ALLOWED_CSV_EXTENSIONS,
    MAX_CSV_SIZE,
    sanitize_filename,
    validate_upload_file,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["imports"])


@router.post("/upload-csv", dependencies=[require_write_auth])
async def upload_csv(file: UploadFile = File(...)):
    """Upload a CSV file to imports folder."""
    content = await validate_upload_file(file, ALLOWED_CSV_EXTENSIONS, MAX_CSV_SIZE, "CSV file")

    filename = sanitize_filename(file.filename)
    file_path = IMPORTS_DIR / filename

    with open(file_path, "wb") as f:
        f.write(content)

    return JSONResponse({"message": f"Uploaded {filename}", "path": str(file_path)})


@router.post("/import-csv", dependencies=[require_write_auth])
async def import_csv(
    request: Request,
    file: UploadFile = File(...),
    format: str = Form("generic"),
    balance_file: Optional[UploadFile] = File(None),
    input_timezone: str = Form("America/New_York"),
):
    """Upload and import a CSV file (non-blocking)."""
    user = await get_user_from_request(request)
    user_id = user.id if user else None

    # Validate main CSV file
    try:
        content = await validate_upload_file(file, ALLOWED_CSV_EXTENSIONS, MAX_CSV_SIZE, "CSV file")
    except HTTPException as e:
        return JSONResponse(
            {"error": e.detail, "imported": 0, "errors": 1}, status_code=e.status_code
        )

    try:
        # Save main file
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Save balance file if provided (also validate it)
        balance_path = None
        if balance_file and balance_file.filename:
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".csv", delete=False) as bal_tmp:
                bal_content = await balance_file.read()
                bal_tmp.write(bal_content)
                balance_path = bal_tmp.name

        def _do_import():
            ingester = TradeIngester()
            if format == "tv_order_history":
                return ingester._import_tv_order_history(
                    Path(tmp_path),
                    skip_errors=True,
                    balance_file_path=Path(balance_path) if balance_path else None,
                    input_timezone=input_timezone,
                    user_id=user_id,
                )
            if format == "interactive_brokers":
                return ingester._import_interactive_brokers_csv(
                    Path(tmp_path),
                    skip_errors=True,
                    input_timezone=input_timezone,
                    user_id=user_id,
                )
            return ingester.import_csv(tmp_path, format=format, user_id=user_id)

        imported, errors, messages = await asyncio.to_thread(_do_import)

        os.unlink(tmp_path)
        if balance_path:
            os.unlink(balance_path)

        return JSONResponse(
            {
                "imported": imported,
                "errors": errors,
                "total_messages": len(messages),
                "messages": messages[:50],
                "format": format,
                "cross_validated": balance_path is not None,
            }
        )

    except Exception as e:
        logger.error(f"Import error: {e}")
        return JSONResponse({"error": str(e), "imported": 0, "errors": 1})


@router.post("/bulk-import", dependencies=[require_write_auth])
async def bulk_import(request: Request):
    """Import all CSVs from imports folder (non-blocking)."""
    user = await get_user_from_request(request)
    user_id = user.id if user else None

    def _do_import():
        ingester = TradeIngester()
        return ingester.bulk_import_from_folder(user_id=user_id)

    imported, errors, messages = await asyncio.to_thread(_do_import)

    return JSONResponse(
        {
            "imported": imported,
            "errors": errors,
            "total_messages": len(messages),
            "messages": messages[:50],
        }
    )


# ==================== ROBINHOOD API ====================


@router.get("/robinhood/status")
async def robinhood_status():
    """Check Robinhood connection status."""
    ingester = TradeIngester()
    return JSONResponse({"has_session": ingester.robinhood_has_session()})


@router.post("/robinhood/import", dependencies=[require_write_auth])
async def robinhood_import(request: Request):
    """Import trades from Robinhood with login."""
    user = await get_user_from_request(request)
    user_id = user.id if user else None

    try:
        data = await request.json()
        username = data.get("username")
        password = data.get("password")
        mfa_code = data.get("mfa_code")
        days_back = data.get("days_back", 30)

        if not username or not password:
            return JSONResponse(
                {
                    "error": "Username and password required",
                    "imported": 0,
                    "errors": 1,
                    "needs_mfa": False,
                    "needs_device_approval": False,
                }
            )

        ingester = TradeIngester()
        result = ingester.import_from_robinhood(
            username=username,
            password=password,
            mfa_code=mfa_code,
            days_back=days_back,
            user_id=user_id,
        )

        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(
            {
                "error": str(e),
                "imported": 0,
                "errors": 1,
                "needs_mfa": False,
                "needs_device_approval": False,
            }
        )


@router.post("/robinhood/import-session", dependencies=[require_write_auth])
async def robinhood_import_session(request: Request):
    """Import trades from Robinhood using stored session."""
    # Get current user for ownership
    user = await get_user_from_request(request)
    user_id = user.id if user else None

    try:
        data = await request.json()
        days_back = data.get("days_back", 30)

        ingester = TradeIngester()

        # Try to login with stored session
        if not ingester.robinhood_login_with_session():
            return JSONResponse(
                {"error": "No stored session. Please login first.", "imported": 0, "errors": 1}
            )

        from app.data.robinhood import get_robinhood_client

        client = get_robinhood_client()

        orders = client.get_stock_orders(days_back=days_back)

        if not orders:
            return JSONResponse(
                {
                    "imported": 0,
                    "errors": 0,
                    "messages": [f"No filled orders found in the last {days_back} days."],
                }
            )

        messages = [f"Found {len(orders)} filled orders from Robinhood"]
        imported = 0
        errors = 0

        for order in orders:
            try:
                parsed = client.parse_stock_order_to_trade(order)
                if parsed:
                    trade = ingester.add_trade_manual(
                        ticker=parsed["ticker"],
                        trade_date=parsed["trade_date"],
                        direction=parsed["direction"],
                        entry_price=parsed["entry_price"],
                        exit_price=parsed["exit_price"],
                        size=parsed.get("size", 1),
                        entry_time=parsed.get("entry_time"),
                        exit_time=parsed.get("exit_time"),
                        user_id=user_id,
                        skip_duplicates=True,
                    )
                    if trade:
                        imported += 1
                    else:
                        messages.append(f"Skipped duplicate: {parsed['ticker']}")
            except Exception as e:
                errors += 1
                messages.append(f"Error processing order: {str(e)}")

        messages.append(f"Imported {imported} trades, {errors} errors")

        return JSONResponse({"imported": imported, "errors": errors, "messages": messages[:20]})

    except Exception as e:
        return JSONResponse({"error": str(e), "imported": 0, "errors": 1})


@router.post("/robinhood/disconnect", dependencies=[require_write_auth])
async def robinhood_disconnect():
    """Disconnect Robinhood (clear stored session)."""
    ingester = TradeIngester()
    ingester.robinhood_clear_session()
    return JSONResponse({"message": "Disconnected"})


# ==================== IBKR FLEX (AUTO-IMPORT) ====================


@router.get("/ibkr/status")
async def ibkr_status():
    """Check whether IBKR Flex is configured (env vars present)."""
    from app.data.ibkr_flex import get_ibkr_flex_query_id, get_ibkr_flex_token

    token = get_ibkr_flex_token()
    query_id = get_ibkr_flex_query_id()
    return JSONResponse(
        {
            "configured": bool(token and query_id),
            "has_token": bool(token),
            "has_query_id": bool(query_id),
        }
    )


@router.post("/ibkr/import", dependencies=[require_write_auth])
async def ibkr_import(request: Request):
    """Import trades from IBKR Flex Web Service (read-only)."""
    user = await get_user_from_request(request)
    user_id = user.id if user else None

    try:
        data = await request.json()
    except Exception:
        data = {}

    days_back = data.get("days_back", 30)
    token = data.get("token")
    query_id = data.get("query_id")
    input_timezone = data.get("input_timezone") or "America/New_York"

    def _do_import():
        ingester = TradeIngester()
        return ingester.import_from_ibkr_flex(
            days_back=days_back,
            token=token,
            query_id=query_id,
            input_timezone=input_timezone,
            user_id=user_id,
        )

    result = await asyncio.to_thread(_do_import)
    return JSONResponse(result)
