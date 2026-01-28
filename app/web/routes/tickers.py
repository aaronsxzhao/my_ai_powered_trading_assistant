"""
Ticker management API routes.

Handles updating the favorite tickers list.
"""

from __future__ import annotations

from fastapi import APIRouter, Form
from fastapi.responses import RedirectResponse

from app.config import load_tickers_from_file, save_tickers_to_file
from app.web.dependencies import require_write_auth

router = APIRouter(prefix="/api/tickers", tags=["tickers"])


@router.post("", dependencies=[require_write_auth])
async def update_tickers(tickers: str = Form(...)):
    """Replace the tickers list."""
    ticker_list = [
        t.strip().upper()
        for t in tickers.split("\n")
        if t.strip() and not t.strip().startswith("#")
    ]
    save_tickers_to_file(ticker_list)
    return RedirectResponse(url="/tickers", status_code=303)


@router.post("/add", dependencies=[require_write_auth])
async def add_ticker(ticker: str = Form(...)):
    """Add a single ticker."""
    tickers = load_tickers_from_file()
    ticker = ticker.upper().strip()
    if ticker and ticker not in tickers:
        tickers.append(ticker)
        save_tickers_to_file(tickers)
    return RedirectResponse(url="/tickers", status_code=303)


@router.post("/remove/{ticker}", dependencies=[require_write_auth])
async def remove_ticker(ticker: str):
    """Remove a ticker."""
    tickers = load_tickers_from_file()
    ticker = ticker.upper()
    if ticker in tickers:
        tickers.remove(ticker)
        save_tickers_to_file(tickers)
    return RedirectResponse(url="/tickers", status_code=303)
