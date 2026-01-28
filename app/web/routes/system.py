"""
System/health API routes.
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.config import (
    IMPORTS_DIR,
    OUTPUTS_DIR,
    get_llm_api_key,
    get_polygon_api_key,
    is_auth_enabled,
    load_tickers_from_file,
    settings,
)
from app.web.schemas import success_response

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return JSONResponse(success_response(data={"status": "healthy"}, message="Service is running"))


@router.get("/api/status")
async def api_status():
    """Get API status including auth and service availability."""
    return JSONResponse(
        success_response(
            data={
                "version": "0.1.0",
                "auth_enabled": is_auth_enabled(),
                "llm_available": get_llm_api_key() is not None,
                "polygon_available": get_polygon_api_key() is not None,
                "data_provider": settings.data_provider,
                # Extra status fields (kept stable in `data`)
                "tickers": load_tickers_from_file(),
                "outputs_dir": str(OUTPUTS_DIR),
                "imports_dir": str(IMPORTS_DIR),
            }
        )
    )


@router.get("/api/exchange-rate")
async def get_exchange_rate_api(currency: str, date: str | None = None):
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
            trade_date = datetime.fromisoformat(date.replace("Z", "+00:00")).date()
        except ValueError:
            try:
                trade_date = date_type.fromisoformat(date[:10])
            except ValueError:
                trade_date = None

    rate = get_exchange_rate(currency, trade_date)

    if rate is None:
        fallback_rates = {
            "HKD": 7.78,
            "EUR": 0.92,
            "GBP": 0.79,
            "JPY": 149.5,
            "CNY": 7.24,
            "CAD": 1.36,
            "AUD": 1.53,
            "CHF": 0.88,
            "SGD": 1.34,
            "KRW": 1320.0,
            "TWD": 31.5,
        }
        rate = fallback_rates.get(currency, 1.0)
        return JSONResponse(
            {
                "currency": currency,
                "rate": rate,
                "date": date or str(date_type.today()),
                "source": "fallback",
            }
        )

    return JSONResponse(
        {
            "currency": currency,
            "rate": rate,
            "date": date or str(date_type.today()),
            "source": "frankfurter",
        }
    )
