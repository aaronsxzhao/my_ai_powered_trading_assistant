"""
Trade management API routes.

Handles trade CRUD operations, chart data, and trade updates.
"""

import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import JSONResponse

from app.journal.models import Trade, get_session, TradeDirection
from app.journal.ingest import TradeIngester
from app.web.schemas import success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/trades", tags=["trades"])


# Note: Most trade endpoints remain in server.py for now to avoid 
# breaking the tightly coupled template rendering. This module 
# demonstrates the pattern for future refactoring.


@router.post("/{trade_id}/recalculate")
async def recalculate_trade_metrics(trade_id: int):
    """Recalculate metrics for a single trade."""
    session = get_session()
    try:
        trade = session.query(Trade).filter(Trade.id == trade_id).first()
        if not trade:
            raise HTTPException(status_code=404, detail="Trade not found")
        
        trade.compute_metrics()
        session.commit()
        
        return JSONResponse(success_response(
            data={
                "trade_id": trade_id,
                "r_multiple": trade.r_multiple,
                "pnl_dollars": trade.pnl_dollars,
                "outcome": trade.outcome.value if trade.outcome else None,
            },
            message="Metrics recalculated"
        ))
    finally:
        session.close()
