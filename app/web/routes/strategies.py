"""
Strategy management API routes.

Handles strategy CRUD operations, merging, and categorization.
"""

import logging
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.journal.models import Trade, Strategy, get_session
from app.web.schemas import StrategyCreate, StrategyUpdate, StrategyMerge, success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


# Note: Strategy endpoints remain in server.py for now.
# This module demonstrates the pattern for the incremental migration.
# Future work: Move all strategy endpoints here and update imports.
