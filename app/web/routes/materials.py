"""
Training materials and RAG API routes.

Handles material upload, deletion, and RAG indexing.
"""

import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse

from app.config import MATERIALS_DIR
from app.web.schemas import success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/materials", tags=["materials"])


# Note: Materials endpoints remain in server.py for now.
# This module demonstrates the pattern for the incremental migration.
# Future work: Move all materials endpoints here with proper auth.
