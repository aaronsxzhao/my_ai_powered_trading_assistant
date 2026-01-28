"""
Shared utilities for the FastAPI web UI.

Extracted from `app/web/server.py` to keep the main server module smaller.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, UploadFile

from app.config import get_llm_api_key, settings


# ==================== FILE UPLOAD SECURITY ====================

# Maximum file sizes (in bytes)
MAX_CSV_SIZE = 10 * 1024 * 1024  # 10 MB for CSV imports
MAX_MATERIAL_SIZE = 50 * 1024 * 1024  # 50 MB for training materials (PDFs can be large)
MAX_TOTAL_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB total for batch uploads

# Allowed file extensions
ALLOWED_CSV_EXTENSIONS = {".csv"}
ALLOWED_MATERIAL_EXTENSIONS = {".pdf", ".txt", ".md"}


async def validate_upload_file(
    file: UploadFile,
    allowed_extensions: set[str],
    max_size: int,
    error_prefix: str = "File",
) -> bytes:
    """
    Validate and read an uploaded file.

    Args:
        file: The uploaded file
        allowed_extensions: Set of allowed file extensions (e.g., {'.csv', '.pdf'})
        max_size: Maximum file size in bytes
        error_prefix: Prefix for error messages

    Returns:
        File content as bytes

    Raises:
        HTTPException: If validation fails
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail=f"{error_prefix} name is required")

    # Sanitize filename
    filename = Path(file.filename).name  # Strip any path components
    ext = Path(filename).suffix.lower()

    # Check extension
    if ext not in allowed_extensions:
        allowed = ", ".join(sorted(allowed_extensions))
        raise HTTPException(
            status_code=400,
            detail=f"{error_prefix} type '{ext}' not allowed. Allowed types: {allowed}",
        )

    # Read content and check size
    content = await file.read()
    if len(content) > max_size:
        max_mb = max_size / (1024 * 1024)
        actual_mb = len(content) / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"{error_prefix} too large: {actual_mb:.1f} MB (max: {max_mb:.0f} MB)",
        )

    return content


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename to prevent path traversal attacks."""
    # Get just the filename, strip any directory components
    name = Path(filename).name
    # Remove any null bytes or other problematic characters
    name = name.replace("\x00", "").replace("/", "").replace("\\", "")
    return name


# ==================== SIMPLE IN-PROCESS CACHE ====================

_cache: dict[str, tuple[Any, float]] = {}
CACHE_TTL = 60  # seconds


def get_cached(key: str, ttl: int = CACHE_TTL):
    """Get value from cache if not expired."""
    if key in _cache:
        value, timestamp = _cache[key]
        if time.time() - timestamp < ttl:
            return value
    return None


def set_cached(key: str, value: Any) -> None:
    """Store value in cache."""
    _cache[key] = (value, time.time())


def clear_cache(key: str | None = None) -> None:
    """Clear cache (specific key or all)."""
    if key:
        _cache.pop(key, None)
    else:
        _cache.clear()


def get_active_strategies_cached(session) -> list:
    """Get active strategies with caching (reduces DB queries)."""
    from app.journal.models import Strategy

    cached = get_cached("active_strategies")
    if cached is not None:
        return cached
    strategies = (
        session.query(Strategy)
        .filter(Strategy.is_active)
        .order_by(Strategy.category, Strategy.name)
        .all()
    )
    set_cached("active_strategies", strategies)
    return strategies


# ==================== TEMPLATE HELPERS ====================


def ticker_display(ticker: str) -> str:
    """
    Strip exchange prefix from ticker for display.

    Examples:
        \"HKEX:0981\" -> \"0981\"
        \"AMEX:SOXL\" -> \"SOXL\"
        \"SOXL\" -> \"SOXL\"
    """
    if ticker and ":" in ticker:
        return ticker.split(":", 1)[1]
    return ticker or ""


def ticker_exchange(ticker: str) -> str:
    """
    Extract exchange prefix from ticker.

    Examples:
        \"HKEX:0981\" -> \"HKEX\"
        \"AMEX:SOXL\" -> \"AMEX\"
        \"SOXL\" -> \"\"
    """
    if ticker and ":" in ticker:
        return ticker.split(":", 1)[0]
    return ""


async def build_template_context(request: Request, user: Any | None = None) -> dict:
    """
    Build common template context.

    If `user` is not provided, attempts to resolve the current user (non-blocking).
    """
    from app.auth.service import get_current_user_optional

    context: dict[str, Any] = {
        "data_provider": settings.data_provider,
        "llm_available": get_llm_api_key() is not None,
    }

    if user is not None:
        context["current_user"] = user
        return context

    try:
        resolved = await get_current_user_optional(request, None)
        context["current_user"] = resolved
    except Exception:
        context["current_user"] = None

    return context


# Backwards-compatible wrappers (keep server/templates stable)
async def get_template_context(request: Request) -> dict:
    return await build_template_context(request)


async def get_template_context_with_user(request: Request, user: Any) -> dict:
    return await build_template_context(request, user=user)
