"""
Shared FastAPI dependencies for authentication/authorization.

These are extracted from `app/web/server.py` to keep route modules focused.
"""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request, Security
from fastapi.security import APIKeyHeader

from app.config import get_app_api_key


# API Key authentication for sensitive endpoints
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """
    Verify API key for protected endpoints.

    If APP_API_KEY is not set in environment, authentication is disabled.
    If set, the X-API-Key header must match.

    Returns True if authenticated or auth is disabled.
    Raises HTTPException 401 if key is required but missing/invalid.
    """
    expected_key = get_app_api_key()

    # If no API key configured, auth is disabled
    if expected_key is None:
        return True

    # API key is required
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Set X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if api_key != expected_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return True


# Dependency for API key-protected routes (used for DELETE endpoints)
require_auth = Depends(verify_api_key)


async def require_user_or_api_key(request: Request):
    """
    Dependency that requires either:
    1. A logged-in user (via session cookie or Bearer token), OR
    2. A valid API key

    Returns the authenticated user if logged in, or True if API key is valid.
    """
    from app.auth.service import get_current_user_optional

    # First, try to get the logged-in user
    try:
        user = await get_current_user_optional(request, None)
        if user:
            return user
    except Exception:
        pass

    # Fall back to API key authentication
    api_key = request.headers.get("X-API-Key")
    expected_key = get_app_api_key()

    if expected_key and api_key == expected_key:
        return True

    # Neither user nor API key - require authentication
    raise HTTPException(
        status_code=401,
        detail="Authentication required. Please log in or provide a valid API key.",
        headers={"WWW-Authenticate": "Bearer"},
    )


# Dependency for write operations (POST/PATCH)
require_write_auth = Depends(require_user_or_api_key)


async def get_user_from_request(request: Request):
    """
    Get the current logged-in user from a request (if any).
    Returns None if using API key authentication.
    """
    from app.auth.service import get_current_user_optional

    try:
        return await get_current_user_optional(request, None)
    except Exception:
        return None


async def verify_trade_ownership(trade_id: int, request: Request, session):
    """
    Verify that the trade belongs to the current user.
    For API key auth, returns the trade without ownership check.
    Returns the trade if authorized, raises 404 if not found or not authorized.
    """
    from app.journal.models import Trade

    user = await get_user_from_request(request)

    if user:
        # User is logged in - verify ownership
        trade = session.query(Trade).filter(Trade.id == trade_id, Trade.user_id == user.id).first()
    else:
        # API key auth - allow access to any trade
        trade = session.query(Trade).filter(Trade.id == trade_id).first()

    if not trade:
        raise HTTPException(status_code=404, detail="Trade not found")

    return trade, user


async def require_login(request: Request):
    """
    Dependency that requires user to be logged in.
    Redirects to login page if not authenticated.
    Returns the authenticated user.
    """
    from app.auth.service import get_current_user

    try:
        user = await get_current_user(request, None)
        return user
    except HTTPException:
        # Redirect to login with return URL
        login_url = f"/auth/login?next={request.url.path}"
        raise HTTPException(status_code=307, headers={"Location": login_url})
