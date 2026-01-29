"""
Supabase Authentication Service.

Handles user authentication via Supabase Auth, including:
- Sign up with email verification
- Sign in with email/password
- Password reset
- Token validation
- Session management
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Optional
from datetime import datetime

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)


@dataclass
class SupabaseUser:
    """User object from Supabase Auth."""
    id: str  # UUID
    email: str
    name: Optional[str] = None
    is_verified: bool = False
    created_at: Optional[datetime] = None
    last_sign_in_at: Optional[datetime] = None
    
    @property
    def is_active(self) -> bool:
        """Check if user is active (email verified in production)."""
        return True  # Supabase handles verification


def _get_supabase():
    """Get Supabase client (lazy import)."""
    from app.db.supabase_client import get_supabase_client
    return get_supabase_client()


def _get_service_client():
    """Get Supabase service client (lazy import)."""
    from app.db.supabase_client import get_service_client
    return get_service_client()


def is_supabase_auth_enabled() -> bool:
    """Check if Supabase Auth is configured."""
    from app.db.supabase_client import is_supabase_configured
    return is_supabase_configured()


async def sign_up(
    email: str, 
    password: str, 
    name: Optional[str] = None
) -> dict:
    """
    Register a new user with email verification.
    
    Returns:
        dict with 'user' and 'session' keys
    
    Raises:
        HTTPException if registration fails
    """
    def _do_sign_up():
        supabase = _get_supabase()
        
        # Sign up with metadata
        options = {}
        if name:
            options["data"] = {"name": name}
        
        return supabase.auth.sign_up({
            "email": email,
            "password": password,
            "options": options
        })
    
    try:
        result = await asyncio.to_thread(_do_sign_up)
        
        if result.user is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to create account. Please try again."
            )
        
        # Check if email confirmation is required
        if result.session is None:
            # Email confirmation required
            logger.info(f"User registered, verification email sent: {email}")
            return {
                "user": _parse_user(result.user),
                "session": None,
                "message": "Please check your email to verify your account."
            }
        
        # Safely extract expires_at (handle both int and string formats)
        expires_at = result.session.expires_at
        if isinstance(expires_at, str):
            try:
                expires_at = int(expires_at)
            except (ValueError, TypeError):
                expires_at = None
        
        logger.info(f"User registered and logged in: {email}")
        return {
            "user": _parse_user(result.user),
            "session": {
                "access_token": result.session.access_token,
                "refresh_token": result.session.refresh_token,
                "expires_at": expires_at
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Sign up error: {error_msg}")
        
        # Parse common Supabase errors
        if "already registered" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail="This email is already registered."
            )
        if "password" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 6 characters."
            )
        
        raise HTTPException(
            status_code=500,
            detail="Failed to create account. Please try again."
        )


async def sign_in(email: str, password: str) -> dict:
    """
    Sign in with email and password.
    
    Returns:
        dict with 'user' and 'session' keys
    
    Raises:
        HTTPException if login fails
    """
    def _do_sign_in():
        supabase = _get_supabase()
        return supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
    
    def _update_last_login(user_id: str):
        try:
            service = _get_service_client()
            service.table("profiles").update({
                "last_login": datetime.utcnow().isoformat()
            }).eq("id", user_id).execute()
        except Exception as e:
            logger.warning(f"Failed to update last_login: {e}")
    
    try:
        result = await asyncio.to_thread(_do_sign_in)
        
        if result.user is None or result.session is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )
        
        # Update last login in profiles table (non-blocking)
        asyncio.get_event_loop().run_in_executor(
            None, _update_last_login, str(result.user.id)
        )
        
        # Safely extract expires_at (handle both int and string formats)
        expires_at = result.session.expires_at
        if isinstance(expires_at, str):
            try:
                expires_at = int(expires_at)
            except (ValueError, TypeError):
                expires_at = None
        
        logger.info(f"User signed in: {email}")
        return {
            "user": _parse_user(result.user),
            "session": {
                "access_token": result.session.access_token,
                "refresh_token": result.session.refresh_token,
                "expires_at": expires_at
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Sign in error: {error_msg}", exc_info=True)
        
        if "invalid" in error_msg.lower() or "credentials" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )
        if "not confirmed" in error_msg.lower():
            raise HTTPException(
                status_code=401,
                detail="Please verify your email before signing in."
            )
        
        raise HTTPException(
            status_code=500,
            detail="Login failed. Please try again."
        )


async def sign_out(access_token: str) -> bool:
    """
    Sign out the current user.
    
    Returns:
        True if successful
    """
    def _do_sign_out():
        try:
            supabase = _get_supabase()
            supabase.auth.sign_out()
            return True
        except Exception as e:
            logger.warning(f"Sign out error: {e}")
            return True  # Still return True - user is effectively signed out
    
    return await asyncio.to_thread(_do_sign_out)


async def get_user_from_token(access_token: str) -> Optional[SupabaseUser]:
    """
    Validate an access token and get the user.
    
    Returns:
        SupabaseUser if valid, None otherwise
    """
    if not access_token:
        return None
    
    def _do_get_user():
        try:
            supabase = _get_supabase()
            result = supabase.auth.get_user(access_token)
            
            if result.user is None:
                return None
            
            return _parse_user(result.user)
            
        except Exception as e:
            logger.debug(f"Token validation failed: {e}")
            return None
    
    return await asyncio.to_thread(_do_get_user)


async def get_user_by_id(user_id: str) -> Optional[SupabaseUser]:
    """
    Get a user by their ID (using service client).
    
    Returns:
        SupabaseUser if found, None otherwise
    """
    def _do_get_user():
        try:
            service = _get_service_client()
            result = service.auth.admin.get_user_by_id(user_id)
            
            if result.user is None:
                return None
            
            return _parse_user(result.user)
            
        except Exception as e:
            logger.error(f"Failed to get user by ID: {e}")
            return None
    
    return await asyncio.to_thread(_do_get_user)


async def reset_password_request(email: str) -> bool:
    """
    Send a password reset email.
    
    Returns:
        True if email was sent (or would be sent)
    """
    def _do_reset():
        try:
            supabase = _get_supabase()
            
            # Get redirect URL from environment
            app_url = os.getenv("APP_URL", "http://localhost:8000")
            redirect_url = f"{app_url}/auth/reset-password"
            
            supabase.auth.reset_password_email(
                email,
                options={"redirect_to": redirect_url}
            )
            
            logger.info(f"Password reset email requested for: {email}")
            return True
            
        except Exception as e:
            logger.warning(f"Password reset request failed: {e}")
            # Don't reveal if email exists
            return True
    
    return await asyncio.to_thread(_do_reset)


async def update_password(access_token: str, new_password: str) -> bool:
    """
    Update user's password (requires valid session).
    
    Returns:
        True if password was updated
    """
    def _do_update():
        supabase = _get_supabase()
        
        # Set session first
        supabase.auth.set_session(access_token, "")
        
        # Update password
        result = supabase.auth.update_user({
            "password": new_password
        })
        
        if result.user:
            logger.info(f"Password updated for user: {result.user.email}")
            return True
        return False
    
    try:
        return await asyncio.to_thread(_do_update)
    except Exception as e:
        logger.error(f"Password update failed: {e}")
        raise HTTPException(
            status_code=400,
            detail="Failed to update password. Please try again."
        )


async def refresh_session(refresh_token: str) -> Optional[dict]:
    """
    Refresh an access token using a refresh token.
    
    Returns:
        New session dict or None if refresh failed
    """
    def _do_refresh():
        try:
            supabase = _get_supabase()
            result = supabase.auth.refresh_session(refresh_token)
            
            if result.session is None:
                return None
            
            # Safely extract expires_at (handle both int and string formats)
            expires_at = result.session.expires_at
            if isinstance(expires_at, str):
                try:
                    expires_at = int(expires_at)
                except (ValueError, TypeError):
                    expires_at = None
            
            return {
                "access_token": result.session.access_token,
                "refresh_token": result.session.refresh_token,
                "expires_at": expires_at
            }
        except Exception as e:
            logger.debug(f"Session refresh failed: {e}")
            return None
    
    return await asyncio.to_thread(_do_refresh)


async def get_current_user(request: Request) -> SupabaseUser:
    """
    Get the current authenticated user from request.
    
    Checks:
    1. Authorization header (Bearer token)
    2. Session cookie (access_token)
    
    Raises:
        HTTPException 401 if not authenticated
    """
    token = None
    
    # Check Authorization header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    
    # Fall back to session cookie
    if not token:
        token = request.cookies.get("access_token")
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    user = await get_user_from_token(token)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return user


async def get_current_user_optional(request: Request) -> Optional[SupabaseUser]:
    """
    Get the current user if authenticated, None otherwise.
    
    Does not raise exceptions for unauthenticated requests.
    """
    try:
        return await get_current_user(request)
    except HTTPException:
        return None


def _parse_user(supabase_user) -> SupabaseUser:
    """Parse Supabase user object to SupabaseUser dataclass."""
    user_metadata = getattr(supabase_user, "user_metadata", {}) or {}
    
    def _parse_datetime(value) -> Optional[datetime]:
        """Parse a datetime value that could be string or datetime object."""
        if value is None:
            return None
        # Already a datetime object
        if isinstance(value, datetime):
            return value
        # String - parse it
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                pass
        return None
    
    # Parse dates
    created_at = None
    if hasattr(supabase_user, "created_at"):
        created_at = _parse_datetime(supabase_user.created_at)
    
    last_sign_in_at = None
    if hasattr(supabase_user, "last_sign_in_at"):
        last_sign_in_at = _parse_datetime(supabase_user.last_sign_in_at)
    
    return SupabaseUser(
        id=supabase_user.id,
        email=supabase_user.email,
        name=user_metadata.get("name"),
        is_verified=getattr(supabase_user, "email_confirmed_at", None) is not None,
        created_at=created_at,
        last_sign_in_at=last_sign_in_at
    )
