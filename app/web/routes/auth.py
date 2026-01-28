"""
Authentication routes for user management.

Supports both Supabase Auth (production) and local JWT auth (development).

Provides endpoints for:
- Registration
- Login/Logout
- Email verification
- Password reset
"""

import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.web.schemas import success_response
from app.web.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# Get templates from parent
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _is_supabase_enabled() -> bool:
    """Check if Supabase Auth is configured."""
    try:
        from app.db.supabase_client import is_supabase_configured
        return is_supabase_configured()
    except ImportError:
        return False


def _is_email_configured() -> bool:
    """Check if email is configured (local auth) or Supabase handles it."""
    if _is_supabase_enabled():
        return True  # Supabase handles email
    from app.auth.email import is_email_configured
    return is_email_configured()


# ==================== REQUEST MODELS ====================


class RegisterRequest(BaseModel):
    """User registration request."""

    email: EmailStr
    password: str = Field(..., min_length=8)
    name: Optional[str] = None


class LoginRequest(BaseModel):
    """User login request."""

    email: EmailStr
    password: str


class PasswordResetRequest(BaseModel):
    """Password reset request."""

    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""

    token: str
    password: str = Field(..., min_length=8)


# ==================== PAGE ROUTES ====================


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, next: Optional[str] = None, error: Optional[str] = None):
    """Login page."""
    return templates.TemplateResponse(
        request,
        "auth/login.html",
        {
            "next": next or "/",
            "error": error,
            "supabase_enabled": _is_supabase_enabled(),
        },
    )


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request, error: Optional[str] = None):
    """Registration page."""
    return templates.TemplateResponse(
        request,
        "auth/register.html",
        {
            "error": error,
            "email_configured": _is_email_configured(),
            "supabase_enabled": _is_supabase_enabled(),
        },
    )


@router.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request, token: str = "", type: str = ""):
    """Email verification page."""
    success = False
    message = ""
    
    if _is_supabase_enabled():
        # Supabase handles verification via redirect
        # This page is shown after Supabase redirects back
        if type == "signup":
            success = True
            message = "Email verified successfully! You can now log in."
        elif type == "recovery":
            # Redirect to reset password page
            return RedirectResponse(
                url=f"/auth/reset-password?token={token}",
                status_code=303
            )
        else:
            success = True
            message = "Verification complete."
    else:
        # Local auth verification
        from app.auth.service import verify_email
        success = verify_email(token)
        message = "Email verified!" if success else "Invalid or expired verification token."
    
    return templates.TemplateResponse(
        request,
        "auth/verify.html",
        {
            "success": success,
            "message": message,
        },
    )


@router.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request, sent: bool = False):
    """Forgot password page."""
    return templates.TemplateResponse(
        request,
        "auth/forgot_password.html",
        {
            "sent": sent,
            "email_configured": _is_email_configured(),
            "supabase_enabled": _is_supabase_enabled(),
        },
    )


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(
    request: Request, 
    token: str = "", 
    error: Optional[str] = None,
    access_token: Optional[str] = None,
):
    """Password reset page."""
    # Supabase sends access_token for password recovery
    effective_token = access_token or token
    
    return templates.TemplateResponse(
        request,
        "auth/reset_password.html",
        {
            "token": effective_token,
            "error": error,
            "supabase_enabled": _is_supabase_enabled(),
        },
    )


# ==================== API ROUTES ====================


@router.post("/register")
async def register(request: Request):
    """
    Register a new user account.
    
    For Supabase: Sends verification email automatically.
    For local auth: Creates user (auto-verified).
    """
    data = await request.json()

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip() or None

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if _is_supabase_enabled():
        # Use Supabase Auth
        from app.auth.supabase_auth import sign_up
        
        result = await sign_up(email, password, name)
        
        if result.get("session"):
            # User was auto-verified, set cookie
            response = JSONResponse(
                success_response(
                    data={
                        "user_id": result["user"].id,
                        "email": result["user"].email,
                    },
                    message="Account created! You are now logged in.",
                )
            )
            response.set_cookie(
                key="access_token",
                value=result["session"]["access_token"],
                httponly=True,
                secure=request.url.scheme == "https",
                max_age=60 * 60 * 24,
                samesite="lax",
                path="/",
            )
            return response
        else:
            # Email verification required
            return JSONResponse(
                success_response(
                    data={"email": email},
                    message=result.get("message", "Please check your email to verify your account."),
                )
            )
    else:
        # Use local auth
        from app.auth.service import create_user
        
        user = create_user(email, password, name)

        return JSONResponse(
            success_response(
                data={"user_id": user.id, "email": user.email},
                message="Account created! You can now log in.",
            )
        )


@router.post("/login")
async def login(request: Request):
    """
    Login with email and password.

    Returns JWT token and sets session cookie.
    """
    data = await request.json()

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    if _is_supabase_enabled():
        # Use Supabase Auth
        from app.auth.supabase_auth import sign_in
        
        result = await sign_in(email, password)
        
        response = JSONResponse(
            success_response(
                data={
                    "access_token": result["session"]["access_token"],
                    "token_type": "bearer",
                    "user": {
                        "id": result["user"].id,
                        "email": result["user"].email,
                        "name": result["user"].name,
                    },
                },
                message="Login successful",
            )
        )

        # Set HTTP-only cookie for session
        response.set_cookie(
            key="access_token",
            value=result["session"]["access_token"],
            httponly=True,
            secure=request.url.scheme == "https",
            max_age=60 * 60 * 24,  # 24 hours
            samesite="lax",
            path="/",
        )

        # Also store refresh token for session refresh
        if result["session"].get("refresh_token"):
            response.set_cookie(
                key="refresh_token",
                value=result["session"]["refresh_token"],
                httponly=True,
                secure=request.url.scheme == "https",
                max_age=60 * 60 * 24 * 7,  # 7 days
                samesite="lax",
                path="/",
            )

        return response
    else:
        # Use local auth
        from app.auth.service import authenticate_user, create_access_token
        
        user = authenticate_user(email, password)

        if not user:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = create_access_token(user.id, user.email)

        response = JSONResponse(
            success_response(
                data={
                    "access_token": token,
                    "token_type": "bearer",
                    "user": {
                        "id": user.id,
                        "email": user.email,
                        "name": user.name,
                    },
                },
                message="Login successful",
            )
        )

        response.set_cookie(
            key="access_token",
            value=token,
            httponly=True,
            secure=request.url.scheme == "https",
            max_age=60 * 60 * 24,
            samesite="lax",
            path="/",
        )

        return response


@router.post("/logout")
async def logout(request: Request):
    """
    Logout user by clearing session cookie.
    """
    if _is_supabase_enabled():
        try:
            from app.auth.supabase_auth import sign_out
            token = request.cookies.get("access_token")
            if token:
                await sign_out(token)
        except Exception as e:
            logger.warning(f"Supabase sign out failed: {e}")
    
    response = JSONResponse(success_response(message="Logged out"))
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return response


@router.get("/logout")
async def logout_redirect(request: Request):
    """
    Logout via GET (for simple links) - redirects to login.
    """
    if _is_supabase_enabled():
        try:
            from app.auth.supabase_auth import sign_out
            token = request.cookies.get("access_token")
            if token:
                await sign_out(token)
        except Exception:
            pass
    
    response = RedirectResponse(url="/auth/login", status_code=303)
    response.delete_cookie("access_token", path="/")
    response.delete_cookie("refresh_token", path="/")
    return response


@router.post("/forgot-password")
async def forgot_password(request: Request):
    """
    Request password reset email.
    """
    data = await request.json()
    email = data.get("email", "").strip().lower()

    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    if _is_supabase_enabled():
        from app.auth.supabase_auth import reset_password_request
        await reset_password_request(email)
    else:
        from app.auth.service import create_password_reset_token
        from app.auth.email import send_password_reset_email, is_email_configured
        
        token = create_password_reset_token(email)

        if token and is_email_configured():
            from app.journal.models import User, get_session

            session = get_session()
            try:
                user = session.query(User).filter(User.email == email).first()
                send_password_reset_email(email, token, user.name if user else None)
            finally:
                session.close()

    # Always return success to prevent email enumeration
    return JSONResponse(
        success_response(
            message="If an account exists with that email, you will receive a password reset link."
        )
    )


@router.post("/reset-password")
async def do_reset_password(request: Request):
    """
    Reset password with token.
    """
    data = await request.json()
    token = data.get("token", "")
    password = data.get("password", "")

    if not token or not password:
        raise HTTPException(status_code=400, detail="Token and password required")

    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    if _is_supabase_enabled():
        from app.auth.supabase_auth import update_password
        
        success = await update_password(token, password)
        if success:
            return JSONResponse(
                success_response(message="Password reset successful. You can now log in.")
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to reset password. Please try again.")
    else:
        from app.auth.service import reset_password
        
        if reset_password(token, password):
            return JSONResponse(
                success_response(message="Password reset successful. You can now log in.")
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid or expired reset token")


@router.get("/me")
async def get_current_user_info(user=Depends(get_current_user)):
    """
    Get current user information.
    """
    return JSONResponse(
        success_response(
            data={
                "id": str(user.id),
                "email": user.email,
                "name": getattr(user, "name", None),
                "is_verified": getattr(user, "is_verified", True),
            }
        )
    )


@router.post("/resend-verification")
async def resend_verification(request: Request):
    """
    Resend verification email.
    """
    data = await request.json()
    email = data.get("email", "").strip().lower()

    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    if _is_supabase_enabled():
        # Supabase has a resend method
        try:
            from app.db.supabase_client import get_supabase_client
            supabase = get_supabase_client()
            supabase.auth.resend({"type": "signup", "email": email})
        except Exception as e:
            logger.warning(f"Failed to resend verification: {e}")
    else:
        from app.journal.models import User, get_session
        from datetime import datetime, timezone
        from app.auth.service import generate_verification_token, VERIFICATION_TOKEN_HOURS
        from app.auth.email import send_verification_email, is_email_configured

        session = get_session()
        try:
            user = session.query(User).filter(User.email == email).first()

            if user and not user.is_verified:
                # Generate new token
                user.verification_token = generate_verification_token()
                user.verification_token_expires = datetime.now(timezone.utc) + timedelta(
                    hours=VERIFICATION_TOKEN_HOURS
                )
                session.commit()

                if is_email_configured():
                    send_verification_email(email, user.verification_token, user.name)
        finally:
            session.close()

    # Always return success to prevent enumeration
    return JSONResponse(
        success_response(
            message="If your account exists and is not verified, you will receive a new verification email."
        )
    )


@router.post("/refresh")
async def refresh_token(request: Request):
    """
    Refresh access token using refresh token.
    Only applicable for Supabase auth.
    """
    if not _is_supabase_enabled():
        raise HTTPException(status_code=400, detail="Token refresh not supported")
    
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh token")
    
    from app.auth.supabase_auth import refresh_session
    
    result = await refresh_session(refresh_token)
    
    if not result:
        raise HTTPException(status_code=401, detail="Failed to refresh session")
    
    response = JSONResponse(success_response(message="Session refreshed"))
    
    response.set_cookie(
        key="access_token",
        value=result["access_token"],
        httponly=True,
        secure=request.url.scheme == "https",
        max_age=60 * 60 * 24,
        samesite="lax",
        path="/",
    )
    
    if result.get("refresh_token"):
        response.set_cookie(
            key="refresh_token",
            value=result["refresh_token"],
            httponly=True,
            secure=request.url.scheme == "https",
            max_age=60 * 60 * 24 * 7,
            samesite="lax",
            path="/",
        )
    
    return response
