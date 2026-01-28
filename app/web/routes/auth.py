"""
Authentication routes for user management.

Provides endpoints for:
- Registration
- Login/Logout
- Email verification
- Password reset
"""

import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, EmailStr, Field

from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.auth.service import (
    create_user,
    authenticate_user,
    verify_email,
    create_access_token,
    create_password_reset_token,
    reset_password,
    get_current_user,
)
from app.auth.email import send_verification_email, send_password_reset_email, is_email_configured
from app.web.schemas import success_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["authentication"])

# Get templates from parent
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


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
            "email_configured": is_email_configured(),
        },
    )


@router.get("/verify", response_class=HTMLResponse)
async def verify_page(request: Request, token: str):
    """Email verification page."""
    success = verify_email(token)
    return templates.TemplateResponse(
        request,
        "auth/verify.html",
        {
            "success": success,
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
            "email_configured": is_email_configured(),
        },
    )


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request, token: str, error: Optional[str] = None):
    """Password reset page."""
    return templates.TemplateResponse(
        request,
        "auth/reset_password.html",
        {
            "token": token,
            "error": error,
        },
    )


# ==================== API ROUTES ====================


@router.post("/register")
async def register(request: Request):
    """
    Register a new user account.

    Creates user (auto-verified, ready to login immediately).
    """
    data = await request.json()

    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip() or None

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password required")

    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    # Create user (auto-verified)
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

    user = authenticate_user(email, password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create token
    token = create_access_token(user.id, user.email)

    # Set cookie for web UI
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

    # Set HTTP-only cookie for session
    # secure=True only when running on HTTPS
    response.set_cookie(
        key="access_token",
        value=token,
        httponly=True,
        secure=request.url.scheme == "https",
        max_age=60 * 60 * 24,  # 24 hours
        samesite="lax",
        path="/",
    )

    return response


@router.post("/logout")
async def logout():
    """
    Logout user by clearing session cookie.
    """
    response = JSONResponse(success_response(message="Logged out"))
    response.delete_cookie("access_token", path="/")
    return response


@router.get("/logout")
async def logout_redirect():
    """
    Logout via GET (for simple links) - redirects to login.
    """
    response = RedirectResponse(url="/auth/login", status_code=303)
    response.delete_cookie("access_token", path="/")
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

    # Always return success to prevent email enumeration
    token = create_password_reset_token(email)

    if token and is_email_configured():
        from app.journal.models import User, get_session

        session = get_session()
        try:
            user = session.query(User).filter(User.email == email).first()
            send_password_reset_email(email, token, user.name if user else None)
        finally:
            session.close()

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
                "id": user.id,
                "email": user.email,
                "name": user.name,
                "is_verified": user.is_verified,
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

    from app.journal.models import User, get_session
    from datetime import datetime, timezone
    from app.auth.service import generate_verification_token, VERIFICATION_TOKEN_HOURS

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
