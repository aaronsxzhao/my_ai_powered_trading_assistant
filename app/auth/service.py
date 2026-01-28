"""
Authentication service for user management.

Handles:
- Password hashing with bcrypt
- JWT token creation/validation
- User creation and authentication
- Email verification
"""

from __future__ import annotations

import logging
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from app.journal.models import User

# JWT Configuration
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Token expiration
VERIFICATION_TOKEN_HOURS = 24
RESET_TOKEN_HOURS = 1

# Security scheme for protected routes
security = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    import bcrypt

    salt = bcrypt.gensalt()
    return bcrypt.hashpw(password.encode("utf-8"), salt).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    import bcrypt

    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False


def create_access_token(user_id: int, email: str) -> str:
    """Create a JWT access token."""
    import jwt

    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and validate a JWT token."""
    import jwt

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.debug(f"Invalid token: {e}")
        return None


def generate_verification_token() -> str:
    """Generate a random verification token."""
    return secrets.token_urlsafe(32)


def create_user(email: str, password: str, name: Optional[str] = None) -> User:
    """
    Create a new user account.

    Returns the created user (auto-verified, ready to login).
    Raises HTTPException if email already exists.
    """
    from app.journal.models import User, get_session

    session = get_session()
    try:
        # Check if email already exists
        existing = session.query(User).filter(User.email == email.lower()).first()
        if existing:
            raise HTTPException(status_code=400, detail="Email already registered")

        # Create user (auto-verified, no email verification needed)
        user = User(
            email=email.lower().strip(),
            password_hash=hash_password(password),
            name=name,
            verification_token=None,
            verification_token_expires=None,
            is_verified=True,  # Auto-verify immediately
            is_active=True,
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        # Eagerly load attributes before detaching
        _ = user.id, user.email, user.name, user.is_verified, user.is_active
        session.expunge(user)

        logger.info(f"Created user: {email}")
        return user
    except HTTPException:
        raise
    except Exception as e:
        session.rollback()
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")
    finally:
        session.close()


def authenticate_user(email: str, password: str) -> Optional[User]:
    """
    Authenticate a user with email and password.

    Returns the user if authentication successful, None otherwise.
    """
    from app.journal.models import User, get_session

    session = get_session()
    try:
        user = session.query(User).filter(User.email == email.lower()).first()

        if not user:
            logger.debug(f"User not found: {email}")
            return None

        if not verify_password(password, user.password_hash):
            logger.debug(f"Invalid password for: {email}")
            return None

        if not user.is_active:
            logger.debug(f"Inactive user: {email}")
            return None

        # Update last login
        user.last_login = datetime.now(timezone.utc)
        session.commit()

        # Eagerly load all attributes we need before detaching
        _ = user.id, user.email, user.name, user.is_verified, user.is_active

        # Detach from session so it can be used after session closes
        session.expunge(user)

        return user
    finally:
        session.close()


def verify_email(token: str) -> bool:
    """
    Verify a user's email with the verification token.

    Returns True if verification successful.
    """
    from app.journal.models import User, get_session

    session = get_session()
    try:
        user = session.query(User).filter(User.verification_token == token).first()

        if not user:
            logger.debug("Verification token not found")
            return False

        if user.verification_token_expires and user.verification_token_expires < datetime.now(
            timezone.utc
        ):
            logger.debug("Verification token expired")
            return False

        user.is_verified = True
        user.verification_token = None
        user.verification_token_expires = None
        session.commit()

        logger.info(f"Email verified: {user.email}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error verifying email: {e}")
        return False
    finally:
        session.close()


def get_user_by_id(user_id: int) -> Optional[User]:
    """Get a user by ID."""
    from app.journal.models import User, get_session

    session = get_session()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if user:
            # Eagerly load attributes before detaching
            _ = user.id, user.email, user.name, user.is_verified, user.is_active
            session.expunge(user)
        return user
    finally:
        session.close()


async def get_current_user(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Get the current authenticated user from JWT token.

    Checks both:
    1. Authorization header (Bearer token)
    2. Session cookie (for web UI)

    Raises HTTPException 401 if not authenticated.
    """
    from app.journal.models import User, get_session

    token = None

    # Check Authorization header first
    if credentials:
        token = credentials.credentials

    # Fall back to session cookie
    if not token:
        token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate token payload
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        user_id = int(sub)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=401,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    session = get_session()
    try:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        if not user.is_active:
            raise HTTPException(status_code=401, detail="User account is disabled")

        # Eagerly load attributes before detaching
        _ = user.id, user.email, user.name, user.is_verified, user.is_active
        session.expunge(user)

        return user
    finally:
        session.close()


async def get_current_user_optional(
    request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """
    Get the current user if authenticated, None otherwise.

    Does not raise exceptions - returns None for unauthenticated requests.
    """
    try:
        return await get_current_user(request, credentials)
    except HTTPException:
        return None


def create_password_reset_token(email: str) -> Optional[str]:
    """
    Create a password reset token for a user.

    Returns the token if user exists, None otherwise.
    """
    from app.journal.models import User, get_session

    session = get_session()
    try:
        user = session.query(User).filter(User.email == email.lower()).first()
        if not user:
            return None

        token = generate_verification_token()
        user.reset_token = token
        user.reset_token_expires = datetime.now(timezone.utc) + timedelta(hours=RESET_TOKEN_HOURS)
        session.commit()

        return token
    finally:
        session.close()


def reset_password(token: str, new_password: str) -> bool:
    """
    Reset a user's password using the reset token.

    Returns True if successful.
    """
    from app.journal.models import User, get_session

    session = get_session()
    try:
        user = session.query(User).filter(User.reset_token == token).first()

        if not user:
            return False

        if user.reset_token_expires and user.reset_token_expires < datetime.now(timezone.utc):
            return False

        user.password_hash = hash_password(new_password)
        user.reset_token = None
        user.reset_token_expires = None
        session.commit()

        logger.info(f"Password reset for: {user.email}")
        return True
    except Exception as e:
        session.rollback()
        logger.error(f"Error resetting password: {e}")
        return False
    finally:
        session.close()
