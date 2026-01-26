"""
Authentication module for Brooks Trading Coach.

Provides:
- Password hashing and verification
- JWT token creation and validation
- Email verification
- Session management
"""

from .service import (
    hash_password,
    verify_password,
    create_access_token,
    decode_access_token,
    generate_verification_token,
    create_user,
    authenticate_user,
    verify_email,
    get_current_user,
    get_current_user_optional,
)

from .email import send_verification_email, send_password_reset_email

__all__ = [
    'hash_password',
    'verify_password',
    'create_access_token',
    'decode_access_token',
    'generate_verification_token',
    'create_user',
    'authenticate_user',
    'verify_email',
    'get_current_user',
    'get_current_user_optional',
    'send_verification_email',
    'send_password_reset_email',
]
