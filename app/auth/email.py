"""
Email service for authentication.

Handles sending verification and password reset emails.
Requires SMTP configuration in environment variables.
"""

import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional

logger = logging.getLogger(__name__)

# SMTP Configuration from environment
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER)
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"

# App configuration
APP_NAME = "Brooks Trading Coach"
APP_URL = os.getenv("APP_URL", "http://localhost:8000")


def is_email_configured() -> bool:
    """Check if email sending is properly configured."""
    return bool(SMTP_USER and SMTP_PASSWORD)


def send_email(to: str, subject: str, html_body: str, text_body: Optional[str] = None) -> bool:
    """
    Send an email using SMTP.
    
    Args:
        to: Recipient email address
        subject: Email subject
        html_body: HTML content
        text_body: Plain text fallback (optional)
    
    Returns:
        True if sent successfully
    """
    if not is_email_configured():
        logger.warning("Email not configured - skipping send")
        logger.info(f"Would send email to {to}: {subject}")
        return False
    
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{APP_NAME} <{SMTP_FROM}>"
        msg["To"] = to
        
        # Add plain text version
        if text_body:
            msg.attach(MIMEText(text_body, "plain"))
        
        # Add HTML version
        msg.attach(MIMEText(html_body, "html"))
        
        # Connect and send
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_USE_TLS:
                server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM, to, msg.as_string())
        
        logger.info(f"Email sent to {to}: {subject}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send email to {to}: {e}")
        return False


def send_verification_email(email: str, token: str, name: Optional[str] = None) -> bool:
    """
    Send email verification link.
    
    Args:
        email: User's email address
        token: Verification token
        name: User's name (optional)
    
    Returns:
        True if sent successfully
    """
    verify_url = f"{APP_URL}/auth/verify?token={token}"
    greeting = f"Hi {name}," if name else "Hi,"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .button {{ 
                display: inline-block; 
                padding: 12px 24px; 
                background-color: #3B82F6; 
                color: white !important; 
                text-decoration: none; 
                border-radius: 6px; 
                margin: 20px 0;
            }}
            .footer {{ color: #666; font-size: 12px; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Welcome to {APP_NAME}!</h2>
            <p>{greeting}</p>
            <p>Thanks for signing up! Please verify your email address to activate your account.</p>
            <a href="{verify_url}" class="button">Verify Email Address</a>
            <p>Or copy this link into your browser:</p>
            <p style="word-break: break-all; color: #666;">{verify_url}</p>
            <p>This link will expire in 24 hours.</p>
            <div class="footer">
                <p>If you didn't create an account, you can safely ignore this email.</p>
                <p>&copy; {APP_NAME}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    Welcome to {APP_NAME}!
    
    {greeting}
    
    Thanks for signing up! Please verify your email address by clicking the link below:
    
    {verify_url}
    
    This link will expire in 24 hours.
    
    If you didn't create an account, you can safely ignore this email.
    """
    
    return send_email(email, f"Verify your {APP_NAME} account", html_body, text_body)


def send_password_reset_email(email: str, token: str, name: Optional[str] = None) -> bool:
    """
    Send password reset link.
    
    Args:
        email: User's email address
        token: Reset token
        name: User's name (optional)
    
    Returns:
        True if sent successfully
    """
    reset_url = f"{APP_URL}/auth/reset-password?token={token}"
    greeting = f"Hi {name}," if name else "Hi,"
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .button {{ 
                display: inline-block; 
                padding: 12px 24px; 
                background-color: #3B82F6; 
                color: white !important; 
                text-decoration: none; 
                border-radius: 6px; 
                margin: 20px 0;
            }}
            .warning {{ color: #DC2626; }}
            .footer {{ color: #666; font-size: 12px; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>Password Reset Request</h2>
            <p>{greeting}</p>
            <p>We received a request to reset your password. Click the button below to create a new password:</p>
            <a href="{reset_url}" class="button">Reset Password</a>
            <p>Or copy this link into your browser:</p>
            <p style="word-break: break-all; color: #666;">{reset_url}</p>
            <p class="warning"><strong>This link will expire in 1 hour.</strong></p>
            <div class="footer">
                <p>If you didn't request a password reset, you can safely ignore this email. Your password will remain unchanged.</p>
                <p>&copy; {APP_NAME}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    text_body = f"""
    Password Reset Request
    
    {greeting}
    
    We received a request to reset your password. Click the link below to create a new password:
    
    {reset_url}
    
    This link will expire in 1 hour.
    
    If you didn't request a password reset, you can safely ignore this email.
    """
    
    return send_email(email, f"Reset your {APP_NAME} password", html_body, text_body)
