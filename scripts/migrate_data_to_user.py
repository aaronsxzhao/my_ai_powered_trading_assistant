#!/usr/bin/env python3
"""
Migrate existing trade data to a user account.

This script:
1. Creates a user account (if not exists) for the specified email
2. Auto-verifies the account (no email verification needed)
3. Assigns all existing trades (with no user_id) to this user

Usage:
    python scripts/migrate_data_to_user.py

The user will be created with a temporary password that should be changed
via the password reset flow.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.journal.models import init_db, Trade, User, get_session
from app.auth.service import hash_password, generate_verification_token

# Configuration
TARGET_EMAIL = "aaronsxzhao@gmail.com"
TEMP_PASSWORD = "ChangeMeNow123!"  # User should change this immediately


def migrate_data():
    """Migrate existing trades to the specified user account."""
    
    # Initialize database (creates tables if needed)
    print("Initializing database...")
    init_db()
    
    session = get_session()
    try:
        # Check if user already exists
        user = session.query(User).filter(User.email == TARGET_EMAIL.lower()).first()
        
        if user:
            print(f"User already exists: {user.email} (ID: {user.id})")
        else:
            # Create new user
            print(f"Creating user: {TARGET_EMAIL}")
            user = User(
                email=TARGET_EMAIL.lower(),
                password_hash=hash_password(TEMP_PASSWORD),
                name="Aaron",
                is_verified=True,  # Auto-verify
                is_active=True,
            )
            session.add(user)
            session.commit()
            session.refresh(user)
            print(f"User created with ID: {user.id}")
            print(f"Temporary password: {TEMP_PASSWORD}")
            print("⚠️  Please change your password after logging in!")
        
        # Count trades without user_id
        orphan_trades = session.query(Trade).filter(Trade.user_id == None).count()
        print(f"\nFound {orphan_trades} trades without a user assignment")
        
        if orphan_trades > 0:
            # Assign all orphan trades to this user
            print(f"Assigning trades to user {user.email}...")
            session.query(Trade).filter(Trade.user_id == None).update(
                {"user_id": user.id},
                synchronize_session=False
            )
            session.commit()
            print(f"✅ Migrated {orphan_trades} trades to {user.email}")
        else:
            print("No trades to migrate.")
        
        # Summary
        total_user_trades = session.query(Trade).filter(Trade.user_id == user.id).count()
        print(f"\nTotal trades for {user.email}: {total_user_trades}")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        session.close()
    
    print("\n✅ Migration complete!")
    print(f"\nYou can now log in at: http://localhost:8000/auth/login")
    print(f"Email: {TARGET_EMAIL}")
    print(f"Password: {TEMP_PASSWORD}")


if __name__ == "__main__":
    migrate_data()
