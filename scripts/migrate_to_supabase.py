#!/usr/bin/env python3
"""
Migrate local SQLite data to Supabase.

This script:
1. Reads user and trade data from local SQLite database
2. Creates user in Supabase Auth (if not exists)
3. Migrates all trades to Supabase PostgreSQL

Uses direct HTTP requests to avoid dependency on supabase pip package.

Prerequisites:
1. Run the Supabase migrations first (001_initial_schema.sql, 002_storage_policies.sql)
2. Uncomment SUPABASE_* variables in .env

Usage:
    python scripts/migrate_to_supabase.py

Options:
    --dry-run    Show what would be migrated without making changes
    --email      Specific email to migrate (default: aaronsxzhao@gmail.com)
"""

import sys
import os
import argparse
import json
import urllib.request
import urllib.error
import ssl
from datetime import datetime, date
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()


class SupabaseClient:
    """Simple Supabase client using urllib (no external dependencies)."""
    
    def __init__(self, url: str, service_key: str):
        self.url = url.rstrip('/')
        self.service_key = service_key
        # Create SSL context that doesn't verify (for environments with cert issues)
        self.ssl_context = ssl.create_default_context()
        # Uncomment if SSL issues persist:
        # self.ssl_context.check_hostname = False
        # self.ssl_context.verify_mode = ssl.CERT_NONE
    
    def _request(self, method: str, endpoint: str, data: dict = None, headers: dict = None) -> dict:
        """Make HTTP request to Supabase."""
        url = f"{self.url}{endpoint}"
        
        default_headers = {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation"
        }
        if headers:
            default_headers.update(headers)
        
        body = json.dumps(data).encode('utf-8') if data else None
        
        req = urllib.request.Request(url, data=body, headers=default_headers, method=method)
        
        try:
            with urllib.request.urlopen(req, context=self.ssl_context) as response:
                response_data = response.read().decode('utf-8')
                return json.loads(response_data) if response_data else {}
        except urllib.error.HTTPError as e:
            error_body = e.read().decode('utf-8')
            raise Exception(f"HTTP {e.code}: {error_body}")
    
    def list_users(self) -> list:
        """List all users in Supabase Auth."""
        result = self._request("GET", "/auth/v1/admin/users")
        return result.get("users", [])
    
    def create_user(self, email: str, password: str, name: str = None) -> dict:
        """Create a new user in Supabase Auth."""
        data = {
            "email": email,
            "password": password,
            "email_confirm": True,
        }
        if name:
            data["user_metadata"] = {"name": name}
        
        return self._request("POST", "/auth/v1/admin/users", data)
    
    def insert_trade(self, trade_data: dict) -> dict:
        """Insert a trade into the trades table."""
        return self._request("POST", "/rest/v1/trades", trade_data)
    
    def get_trades(self, user_id: str, ticker: str, trade_date: str, entry_price: float) -> list:
        """Check if a trade already exists."""
        endpoint = f"/rest/v1/trades?user_id=eq.{user_id}&ticker=eq.{ticker}&trade_date=eq.{trade_date}&entry_price=eq.{entry_price}&select=id"
        return self._request("GET", endpoint)


def get_local_data(email: str):
    """Fetch user and trades from local SQLite database."""
    from app.journal.models import init_db, Trade, User, get_session
    
    init_db()
    session = get_session()
    
    try:
        # Get user
        user = session.query(User).filter(User.email == email.lower()).first()
        if not user:
            print(f"❌ User not found: {email}")
            return None, []
        
        print(f"✅ Found user: {user.email} (ID: {user.id})")
        
        # Get user's trades
        trades = session.query(Trade).filter(Trade.user_id == str(user.id)).all()
        print(f"✅ Found {len(trades)} trades")
        
        # Convert to dictionaries for transport
        user_data = {
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at,
        }
        
        trade_data = []
        for t in trades:
            trade_dict = {
                "trade_number": t.trade_number,
                "ticker": t.ticker,
                "trade_date": t.trade_date,
                "timeframe": t.timeframe,
                "direction": t.direction.value if t.direction else "long",
                "entry_price": t.entry_price,
                "entry_time": t.entry_time,
                "entry_reason": t.entry_reason,
                "exit_price": t.exit_price,
                "exit_time": t.exit_time,
                "exit_reason": t.exit_reason,
                "size": t.size,
                "stop_loss": t.stop_loss,
                "take_profit": t.take_profit,
                "currency": t.currency or "USD",
                "currency_rate": t.currency_rate or 1.0,
                "market_timezone": t.market_timezone or "America/New_York",
                "input_timezone": t.input_timezone,
                "r_multiple": t.r_multiple,
                "pnl_dollars": t.pnl_dollars,
                "pnl_percent": t.pnl_percent,
                "outcome": t.outcome.value if t.outcome else None,
                "strategy_id": t.strategy_id,
                "setup_type": t.setup_type,
                "ai_setup_classification": t.ai_setup_classification,
                "notes": t.notes,
                "mistakes": t.mistakes,
                "lessons": t.lessons,
                "coach_feedback": t.coach_feedback,
                "cached_review": t.cached_review,
                "review_generated_at": t.review_generated_at,
                "created_at": t.created_at,
                "updated_at": t.updated_at,
                # Additional fields
                "hold_time_minutes": t.hold_time_minutes,
                "mae": t.mae,
                "mfe": t.mfe,
                "account_type": getattr(t, "account_type", "paper"),
            }
            trade_data.append(trade_dict)
        
        return user_data, trade_data
    finally:
        session.close()


def create_supabase_user(client: SupabaseClient, email: str, password: str, name: Optional[str] = None) -> str:
    """Create user in Supabase Auth or get existing user ID."""
    
    # Try to get existing user first
    try:
        users = client.list_users()
        for user in users:
            if user.get("email", "").lower() == email.lower():
                print(f"✅ User already exists in Supabase: {user['id']}")
                return user['id']
        print(f"  No existing user found, will create new one...")
    except Exception as e:
        print(f"⚠️ Could not list users: {e}")
    
    # Create new user
    try:
        response = client.create_user(email, password, name)
        user_id = response.get("id")
        print(f"✅ Created Supabase user: {user_id}")
        return user_id
    except Exception as e:
        error_str = str(e).lower()
        if "already been registered" in error_str or "already exists" in error_str or "duplicate" in error_str:
            # User exists, try to find their ID
            print(f"⚠️ User may exist, fetching ID...")
            try:
                users = client.list_users()
                for user in users:
                    if user.get("email", "").lower() == email.lower():
                        return user['id']
            except:
                pass
        
        # If it's a database error, it might be because the profiles table trigger failed
        # But the user might have been created - try to list again
        if "database error" in error_str:
            print(f"⚠️ Database error during user creation, checking if user was created anyway...")
            try:
                users = client.list_users()
                for user in users:
                    if user.get("email", "").lower() == email.lower():
                        print(f"✅ User was created despite error: {user['id']}")
                        return user['id']
            except Exception as e2:
                print(f"  Could not verify: {e2}")
        
        raise


def migrate_trades_to_supabase(client: SupabaseClient, user_id: str, trades: list, dry_run: bool = False):
    """Migrate trades to Supabase PostgreSQL."""
    
    migrated = 0
    skipped = 0
    errors = 0
    
    for trade in trades:
        try:
            # Convert dates to ISO strings
            trade_data = {
                "user_id": user_id,
                "trade_number": trade["trade_number"],
                "ticker": trade["ticker"],
                "trade_date": trade["trade_date"].isoformat() if isinstance(trade["trade_date"], date) else trade["trade_date"],
                "timeframe": trade["timeframe"],
                "direction": trade["direction"],
                "entry_price": trade["entry_price"],
                "exit_price": trade["exit_price"],
                "size": trade["size"],
                "stop_loss": trade["stop_loss"],
                "take_profit": trade["take_profit"],
                "currency": trade["currency"],
                "currency_rate": trade["currency_rate"],
                "market_timezone": trade["market_timezone"],
                "input_timezone": trade["input_timezone"],
                "r_multiple": trade["r_multiple"],
                "pnl_dollars": trade["pnl_dollars"],
                "pnl_percent": trade["pnl_percent"],
                "outcome": trade["outcome"],
                "setup_type": trade["setup_type"],
                "ai_setup_classification": trade["ai_setup_classification"],
                "notes": trade["notes"],
                "mistakes": trade["mistakes"],
                "lessons": trade["lessons"],
                "coach_feedback": trade["coach_feedback"],
                "cached_review": trade["cached_review"],
                "account_type": trade.get("account_type", "paper"),
                "hold_time_minutes": trade.get("hold_time_minutes"),
                "mae": trade.get("mae"),
                "mfe": trade.get("mfe"),
            }
            
            # Handle datetime fields
            if trade.get("entry_time"):
                trade_data["entry_time"] = trade["entry_time"].isoformat() if isinstance(trade["entry_time"], datetime) else trade["entry_time"]
            if trade.get("exit_time"):
                trade_data["exit_time"] = trade["exit_time"].isoformat() if isinstance(trade["exit_time"], datetime) else trade["exit_time"]
            if trade.get("entry_reason"):
                trade_data["entry_reason"] = trade["entry_reason"]
            if trade.get("exit_reason"):
                trade_data["exit_reason"] = trade["exit_reason"]
            if trade.get("review_generated_at"):
                trade_data["review_generated_at"] = trade["review_generated_at"].isoformat() if isinstance(trade["review_generated_at"], datetime) else trade["review_generated_at"]
            
            # Handle strategy_id (needs to match Supabase strategies table)
            # Supabase only has strategies 1-25, skip if higher
            if trade.get("strategy_id") and trade["strategy_id"] <= 25:
                trade_data["strategy_id"] = trade["strategy_id"]
            
            # Remove None values
            trade_data = {k: v for k, v in trade_data.items() if v is not None}
            
            if dry_run:
                print(f"  [DRY-RUN] Would insert trade #{trade['trade_number']}: {trade['ticker']} {trade['direction']} on {trade['trade_date']}")
                migrated += 1
            else:
                # Check if trade already exists (by ticker, date, entry_price, direction)
                try:
                    existing = client.get_trades(user_id, trade["ticker"], trade_data["trade_date"], trade["entry_price"])
                    if existing:
                        print(f"  ⏭️  Trade #{trade['trade_number']} already exists, skipping")
                        skipped += 1
                        continue
                except:
                    pass  # If check fails, try to insert anyway
                
                client.insert_trade(trade_data)
                print(f"  ✅ Migrated trade #{trade['trade_number']}: {trade['ticker']} {trade['direction']}")
                migrated += 1
                    
        except Exception as e:
            print(f"  ❌ Error migrating trade #{trade.get('trade_number', '?')}: {e}")
            errors += 1
    
    return migrated, skipped, errors


def main():
    parser = argparse.ArgumentParser(description="Migrate data to Supabase")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be migrated without making changes")
    parser.add_argument("--email", default="aaronsxzhao@gmail.com", help="Email address to migrate")
    parser.add_argument("--password", default="Trading123!", help="Password for Supabase user (if creating)")
    parser.add_argument("--user-id", help="Supabase user UUID (skip user creation, use this ID directly)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Migrating data to Supabase")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 DRY-RUN MODE - No changes will be made\n")
    
    # Check Supabase configuration
    url = os.getenv("SUPABASE_URL")
    service_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not url or not service_key:
        print("❌ Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        print("   Uncomment these variables in your .env file first.")
        sys.exit(1)
    
    print(f"📦 Supabase URL: {url}")
    print(f"📧 Migrating user: {args.email}\n")
    
    # Step 1: Get local data
    print("Step 1: Reading local data...")
    user_data, trades = get_local_data(args.email)
    
    if not user_data:
        sys.exit(1)
    
    if not trades:
        print("⚠️  No trades to migrate")
        sys.exit(0)
    
    # Step 2: Create/get Supabase user
    print("\nStep 2: Setting up Supabase user...")
    if args.dry_run:
        print(f"  [DRY-RUN] Would create/get Supabase user for {args.email}")
        supabase_user_id = "dry-run-uuid"
        client = None
    elif args.user_id:
        # Use provided user ID directly
        print(f"  Using provided user ID: {args.user_id}")
        supabase_user_id = args.user_id
        client = SupabaseClient(url, service_key)
    else:
        client = SupabaseClient(url, service_key)
        supabase_user_id = create_supabase_user(
            client,
            email=args.email,
            password=args.password,
            name=user_data.get("name")
        )
    
    # Step 3: Migrate trades
    print(f"\nStep 3: Migrating {len(trades)} trades...")
    migrated, skipped, errors = migrate_trades_to_supabase(
        client=client,
        user_id=supabase_user_id,
        trades=trades,
        dry_run=args.dry_run
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"  Migrated: {migrated}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")
    
    if not args.dry_run and migrated > 0:
        print("\n✅ Migration complete!")
        print(f"\nTo use Supabase, update your .env file:")
        print("  1. SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_KEY are already set")
        print("  2. Comment out DATABASE_URL (SQLite) if you want to switch to Supabase DB")
        print("  3. Restart the application")
        print(f"\nLogin with: {args.email} / {args.password}")
    elif args.dry_run:
        print("\n🔍 This was a dry run. Run without --dry-run to actually migrate.")


if __name__ == "__main__":
    main()
