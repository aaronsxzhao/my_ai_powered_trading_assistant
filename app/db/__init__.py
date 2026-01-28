"""Database module for Supabase integration."""

from app.db.supabase_client import get_supabase_client, get_service_client

__all__ = ["get_supabase_client", "get_service_client"]
