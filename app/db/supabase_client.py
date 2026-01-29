"""
Supabase client initialization and utilities.

Provides both anon (user-facing) and service (server-side) clients.
"""

import os
import logging
from functools import lru_cache
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy imports to avoid import errors if supabase not installed
_supabase_client: Optional["Client"] = None
_service_client: Optional["Client"] = None


def get_supabase_url() -> str:
    """Get Supabase project URL."""
    url = os.getenv("SUPABASE_URL", "").strip()
    if not url:
        raise ValueError("SUPABASE_URL environment variable is required")
    return url


def get_supabase_anon_key() -> str:
    """Get Supabase anonymous/public key."""
    key = os.getenv("SUPABASE_ANON_KEY", "").strip()
    if not key:
        raise ValueError("SUPABASE_ANON_KEY environment variable is required")
    return key


def get_supabase_service_key() -> Optional[str]:
    """Get Supabase service role key (for server-side operations)."""
    return os.getenv("SUPABASE_SERVICE_KEY", "").strip() or None


def is_supabase_configured() -> bool:
    """
    Check if Supabase is properly configured.
    
    Returns True only if:
    - Running on Render (production), OR
    - FORCE_SUPABASE=true is set (for local testing with Supabase)
    
    AND the required environment variables are set.
    """
    try:
        # Check if we should use Supabase
        is_render = os.getenv("RENDER", "").lower() == "true"
        is_production = os.getenv("ENVIRONMENT", "").lower() == "production"
        force_supabase = os.getenv("FORCE_SUPABASE", "").lower() == "true"
        
        # Only use Supabase in production or when explicitly forced
        if not (is_render or is_production or force_supabase):
            return False
        
        # Check if credentials are available
        url = os.getenv("SUPABASE_URL", "").strip()
        key = os.getenv("SUPABASE_ANON_KEY", "").strip()
        return bool(url and key)
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_supabase_client() -> "Client":
    """
    Get the Supabase client with anonymous key.
    
    Use this for user-facing operations where RLS applies.
    The client is cached for reuse.
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        raise ImportError(
            "supabase package not installed. Run: pip install supabase"
        )
    
    global _supabase_client
    if _supabase_client is None:
        url = get_supabase_url()
        key = get_supabase_anon_key()
        logger.info(f"Initializing Supabase client with URL: {url[:30]}...")
        _supabase_client = create_client(url, key)
        logger.info("Supabase client initialized successfully")
    
    return _supabase_client


@lru_cache(maxsize=1)
def get_service_client() -> "Client":
    """
    Get the Supabase client with service role key.
    
    Use this for server-side operations that bypass RLS.
    Should only be used for administrative tasks.
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        raise ImportError(
            "supabase package not installed. Run: pip install supabase"
        )
    
    global _service_client
    if _service_client is None:
        url = get_supabase_url()
        key = get_supabase_service_key()
        if not key:
            raise ValueError(
                "SUPABASE_SERVICE_KEY environment variable is required for service client"
            )
        _service_client = create_client(url, key)
        logger.info("Supabase service client initialized")
    
    return _service_client


def get_client_with_token(access_token: str) -> "Client":
    """
    Create a Supabase client authenticated with a user's access token.
    
    This allows server-side code to make requests on behalf of a user,
    respecting RLS policies.
    """
    try:
        from supabase import create_client, Client
    except ImportError:
        raise ImportError(
            "supabase package not installed. Run: pip install supabase"
        )
    
    url = get_supabase_url()
    key = get_supabase_anon_key()
    
    # Create client with user's token
    client = create_client(url, key)
    client.auth.set_session(access_token, "")
    
    return client


class SupabaseStorage:
    """Helper class for Supabase Storage operations."""
    
    def __init__(self, client: "Client", bucket: str = "materials"):
        self.client = client
        self.bucket = bucket
    
    def upload_file(
        self, 
        user_id: str, 
        filename: str, 
        file_data: bytes,
        content_type: str = "application/octet-stream"
    ) -> str:
        """
        Upload a file to user's folder.
        
        Returns the storage path.
        """
        path = f"{user_id}/{filename}"
        
        # Check if file exists and remove it first
        try:
            self.client.storage.from_(self.bucket).remove([path])
        except Exception:
            pass  # File doesn't exist, that's fine
        
        # Upload new file
        result = self.client.storage.from_(self.bucket).upload(
            path=path,
            file=file_data,
            file_options={"content-type": content_type}
        )
        
        return path
    
    def download_file(self, storage_path: str) -> bytes:
        """Download a file from storage."""
        result = self.client.storage.from_(self.bucket).download(storage_path)
        return result
    
    def list_files(self, user_id: str) -> list[dict]:
        """List all files in user's folder."""
        result = self.client.storage.from_(self.bucket).list(user_id)
        return result or []
    
    def delete_file(self, storage_path: str) -> bool:
        """Delete a file from storage."""
        try:
            self.client.storage.from_(self.bucket).remove([storage_path])
            return True
        except Exception:
            return False
    
    def get_public_url(self, storage_path: str) -> str:
        """Get public URL for a file (if bucket is public)."""
        result = self.client.storage.from_(self.bucket).get_public_url(storage_path)
        return result
    
    def get_signed_url(self, storage_path: str, expires_in: int = 3600) -> str:
        """Get a signed URL for temporary access."""
        result = self.client.storage.from_(self.bucket).create_signed_url(
            storage_path, expires_in
        )
        return result.get("signedURL", "")
