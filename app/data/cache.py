"""
OHLCV data caching layer.

Caches market data to parquet files to reduce API calls and improve performance.
"""

import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from app.config import CACHE_DIR, settings
from app.data.providers import DataProvider, Timeframe, get_provider

logger = logging.getLogger(__name__)


class OHLCVCache:
    """
    Local file cache for OHLCV data.

    Stores data as parquet files with TTL-based invalidation.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_minutes = settings.cache_ttl_minutes

    def _get_cache_key(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> str:
        """Generate a unique cache key for the request."""
        key_str = f"{ticker}_{timeframe}_{start.date()}_{end.date()}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.parquet"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if cached file is still valid based on TTL."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        age = datetime.now() - mtime

        # For daily data, cache is valid for longer
        # For intraday, use the configured TTL
        return age < timedelta(minutes=self.ttl_minutes)

    def get(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        provider: DataProvider | None = None,
        cancellation_check: callable = None,
    ) -> pd.DataFrame:
        """
        Get OHLCV data, using cache if available.

        Args:
            ticker: Stock symbol
            timeframe: Candle timeframe
            start: Start datetime
            end: End datetime
            provider: Data provider (uses default if None)
            cancellation_check: Optional callable to check for cancellation

        Returns:
            DataFrame with OHLCV data
        """
        # Check for cancellation early
        if cancellation_check and cancellation_check():
            return pd.DataFrame()
        
        if not settings.cache_enabled:
            provider = provider or get_provider()
            return provider.get_ohlcv(ticker, timeframe, start, end, cancellation_check)

        cache_key = self._get_cache_key(ticker, timeframe, start, end)
        cache_path = self._get_cache_path(cache_key)

        # Try to load from cache
        if self._is_cache_valid(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                logger.debug(f"Cache hit for {ticker} ({timeframe})")
                return df
            except Exception as e:
                logger.warning(f"Failed to read cache: {e}")

        # Check for cancellation before expensive fetch
        if cancellation_check and cancellation_check():
            return pd.DataFrame()

        # Fetch from provider
        provider = provider or get_provider()
        df = provider.get_ohlcv(ticker, timeframe, start, end, cancellation_check)

        # Save to cache (only if not cancelled and not empty)
        if not df.empty:
            try:
                df.to_parquet(cache_path, index=False)
                logger.debug(f"Cached {ticker} ({timeframe})")
            except Exception as e:
                logger.warning(f"Failed to write cache: {e}")

        return df

    def invalidate(self, ticker: str | None = None) -> int:
        """
        Invalidate cached data.

        Args:
            ticker: If provided, only invalidate for this ticker.
                    If None, invalidate all cached data.

        Returns:
            Number of cache files removed
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.parquet"):
            if ticker is None or ticker.lower() in cache_file.stem.lower():
                cache_file.unlink()
                count += 1
        logger.info(f"Invalidated {count} cache files")
        return count

    def get_cache_stats(self) -> dict:
        """Get statistics about the cache."""
        files = list(self.cache_dir.glob("*.parquet"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
        }


# Global cache instance
_cache: OHLCVCache | None = None


def get_cache() -> OHLCVCache:
    """Get the global cache instance."""
    global _cache
    if _cache is None:
        _cache = OHLCVCache()
    return _cache


def get_cached_ohlcv(
    ticker: str,
    timeframe: Timeframe,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """
    Convenience function to get cached OHLCV data.

    Args:
        ticker: Stock symbol
        timeframe: Candle timeframe
        start: Start datetime
        end: End datetime

    Returns:
        DataFrame with OHLCV data
    """
    cache = get_cache()
    return cache.get(ticker, timeframe, start, end)
