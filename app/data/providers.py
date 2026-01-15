"""
Market data providers for OHLCV data.

Supports yfinance (default), with placeholders for Polygon and Alpaca.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Literal
import logging
import time
import threading

import pandas as pd
import pytz

from app.config import settings, get_env, get_polygon_api_key

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter with per-second limits."""
    
    def __init__(self, calls_per_second: float = 0.5):
        """
        Initialize rate limiter.
        
        Args:
            calls_per_second: Maximum calls per second (default 0.5 = 1 call per 2 seconds)
        """
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0.0
        self._lock = threading.Lock()
    
    def wait(self):
        """Wait if necessary to respect rate limit."""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_call = time.time()


# Global rate limiter for yfinance (1 call per second - yfinance is more tolerant)
_yfinance_rate_limiter = RateLimiter(calls_per_second=1.0)

# Standard column names for OHLCV data
OHLCV_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]


def normalize_ticker(ticker: str) -> tuple[str, str]:
    """
    Normalize ticker symbol and detect exchange.
    
    Handles various formats:
    - "HKEX:0700" -> "0700.HK", 'HK'
    - "0700.HK" -> "0700.HK", 'HK'
    - "0700" (4-5 digits) -> "0700.HK", 'HK'
    - "AMEX:SPY" -> "SPY", 'US'
    - "SPY" -> "SPY", 'US'
    
    Returns:
        tuple of (normalized_ticker, exchange)
        exchange can be: 'US', 'HK', 'UK', 'JP', 'CN'
    """
    ticker = ticker.upper().strip()
    
    # Handle exchange prefix (e.g., "HKEX:0700", "AMEX:SPY")
    if ':' in ticker:
        exchange_prefix, symbol = ticker.split(':', 1)
        
        # Hong Kong Exchange
        if exchange_prefix in ['HKEX', 'HKG', 'SEHK']:
            # HK tickers need .HK suffix for yfinance
            # Pad with leading zeros if needed (e.g., 700 -> 0700)
            symbol = symbol.lstrip('0') or '0'  # Remove leading zeros first
            symbol = symbol.zfill(4)  # Pad to 4 digits
            return f"{symbol}.HK", 'HK'
        
        # China mainland exchanges
        if exchange_prefix in ['SSE', 'SHA']:  # Shanghai
            return f"{symbol}.SS", 'CN'
        if exchange_prefix in ['SZSE', 'SHE']:  # Shenzhen
            return f"{symbol}.SZ", 'CN'
        
        # US exchanges - just return the symbol
        if exchange_prefix in ['AMEX', 'NYSE', 'NASDAQ', 'ARCA', 'BATS', 'OTC']:
            return symbol, 'US'
        
        # Unknown exchange prefix - try to handle as regular ticker
        ticker = symbol
    
    # Already has exchange suffix
    if '.HK' in ticker:
        return ticker, 'HK'
    if '.SS' in ticker:
        return ticker, 'CN'
    if '.SZ' in ticker:
        return ticker, 'CN'
    if '.L' in ticker:
        return ticker, 'UK'
    if '.T' in ticker:
        return ticker, 'JP'
    
    # Detect HK stocks (numeric only, 4-5 digits)
    if ticker.isdigit() and len(ticker) in [4, 5]:
        ticker = ticker.zfill(4)  # Ensure 4 digits with leading zeros
        return f"{ticker}.HK", 'HK'
    
    # Default to US
    return ticker, 'US'


def is_international_ticker(ticker: str) -> bool:
    """Check if ticker is for an international (non-US) market."""
    _, exchange = normalize_ticker(ticker)
    return exchange in ['HK', 'CN', 'UK', 'JP']

Timeframe = Literal["1d", "2h", "1h", "30m", "15m", "5m", "1m"]


class DataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a ticker.

        Args:
            ticker: Stock symbol (e.g., 'SPY', 'AAPL')
            timeframe: Candle timeframe ('1d', '2h', '5m', etc.)
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with columns: datetime, open, high, low, close, volume
            Index is datetime in America/New_York timezone.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    def _normalize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize DataFrame to standard format."""
        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()

        # Rename common variations
        column_map = {
            "date": "datetime",
            "time": "datetime",
            "timestamp": "datetime",
            "vol": "volume",
        }
        df = df.rename(columns=column_map)

        # Ensure datetime column exists
        if "datetime" not in df.columns:
            if df.index.name in ["datetime", "date", "Date", "Datetime"]:
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "datetime"})
            elif isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "datetime"})

        # Convert to NYC timezone
        ny_tz = pytz.timezone("America/New_York")
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(ny_tz)
        else:
            df["datetime"] = df["datetime"].dt.tz_convert(ny_tz)

        # Select only OHLCV columns
        available_cols = [c for c in OHLCV_COLUMNS if c in df.columns]
        df = df[available_cols].copy()

        # Sort by datetime
        df = df.sort_values("datetime").reset_index(drop=True)

        return df


class YFinanceProvider(DataProvider):
    """Yahoo Finance data provider using yfinance library."""

    def __init__(self):
        """Initialize with rate limiter."""
        self.rate_limiter = _yfinance_rate_limiter
        self.max_retries = 3
        self.base_delay = 5.0  # Base delay for exponential backoff

    @property
    def name(self) -> str:
        return "yfinance"

    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance with rate limiting and retry."""
        import yfinance as yf

        # Normalize ticker (e.g., "9988" -> "9988.HK" for Hong Kong stocks)
        normalized_ticker, exchange = normalize_ticker(ticker)

        # Map timeframe to yfinance interval
        interval_map = {
            "1d": "1d", "2h": "2h", "1h": "1h",
            "30m": "30m", "15m": "15m", "5m": "5m", "1m": "1m",
        }
        interval = interval_map.get(timeframe, "1d")

        last_error = None
        for attempt in range(self.max_retries):
            try:
                self.rate_limiter.wait()
                ticker_obj = yf.Ticker(normalized_ticker)
                df = ticker_obj.history(
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval=interval,
                    actions=False,
                )

                if df.empty:
                    # Try shorter date range for intraday
                    if interval in ['1m', '5m', '15m', '30m', '1h', '2h']:
                        df = ticker_obj.history(period="5d", interval=interval, actions=False)
                    
                    if df.empty:
                        return pd.DataFrame(columns=OHLCV_COLUMNS)

                # Reset index to get datetime as column
                df = df.reset_index()
                df = df.rename(columns={"Date": "datetime", "Datetime": "datetime"})

                return self._normalize_dataframe(df)

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if rate limited
                if "rate" in error_str or "too many" in error_str or "429" in error_str:
                    delay = self.base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Rate limited for {ticker}, attempt {attempt + 1}/{self.max_retries}. "
                        f"Waiting {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(f"Error fetching {ticker} from yfinance: {e}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)
        
        # All retries exhausted
        logger.error(f"Failed to fetch {ticker} after {self.max_retries} retries: {last_error}")
        return pd.DataFrame(columns=OHLCV_COLUMNS)


class PolygonRateLimiter:
    """Rate limiter for Polygon API to avoid 429 errors.
    
    Polygon free tier: 5 requests/minute
    Polygon paid tiers: Higher limits
    """
    
    def __init__(self, requests_per_minute: int = 5):
        self._lock = threading.Lock()
        self._last_request_time = 0
        self._min_interval = 60.0 / requests_per_minute  # seconds between requests
    
    def wait(self):
        """Wait if necessary to respect rate limits."""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_interval:
                sleep_time = self._min_interval - elapsed
                time.sleep(sleep_time)
            self._last_request_time = time.time()


# Global rate limiter for Polygon (5 req/min for free tier, adjust if you have paid)
_polygon_rate_limiter = PolygonRateLimiter(requests_per_minute=5)


class PolygonProvider(DataProvider):
    """Polygon.io data provider with rate limiting and retry logic."""

    def __init__(self):
        self.api_key = get_polygon_api_key()
        self.rate_limiter = _polygon_rate_limiter
        self.max_retries = 5
        self.base_delay = 2.0  # Base delay for exponential backoff
        if not self.api_key:
            logger.warning("POLYGON_API_KEY not set. Add it to .env file.")

    @property
    def name(self) -> str:
        return "polygon"

    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Polygon.io with rate limiting and retry."""
        
        # Polygon only supports US stocks - fall back to yfinance for international
        if is_international_ticker(ticker):
            return YFinanceProvider().get_ohlcv(ticker, timeframe, start, end)
        
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY is required for Polygon provider")

        import httpx

        # Map timeframe to Polygon format
        timespan_map = {
            "1d": ("1", "day"),
            "2h": ("2", "hour"),
            "1h": ("1", "hour"),
            "30m": ("30", "minute"),
            "15m": ("15", "minute"),
            "5m": ("5", "minute"),
            "1m": ("1", "minute"),
        }
        multiplier, timespan = timespan_map.get(timeframe, ("1", "day"))

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/"
            f"{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        )
        params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 50000}

        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Rate limit before making request
                self.rate_limiter.wait()
                
                response = httpx.get(url, params=params, timeout=30)
                
                # Handle rate limiting (429)
                if response.status_code == 429:
                    # Exponential backoff with jitter
                    delay = self.base_delay * (2 ** attempt) + (time.time() % 1)
                    logger.warning(
                        f"Polygon rate limit hit for {ticker}, "
                        f"waiting {delay:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                    continue
                
                response.raise_for_status()
                data = response.json()

                if data.get("resultsCount", 0) == 0:
                    logger.debug(f"No data returned for {ticker} ({timeframe})")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)

                df = pd.DataFrame(data["results"])
                df = df.rename(
                    columns={
                        "t": "datetime",
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                    }
                )
                df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")

                return self._normalize_dataframe(df)

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:
                    delay = self.base_delay * (2 ** attempt) + (time.time() % 1)
                    logger.warning(f"Polygon 429 for {ticker}, retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.error(f"Error fetching {ticker} from Polygon: {e}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)
            except Exception as e:
                last_error = e
                logger.error(f"Error fetching {ticker} from Polygon: {e}")
                return pd.DataFrame(columns=OHLCV_COLUMNS)
        
        # All retries exhausted
        logger.error(f"Failed to fetch {ticker} from Polygon after {self.max_retries} retries: {last_error}")
        return pd.DataFrame(columns=OHLCV_COLUMNS)


class AlpacaProvider(DataProvider):
    """Alpaca Markets data provider (placeholder - requires API key)."""

    def __init__(self):
        self.api_key = get_env("ALPACA_API_KEY")
        self.secret_key = get_env("ALPACA_SECRET_KEY")
        if not self.api_key:
            logger.warning("ALPACA_API_KEY not set. Alpaca provider will not work.")

    @property
    def name(self) -> str:
        return "alpaca"

    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpaca Markets."""
        if not self.api_key:
            raise ValueError("ALPACA_API_KEY is required for Alpaca provider")

        # Placeholder implementation
        import httpx

        # Map timeframe to Alpaca format
        tf_map = {
            "1d": "1Day",
            "2h": "2Hour",
            "1h": "1Hour",
            "30m": "30Min",
            "15m": "15Min",
            "5m": "5Min",
            "1m": "1Min",
        }
        alpaca_tf = tf_map.get(timeframe, "1Day")

        url = f"https://data.alpaca.markets/v2/stocks/{ticker}/bars"
        headers = {"APCA-API-KEY-ID": self.api_key, "APCA-API-SECRET-KEY": self.secret_key}
        params = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "timeframe": alpaca_tf,
            "adjustment": "split",
            "limit": 10000,
        }

        try:
            response = httpx.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if not data.get("bars"):
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            df = pd.DataFrame(data["bars"])
            df = df.rename(
                columns={
                    "t": "datetime",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                }
            )
            df["datetime"] = pd.to_datetime(df["datetime"])

            return self._normalize_dataframe(df)

        except Exception as e:
            logger.error(f"Error fetching {ticker} from Alpaca: {e}")
            return pd.DataFrame(columns=OHLCV_COLUMNS)


def get_provider(name: str | None = None) -> DataProvider:
    """
    Get a data provider instance.

    Args:
        name: Provider name ('yfinance', 'polygon', 'alpaca').
              Defaults to config setting.

    Returns:
        DataProvider instance
    """
    provider_name = name or settings.data_provider

    providers = {
        "yfinance": YFinanceProvider,
        "polygon": PolygonProvider,
        "alpaca": AlpacaProvider,
    }

    if provider_name not in providers:
        logger.warning(f"Unknown provider '{provider_name}', using yfinance")
        provider_name = "yfinance"

    return providers[provider_name]()


def get_daily_data(ticker: str, days: int = 252) -> pd.DataFrame:
    """Convenience function to get daily OHLCV data."""
    provider = get_provider()
    end = datetime.now()
    start = end - timedelta(days=days * 1.5)  # Account for weekends/holidays
    return provider.get_ohlcv(ticker, "1d", start, end)


def get_intraday_data(ticker: str, timeframe: Timeframe = "5m", days: int = 5) -> pd.DataFrame:
    """Convenience function to get intraday OHLCV data."""
    provider = get_provider()
    end = datetime.now()
    start = end - timedelta(days=days)
    return provider.get_ohlcv(ticker, timeframe, start, end)
