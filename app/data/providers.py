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


def get_alltick_token() -> str | None:
    """Get AllTick API token from environment."""
    return get_env("ALLTICK_TOKEN")


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


# Global rate limiter for yfinance (0.2 calls per second = 1 call per 5 seconds for HK stocks)
# Yahoo Finance is very strict with rate limiting, especially for international stocks
_yfinance_rate_limiter = RateLimiter(calls_per_second=0.2)

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
        cancellation_check: callable = None,
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
        cancellation_check: callable = None,
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
            # Check for cancellation before each attempt
            if cancellation_check and cancellation_check():
                logger.info(f"⏹️ Cancelled fetching {ticker}")
                return pd.DataFrame(columns=OHLCV_COLUMNS)
            
            try:
                self.rate_limiter.wait()
                
                # Check again after rate limit wait
                if cancellation_check and cancellation_check():
                    logger.info(f"⏹️ Cancelled fetching {ticker}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)
                
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
                    # Check cancellation during wait (in 1-second intervals)
                    for _ in range(int(delay)):
                        if cancellation_check and cancellation_check():
                            logger.info(f"⏹️ Cancelled during retry wait for {ticker}")
                            return pd.DataFrame(columns=OHLCV_COLUMNS)
                        time.sleep(1)
                    # Sleep remaining fraction
                    remaining = delay - int(delay)
                    if remaining > 0:
                        time.sleep(remaining)
                else:
                    # Non-rate-limit error, don't retry
                    logger.error(f"Error fetching {ticker} from yfinance: {e}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)

        # All retries exhausted (reuse exchange from earlier normalization)
        if exchange == 'HK':
            logger.error(
                f"Failed to fetch HK stock {ticker} after {self.max_retries} retries: {last_error}. "
                f"TIP: Configure ALLTICK_TOKEN in .env for better HK stock data."
            )
        else:
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
        cancellation_check: callable = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Polygon.io with rate limiting and retry."""
        
        # Polygon only supports US stocks - fall back to yfinance for international
        if is_international_ticker(ticker):
            return YFinanceProvider().get_ohlcv(ticker, timeframe, start, end, cancellation_check)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            logger.info(f"⏹️ Cancelled fetching {ticker}")
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        
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
        cancellation_check: callable = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Alpaca Markets."""
        if cancellation_check and cancellation_check():
            return pd.DataFrame(columns=OHLCV_COLUMNS)
            
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


class AllTickProvider(DataProvider):
    """
    AllTick data provider for HK, US, and A-share stocks.
    
    Uses AllTick API: https://alltick.co
    Best for Hong Kong stocks with real-time and historical K-line data.
    
    API Reference: https://en.apis.alltick.co/rest-api/http-interface-api/get-single-product-k-line-query
    """

    def __init__(self):
        """Initialize with API token."""
        self.token = get_alltick_token()
        self.base_url = "https://quote.alltick.co/quote-stock-b-api"
        # AllTick free tier: 1 /kline request every 10 seconds (6 calls per minute)
        self.rate_limiter = RateLimiter(calls_per_second=0.1)  # 1 call per 10 seconds
        self.max_retries = 3

    @property
    def name(self) -> str:
        return "alltick"

    @property
    def is_available(self) -> bool:
        return self.token is not None

    def _normalize_ticker_for_alltick(self, ticker: str) -> tuple[str, bool]:
        """
        Normalize ticker to AllTick format.
        
        AllTick uses:
        - HK stocks: 700.HK, 9988.HK (no leading zeros needed)
        - US stocks: AAPL, MSFT
        - A-shares: 600519.SH, 000001.SZ
        
        Returns:
            tuple of (alltick_code, is_supported)
        """
        normalized, exchange = normalize_ticker(ticker)
        
        if exchange == 'HK':
            # AllTick HK format: remove leading zeros and add .HK
            # e.g., 0700.HK -> 700.HK
            symbol = normalized.replace('.HK', '').lstrip('0') or '0'
            return f"{symbol}.HK", True
        elif exchange == 'US':
            return normalized, True
        elif exchange == 'CN':
            # A-shares: .SS -> .SH for AllTick
            if '.SS' in normalized:
                return normalized.replace('.SS', '.SH'), True
            return normalized, True
        else:
            return normalized, False

    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        cancellation_check: callable = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from AllTick API."""
        import httpx
        import json
        from urllib.parse import quote
        
        if not self.is_available:
            logger.warning("AllTick token not configured")
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        
        # Check for cancellation
        if cancellation_check and cancellation_check():
            logger.info(f"⏹️ Cancelled fetching {ticker}")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        alltick_code, is_supported = self._normalize_ticker_for_alltick(ticker)
        if not is_supported:
            logger.warning(f"Ticker {ticker} not supported by AllTick")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Map timeframe to AllTick kline_type
        # 1=1m, 2=5m, 3=15m, 4=30m, 5=1h, 6=2h (forex only), 8=day, 9=week, 10=month
        kline_type_map = {
            "1m": 1,
            "5m": 2,
            "15m": 3,
            "30m": 4,
            "1h": 5,
            "2h": 5,  # AllTick doesn't support 2h for stocks, use 1h
            "1d": 8,
        }
        kline_type = kline_type_map.get(timeframe, 8)

        # Calculate number of bars needed (max 500 per request)
        if timeframe == "1d":
            days_needed = (end - start).days + 1
            query_num = min(days_needed, 500)
        elif timeframe in ["1h", "2h"]:
            hours_needed = int((end - start).total_seconds() / 3600) + 1
            query_num = min(hours_needed, 500)
        elif timeframe in ["5m", "15m", "30m"]:
            minutes_needed = int((end - start).total_seconds() / 60)
            bars_per_minute = {"5m": 1/5, "15m": 1/15, "30m": 1/30}[timeframe]
            query_num = min(int(minutes_needed * bars_per_minute) + 1, 500)
        else:
            query_num = 500

        # Build query payload
        query_payload = {
            "trace": f"trade_coach_{int(time.time())}",
            "data": {
                "code": alltick_code,
                "kline_type": kline_type,
                "kline_timestamp_end": 0,  # Get from latest
                "query_kline_num": query_num,
                "adjust_type": 0  # Ex-rights adjustment
            }
        }

        query_json = json.dumps(query_payload)
        encoded_query = quote(query_json)
        
        url = f"{self.base_url}/kline?token={self.token}&query={encoded_query}"

        for attempt in range(self.max_retries):
            # Check for cancellation before each attempt
            if cancellation_check and cancellation_check():
                logger.info(f"⏹️ Cancelled fetching {ticker}")
                return pd.DataFrame(columns=OHLCV_COLUMNS)
            
            try:
                self.rate_limiter.wait()
                
                # Check again after rate limit wait
                if cancellation_check and cancellation_check():
                    logger.info(f"⏹️ Cancelled fetching {ticker}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)
                
                response = httpx.get(url, timeout=30)
                response.raise_for_status()
                data = response.json()

                # AllTick uses "ret" for return code (0 = success)
                ret_code = data.get("ret", data.get("code", -1))
                if ret_code != 0:
                    error_msg = data.get("msg", "Unknown error")
                    logger.warning(f"AllTick API error for {alltick_code}: ret={ret_code}, msg={error_msg}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)

                kline_list = data.get("data", {}).get("kline_list", [])
                
                if not kline_list:
                    logger.warning(f"No K-line data returned for {alltick_code}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)

                # Convert to DataFrame
                # AllTick returns all values as strings
                records = []
                for k in kline_list:
                    try:
                        # timestamp can be string or int
                        ts = k.get("timestamp", 0)
                        if isinstance(ts, str):
                            ts = int(ts) if ts else 0
                        
                        records.append({
                            "datetime": pd.to_datetime(ts, unit="s"),
                            "open": float(k.get("open_price", 0)),
                            "high": float(k.get("high_price", 0)),
                            "low": float(k.get("low_price", 0)),
                            "close": float(k.get("close_price", 0)),
                            "volume": int(float(k.get("volume", 0))),  # volume is string
                        })
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Skipping invalid kline data: {e}")
                        continue

                if not records:
                    logger.warning(f"No valid kline records parsed for {alltick_code}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)
                
                df = pd.DataFrame(records)

                # Filter to requested date range
                df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]

                logger.debug(f"📈 AllTick: Fetched {len(df)} {timeframe} bars for {alltick_code}")
                
                return self._normalize_dataframe(df)

            except httpx.HTTPStatusError as e:
                logger.warning(f"AllTick HTTP error (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
            except Exception as e:
                logger.error(f"AllTick error for {alltick_code}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)

        return pd.DataFrame(columns=OHLCV_COLUMNS)


# Cache for provider instances (avoid recreating on every call)
_provider_cache: dict[str, DataProvider] = {}


def get_provider(name: str | None = None) -> DataProvider:
    """
    Get a data provider instance (cached).

    Args:
        name: Provider name ('yfinance', 'polygon', 'alpaca', 'alltick').
              Defaults to config setting.

    Returns:
        DataProvider instance (cached)
    """
    provider_name = name or settings.data_provider

    # Return cached instance if available
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    providers = {
        "yfinance": YFinanceProvider,
        "polygon": PolygonProvider,
        "alpaca": AlpacaProvider,
        "alltick": AllTickProvider,
    }

    if provider_name not in providers:
        logger.warning(f"Unknown provider '{provider_name}', using yfinance")
        provider_name = "yfinance"

    # Create and cache the provider
    _provider_cache[provider_name] = providers[provider_name]()
    return _provider_cache[provider_name]


def get_provider_for_ticker(ticker: str) -> DataProvider:
    """
    Get the best data provider for a specific ticker (cached).
    
    Uses AllTick for HK stocks (if token configured), yfinance for others.
    
    Args:
        ticker: Stock symbol
        
    Returns:
        Best DataProvider for this ticker (cached)
    """
    _, exchange = normalize_ticker(ticker)
    
    # For HK stocks, prefer AllTick if available
    if exchange == 'HK':
        alltick_token = get_alltick_token()
        if alltick_token:
            return get_provider("alltick")
    
    # For other international stocks, use yfinance
    if exchange in ['HK', 'CN', 'JP', 'UK']:
        return get_provider("yfinance")
    
    # For US stocks, use configured provider
    return get_provider()


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
