"""
Market data providers for OHLCV data.

Provider Strategy:
- US stocks: Polygon (primary), YFinance (fallback)
- HK stocks: Tencent (primary for daily), YFinance (fallback/intraday)
- Futures: Databento (primary), YFinance (fallback)
- Other international: YFinance

Supports: Polygon, YFinance, AllTick, Alpaca (placeholder), Tencent HK, Databento
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, date
from typing import Literal
import logging
import time
import threading
import warnings

import pandas as pd
import pytz

from app.config import settings, get_env, get_polygon_api_key

logger = logging.getLogger(__name__)


def get_alltick_token() -> str | None:
    """Get AllTick API token from environment."""
    return get_env("ALLTICK_TOKEN")


def get_databento_api_key() -> str | None:
    """Get Databento API key from environment."""
    return get_env("DATABENTO_API_KEY")


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


# Global rate limiter for yfinance (0.5 calls per second = 1 call per 2 seconds)
# Matches Polygon's effective rate for consistency across US and HK stocks
_yfinance_rate_limiter = RateLimiter(calls_per_second=0.5)

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
    - "MES=F", "ES=F" -> "MES=F", 'FUTURES'
    - "CME_MINI:MES1!" -> "MES=F", 'FUTURES'

    Returns:
        tuple of (normalized_ticker, exchange)
        exchange can be: 'US', 'HK', 'UK', 'JP', 'CN', 'FUTURES'
    """
    ticker = ticker.upper().strip()

    # Detect futures by =F suffix (Yahoo Finance format)
    if "=F" in ticker:
        return ticker, "FUTURES"

    # Handle exchange prefix (e.g., "HKEX:0700", "AMEX:SPY", "CME_MINI:MES1!")
    if ":" in ticker:
        exchange_prefix, symbol = ticker.split(":", 1)

        # CME Futures (from TradingView format like "CME_MINI:MES1!")
        if exchange_prefix in ["CME", "CME_MINI", "CBOT", "NYMEX", "COMEX"]:
            # Convert TradingView format to Yahoo Finance format
            # MES1! -> MES=F, ES1! -> ES=F, NQ1! -> NQ=F
            base_symbol = symbol.rstrip("0123456789!").upper()
            return f"{base_symbol}=F", "FUTURES"

        # Hong Kong Exchange
        if exchange_prefix in ["HKEX", "HKG", "SEHK"]:
            # HK tickers need .HK suffix for yfinance
            # Pad with leading zeros if needed (e.g., 700 -> 0700)
            symbol = symbol.lstrip("0") or "0"  # Remove leading zeros first
            symbol = symbol.zfill(4)  # Pad to 4 digits
            return f"{symbol}.HK", "HK"

        # China mainland exchanges
        if exchange_prefix in ["SSE", "SHA"]:  # Shanghai
            return f"{symbol}.SS", "CN"
        if exchange_prefix in ["SZSE", "SHE"]:  # Shenzhen
            return f"{symbol}.SZ", "CN"

        # US exchanges - just return the symbol
        if exchange_prefix in ["AMEX", "NYSE", "NASDAQ", "ARCA", "BATS", "OTC"]:
            return symbol, "US"

        # Unknown exchange prefix - try to handle as regular ticker
        ticker = symbol

    # Already has exchange suffix
    if ".HK" in ticker:
        return ticker, "HK"
    if ".SS" in ticker:
        return ticker, "CN"
    if ".SZ" in ticker:
        return ticker, "CN"
    if ".L" in ticker:
        return ticker, "UK"
    if ".T" in ticker:
        return ticker, "JP"

    # Detect HK stocks (numeric only, 1-5 digits)
    # HK stock codes range from 1 to 99999 (e.g., 5, 16, 700, 981, 9988)
    if ticker.isdigit() and 1 <= len(ticker) <= 5:
        ticker = ticker.zfill(4)  # Pad to 4 digits with leading zeros (e.g., 981 -> 0981)
        return f"{ticker}.HK", "HK"

    # Default to US
    return ticker, "US"


def is_international_ticker(ticker: str) -> bool:
    """Check if ticker is for an international (non-US) market.

    Note: Futures are NOT considered international (they trade on CME in Chicago).
    """
    _, exchange = normalize_ticker(ticker)
    return exchange in ["HK", "CN", "UK", "JP"]


def is_futures_ticker(ticker: str) -> bool:
    """Check if ticker is for a futures contract."""
    _, exchange = normalize_ticker(ticker)
    return exchange == "FUTURES"


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
    """Yahoo Finance data provider.

    Strategy:
    1. First try direct Yahoo Finance API (faster, less rate-limited)
    2. Fall back to yfinance library if direct API fails
    """

    def __init__(self):
        """Initialize with rate limiter."""
        self.rate_limiter = _yfinance_rate_limiter
        self.max_retries = 3
        self.base_delay = 5.0  # Base delay for exponential backoff

    @property
    def name(self) -> str:
        return "yfinance"

    def _fetch_via_direct_api(
        self,
        normalized_ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        """
        Fetch data using direct Yahoo Finance API.

        This is faster and less prone to rate limiting than the yfinance library.
        """
        import requests

        # Map our interval format to Yahoo Finance API format
        # Yahoo accepts: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "2h": "60m",  # Yahoo doesn't have 2h, use 1h as closest
            "1d": "1d",
        }
        yahoo_interval = interval_map.get(interval, interval)

        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{normalized_ticker}"

        # Calculate days for range parameter
        days_diff = (end - start).days + 1

        # Yahoo Finance range options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        if yahoo_interval in ["1m", "2m", "5m", "15m", "30m"]:
            # Intraday data limited to recent days
            range_param = "5d" if days_diff <= 5 else "1mo"
        elif yahoo_interval in ["60m", "90m", "1h"]:
            range_param = "1mo" if days_diff <= 30 else "3mo"
        else:
            # Daily data
            if days_diff <= 30:
                range_param = "1mo"
            elif days_diff <= 90:
                range_param = "3mo"
            elif days_diff <= 180:
                range_param = "6mo"
            elif days_diff <= 365:
                range_param = "1y"
            else:
                range_param = "2y"

        params = {
            "interval": yahoo_interval,
            "range": range_param,
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }

        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)

            if response.status_code == 429:
                logger.warning(f"Direct API rate limited for {normalized_ticker}")
                return pd.DataFrame()

            if response.status_code != 200:
                logger.warning(f"Direct API error {response.status_code} for {normalized_ticker}")
                return pd.DataFrame()

            data = response.json()
            result = data.get("chart", {}).get("result", [])

            if not result:
                return pd.DataFrame()

            timestamps = result[0].get("timestamp", [])
            quote = result[0].get("indicators", {}).get("quote", [{}])[0]

            if not timestamps:
                return pd.DataFrame()

            # Build DataFrame
            df = pd.DataFrame(
                {
                    "datetime": pd.to_datetime(timestamps, unit="s"),
                    "open": quote.get("open", []),
                    "high": quote.get("high", []),
                    "low": quote.get("low", []),
                    "close": quote.get("close", []),
                    "volume": quote.get("volume", []),
                }
            )

            # Remove rows with None values
            df = df.dropna(subset=["open", "high", "low", "close"])

            if df.empty:
                return df

            # Log the data range we got from the API (timestamps are in UTC)
            api_start = df["datetime"].min()
            api_end = df["datetime"].max()

            # Filter to requested date range
            # IMPORTANT: Yahoo API returns timestamps in UTC, so we must convert
            # start/end to UTC before comparing
            pre_filter_count = len(df)
            start_ts = pd.Timestamp(start)
            end_ts = pd.Timestamp(end)

            # Convert timezone-aware timestamps to UTC, then make naive for comparison
            if start_ts.tzinfo is not None:
                start_ts = start_ts.tz_convert("UTC").tz_localize(None)
            if end_ts.tzinfo is not None:
                end_ts = end_ts.tz_convert("UTC").tz_localize(None)

            df = df[(df["datetime"] >= start_ts) & (df["datetime"] <= end_ts)]

            if df.empty and pre_filter_count > 0:
                # The API returned data but it's outside the requested range
                # This usually means the trade is older than the available data
                logger.warning(
                    f"Direct API: got {pre_filter_count} bars for {normalized_ticker} "
                    f"(API range: {api_start} to {api_end} UTC) but requested range "
                    f"({start_ts} to {end_ts} UTC) has no overlap"
                )

            return df

        except Exception as e:
            logger.warning(f"Direct API exception for {normalized_ticker}: {e}")
            return pd.DataFrame()

    def _fetch_via_yfinance_library(
        self,
        normalized_ticker: str,
        interval: str,
        start: datetime,
        end: datetime,
        cancellation_check: callable = None,
    ) -> pd.DataFrame:
        """
        Fetch data using yfinance library (fallback).
        """
        import yfinance as yf

        # Map our interval format to yfinance format
        interval_map = {
            "1m": "1m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "1h",
            "2h": "1h",  # yfinance doesn't have 2h, use 1h as closest
            "1d": "1d",
        }
        yf_interval = interval_map.get(interval, interval)

        try:
            ticker_obj = yf.Ticker(normalized_ticker)
            df = ticker_obj.history(
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=yf_interval,
                actions=False,
            )

            if df.empty:
                # Try shorter date range for intraday
                if yf_interval in ["1m", "5m", "15m", "30m", "1h"]:
                    df = ticker_obj.history(period="5d", interval=yf_interval, actions=False)

            if df.empty:
                return pd.DataFrame()

            # Reset index to get datetime as column
            df = df.reset_index()
            df = df.rename(columns={"Date": "datetime", "Datetime": "datetime"})

            logger.debug(f"yfinance library: got {len(df)} bars for {normalized_ticker}")
            return df

        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "too many" in error_str or "429" in error_str:
                logger.warning(f"yfinance library rate limited for {normalized_ticker}")
            else:
                logger.warning(f"yfinance library error for {normalized_ticker}: {e}")
            return pd.DataFrame()

    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        cancellation_check: callable = None,
    ) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance.

        Strategy:
        1. First try direct Yahoo Finance API (faster, less rate-limited)
        2. Fall back to yfinance library if direct API fails
        """
        # Normalize ticker (e.g., "9988" -> "9988.HK" for Hong Kong stocks)
        normalized_ticker, exchange = normalize_ticker(ticker)

        # Map timeframe to interval
        interval_map = {
            "1d": "1d",
            "2h": "2h",
            "1h": "1h",
            "30m": "30m",
            "15m": "15m",
            "5m": "5m",
            "1m": "1m",
        }
        interval = interval_map.get(timeframe, "1d")

        # Check for cancellation
        if cancellation_check and cancellation_check():
            logger.info(f"⏹️ Cancelled fetching {ticker}")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Rate limit before making request
        self.rate_limiter.wait()

        # Check again after rate limit wait
        if cancellation_check and cancellation_check():
            logger.info(f"⏹️ Cancelled fetching {ticker}")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Strategy 1: Try direct Yahoo Finance API first (faster, less rate-limited)
        logger.info(f"📡 Fetching {normalized_ticker} via direct Yahoo API...")
        df = self._fetch_via_direct_api(normalized_ticker, interval, start, end)

        if not df.empty:
            logger.info(f"✅ Direct API: got {len(df)} bars for {normalized_ticker}")
            return self._normalize_dataframe(df)

        # Strategy 2: Fall back to yfinance library
        logger.info(
            f"⚠️ Direct API returned no data for {normalized_ticker}, trying yfinance library..."
        )

        for attempt in range(self.max_retries):
            if cancellation_check and cancellation_check():
                logger.info(f"⏹️ Cancelled fetching {ticker}")
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            df = self._fetch_via_yfinance_library(
                normalized_ticker, interval, start, end, cancellation_check
            )

            if not df.empty:
                return self._normalize_dataframe(df)

            # Wait before retry with exponential backoff
            if attempt < self.max_retries - 1:
                delay = self.base_delay * (2**attempt)
                logger.debug(
                    f"Retry {attempt + 1}/{self.max_retries} for {ticker}, waiting {delay:.1f}s..."
                )

                # Check cancellation during wait
                for _ in range(int(delay)):
                    if cancellation_check and cancellation_check():
                        logger.info(f"⏹️ Cancelled during retry wait for {ticker}")
                        return pd.DataFrame(columns=OHLCV_COLUMNS)
                    time.sleep(1)

                # Rate limit before next attempt
                self.rate_limiter.wait()

        # All retries exhausted
        if exchange == "HK":
            logger.warning(
                f"Failed to fetch HK stock {ticker} from Yahoo Finance. "
                f"TIP: Configure ALLTICK_TOKEN in .env for better HK stock data."
            )
        else:
            logger.warning(
                f"Failed to fetch {ticker} from Yahoo Finance after {self.max_retries} retries"
            )

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

        # Normalize ticker (strip exchange prefix like "AMEX:SOXL" -> "SOXL")
        normalized_ticker, _ = normalize_ticker(ticker)

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
            f"https://api.polygon.io/v2/aggs/ticker/{normalized_ticker}/range/"
            f"{multiplier}/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        )
        params = {"apiKey": self.api_key, "adjusted": "true", "sort": "asc", "limit": 50000}

        for attempt in range(self.max_retries):
            try:
                # Rate limit before making request
                self.rate_limiter.wait()

                response = httpx.get(url, params=params, timeout=30)

                # Handle rate limiting (429)
                if response.status_code == 429:
                    # Exponential backoff with jitter
                    delay = self.base_delay * (2**attempt) + (time.time() % 1)
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
                if e.response.status_code == 429:
                    delay = self.base_delay * (2**attempt) + (time.time() % 1)
                    logger.warning(f"Polygon 429 for {ticker}, retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue
                else:
                    logger.warning(f"Polygon error for {ticker}: {e}, falling back to YFinance")
                    return YFinanceProvider().get_ohlcv(
                        ticker, timeframe, start, end, cancellation_check
                    )
            except Exception as e:
                logger.warning(f"Polygon error for {ticker}: {e}, falling back to YFinance")
                return YFinanceProvider().get_ohlcv(
                    ticker, timeframe, start, end, cancellation_check
                )

        # All retries exhausted - fall back to YFinance
        logger.warning(
            f"Polygon failed for {ticker} after {self.max_retries} retries, falling back to YFinance"
        )
        return YFinanceProvider().get_ohlcv(ticker, timeframe, start, end, cancellation_check)


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


class TencentHKProvider(DataProvider):
    """
    Tencent/East Money data provider for Hong Kong stocks.

    Uses free APIs for HK stock data:
    - Daily K-line: Tencent API (reliable, no rate limits)
    - Intraday K-line: East Money API (reliable, supports 1m/5m/15m/30m/1h)

    API Reference:
    - Tencent: http://web.ifzq.gtimg.cn/appstock/app/kline/kline
    - East Money: http://push2his.eastmoney.com/api/qt/stock/kline/get
    """

    # Map our timeframes to East Money klt parameter
    # klt: 1=1m, 5=5m, 15=15m, 30=30m, 60=1h, 101=daily
    EASTMONEY_KLT_MAP = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 60,  # Use 1h and aggregate
    }

    def __init__(self):
        """Initialize Tencent/Sina HK provider."""
        self.rate_limiter = RateLimiter(calls_per_second=2.0)  # 2 calls per second (generous)
        self.max_retries = 3
        self.base_delay = 1.0

    @property
    def name(self) -> str:
        return "tencent_hk"

    def _normalize_hk_code(self, ticker: str) -> str:
        """
        Normalize ticker to 5-digit HK format.

        Examples:
            '0700.HK' -> '00700'
            'HKEX:0700' -> '00700'
            '700' -> '00700'
            '9988.HK' -> '09988'
        """
        normalized, exchange = normalize_ticker(ticker)
        if exchange != "HK":
            return None

        # Extract numeric part
        code = normalized.replace(".HK", "").lstrip("0") or "0"
        # Pad to 5 digits
        return code.zfill(5)

    def _fetch_daily_kline(self, hk_code: str, bars: int = 500) -> pd.DataFrame:
        """
        Fetch daily K-line data from Tencent API.

        Tries multiple endpoints:
        1. hkfqkline (easyquotation style): http://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get
        2. fqkline (go-stock style): https://web.ifzq.gtimg.cn/appstock/app/fqkline/get

        References:
        - https://github.com/shidenggui/easyquotation
        - https://github.com/ArvinLovegood/go-stock
        """
        import requests
        import re
        import json

        # Strategy 1: Try hkfqkline endpoint first (easyquotation style)
        df = self._fetch_daily_via_hkfqkline(hk_code, bars)
        if not df.empty:
            return df

        # Strategy 2: Try fqkline endpoint (go-stock style)
        df = self._fetch_daily_via_fqkline(hk_code, bars)
        if not df.empty:
            return df

        logger.warning(f"All Tencent daily endpoints failed for HK{hk_code}")
        return pd.DataFrame()

    def _fetch_daily_via_hkfqkline(self, hk_code: str, bars: int) -> pd.DataFrame:
        """Fetch from hkfqkline endpoint (easyquotation style)."""
        import requests
        import re
        import json

        url = "http://web.ifzq.gtimg.cn/appstock/app/hkfqkline/get"
        params = {
            "_var": "kline_dayqfq",
            "param": f"hk{hk_code},day,,,{bars},qfq",
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            raw_text = response.text
            json_match = re.search(r"=(.*)$", raw_text)
            if not json_match:
                return pd.DataFrame()

            data = json.loads(json_match.group(1))
            if data.get("code") != 0:
                return pd.DataFrame()

            return self._parse_tencent_kline_response(data, f"hk{hk_code}", "hkfqkline")

        except Exception as e:
            logger.debug(f"hkfqkline failed for HK{hk_code}: {e}")
            return pd.DataFrame()

    def _fetch_daily_via_fqkline(self, hk_code: str, bars: int) -> pd.DataFrame:
        """Fetch from fqkline endpoint (go-stock style)."""
        import requests
        import json

        # go-stock uses: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=hk{code},day,,,{days},qfq
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {"param": f"hk{hk_code},day,,,{bars},qfq"}

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()

            data = response.json()
            if data.get("code") != 0:
                return pd.DataFrame()

            return self._parse_tencent_kline_response(data, f"hk{hk_code}", "fqkline")

        except Exception as e:
            logger.debug(f"fqkline failed for HK{hk_code}: {e}")
            return pd.DataFrame()

    def _parse_tencent_kline_response(self, data: dict, stock_key: str, source: str) -> pd.DataFrame:
        """Parse Tencent K-line API response into DataFrame."""
        stock_data = data.get("data", {})

        if stock_key not in stock_data:
            return pd.DataFrame()

        # Try qfqday first (forward-adjusted), then day
        kline_data = stock_data[stock_key].get("qfqday", [])
        if not kline_data:
            kline_data = stock_data[stock_key].get("day", [])

        if not kline_data:
            return pd.DataFrame()

        # Build DataFrame: [date, open, close, high, low, volume, ...]
        records = []
        for k in kline_data:
            try:
                if len(k) >= 6:
                    records.append({
                        "datetime": pd.to_datetime(k[0]),
                        "open": float(k[1]),
                        "close": float(k[2]),
                        "high": float(k[3]),
                        "low": float(k[4]),
                        "volume": int(float(k[5])) if k[5] else 0,
                    })
            except (ValueError, TypeError, IndexError) as e:
                continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        logger.info(f"✅ Tencent {source}: got {len(df)} daily bars for {stock_key.upper()}")
        return df

    def _fetch_intraday_kline(self, hk_code: str, klt: int, bars: int = 500) -> pd.DataFrame:
        """
        Fetch intraday K-line data with multiple API fallbacks.

        Strategy:
        1. East Money API (primary)
        2. Tencent fqkline API (fallback, from go-stock)

        Args:
            hk_code: 5-digit HK stock code (e.g., '00700')
            klt: K-line type (1=1m, 5=5m, 15=15m, 30=30m, 60=1h)
            bars: Number of bars to fetch
        """
        # Try East Money first
        df = self._fetch_intraday_eastmoney(hk_code, klt, bars)
        if not df.empty:
            return df

        # Try Tencent fqkline for intraday (go-stock style)
        tencent_klt_map = {1: "m1", 5: "m5", 15: "m15", 30: "m30", 60: "m60"}
        tencent_klt = tencent_klt_map.get(klt, "m5")
        df = self._fetch_intraday_tencent(hk_code, tencent_klt, bars)
        if not df.empty:
            return df

        logger.warning(f"All intraday APIs failed for HK{hk_code}")
        return pd.DataFrame()

    def _fetch_intraday_eastmoney(self, hk_code: str, klt: int, bars: int) -> pd.DataFrame:
        """
        Fetch intraday data from East Money API with fallback endpoints.

        Tries both HTTP and HTTPS endpoints with proper headers for reliability.
        Reference: https://github.com/ArvinLovegood/go-stock
        """
        import requests

        # Headers from go-stock for better reliability
        headers = {
            "Host": "push2his.eastmoney.com",
            "Referer": "https://quote.eastmoney.com/",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        }

        params = {
            "secid": f"116.{hk_code}",  # 116 = HK market
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": klt,
            "fqt": 1,  # Forward adjustment
            "end": "20500101",  # Far future to get latest data
            "lmt": bars,
        }

        # Try multiple endpoints for reliability
        endpoints = [
            "https://push2his.eastmoney.com/api/qt/stock/kline/get",
            "http://push2his.eastmoney.com/api/qt/stock/kline/get",
            "https://push2.eastmoney.com/api/qt/stock/kline/get",
        ]

        for url in endpoints:
            try:
                # Update Host header based on URL
                if "push2." in url:
                    headers["Host"] = "push2.eastmoney.com"
                else:
                    headers["Host"] = "push2his.eastmoney.com"

                response = requests.get(url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                data = response.json()

                klines = data.get("data", {}).get("klines", [])
                if not klines:
                    continue  # Try next endpoint

                # Parse K-lines: "2026-01-29 14:40,open,close,high,low,volume,amount"
                records = []
                for kline_str in klines:
                    try:
                        parts = kline_str.split(",")
                        if len(parts) >= 6:
                            records.append({
                                "datetime": pd.to_datetime(parts[0]),
                                "open": float(parts[1]),
                                "close": float(parts[2]),
                                "high": float(parts[3]),
                                "low": float(parts[4]),
                                "volume": int(float(parts[5])) if parts[5] else 0,
                            })
                    except (ValueError, TypeError, IndexError):
                        continue

                if records:
                    df = pd.DataFrame(records)
                    logger.info(f"✅ East Money: got {len(df)} intraday bars for HK{hk_code}")
                    return df

            except Exception as e:
                logger.debug(f"East Money endpoint {url} failed for HK{hk_code}: {e}")
                continue

        logger.warning(f"All East Money endpoints failed for HK{hk_code} (klt={klt})")
        return pd.DataFrame()

    def _fetch_intraday_tencent(self, hk_code: str, klt: str, bars: int) -> pd.DataFrame:
        """
        Fetch intraday data from Tencent APIs (go-stock style).

        Tries:
        1. Tencent minute API (TODAY's 1-minute data) - from go-stock GetStockMinutePriceData
        2. Tencent fqkline API (limited intraday)

        Reference: https://github.com/ArvinLovegood/go-stock
        """
        import requests
        from datetime import datetime

        # Strategy 1: Try Tencent minute API for TODAY's data (go-stock style)
        df = self._fetch_today_minute_data(hk_code)
        if not df.empty:
            return df

        # Strategy 2: Try Tencent fqkline (limited to current day summary)
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {"param": f"hk{hk_code},{klt},,,{bars},qfq"}

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                return pd.DataFrame()

            stock_key = f"hk{hk_code}"
            stock_data = data.get("data", {})

            if stock_key not in stock_data:
                return pd.DataFrame()

            # Try qfq{klt} first, then {klt}
            kline_data = stock_data[stock_key].get(f"qfq{klt}", [])
            if not kline_data:
                kline_data = stock_data[stock_key].get(klt, [])

            if not kline_data:
                return pd.DataFrame()

            records = []
            for k in kline_data:
                try:
                    if len(k) >= 6:
                        records.append({
                            "datetime": pd.to_datetime(k[0]),
                            "open": float(k[1]),
                            "close": float(k[2]),
                            "high": float(k[3]),
                            "low": float(k[4]),
                            "volume": int(float(k[5])) if k[5] else 0,
                        })
                except (ValueError, TypeError, IndexError):
                    continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            logger.info(f"✅ Tencent fqkline: got {len(df)} intraday ({klt}) bars for HK{hk_code}")
            return df

        except Exception as e:
            logger.debug(f"Tencent intraday failed for HK{hk_code}: {e}")
            return pd.DataFrame()

    def _fetch_today_minute_data(self, hk_code: str) -> pd.DataFrame:
        """
        Fetch TODAY's 1-minute data from Tencent minute API.

        API (from go-stock): https://web.ifzq.gtimg.cn/appstock/app/minute/query?code=hk{code}

        Reference: https://github.com/ArvinLovegood/go-stock - GetStockMinutePriceData
        """
        import requests
        from datetime import datetime

        url = f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code=hk{hk_code}"
        headers = {
            "Host": "web.ifzq.gtimg.cn",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }

        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            if data.get("code") != 0:
                return pd.DataFrame()

            stock_key = f"hk{hk_code}"
            stock_data = data.get("data", {}).get(stock_key, {})
            minute_info = stock_data.get("data", {})

            if not minute_info:
                return pd.DataFrame()

            date_str = minute_info.get("date", "")
            minute_data = minute_info.get("data", [])

            if not minute_data or not date_str:
                return pd.DataFrame()

            # Parse minute data: "0930 614.500 1152994 709518524.740"
            # Format: HHMM price volume amount
            records = []
            for item in minute_data:
                try:
                    parts = item.split()
                    if len(parts) >= 3:
                        time_str = parts[0]
                        hour = int(time_str[:2])
                        minute = int(time_str[2:4])

                        dt = datetime.strptime(date_str, "%Y%m%d").replace(
                            hour=hour, minute=minute
                        )
                        records.append({
                            "datetime": pd.to_datetime(dt),
                            "open": float(parts[1]),
                            "high": float(parts[1]),
                            "low": float(parts[1]),
                            "close": float(parts[1]),
                            "volume": int(float(parts[2])) if len(parts) > 2 else 0,
                        })
                except (ValueError, TypeError, IndexError):
                    continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            logger.info(f"✅ Tencent minute: got {len(df)} today's 1m bars for HK{hk_code}")
            return df

        except Exception as e:
            logger.debug(f"Tencent minute API failed for HK{hk_code}: {e}")
            return pd.DataFrame()

    def _fetch_intraday_kline_legacy(self, hk_code: str, klt: int, bars: int = 500) -> pd.DataFrame:
        """Legacy East Money intraday fetch (kept for reference)."""
        import requests

        url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            "secid": f"116.{hk_code}",
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": klt,
            "fqt": 1,
            "end": "20500101",
            "lmt": bars,
        }

        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()

            klines = data.get("data", {}).get("klines", [])
            if not klines:
                logger.warning(f"East Money returned no intraday data for HK{hk_code}")
                return pd.DataFrame()

            records = []
            for kline_str in klines:
                try:
                    parts = kline_str.split(",")
                    if len(parts) >= 6:
                        records.append({
                            "datetime": pd.to_datetime(parts[0]),
                            "open": float(parts[1]),
                            "close": float(parts[2]),
                            "high": float(parts[3]),
                            "low": float(parts[4]),
                            "volume": int(float(parts[5])),
                        })
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Skipping invalid kline: {e}")
                    continue

            if not records:
                return pd.DataFrame()

            df = pd.DataFrame(records)
            logger.info(f"✅ East Money: got {len(df)} {klt}m bars for HK{hk_code}")
            return df

        except Exception as e:
            logger.warning(f"East Money API error for HK{hk_code} (klt={klt}): {e}")
            return pd.DataFrame()

    def _fetch_realtime_quote(self, hk_code: str) -> dict | None:
        """
        Fetch real-time quote from Tencent API.

        API: http://qt.gtimg.cn/q=r_hk{code}

        Returns:
            Dict with price info or None if failed
        """
        import requests

        url = f"http://qt.gtimg.cn/q=r_hk{hk_code}"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            # Parse response: v_r_hk00700="100~腾讯控股~00700~413.200~419.200~422.200~21351010~...";
            text = response.text.strip()
            if "=" not in text:
                return None

            # Extract the quoted part
            data_part = text.split("=")[1].strip('"').strip(";").strip('"')
            fields = data_part.split("~")

            if len(fields) < 10:
                return None

            return {
                "code": hk_code,
                "name": fields[1],
                "price": float(fields[3]) if fields[3] else 0,
                "last_close": float(fields[4]) if fields[4] else 0,
                "open": float(fields[5]) if fields[5] else 0,
                "volume": int(float(fields[6])) if fields[6] else 0,
                "high": float(fields[33]) if len(fields) > 33 and fields[33] else 0,
                "low": float(fields[34]) if len(fields) > 34 and fields[34] else 0,
            }

        except Exception as e:
            logger.debug(f"Tencent realtime quote error for HK{hk_code}: {e}")
            return None

    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        cancellation_check: callable = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for HK stocks.

        Uses:
        - Tencent API for daily data (reliable, no rate limits)
        - Sina API for intraday data (1m, 5m, 15m, 30m, 1h)
        - Falls back to Yahoo Finance only if both fail
        """
        # Check for cancellation
        if cancellation_check and cancellation_check():
            logger.info(f"⏹️ Cancelled fetching {ticker}")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        hk_code = self._normalize_hk_code(ticker)
        if hk_code is None:
            logger.warning(f"Ticker {ticker} is not an HK stock")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Rate limit
        self.rate_limiter.wait()

        if cancellation_check and cancellation_check():
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Calculate how old the requested data is
        try:
            if start.tzinfo is not None:
                from zoneinfo import ZoneInfo
                now = datetime.now(ZoneInfo("UTC"))
            else:
                now = datetime.now()
            days_ago = (now - start).days
        except Exception:
            days_ago = 100  # Fallback: assume historical

        # OPTIMIZATION: For old trades (>60 days), skip intraday entirely
        # Free APIs only keep 30-60 days of intraday data
        if timeframe != "1d" and days_ago > 60:
            logger.info(
                f"⏭️ Skipping intraday fetch for {ticker} - trade is {days_ago} days old "
                f"(intraday data not available). Use daily timeframe."
            )
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Calculate bars needed
        # IMPORTANT: Tencent API returns the LATEST N bars, not bars around a date range.
        # For historical daily data, we MUST request max bars (500) to ensure coverage.
        if timeframe == "1d":
            if days_ago > 30:
                bars_needed = 500  # Historical - need max bars
            else:
                bars_needed = (end - start).days + 60  # Recent data
        elif timeframe in ["1h", "2h"]:
            hours_needed = int((end - start).total_seconds() / 3600) + 48
            bars_needed = hours_needed
        else:
            # Minute data - estimate bars needed
            minutes_map = {"1m": 1, "5m": 5, "15m": 15, "30m": 30}
            minutes_per_bar = minutes_map.get(timeframe, 5)
            total_minutes = int((end - start).total_seconds() / 60)
            bars_needed = (total_minutes // minutes_per_bar) + 100  # Buffer

        for attempt in range(self.max_retries):
            if cancellation_check and cancellation_check():
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            # Use appropriate API based on timeframe
            if timeframe == "1d":
                df_raw = self._fetch_daily_kline(hk_code, min(bars_needed, 500))
            else:
                # Intraday - use East Money API
                klt = self.EASTMONEY_KLT_MAP.get(timeframe, 5)
                df_raw = self._fetch_intraday_kline(hk_code, klt, min(bars_needed, 500))

            if not df_raw.empty:
                # Filter to requested date range
                # Convert start/end to pandas Timestamps and ensure timezone compatibility
                try:
                    start_ts = pd.Timestamp(start)
                    end_ts = pd.Timestamp(end)

                    # Check if df datetime column has timezone
                    df_has_tz = df_raw["datetime"].dt.tz is not None

                    if df_has_tz:
                        # If df has timezone, make start/end timezone-aware
                        if start_ts.tzinfo is None:
                            start_ts = start_ts.tz_localize(df_raw["datetime"].dt.tz)
                        else:
                            start_ts = start_ts.tz_convert(df_raw["datetime"].dt.tz)
                        if end_ts.tzinfo is None:
                            end_ts = end_ts.tz_localize(df_raw["datetime"].dt.tz)
                        else:
                            end_ts = end_ts.tz_convert(df_raw["datetime"].dt.tz)
                    else:
                        # If df is timezone-naive, make start/end naive too
                        if start_ts.tzinfo is not None:
                            start_ts = start_ts.tz_convert("UTC").tz_localize(None)
                        if end_ts.tzinfo is not None:
                            end_ts = end_ts.tz_convert("UTC").tz_localize(None)

                    df_filtered = df_raw[(df_raw["datetime"] >= start_ts) & (df_raw["datetime"] <= end_ts)]

                    if not df_filtered.empty:
                        return self._normalize_dataframe(df_filtered)

                    # API returned data but it doesn't overlap with requested range
                    # This means the trade is too old for this API - don't retry, go to Yahoo
                    api_start = df_raw["datetime"].min()
                    api_end = df_raw["datetime"].max()
                    logger.info(
                        f"Intraday API returned {len(df_raw)} bars ({api_start} to {api_end}) "
                        f"but requested range ({start_ts} to {end_ts}) has no overlap. "
                        f"Trade is too old for intraday data, falling back to Yahoo."
                    )
                    break  # Exit retry loop - no point retrying for old data

                except Exception as e:
                    logger.warning(f"Date filtering error: {e}, returning unfiltered data")
                    return self._normalize_dataframe(df_raw)

            # Only retry if API returned empty (actual failure)
            if attempt < self.max_retries - 1:
                delay = self.base_delay * (2**attempt)
                logger.debug(f"Retry {attempt + 1} for {ticker}, waiting {delay:.1f}s")
                time.sleep(delay)

        # Fall back to Yahoo Finance
        logger.info(f"Falling back to Yahoo Finance for {ticker} ({timeframe})")
        return YFinanceProvider().get_ohlcv(ticker, timeframe, start, end, cancellation_check)


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

        if exchange == "HK":
            # AllTick HK format: remove leading zeros and add .HK
            # e.g., 0700.HK -> 700.HK
            symbol = normalized.replace(".HK", "").lstrip("0") or "0"
            return f"{symbol}.HK", True
        elif exchange == "US":
            return normalized, True
        elif exchange == "CN":
            # A-shares: .SS -> .SH for AllTick
            if ".SS" in normalized:
                return normalized.replace(".SS", ".SH"), True
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
            bars_per_minute = {"5m": 1 / 5, "15m": 1 / 15, "30m": 1 / 30}[timeframe]
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
                "adjust_type": 0,  # Ex-rights adjustment
            },
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
                    logger.warning(
                        f"AllTick API error for {alltick_code}: ret={ret_code}, msg={error_msg}"
                    )
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

                        records.append(
                            {
                                "datetime": pd.to_datetime(ts, unit="s"),
                                "open": float(k.get("open_price", 0)),
                                "high": float(k.get("high_price", 0)),
                                "low": float(k.get("low_price", 0)),
                                "close": float(k.get("close_price", 0)),
                                "volume": int(float(k.get("volume", 0))),  # volume is string
                            }
                        )
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
                logger.warning(f"AllTick HTTP error (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)
            except Exception as e:
                logger.error(f"AllTick error for {alltick_code}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2**attempt)

        return pd.DataFrame(columns=OHLCV_COLUMNS)


class DatabentoProvider(DataProvider):
    """
    Databento data provider for futures (ES, MES, NQ, etc.).

    Uses Databento API: https://databento.com
    Best for CME futures with professional-grade data quality.
    Pay-as-you-go pricing starting at < $0.01 per query.

    Data Loading Priority:
    1. Local DBN files in data/databento/ (fastest, no API calls)
    2. Databento API (if API key configured)
    3. YFinance fallback (if all else fails)

    API Reference: https://databento.com/docs
    """

    # Map common futures symbols to Databento instrument IDs
    # Format: Yahoo symbol -> (Databento dataset, instrument pattern)
    FUTURES_MAP = {
        # Micro E-mini S&P 500
        "MES=F": ("GLBX.MDP3", "MES"),
        # E-mini S&P 500
        "ES=F": ("GLBX.MDP3", "ES"),
        # Micro E-mini Nasdaq-100
        "MNQ=F": ("GLBX.MDP3", "MNQ"),
        # E-mini Nasdaq-100
        "NQ=F": ("GLBX.MDP3", "NQ"),
        # Micro E-mini Dow Jones
        "MYM=F": ("GLBX.MDP3", "MYM"),
        # E-mini Dow Jones
        "YM=F": ("GLBX.MDP3", "YM"),
        # Micro E-mini Russell 2000
        "M2K=F": ("GLBX.MDP3", "M2K"),
        # E-mini Russell 2000
        "RTY=F": ("GLBX.MDP3", "RTY"),
        # Crude Oil
        "CL=F": ("GLBX.MDP3", "CL"),
        # Gold
        "GC=F": ("GLBX.MDP3", "GC"),
        # Silver
        "SI=F": ("GLBX.MDP3", "SI"),
        # Natural Gas
        "NG=F": ("GLBX.MDP3", "NG"),
        # 10-Year Treasury Note
        "ZN=F": ("GLBX.MDP3", "ZN"),
        # 30-Year Treasury Bond
        "ZB=F": ("GLBX.MDP3", "ZB"),
        # Euro FX
        "6E=F": ("GLBX.MDP3", "6E"),
    }

    # Map timeframe to Databento schema names
    SCHEMA_MAP = {
        "1m": "ohlcv-1m",
        "5m": "ohlcv-1m",  # Aggregate from 1m
        "15m": "ohlcv-1m",
        "30m": "ohlcv-1m",
        "1h": "ohlcv-1h",
        "2h": "ohlcv-1h",  # Aggregate from 1h
        "1d": "ohlcv-1d",
    }

    def __init__(self):
        """Initialize with API key and local data directory."""
        self.api_key = get_databento_api_key()
        self.rate_limiter = RateLimiter(calls_per_second=1.0)  # Conservative rate limit
        self.max_retries = 3
        self._client = None

        # Local DBN file directory
        import pathlib

        self.data_dir = pathlib.Path(__file__).parent.parent.parent / "data" / "databento"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "databento"

    @property
    def is_available(self) -> bool:
        """Check if Databento is available (local files OR API key)."""
        return self.api_key is not None or self._has_local_data()

    def _has_local_data(self) -> bool:
        """Check if we have any local DBN files."""
        if not self.data_dir.exists():
            return False
        return any(self.data_dir.glob("*.dbn.zst")) or any(self.data_dir.glob("*.dbn"))

    def _get_client(self):
        """Get or create Databento client (lazy initialization)."""
        if self._client is None and self.api_key:
            try:
                # Suppress SWIG warnings right before import
                import warnings

                warnings.filterwarnings("ignore", category=DeprecationWarning)
                import databento as db

                self._client = db.Historical(key=self.api_key)
            except ImportError:
                logger.error("databento package not installed. Install with: pip install databento")
                return None
        return self._client

    def _get_front_month_symbol(self, base_symbol: str) -> str:
        """
        Get the front-month continuous contract symbol for Databento.

        Databento uses patterns like "ES.c.0" for front month.
        """
        # Use continuous contract notation
        return f"{base_symbol}.c.0"

    def _find_local_dbn_files(
        self, base_symbol: str, schema: str, start: datetime, end: datetime
    ) -> list:
        """
        Find local DBN files that cover the requested date range.

        Supports two file naming conventions:
        1. Custom format: {SYMBOL}_{SCHEMA}_{START}_{END}.dbn.zst
           Example: MES_ohlcv-1d_2025-01-01_2025-12-31.dbn.zst

        2. Databento download format (daily split): glbx-mdp3-{YYYYMMDD}.{schema}.dbn.zst
           Located in subfolders: GLBX-{date}-{id}/
           Example: GLBX-20260122-ABC123/glbx-mdp3-20250121.ohlcv-1m.dbn.zst
        """
        matching_files = []

        # Convert date range to date objects for comparison
        start_date = start.date() if hasattr(start, "date") else start
        end_date = end.date() if hasattr(end, "date") else end

        # ===== Format 1: Custom format (symbol_schema_start_end.dbn.zst) =====
        patterns = [
            f"{base_symbol}_{schema}_*.dbn.zst",
            f"{base_symbol}_{schema}_*.dbn",
        ]

        for pattern in patterns:
            for file_path in self.data_dir.glob(pattern):
                try:
                    name = file_path.stem.replace(".dbn", "")
                    parts = name.split("_")
                    if len(parts) >= 4:
                        file_start = datetime.strptime(parts[-2], "%Y-%m-%d")
                        file_end = datetime.strptime(parts[-1], "%Y-%m-%d")

                        if file_start.date() <= end_date and file_end.date() >= start_date:
                            matching_files.append(
                                {
                                    "path": file_path,
                                    "start": file_start,
                                    "end": file_end,
                                }
                            )
                except (ValueError, IndexError) as e:
                    logger.debug(f"Could not parse filename {file_path}: {e}")
                    continue

        # ===== Format 2: Databento download format (daily split in subfolders) =====
        # Pattern: GLBX-*/glbx-mdp3-{YYYYMMDD}.{schema}.dbn.zst
        # Note: These files contain data for ALL symbols in the dataset
        daily_pattern = f"*/glbx-mdp3-*.{schema}.dbn.zst"

        for file_path in self.data_dir.glob(daily_pattern):
            try:
                # Parse date from filename: glbx-mdp3-20250121.ohlcv-1m.dbn.zst
                name = file_path.name
                # Extract date part
                date_part = name.split(".")[0].split("-")[-1]  # Get '20250121'
                file_date = datetime.strptime(date_part, "%Y%m%d").date()

                # Check if this file's date is within requested range
                if start_date <= file_date <= end_date:
                    matching_files.append(
                        {
                            "path": file_path,
                            "start": datetime.combine(file_date, datetime.min.time()),
                            "end": datetime.combine(file_date, datetime.max.time()),
                        }
                    )
            except (ValueError, IndexError) as e:
                logger.debug(f"Could not parse filename {file_path}: {e}")
                continue

        # Sort by start date and remove duplicates
        matching_files.sort(key=lambda x: x["start"])

        # Remove duplicate files (same date from different folders)
        seen_dates = set()
        unique_files = []
        for f in matching_files:
            date_key = f["start"].date()
            if date_key not in seen_dates:
                seen_dates.add(date_key)
                unique_files.append(f)

        return unique_files

    def get_available_dates(self, schema: str) -> set:
        """
        Get set of dates that have local data for a given schema.

        Returns:
            Set of datetime.date objects for which we have local data
        """
        available_dates = set()

        # Check custom format files
        for file_path in self.data_dir.glob(f"*_{schema}_*.dbn.zst"):
            try:
                name = file_path.stem.replace(".dbn", "")
                parts = name.split("_")
                if len(parts) >= 4:
                    file_start = datetime.strptime(parts[-2], "%Y-%m-%d").date()
                    file_end = datetime.strptime(parts[-1], "%Y-%m-%d").date()
                    # Add all dates in range
                    current = file_start
                    while current <= file_end:
                        available_dates.add(current)
                        current += timedelta(days=1)
            except (ValueError, IndexError):
                continue

        # Check Databento daily format files
        for file_path in self.data_dir.glob(f"*/glbx-mdp3-*.{schema}.dbn.zst"):
            try:
                name = file_path.name
                date_part = name.split(".")[0].split("-")[-1]
                file_date = datetime.strptime(date_part, "%Y%m%d").date()
                available_dates.add(file_date)
            except (ValueError, IndexError):
                continue

        return available_dates

    def _load_from_local_dbn(
        self,
        base_symbol: str,
        schema: str,
        start: datetime,
        end: datetime,
        target_price: float = None,
        trade_date: date = None,
    ) -> pd.DataFrame:
        """
        Load data from local DBN files.

        Args:
            base_symbol: Base symbol (e.g., 'MES')
            schema: Data schema (e.g., 'ohlcv-1m')
            start: Start datetime
            end: End datetime
            target_price: Optional target price to help select correct contract
                         (useful when multiple contracts exist at different price levels)
            trade_date: Optional trade date to use for contract price comparison
                       (more accurate than using highest volume day)

        Returns:
            DataFrame with OHLCV data, or empty DataFrame if no data found
        """
        # Suppress SWIG warnings before import
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import databento as db

        matching_files = self._find_local_dbn_files(base_symbol, schema, start, end)

        if not matching_files:
            return pd.DataFrame()

        all_dfs = []

        for file_info in matching_files:
            try:
                logger.info(f"📂 Loading local DBN: {file_info['path'].name}")

                # Read DBN file
                data = db.DBNStore.from_file(str(file_info["path"]))
                df = data.to_df()

                if not df.empty:
                    all_dfs.append(df)

            except Exception as e:
                logger.warning(f"Error reading {file_info['path']}: {e}")
                continue

        if not all_dfs:
            return pd.DataFrame()

        # Combine all dataframes
        combined = pd.concat(all_dfs, ignore_index=False)

        # Filter to single contract (DBN files contain all contracts)
        # We filter out spreads and select appropriate contract
        if "symbol" in combined.columns:
            # Filter out spreads (symbols with '-' like MESZ5-MESH6)
            combined = combined[~combined["symbol"].str.contains("-", na=False)]

            if not combined.empty and len(combined["symbol"].unique()) > 1:
                unique_symbols = combined["symbol"].unique()
                selected_contract = None

                # If target price provided, select contract with closest price
                # Use trade_date if provided (most accurate), else fall back to heuristics
                if target_price is not None:
                    best_match = None
                    best_diff = float("inf")

                    # Determine reference date for price comparison
                    if trade_date is not None:
                        # Use the specific trade date (most accurate)
                        reference_date = trade_date
                    else:
                        # Fallback: find the date with most trading volume
                        volume_by_date = combined.groupby(combined.index.date)["volume"].sum()
                        if not volume_by_date.empty:
                            reference_date = volume_by_date.idxmax()
                        else:
                            all_dates = sorted(set(combined.index.date))
                            reference_date = all_dates[len(all_dates) // 2] if all_dates else None

                    if reference_date:
                        reference_day_data = combined[combined.index.date == reference_date]
                    else:
                        reference_day_data = combined

                    if reference_day_data.empty:
                        reference_day_data = combined  # Fallback to all data

                    logger.debug(
                        f"Using {reference_date} for contract selection (target={target_price})"
                    )

                    for symbol in unique_symbols:
                        sym_data = reference_day_data[reference_day_data["symbol"] == symbol]
                        if sym_data.empty:
                            # If no data on reference day for this symbol, use all data
                            sym_data = combined[combined["symbol"] == symbol]

                        avg_price = sym_data["close"].mean()
                        diff = abs(avg_price - target_price)

                        logger.debug(
                            f"Contract {symbol}: avg_price={avg_price:.2f}, diff from target={diff:.2f}"
                        )

                        if diff < best_diff:
                            best_diff = diff
                            best_match = symbol

                    if best_match and best_diff < target_price * 0.05:  # Within 5%
                        selected_contract = best_match
                        logger.info(
                            f"Selected contract {selected_contract} (closest to target price {target_price} on {reference_date}, diff={best_diff:.2f})"
                        )

                # Fallback: select by highest volume
                if selected_contract is None:
                    symbol_volumes = combined.groupby("symbol")["volume"].sum()
                    if not symbol_volumes.empty:
                        selected_contract = symbol_volumes.idxmax()
                        logger.info(f"Selected contract {selected_contract} (highest volume)")

                if selected_contract:
                    original_count = len(combined)
                    combined = combined[combined["symbol"] == selected_contract]
                    logger.debug(
                        f"Filtered to {selected_contract}: {original_count} -> {len(combined)} bars"
                    )

        # Remove duplicates
        combined = combined[~combined.index.duplicated(keep="first")]

        # Sort by datetime
        combined = combined.sort_index()

        # Filter to requested date range
        # Handle timezone conversion carefully
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        index_tz = getattr(combined.index, "tz", None)

        if index_tz is not None:
            # Index is timezone-aware, make start/end match
            if start_ts.tz is None:
                start_ts = start_ts.tz_localize(index_tz)
            else:
                start_ts = start_ts.tz_convert(index_tz)

            if end_ts.tz is None:
                end_ts = end_ts.tz_localize(index_tz)
            else:
                end_ts = end_ts.tz_convert(index_tz)
        else:
            # Index is timezone-naive, make start/end naive too
            if start_ts.tz is not None:
                # Convert to UTC then remove timezone info using replace
                start_ts = start_ts.tz_convert("UTC").replace(tzinfo=None)
            if end_ts.tz is not None:
                end_ts = end_ts.tz_convert("UTC").replace(tzinfo=None)

        combined = combined[(combined.index >= start_ts) & (combined.index <= end_ts)]

        if not combined.empty:
            logger.info(f"✅ Loaded {len(combined)} bars from local DBN files")

        return combined

    def _process_databento_df(self, df: pd.DataFrame, timeframe: str, schema: str) -> pd.DataFrame:
        """
        Process a Databento DataFrame into standard OHLCV format.

        Handles column renaming and resampling as needed.
        """
        if df.empty:
            return df

        # Reset index if needed (DBN files have datetime as index)
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index()
            # The index becomes a column, rename it
            if "ts_event" in df.columns:
                df = df.rename(columns={"ts_event": "datetime"})
            elif df.columns[0] not in ["datetime", "open", "high", "low", "close", "volume"]:
                df = df.rename(columns={df.columns[0]: "datetime"})

        # Rename Databento columns to standard format
        column_map = {
            "ts_event": "datetime",
            "ts_recv": "datetime",
        }
        df = df.rename(columns=column_map)

        # If datetime column doesn't exist, try first column
        if "datetime" not in df.columns and len(df.columns) > 0:
            if df.columns[0] not in ["open", "high", "low", "close", "volume"]:
                df = df.rename(columns={df.columns[0]: "datetime"})

        # Ensure we have required columns
        required_cols = ["datetime", "open", "high", "low", "close", "volume"]
        available_cols = [c for c in required_cols if c in df.columns]

        if len(available_cols) < 6:
            logger.warning(f"Databento data missing columns. Have: {list(df.columns)}")
            return pd.DataFrame()

        df = df[available_cols].copy()

        # Resample if needed (e.g., 5m from 1m data)
        if timeframe in ["5m", "15m", "30m"] and schema == "ohlcv-1m":
            df = df.set_index("datetime")
            resample_map = {"5m": "5min", "15m": "15min", "30m": "30min"}
            df = (
                df.resample(resample_map[timeframe])
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )
            df = df.reset_index()

        if timeframe == "2h" and schema == "ohlcv-1h":
            df = df.set_index("datetime")
            df = (
                df.resample("2h")
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                        "volume": "sum",
                    }
                )
                .dropna()
            )
            df = df.reset_index()

        return df

    def get_ohlcv(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime,
        end: datetime,
        cancellation_check: callable = None,
        target_price: float = None,
        trade_date: date = None,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data from Databento.

        Loading priority:
        1. Local DBN files (fastest, no API calls needed)
        2. Databento API (if API key configured)
        3. YFinance fallback (if all else fails)

        Args:
            target_price: Optional price to help select correct contract
                         (e.g., entry price to distinguish between different contract months)
            trade_date: Optional trade date to use for contract price comparison
        """
        # Check for cancellation
        if cancellation_check and cancellation_check():
            logger.info(f"⏹️ Cancelled fetching {ticker}")
            return pd.DataFrame(columns=OHLCV_COLUMNS)

        # Normalize ticker to get the Yahoo format
        normalized_ticker, exchange = normalize_ticker(ticker)

        # Check if this is a supported futures contract
        if normalized_ticker not in self.FUTURES_MAP:
            logger.warning(
                f"Futures {normalized_ticker} not in Databento map, falling back to YFinance"
            )
            return YFinanceProvider().get_ohlcv(ticker, timeframe, start, end, cancellation_check)

        dataset, base_symbol = self.FUTURES_MAP[normalized_ticker]

        # Get the schema for this timeframe
        schema = self.SCHEMA_MAP.get(timeframe, "ohlcv-1d")

        # ===== PRIORITY 1: Try local DBN files first =====
        local_df = self._load_from_local_dbn(
            base_symbol, schema, start, end, target_price=target_price, trade_date=trade_date
        )

        if not local_df.empty:
            # Process and return local data
            df = self._process_databento_df(local_df, timeframe, schema)
            if not df.empty:
                return self._normalize_dataframe(df)

        # ===== PRIORITY 2: Try Databento API =====
        if not self.api_key:
            logger.warning(
                f"No local DBN files for {base_symbol} and no DATABENTO_API_KEY configured. "
                f"Falling back to YFinance. "
                f"TIP: Run 'python scripts/download_databento.py --symbols {base_symbol}' to download data."
            )
            return YFinanceProvider().get_ohlcv(ticker, timeframe, start, end, cancellation_check)

        client = self._get_client()
        if client is None:
            logger.warning("Failed to initialize Databento client, falling back to YFinance")
            return YFinanceProvider().get_ohlcv(ticker, timeframe, start, end, cancellation_check)

        for attempt in range(self.max_retries):
            if cancellation_check and cancellation_check():
                logger.info(f"⏹️ Cancelled fetching {ticker}")
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            try:
                self.rate_limiter.wait()

                if cancellation_check and cancellation_check():
                    logger.info(f"⏹️ Cancelled fetching {ticker}")
                    return pd.DataFrame(columns=OHLCV_COLUMNS)

                # Use [ROOT].FUT format for futures parent symbol
                parent_symbol = f"{base_symbol}.FUT"
                logger.info(f"📡 Fetching {parent_symbol} from Databento ({schema})...")

                # Fetch data using Databento API
                # Use stype_in="parent" with [ROOT].FUT format to get all contracts
                data = client.timeseries.get_range(
                    dataset=dataset,
                    symbols=[parent_symbol],
                    stype_in="parent",
                    schema=schema,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )

                # Convert to pandas DataFrame
                df = data.to_df()

                if df.empty:
                    logger.warning(f"No data returned from Databento for {base_symbol}")
                    return YFinanceProvider().get_ohlcv(
                        ticker, timeframe, start, end, cancellation_check
                    )

                # Process the DataFrame
                df = self._process_databento_df(df, timeframe, schema)

                if df.empty:
                    return YFinanceProvider().get_ohlcv(
                        ticker, timeframe, start, end, cancellation_check
                    )

                logger.info(f"✅ Databento API: got {len(df)} bars for {base_symbol}")
                return self._normalize_dataframe(df)

            except ImportError:
                logger.error("databento package not installed. Install with: pip install databento")
                return YFinanceProvider().get_ohlcv(
                    ticker, timeframe, start, end, cancellation_check
                )
            except Exception as e:
                error_str = str(e).lower()
                if "rate" in error_str or "limit" in error_str or "429" in error_str:
                    delay = 2**attempt
                    logger.warning(
                        f"Databento rate limit for {ticker}, waiting {delay}s "
                        f"(attempt {attempt + 1}/{self.max_retries})"
                    )
                    time.sleep(delay)
                elif "dataset_unavailable_range" in error_str or "subscription" in error_str:
                    # Free tier doesn't have recent data - fall back immediately
                    logger.info(
                        f"Databento free tier doesn't include recent data for {ticker}. "
                        f"Falling back to YFinance. Consider subscribing at databento.com/pricing#cme"
                    )
                    return YFinanceProvider().get_ohlcv(
                        ticker, timeframe, start, end, cancellation_check
                    )
                else:
                    logger.warning(f"Databento error for {ticker}: {e}")
                    if attempt == self.max_retries - 1:
                        logger.info("Falling back to YFinance for futures data")
                        return YFinanceProvider().get_ohlcv(
                            ticker, timeframe, start, end, cancellation_check
                        )
                    time.sleep(1)

        # All retries exhausted
        logger.warning(f"Databento failed for {ticker}, falling back to YFinance")
        return YFinanceProvider().get_ohlcv(ticker, timeframe, start, end, cancellation_check)


# Cache for provider instances (avoid recreating on every call)
_provider_cache: dict[str, DataProvider] = {}


def get_provider(name: str | None = None) -> DataProvider:
    """
    Get a data provider instance (cached).

    Args:
        name: Provider name ('yfinance', 'polygon', 'alpaca', 'alltick', 'databento').
              Defaults to 'yfinance'.

    Returns:
        DataProvider instance (cached)
    """
    provider_name = name or "yfinance"

    # If Polygon is selected but no API key is configured, fall back to yfinance.
    if provider_name == "polygon" and not get_polygon_api_key():
        provider_name = "yfinance"

    # Return cached instance if available
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    providers = {
        "yfinance": YFinanceProvider,
        "polygon": PolygonProvider,
        "alpaca": AlpacaProvider,
        "alltick": AllTickProvider,
        "databento": DatabentoProvider,
        "tencent_hk": TencentHKProvider,
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

    Provider routing:
    - Futures (ES, MES, NQ, etc.): Databento (primary), YFinance fallback
    - US stocks: Polygon (primary), with YFinance fallback on error
    - HK stocks: Tencent (primary), AllTick if configured, YFinance fallback
    - Other international (CN, JP, UK): YFinance

    Args:
        ticker: Stock symbol

    Returns:
        Best DataProvider for this ticker (cached)
    """
    _, exchange = normalize_ticker(ticker)

    # For futures, prefer Databento if available, otherwise YFinance
    if exchange == "FUTURES":
        databento_key = get_databento_api_key()
        if databento_key:
            return get_provider("databento")
        logger.info(f"No DATABENTO_API_KEY configured, using YFinance for {ticker}")
        return get_provider("yfinance")

    # For HK stocks: Tencent (primary) > AllTick > YFinance (fallback)
    if exchange == "HK":
        # Primary: Tencent HK (free, no rate limits, reliable for daily data)
        # Note: Tencent internally falls back to Yahoo for intraday data
        return get_provider("tencent_hk")

    # For other international stocks, use YFinance
    if exchange in ["CN", "JP", "UK"]:
        return get_provider("yfinance")

    # For US stocks, prefer configured provider when possible.
    provider_name = settings.data_provider
    if provider_name == "polygon" and get_polygon_api_key():
        return get_provider("polygon")
    return get_provider("yfinance")


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
