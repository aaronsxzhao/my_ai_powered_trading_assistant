"""
Currency exchange rate service.

Fetches historical exchange rates from frankfurter.app (free, uses ECB data).
"""

import logging
from datetime import date, datetime
from typing import Optional
import httpx

logger = logging.getLogger(__name__)

# Base currency is USD for our app
BASE_CURRENCY = "USD"

# frankfurter.app API (free, no key needed)
FRANKFURTER_API = "https://api.frankfurter.app"

# Cache for rates to avoid repeated API calls
_rate_cache: dict[str, dict[str, float]] = {}


def get_exchange_rate(
    target_currency: str,
    trade_date: Optional[date] = None,
) -> Optional[float]:
    """
    Get exchange rate from USD to target currency for a specific date.

    Args:
        target_currency: Target currency code (e.g., 'HKD', 'EUR')
        trade_date: Date for historical rate (defaults to today)

    Returns:
        Exchange rate (1 USD = X target_currency), or None if failed
    """
    if target_currency == "USD":
        return 1.0

    if trade_date is None:
        trade_date = date.today()

    # Ensure we have a date object
    if isinstance(trade_date, datetime):
        trade_date = trade_date.date()

    date_str = trade_date.isoformat()
    cache_key = f"{date_str}_{target_currency}"

    # Check cache first
    if cache_key in _rate_cache:
        return _rate_cache[cache_key].get(target_currency)

    try:
        # frankfurter.app uses EUR as base, so we need to convert
        # First get EUR/USD and EUR/target, then calculate USD/target
        url = f"{FRANKFURTER_API}/{date_str}"
        params = {"from": "USD", "to": target_currency}

        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        rates = data.get("rates", {})
        rate = rates.get(target_currency)

        if rate:
            _rate_cache[cache_key] = {target_currency: rate}
            logger.info(f"Fetched rate for {date_str}: 1 USD = {rate} {target_currency}")
            return rate
        else:
            logger.warning(f"No rate found for {target_currency} on {date_str}")
            return None

    except httpx.HTTPStatusError as e:
        # If date is too recent or weekend, try previous day
        if e.response.status_code == 404:
            logger.info(f"No data for {date_str}, trying previous day")
            from datetime import timedelta

            prev_date = trade_date - timedelta(days=1)
            if prev_date.year >= 2000:  # Don't go too far back
                return get_exchange_rate(target_currency, prev_date)
        logger.error(f"HTTP error fetching rate: {e}")
        return None
    except Exception as e:
        logger.error(f"Error fetching exchange rate: {e}")
        return None


def get_multiple_rates(
    currencies: list[str],
    trade_date: Optional[date] = None,
) -> dict[str, float]:
    """
    Get exchange rates for multiple currencies at once.

    Args:
        currencies: List of currency codes
        trade_date: Date for historical rates

    Returns:
        Dict of currency -> rate
    """
    if trade_date is None:
        trade_date = date.today()

    if isinstance(trade_date, datetime):
        trade_date = trade_date.date()

    # Filter out USD
    target_currencies = [c for c in currencies if c != "USD"]

    if not target_currencies:
        return {"USD": 1.0}

    date_str = trade_date.isoformat()

    try:
        url = f"{FRANKFURTER_API}/{date_str}"
        params = {"from": "USD", "to": ",".join(target_currencies)}

        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        rates = data.get("rates", {})
        rates["USD"] = 1.0

        # Cache all rates
        for currency, rate in rates.items():
            cache_key = f"{date_str}_{currency}"
            _rate_cache[cache_key] = {currency: rate}

        return rates

    except Exception as e:
        logger.error(f"Error fetching multiple rates: {e}")
        return {"USD": 1.0}


# Supported currencies with their symbols
SUPPORTED_CURRENCIES = {
    "USD": {"symbol": "$", "name": "US Dollar"},
    "HKD": {"symbol": "HK$", "name": "Hong Kong Dollar"},
    "EUR": {"symbol": "€", "name": "Euro"},
    "GBP": {"symbol": "£", "name": "British Pound"},
    "JPY": {"symbol": "¥", "name": "Japanese Yen"},
    "CNY": {"symbol": "¥", "name": "Chinese Yuan"},
    "CAD": {"symbol": "C$", "name": "Canadian Dollar"},
    "AUD": {"symbol": "A$", "name": "Australian Dollar"},
    "CHF": {"symbol": "CHF", "name": "Swiss Franc"},
    "SGD": {"symbol": "S$", "name": "Singapore Dollar"},
    "KRW": {"symbol": "₩", "name": "South Korean Won"},
    "TWD": {"symbol": "NT$", "name": "Taiwan Dollar"},
}


def get_currency_symbol(currency_code: str) -> str:
    """Get the symbol for a currency code."""
    return SUPPORTED_CURRENCIES.get(currency_code, {}).get("symbol", currency_code)
