"""
Configuration management for Brooks Trading Coach.

Loads settings from config.yaml and environment variables.
"""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.yaml"
TICKERS_FILE = PROJECT_ROOT / "tickers.txt"
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
IMPORTS_DIR = PROJECT_ROOT / "imports"
MATERIALS_DIR = PROJECT_ROOT / "materials"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
IMPORTS_DIR.mkdir(exist_ok=True)
MATERIALS_DIR.mkdir(exist_ok=True)


def load_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    return get_default_config()


def get_default_config() -> dict[str, Any]:
    """Return default configuration if config.yaml doesn't exist."""
    return {
        "tickers": ["SPY", "QQQ"],
        "timezone": "America/New_York",
        "data": {
            "provider": "yfinance",
            "cache_enabled": True,
            "cache_ttl_minutes": 15,
        },
        "timeframes": {
            "daily": {"lookback_days": 252},
            "intraday_2h": {"lookback_days": 60},
            "intraday_5m": {"lookback_days": 5},
        },
        "analysis": {
            "ema_period": 20,
            "sma_periods": [10, 20, 50, 200],
            "atr_period": 14,
            "swing_lookback": 5,
            "trend_ema_slope_threshold": 0.001,
            "trend_closes_above_ema_pct": 0.6,
            "overlap_ratio_range_threshold": 0.5,
        },
        "risk": {
            "max_daily_loss_r": 3.0,
            "max_losing_streak": 3,
            "default_risk_per_trade_pct": 1.0,
            "warn_after_consecutive_losses": 2,
            "no_trade_after_daily_loss_r": 4.0,
        },
        "reports": {
            "output_dir": "outputs",
            "format": "markdown",
            "include_charts": True,
            "chart_style": "classic",
        },
        "llm": {
            "enabled": True,
            "model": "claude-sonnet-4.5",
            "max_tokens": 1500,
            "temperature": 0.3,
        },
    }


def save_config(config: dict[str, Any]) -> None:
    """Save configuration to YAML file."""
    with open(CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_tickers_from_file() -> list[str]:
    """
    Load tickers from tickers.txt file.

    Returns:
        List of ticker symbols
    """
    if not TICKERS_FILE.exists():
        return ["SPY", "QQQ"]

    tickers = []
    with open(TICKERS_FILE, "r") as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith("#"):
                tickers.append(line.upper())

    return tickers if tickers else ["SPY", "QQQ"]


def save_tickers_to_file(tickers: list[str]) -> None:
    """
    Save tickers to tickers.txt file.

    Args:
        tickers: List of ticker symbols
    """
    with open(TICKERS_FILE, "w") as f:
        f.write("# Brooks Trading Coach - Favorite Tickers\n")
        f.write("# =========================================\n")
        f.write("# Add one ticker per line. Lines starting with # are comments.\n")
        f.write("# Edit this file directly to add/remove tickers.\n")
        f.write("#\n\n")
        for ticker in tickers:
            f.write(f"{ticker.upper()}\n")


def add_ticker_to_file(ticker: str) -> None:
    """Add a ticker to tickers.txt."""
    tickers = load_tickers_from_file()
    ticker = ticker.upper()
    if ticker not in tickers:
        tickers.append(ticker)
        save_tickers_to_file(tickers)


def remove_ticker_from_file(ticker: str) -> bool:
    """Remove a ticker from tickers.txt. Returns True if removed."""
    tickers = load_tickers_from_file()
    ticker = ticker.upper()
    if ticker in tickers:
        tickers.remove(ticker)
        save_tickers_to_file(tickers)
        return True
    return False


class Settings:
    """Application settings singleton."""

    _instance = None
    _config: dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config = load_config()
        return cls._instance

    def reload(self) -> None:
        """Reload configuration from file."""
        self._config = load_config()

    @property
    def tickers(self) -> list[str]:
        """Load tickers from tickers.txt file (not config.yaml)."""
        return load_tickers_from_file()

    @property
    def timezone(self) -> str:
        return self._config.get("timezone", "America/New_York")

    @property
    def data_provider(self) -> str:
        # Check environment first, then config file
        env_provider = os.getenv("DATA_PROVIDER")
        if env_provider:
            return env_provider.lower()
        return self._config.get("data", {}).get("provider", "yfinance")

    @property
    def cache_enabled(self) -> bool:
        return self._config.get("data", {}).get("cache_enabled", True)

    @property
    def cache_ttl_minutes(self) -> int:
        return self._config.get("data", {}).get("cache_ttl_minutes", 15)

    @property
    def ema_period(self) -> int:
        return self._config.get("analysis", {}).get("ema_period", 20)

    @property
    def atr_period(self) -> int:
        return self._config.get("analysis", {}).get("atr_period", 14)

    @property
    def swing_lookback(self) -> int:
        return self._config.get("analysis", {}).get("swing_lookback", 5)

    @property
    def max_daily_loss_r(self) -> float:
        return self._config.get("risk", {}).get("max_daily_loss_r", 3.0)

    @property
    def max_losing_streak(self) -> int:
        return self._config.get("risk", {}).get("max_losing_streak", 3)

    @property
    def llm_enabled(self) -> bool:
        return self._config.get("llm", {}).get("enabled", False)

    @property
    def llm_model(self) -> str:
        return self._config.get("llm", {}).get("model", "gpt-4o")

    @property
    def outputs_dir(self) -> Path:
        return OUTPUTS_DIR

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-notation key."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def add_ticker(self, ticker: str) -> None:
        """Add a ticker to tickers.txt."""
        add_ticker_to_file(ticker)

    def remove_ticker(self, ticker: str) -> bool:
        """Remove a ticker from tickers.txt."""
        return remove_ticker_from_file(ticker)


# Global settings instance
settings = Settings()


# Environment variable helpers
def get_env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.getenv(key, default)


# LLM API Configuration (LiteLLM Proxy - OpenAI compatible)
def get_llm_api_key() -> str | None:
    """
    Get LLM API key.

    No default/fallback key is provided. For backwards compatibility, this
    also checks common alternate env var names.
    """
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    key = key.strip() if key else None
    return key or None


def get_llm_base_url() -> str:
    """Get LLM base URL."""
    return os.getenv("LLM_BASE_URL", "https://duet-litellm-api.winktech.net/v1")


def get_llm_model() -> str:
    """Get LLM model name."""
    return os.getenv("LLM_MODEL", "claude-sonnet-4.5")


# Backwards compatibility aliases
def get_anthropic_api_key() -> str | None:
    """Alias for get_llm_api_key."""
    return get_llm_api_key()


def get_openai_api_key() -> str | None:
    """Alias for get_llm_api_key."""
    return get_llm_api_key()


def get_database_url() -> str:
    """Get database URL from environment or default."""
    default_db = f"sqlite:///{DATA_DIR}/trades.db"
    return os.getenv("DATABASE_URL", default_db)


def get_polygon_api_key() -> str | None:
    """Get Polygon.io API key from environment."""
    return os.getenv("POLYGON_API_KEY")


def get_llm_workers() -> int:
    """Get number of concurrent LLM workers from environment."""
    try:
        return int(os.getenv("LLM_WORKERS", "20"))
    except ValueError:
        return 20


def get_app_api_key() -> str | None:
    """
    Get application API key for authentication.

    When set, destructive API endpoints (DELETE, sensitive POST/PATCH)
    require this key in the X-API-Key header.

    Returns None if no key is configured (auth disabled).
    """
    key = os.getenv("APP_API_KEY", "").strip()
    return key if key else None


def is_auth_enabled() -> bool:
    """Check if API key authentication is enabled."""
    return get_app_api_key() is not None
