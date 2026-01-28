"""
Pydantic schemas for API request/response validation.

Provides standardized request and response models for the trading coach API.
"""

from typing import Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field


# ==================== RESPONSE MODELS ====================


class APIResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, data: Any = None, message: str = None) -> "APIResponse":
        """Create a successful response."""
        return cls(success=True, data=data, message=message)

    @classmethod
    def fail(cls, error: str, data: Any = None) -> "APIResponse":
        """Create a failure response."""
        return cls(success=False, error=error, data=data)


class PaginatedResponse(BaseModel):
    """Response with pagination info."""

    success: bool = True
    data: List[Any]
    page: int = 1
    per_page: int = 50
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


# ==================== TRADE MODELS ====================


class TradeDirection(str, Enum):
    """Trade direction enum."""

    long = "long"
    short = "short"


class TradeOutcome(str, Enum):
    """Trade outcome enum."""

    win = "win"
    loss = "loss"
    breakeven = "breakeven"


class TradeUpdate(BaseModel):
    """Model for updating trade fields."""

    # Basic trade info
    ticker: Optional[str] = None
    direction: Optional[TradeDirection] = None
    size: Optional[float] = None

    # Prices
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    # Times (as strings, parsed on server)
    entry_time: Optional[str] = None
    exit_time: Optional[str] = None

    # Currency
    currency: Optional[str] = None
    exchange_rate_to_usd: Optional[float] = None

    # Notes and analysis
    notes: Optional[str] = None
    entry_reason: Optional[str] = None
    exit_reason: Optional[str] = None
    setup_type: Optional[str] = None
    signal_reason: Optional[str] = None

    # Risk management
    r_target: Optional[float] = None
    was_signal_present: Optional[str] = None
    entry_tp_far: Optional[str] = None

    # Strategy
    strategy_id: Optional[int] = None

    class Config:
        use_enum_values = True


class TradeNotesUpdate(BaseModel):
    """Model for updating just trade notes."""

    notes: Optional[str] = None
    entry_reason: Optional[str] = None
    lesson_learned: Optional[str] = None
    mistake: Optional[str] = None


class TradeStrategyUpdate(BaseModel):
    """Model for updating trade strategy assignment."""

    strategy_id: Optional[int] = None


# ==================== STRATEGY MODELS ====================


class StrategyCreate(BaseModel):
    """Model for creating a strategy."""

    name: str = Field(..., min_length=1, max_length=100)
    category: str = Field(default="with_trend")
    description: Optional[str] = None
    is_active: bool = True


class StrategyUpdate(BaseModel):
    """Model for updating a strategy."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    category: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None


class StrategyMerge(BaseModel):
    """Model for merging strategies."""

    source_id: int
    target_id: int


# ==================== SETTINGS MODELS ====================


class CacheSettings(BaseModel):
    """Cache configuration settings."""

    enable_review_cache: bool = True
    auto_regenerate: bool = False


class CandleSettings(BaseModel):
    """Candle count settings by timeframe."""

    daily: int = Field(default=60, ge=10, le=500)
    h2: int = Field(default=60, ge=10, le=500)
    m5: int = Field(default=80, ge=10, le=500)


class PromptSettings(BaseModel):
    """LLM prompt settings."""

    prompt_type: str
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None


# ==================== IMPORT MODELS ====================


class RobinhoodCredentials(BaseModel):
    """Robinhood login credentials."""

    username: str
    password: str
    days: int = Field(default=30, ge=1, le=365)


# ==================== ANALYSIS MODELS ====================


class BulkAnalysisRequest(BaseModel):
    """Request for bulk trade analysis."""

    trade_ids: List[int]
    force: bool = False


# ==================== HELPER FUNCTIONS ====================


def success_response(data: Any = None, message: str = None) -> dict:
    """Create a standardized success response dict."""
    response = {"success": True}
    if data is not None:
        response["data"] = data
    if message:
        response["message"] = message
    return response


def error_response(error: str, data: Any = None, status_code: int = 400) -> tuple[dict, int]:
    """Create a standardized error response dict with status code."""
    response = {"success": False, "error": error}
    if data is not None:
        response["data"] = data
    return response, status_code
