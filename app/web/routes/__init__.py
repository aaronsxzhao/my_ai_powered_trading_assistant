"""
Route modules for AI Trading Coach API.

This package contains modular route definitions split by functionality.
"""

# Import all routers for easy access
from .system import router as system_router
from .trades import router as trades_router
from .strategies import router as strategies_router
from .materials import router as materials_router
from .tickers import router as tickers_router
from .imports import router as imports_router
from .reports import router as reports_router
from .settings import router as settings_router
from .auth import router as auth_router

__all__ = [
    "system_router",
    "trades_router",
    "strategies_router",
    "materials_router",
    "tickers_router",
    "imports_router",
    "reports_router",
    "settings_router",
    "auth_router",
]
