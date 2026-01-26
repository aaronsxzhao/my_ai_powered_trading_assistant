"""
Route modules for Brooks Trading Coach API.

This package contains modular route definitions split by functionality.
"""

from fastapi import APIRouter

# Import all routers for easy access
from .trades import router as trades_router
from .strategies import router as strategies_router
from .materials import router as materials_router
from .auth import router as auth_router

__all__ = [
    'trades_router',
    'strategies_router', 
    'materials_router',
    'auth_router',
]
