"""
Settings API routes.

Handles prompt customization, candle count configuration, and cache settings.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.web.dependencies import require_write_auth

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.post("/candles", dependencies=[require_write_auth])
async def update_candle_settings(request: Request):
    """Update candle count settings."""
    from app.config_prompts import update_candles

    data = await request.json()
    success = update_candles(
        daily=data.get("daily"),
        hourly=data.get("hourly"),
        five_min=data.get("5min"),
    )

    if success:
        return JSONResponse({"message": "Candle settings saved"})
    raise HTTPException(status_code=500, detail="Failed to save settings")


@router.post("/prompts", dependencies=[require_write_auth])
async def update_prompt_settings(request: Request):
    """Update a specific prompt (system or user)."""
    from app.config_prompts import update_prompt

    data = await request.json()
    prompt_type = data.get("prompt_type")
    prompt_text = data.get("prompt_text")
    is_user_prompt = data.get("is_user_prompt", False)

    if not prompt_type or prompt_text is None:
        raise HTTPException(status_code=400, detail="Missing prompt_type or prompt_text")

    success = update_prompt(prompt_type, prompt_text, is_user_prompt=is_user_prompt)

    prompt_label = "User" if is_user_prompt else "System"
    if success:
        return JSONResponse({"message": f"{prompt_label} prompt '{prompt_type}' saved"})
    raise HTTPException(status_code=500, detail="Failed to save prompt")


@router.get("/cache")
async def get_cache_settings_api():
    """Get cache settings."""
    from app.config_prompts import get_cache_settings

    return JSONResponse(get_cache_settings())


@router.post("/cache", dependencies=[require_write_auth])
async def update_cache_settings_api(request: Request):
    """Update cache settings."""
    from app.config_prompts import update_cache_settings

    data = await request.json()
    success = update_cache_settings(
        enable_review_cache=data.get("enable_review_cache"),
        auto_regenerate=data.get("auto_regenerate"),
    )

    if success:
        return JSONResponse({"message": "Cache settings saved"})
    raise HTTPException(status_code=500, detail="Failed to save cache settings")


@router.post("/reset", dependencies=[require_write_auth])
async def reset_settings():
    """Reset all settings to defaults."""
    from app.config_prompts import DEFAULT_SETTINGS, save_settings

    success = save_settings(DEFAULT_SETTINGS.copy())

    if success:
        return JSONResponse({"message": "Settings reset to defaults"})
    raise HTTPException(status_code=500, detail="Failed to reset settings")
