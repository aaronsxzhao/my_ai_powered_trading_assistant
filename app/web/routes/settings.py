"""
Settings API routes.

Handles prompt customization, candle count configuration, and cache settings.
Supports per-user settings with Supabase, falls back to global settings locally.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse

from app.web.dependencies import require_write_auth, get_current_user, get_user_id

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _is_supabase_enabled() -> bool:
    """Check if Supabase is configured."""
    try:
        from app.db.supabase_client import is_supabase_configured
        return is_supabase_configured()
    except ImportError:
        return False


@router.get("")
async def get_all_settings(request: Request, user = Depends(get_current_user)):
    """Get all settings for the current user."""
    user_id = get_user_id(user)
    
    if _is_supabase_enabled():
        from app.config_prompts_db import get_all_user_settings
        settings = get_all_user_settings(user_id)
    else:
        from app.config_prompts import load_settings
        settings = load_settings()
    
    return JSONResponse(settings)


@router.post("/candles")
async def update_candle_settings(request: Request, user = Depends(get_current_user)):
    """Update candle count settings."""
    user_id = get_user_id(user)
    data = await request.json()
    
    if _is_supabase_enabled():
        from app.config_prompts_db import save_user_candle_counts
        
        success = save_user_candle_counts(
            user_id,
            daily=data.get("daily"),
            hourly=data.get("hourly"),
            five_min=data.get("5min"),
        )
    else:
        from app.config_prompts import update_candles
        
        success = update_candles(
            daily=data.get("daily"),
            hourly=data.get("hourly"),
            five_min=data.get("5min"),
        )

    if success:
        return JSONResponse({"message": "Candle settings saved"})
    raise HTTPException(status_code=500, detail="Failed to save settings")


@router.get("/prompts/{prompt_type}")
async def get_prompt(
    prompt_type: str, 
    request: Request,
    is_user_prompt: bool = False,
    user = Depends(get_current_user)
):
    """Get a specific prompt for the current user."""
    user_id = get_user_id(user)
    
    if _is_supabase_enabled():
        from app.config_prompts_db import get_user_editable_prompt
        prompt_text = get_user_editable_prompt(user_id, prompt_type, is_user_prompt)
    else:
        from app.config_prompts import get_editable_prompt
        prompt_text = get_editable_prompt(prompt_type, is_user_prompt)
    
    return JSONResponse({
        "prompt_type": prompt_type,
        "is_user_prompt": is_user_prompt,
        "text": prompt_text,
    })


@router.post("/prompts")
async def update_prompt_settings(request: Request, user = Depends(get_current_user)):
    """Update a specific prompt (system or user)."""
    user_id = get_user_id(user)
    data = await request.json()
    
    prompt_type = data.get("prompt_type")
    prompt_text = data.get("prompt_text")
    is_user_prompt = data.get("is_user_prompt", False)

    if not prompt_type or prompt_text is None:
        raise HTTPException(status_code=400, detail="Missing prompt_type or prompt_text")

    if _is_supabase_enabled():
        from app.config_prompts_db import save_user_system_prompt, save_user_user_prompt
        
        if is_user_prompt:
            success = save_user_user_prompt(user_id, prompt_type, prompt_text)
        else:
            success = save_user_system_prompt(user_id, prompt_type, prompt_text)
    else:
        from app.config_prompts import update_prompt
        success = update_prompt(prompt_type, prompt_text, is_user_prompt=is_user_prompt)

    prompt_label = "User" if is_user_prompt else "System"
    if success:
        return JSONResponse({"message": f"{prompt_label} prompt '{prompt_type}' saved"})
    raise HTTPException(status_code=500, detail="Failed to save prompt")


@router.delete("/prompts/{prompt_type}")
async def reset_prompt(
    prompt_type: str,
    request: Request,
    is_user_prompt: bool = False,
    user = Depends(get_current_user)
):
    """Reset a specific prompt to default."""
    user_id = get_user_id(user)
    
    if _is_supabase_enabled():
        from app.config_prompts_db import reset_user_prompt
        success = reset_user_prompt(user_id, prompt_type, is_user_prompt)
        
        if success:
            return JSONResponse({"message": f"Prompt '{prompt_type}' reset to default"})
        raise HTTPException(status_code=500, detail="Failed to reset prompt")
    else:
        # For local storage, we can't easily reset individual prompts
        # Just return success and let the UI handle it
        return JSONResponse({"message": f"Prompt '{prompt_type}' reset to default"})


@router.get("/cache")
async def get_cache_settings_api(request: Request, user = Depends(get_current_user)):
    """Get cache settings."""
    user_id = get_user_id(user)
    
    if _is_supabase_enabled():
        from app.config_prompts_db import get_user_cache_settings
        settings = get_user_cache_settings(user_id)
    else:
        from app.config_prompts import get_cache_settings
        settings = get_cache_settings()

    return JSONResponse(settings)


@router.post("/cache")
async def update_cache_settings_api(request: Request, user = Depends(get_current_user)):
    """Update cache settings."""
    user_id = get_user_id(user)
    data = await request.json()
    
    if _is_supabase_enabled():
        from app.config_prompts_db import save_user_cache_settings
        
        success = save_user_cache_settings(
            user_id,
            enable_review_cache=data.get("enable_review_cache"),
            auto_regenerate=data.get("auto_regenerate"),
        )
    else:
        from app.config_prompts import update_cache_settings
        
        success = update_cache_settings(
            enable_review_cache=data.get("enable_review_cache"),
            auto_regenerate=data.get("auto_regenerate"),
        )

    if success:
        return JSONResponse({"message": "Cache settings saved"})
    raise HTTPException(status_code=500, detail="Failed to save cache settings")


@router.post("/reset")
async def reset_settings(request: Request, user = Depends(get_current_user)):
    """Reset all settings to defaults for current user."""
    user_id = get_user_id(user)
    
    if _is_supabase_enabled():
        from app.db.supabase_client import get_service_client
        
        try:
            client = get_service_client()
            # Delete all user settings
            client.table("user_settings").delete().eq(
                "user_id", user_id
            ).execute()
            return JSONResponse({"message": "Settings reset to defaults"})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to reset settings: {e}")
    else:
        from app.config_prompts import DEFAULT_SETTINGS, save_settings
        
        success = save_settings(DEFAULT_SETTINGS.copy())

        if success:
            return JSONResponse({"message": "Settings reset to defaults"})
        raise HTTPException(status_code=500, detail="Failed to reset settings")


@router.get("/candles")
async def get_candle_settings(request: Request, user = Depends(get_current_user)):
    """Get candle count settings."""
    user_id = get_user_id(user)
    
    if _is_supabase_enabled():
        from app.config_prompts_db import get_user_candle_counts
        settings = get_user_candle_counts(user_id)
    else:
        from app.config_prompts import load_settings, DEFAULT_SETTINGS
        all_settings = load_settings()
        settings = all_settings.get("candles", DEFAULT_SETTINGS.get("candles", {}))
    
    return JSONResponse(settings)
