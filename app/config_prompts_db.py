"""
Per-user prompt configuration using Supabase database.

Stores user-customized prompts in the user_settings table.
Falls back to default prompts from config_prompts.py.
"""

import logging
from typing import Optional, Any

from app.config_prompts import (
    DEFAULT_SETTINGS,
    PROTECTED_JSON_SCHEMAS,
    get_materials_content,
)

logger = logging.getLogger(__name__)


def _is_supabase_enabled() -> bool:
    """Check if Supabase is configured."""
    try:
        from app.db.supabase_client import is_supabase_configured
        return is_supabase_configured()
    except ImportError:
        return False


def _get_supabase_client():
    """Get Supabase service client."""
    from app.db.supabase_client import get_service_client
    return get_service_client()


# ==================== USER SETTINGS ====================


def get_user_setting(user_id: str, setting_key: str) -> Optional[Any]:
    """
    Get a user setting from the database.
    
    Args:
        user_id: User's UUID
        setting_key: Setting key (e.g., 'prompt_trade_analysis')
    
    Returns:
        Setting value or None if not found
    """
    if not _is_supabase_enabled():
        return None
    
    try:
        client = _get_supabase_client()
        result = client.table("user_settings").select(
            "setting_value"
        ).eq("user_id", user_id).eq("setting_key", setting_key).single().execute()
        
        if result.data:
            return result.data["setting_value"]
        return None
    except Exception as e:
        logger.warning(f"Failed to get user setting {setting_key}: {e}")
        return None


def set_user_setting(user_id: str, setting_key: str, value: Any) -> bool:
    """
    Set a user setting in the database.
    
    Args:
        user_id: User's UUID
        setting_key: Setting key
        value: Setting value (will be stored as JSONB)
    
    Returns:
        True if successful
    """
    if not _is_supabase_enabled():
        return False
    
    try:
        client = _get_supabase_client()
        client.table("user_settings").upsert({
            "user_id": user_id,
            "setting_key": setting_key,
            "setting_value": value,
        }).execute()
        return True
    except Exception as e:
        logger.error(f"Failed to set user setting {setting_key}: {e}")
        return False


def delete_user_setting(user_id: str, setting_key: str) -> bool:
    """Delete a user setting (revert to default)."""
    if not _is_supabase_enabled():
        return False
    
    try:
        client = _get_supabase_client()
        client.table("user_settings").delete().eq(
            "user_id", user_id
        ).eq("setting_key", setting_key).execute()
        return True
    except Exception:
        return False


# ==================== PROMPTS ====================


def get_user_system_prompt(
    user_id: str, 
    prompt_type: str, 
    include_materials: bool = True
) -> str:
    """
    Get a system prompt for a user.
    
    Checks for user customization first, then falls back to default.
    Automatically appends protected JSON schema if applicable.
    
    Args:
        user_id: User's UUID
        prompt_type: Prompt type ('trade_analysis', 'trade_classification', 'market_context')
        include_materials: Whether to include training materials content
    
    Returns:
        Complete system prompt
    """
    # Try to get user's custom prompt
    setting = get_user_setting(user_id, f"system_prompt_{prompt_type}")
    
    if setting and setting.get("text"):
        base_prompt = setting["text"]
    else:
        # Fall back to default
        base_prompt = DEFAULT_SETTINGS.get("system_prompts", {}).get(prompt_type, "")
    
    # Append protected JSON schema
    json_schema = PROTECTED_JSON_SCHEMAS.get(prompt_type, "")
    full_prompt = base_prompt + json_schema
    
    # Optionally append materials
    if include_materials:
        if _is_supabase_enabled():
            # For Supabase, materials are retrieved via RAG (not appended here)
            # The LLM analyzer should call get_relevant_materials_supabase separately
            pass
        else:
            # For local, append materials content
            materials = get_materials_content()
            if materials:
                full_prompt += f"""

=== TRAINING MATERIALS (Use for context) ===
{materials}
=== END TRAINING MATERIALS ===
"""
    
    return full_prompt


def get_user_user_prompt(user_id: str, prompt_type: str) -> str:
    """
    Get a user prompt template for a user.
    
    Args:
        user_id: User's UUID
        prompt_type: Prompt type
    
    Returns:
        User prompt template
    """
    setting = get_user_setting(user_id, f"user_prompt_{prompt_type}")
    
    if setting and setting.get("text"):
        return setting["text"]
    
    # Fall back to default
    return DEFAULT_SETTINGS.get("user_prompts", {}).get(prompt_type, "")


def save_user_system_prompt(user_id: str, prompt_type: str, text: str) -> bool:
    """
    Save a user's custom system prompt.
    
    Args:
        user_id: User's UUID
        prompt_type: Prompt type
        text: Prompt text (without JSON schema)
    
    Returns:
        True if successful
    """
    # Strip any JSON format section that might be present
    from app.config_prompts import _strip_json_format
    clean_text = _strip_json_format(text)
    
    return set_user_setting(user_id, f"system_prompt_{prompt_type}", {"text": clean_text})


def save_user_user_prompt(user_id: str, prompt_type: str, text: str) -> bool:
    """
    Save a user's custom user prompt template.
    """
    return set_user_setting(user_id, f"user_prompt_{prompt_type}", {"text": text})


def reset_user_prompt(user_id: str, prompt_type: str, is_user_prompt: bool = False) -> bool:
    """
    Reset a user's prompt to default (delete customization).
    """
    prefix = "user_prompt_" if is_user_prompt else "system_prompt_"
    return delete_user_setting(user_id, f"{prefix}{prompt_type}")


def get_user_editable_prompt(user_id: str, prompt_type: str, is_user_prompt: bool = False) -> str:
    """
    Get only the editable part of a prompt (without JSON schema).
    For use in settings UI.
    """
    if is_user_prompt:
        return get_user_user_prompt(user_id, prompt_type)
    
    setting = get_user_setting(user_id, f"system_prompt_{prompt_type}")
    
    if setting and setting.get("text"):
        return setting["text"]
    
    # Fall back to default (already without JSON schema)
    from app.config_prompts import get_editable_prompt
    return get_editable_prompt(prompt_type, is_user_prompt)


# ==================== CACHE SETTINGS ====================


def get_user_cache_settings(user_id: str) -> dict:
    """Get user's cache settings."""
    setting = get_user_setting(user_id, "cache_settings")
    
    defaults = DEFAULT_SETTINGS.get("cache", {})
    
    if setting:
        return {
            "enable_review_cache": setting.get(
                "enable_review_cache", 
                defaults.get("enable_review_cache", True)
            ),
            "auto_regenerate": setting.get(
                "auto_regenerate",
                defaults.get("auto_regenerate", False)
            ),
        }
    
    return defaults


def save_user_cache_settings(
    user_id: str, 
    enable_review_cache: Optional[bool] = None, 
    auto_regenerate: Optional[bool] = None
) -> bool:
    """Update user's cache settings."""
    current = get_user_cache_settings(user_id)
    
    if enable_review_cache is not None:
        current["enable_review_cache"] = enable_review_cache
    if auto_regenerate is not None:
        current["auto_regenerate"] = auto_regenerate
    
    return set_user_setting(user_id, "cache_settings", current)


# ==================== CANDLE SETTINGS ====================


def get_user_candle_counts(user_id: str) -> dict:
    """Get user's candle count settings."""
    setting = get_user_setting(user_id, "candle_counts")
    
    defaults = DEFAULT_SETTINGS.get("candles", {})
    
    if setting:
        return {
            "daily": setting.get("daily", defaults.get("daily", 30)),
            "hourly": setting.get("hourly", defaults.get("hourly", 48)),
            "5min": setting.get("5min", defaults.get("5min", 78)),
        }
    
    return defaults


def save_user_candle_counts(
    user_id: str,
    daily: Optional[int] = None,
    hourly: Optional[int] = None,
    five_min: Optional[int] = None
) -> bool:
    """Update user's candle count settings."""
    current = get_user_candle_counts(user_id)
    
    if daily is not None:
        current["daily"] = daily
    if hourly is not None:
        current["hourly"] = hourly
    if five_min is not None:
        current["5min"] = five_min
    
    return set_user_setting(user_id, "candle_counts", current)


# ==================== ALL SETTINGS ====================


def get_all_user_settings(user_id: str) -> dict:
    """Get all user settings as a dictionary."""
    return {
        "cache": get_user_cache_settings(user_id),
        "candles": get_user_candle_counts(user_id),
        "system_prompts": {
            "trade_classification": get_user_editable_prompt(user_id, "trade_classification", False),
            "trade_analysis": get_user_editable_prompt(user_id, "trade_analysis", False),
            "market_context": get_user_editable_prompt(user_id, "market_context", False),
        },
        "user_prompts": {
            "trade_classification": get_user_editable_prompt(user_id, "trade_classification", True),
            "trade_analysis": get_user_editable_prompt(user_id, "trade_analysis", True),
            "market_context": get_user_editable_prompt(user_id, "market_context", True),
        },
    }
