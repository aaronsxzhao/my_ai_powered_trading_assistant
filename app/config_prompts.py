"""
Configurable prompts and settings for the trading coach.

These can be modified via the web UI Settings page.
"""

import yaml
from pathlib import Path
from typing import Optional
import logging

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Settings file path
SETTINGS_FILE = PROJECT_ROOT / "settings.yaml"

# Default settings
DEFAULT_SETTINGS = {
    # Candle counts for different timeframes
    "candles": {
        "daily": 30,       # Number of daily candles to fetch
        "hourly": 48,      # Number of hourly candles (2 days)
        "5min": 78,        # Number of 5-min candles (1 trading day = 6.5 hours)
    },
    
    # LLM Prompts
    "prompts": {
        "trade_classification": """You are an expert Al Brooks price action analyst. Your job is to classify trades into Brooks-style setups.

Available strategies:
WITH-TREND:
- breakout_pullback_long/short: Entry on pullback after breakout
- second_entry_buy/sell: 2nd attempt after failed first pullback entry
- trend_resumption_long/short: Continuation after pause in trend
- measured_move_long/short: Entry at measured move target

COUNTERTREND:
- failed_breakout_long/short: Fade after breakout fails
- wedge_reversal_long/short: Reversal at 3-push wedge
- double_top_short / double_bottom_long: Classic reversal patterns
- climax_reversal_long/short: Reversal after exhaustion move

TRADING RANGE:
- range_fade_high/low: Fade at range extremes
- range_scalp_long/short: Quick scalp within range

SPECIAL:
- trend_from_open_long/short: Strong directional move from open
- opening_reversal_long/short: Reversal of opening move
- gap_fill_long/short: Trading gap fills

Respond in JSON format:
{
    "strategy_name": "the_strategy_name",
    "strategy_category": "with_trend|countertrend|trading_range|special",
    "confidence": "low|medium|high",
    "reasoning": "Brief explanation of why this classification"
}""",

        "trade_analysis": """You are Al Brooks, the legendary price action trader and author. 
You are coaching a trader on their completed trade using your price action methodology.

Your analysis must include:
1. CONTEXT: What was the likely market regime? (trend up, trend down, trading range)
2. ALWAYS-IN: What was the always-in direction? Did the trade align?
3. SETUP QUALITY: Was this a high-probability setup or low-probability?
4. TRADER'S EQUATION: Did the math make sense? (probability × reward vs risk)
5. ERRORS: What Brooks-style errors were made, if any?
6. WHAT WAS GOOD: Positive aspects of the trade
7. WHAT WAS FLAWED: Areas for improvement
8. RULE FOR NEXT TIME: One specific, actionable rule

Be direct and specific. Use Brooks terminology (always-in, 2nd entry, breakout pullback, etc.)

Respond in JSON format:
{
    "regime": "trend_up|trend_down|trading_range|unclear",
    "always_in": "long|short|neutral",
    "trade_aligned_with_context": true|false,
    "setup_classification": "strategy_name",
    "setup_quality": "good|marginal|poor",
    "probability_assessment": "HIGH|MEDIUM|LOW - explanation",
    "risk_reward_assessment": "explanation",
    "errors": ["list of errors if any"],
    "what_was_good": "positive aspects",
    "what_was_flawed": "areas for improvement", 
    "rule_for_next_time": "specific actionable rule",
    "overall_grade": "A|B|C|D|F",
    "coaching_summary": "2-3 sentence summary"
}""",

        "market_context": """You are an expert Al Brooks price action analyst providing a premarket briefing.

Analyze the provided OHLCV data and provide:
1. REGIME: Current market regime (strong trend, weak trend, trading range, breakout)
2. ALWAYS-IN DIRECTION: The dominant direction based on price action
3. KEY LEVELS: Important support/resistance levels
4. BEST SETUPS: What setups to look for today
5. AVOID LIST: What to avoid today
6. PLAN A vs PLAN B: Primary expectation vs alternative scenario

Use Brooks terminology and be specific about price levels.

Respond in JSON format:
{
    "regime": "description",
    "always_in": "long|short|neutral",
    "key_levels": {"support": [], "resistance": [], "magnets": []},
    "best_setups": ["list of setups to look for"],
    "avoid_list": ["what to avoid"],
    "plan_a": "primary scenario",
    "plan_b": "alternative if plan A fails",
    "risk_notes": "any special risk considerations"
}"""
    }
}


def load_settings() -> dict:
    """Load settings from file, or return defaults if not found."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = yaml.safe_load(f) or {}
            # Merge with defaults to ensure all keys exist
            merged = DEFAULT_SETTINGS.copy()
            if 'candles' in settings:
                merged['candles'].update(settings['candles'])
            if 'prompts' in settings:
                merged['prompts'].update(settings['prompts'])
            return merged
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
            return DEFAULT_SETTINGS.copy()
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> bool:
    """Save settings to file."""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            yaml.dump(settings, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"Settings saved to {SETTINGS_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving settings: {e}")
        return False


def get_candle_count(timeframe: str) -> int:
    """Get the configured candle count for a timeframe."""
    settings = load_settings()
    return settings.get('candles', {}).get(timeframe, DEFAULT_SETTINGS['candles'].get(timeframe, 30))


def get_materials_content() -> str:
    """Load training materials content for prompt enhancement."""
    from app.config import MATERIALS_DIR
    
    content_parts = []
    
    if not MATERIALS_DIR.exists():
        return ""
    
    for f in sorted(MATERIALS_DIR.iterdir()):
        if f.is_file() and not f.name.startswith('.'):
            try:
                ext = f.suffix.lower()
                
                if ext == '.pdf':
                    # Try to extract PDF text
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(str(f))
                        text = ""
                        for page in doc:
                            text += page.get_text()
                        doc.close()
                        if text.strip():
                            content_parts.append(f"=== {f.name} ===\n{text[:8000]}")  # Limit per file
                    except ImportError:
                        logger.warning(f"PyMuPDF not installed, skipping PDF: {f.name}")
                    except Exception as e:
                        logger.error(f"Error reading PDF {f.name}: {e}")
                        
                elif ext in {'.txt', '.md'}:
                    text = f.read_text(encoding='utf-8', errors='ignore')
                    if text.strip():
                        content_parts.append(f"=== {f.name} ===\n{text[:8000]}")
                        
            except Exception as e:
                logger.error(f"Error reading material {f.name}: {e}")
    
    if content_parts:
        # Limit total materials to 20k chars
        full_content = "\n\n".join(content_parts)
        return full_content[:20000]
    return ""


def get_prompt(prompt_type: str, include_materials: bool = True) -> str:
    """Get a configured prompt by type, optionally with training materials."""
    settings = load_settings()
    base_prompt = settings.get('prompts', {}).get(prompt_type, DEFAULT_SETTINGS['prompts'].get(prompt_type, ''))
    
    if include_materials:
        materials = get_materials_content()
        if materials:
            materials_section = f"""

=== TRAINING MATERIALS (Use for context) ===
{materials}
=== END TRAINING MATERIALS ===
"""
            return base_prompt + materials_section
    
    return base_prompt


def update_candles(daily: Optional[int] = None, hourly: Optional[int] = None, five_min: Optional[int] = None) -> bool:
    """Update candle count settings."""
    settings = load_settings()
    if daily is not None:
        settings['candles']['daily'] = daily
    if hourly is not None:
        settings['candles']['hourly'] = hourly
    if five_min is not None:
        settings['candles']['5min'] = five_min
    return save_settings(settings)


def update_prompt(prompt_type: str, prompt_text: str) -> bool:
    """Update a specific prompt."""
    settings = load_settings()
    settings['prompts'][prompt_type] = prompt_text
    return save_settings(settings)
