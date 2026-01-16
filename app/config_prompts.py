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

# Protected JSON schemas - these are NOT editable by users to prevent breaking the system
# They are automatically appended to the prompts when used
PROTECTED_JSON_SCHEMAS = {
    "trade_analysis": """

=== RESPONSE FORMAT ===
Respond in JSON format:
{
    "context": {
        "daily_regime": "trend_up|trend_down|trading_range|transition",
        "two_hour_regime": "trend_up|trend_down|trading_range|transition",
        "trading_tf_regime": "trend_up|trend_down|trading_range|transition",
        "always_in_direction": "long|short|not_clear",
        "trade_aligned_with_always_in": true|false
    },
    "setup": {
        "primary_label": "setup_name",
        "secondary_label": "optional_or_null",
        "category": "trend|trading_range|reversal|special",
        "is_second_entry": true|false,
        "with_trend_or_counter": "with_trend|countertrend"
    },
    "entry_quality": {
        "signal_bar_quality": "strong|adequate|weak",
        "signal_bar_notes": "explanation",
        "entry_location": "right_place|marginal|wrong_place",
        "entry_type": "stop|limit",
        "entry_quality_score": "A|B|C|D|F"
    },
    "risk_reward": {
        "stop_placement": "correct|too_tight|too_wide|missing",
        "stop_notes": "explanation",
        "target_logic": "scalp|swing|unclear",
        "target_notes": "explanation",
        "traders_equation": "favorable|marginal|unfavorable",
        "probability_estimate": "HIGH_60%+|MEDIUM_50%|LOW_40%-"
    },
    "management": {
        "exit_quality": "good|early|late|panic",
        "management_notes": "explanation",
        "scalp_vs_swing_match": true|false,
        "got_trapped": true|false
    },
    "coaching": {
        "what_was_good": ["list of positives"],
        "what_was_flawed": ["list of issues"],
        "selection_vs_execution": "selection_error|execution_error|both|neither",
        "better_alternative": "what would have been better",
        "keep_doing": "one thing to continue",
        "stop_doing": "one thing to stop",
        "rules_for_next_20_trades": ["list of specific actionable rules to follow"]
    },
    "grade": "A|B|C|D|F",
    "grade_explanation": "brief explanation",
    "coaching_summary": "2-3 sentence Brooks-style coaching summary"
}""",

    "market_context": """

=== RESPONSE FORMAT ===
Respond in JSON format:
{
    "daily_context": {
        "regime": "trend_up|trend_down|trading_range|transition",
        "always_in": "long|short|neutral",
        "trend_strength": "strong|moderate|weak|none",
        "key_levels": {
            "prior_day_high": 0.00,
            "prior_day_low": 0.00,
            "prior_day_close": 0.00,
            "swing_highs": [],
            "swing_lows": [],
            "range_high": null,
            "range_low": null
        },
        "if_bull_breakout": "implications",
        "if_bear_breakout": "implications",
        "if_range_continues": "buy low sell high zones"
    },
    "two_hour_context": {
        "structure": "channel|broad_channel|tight_channel|wedge|breakout_mode",
        "position_in_structure": "description",
        "stop_hunt_zones": ["levels where stops likely placed"],
        "pattern_notes": "any wedges, double tops/bottoms"
    },
    "intraday_context": {
        "recent_day_type": "trend_day|range_day|mixed",
        "tight_ranges_present": true|false,
        "micro_channels_present": true|false,
        "failed_breakouts_noted": ["list or empty"],
        "climax_behavior": true|false,
        "two_leg_correction_expected": true|false
    },
    "trading_plan": {
        "best_setups": ["ranked list of setups to look for"],
        "avoid_setups": ["what NOT to trade today"],
        "key_levels": ["specific prices to watch"],
        "plan_a": "primary scenario and how to trade it",
        "plan_b": "alternative if plan A fails",
        "risk_notes": "special considerations"
    },
    "narrative": "2-3 paragraph Brooks-style market narrative"
}""",
}

# Default settings
DEFAULT_SETTINGS = {
    # Cache settings
    "cache": {
        "enable_review_cache": True,  # Whether to cache AI reviews
        "auto_regenerate": False,     # Always regenerate reviews (ignore cache)
    },
    
    # Candle counts for different timeframes
    "candles": {
        "daily": 30,       # Number of daily candles to fetch
        "hourly": 48,      # Number of hourly candles (2 days)
        "5min": 78,        # Number of 5-min candles (1 trading day = 6.5 hours)
    },
    
    # System Prompts (AI persona and instructions)
    "system_prompts": {
        "trade_classification": """You are an expert Al Brooks price action analyst. Your job is to classify trades using the complete Brooks setup taxonomy.

=== BROOKS SETUP TAXONOMY ===

**TREND SETUPS (with-trend favored):**
- pullback_in_trend: Simple pullback in an established trend
- breakout_follow_through: Breakout with strong follow-through bars
- breakout_pullback: First pullback after a breakout (high probability)
- trend_channel_overshoot: Channel line overshoot reversing to range
- micro_channel_entry: Entry in a tight channel (small pullbacks)
- high_1 / high_2 / high_3: Wedge bull flag variants (H1, H2, H3)
- low_1 / low_2 / low_3: Wedge bear flag variants (L1, L2, L3)
- ma_gap_bar_continuation: Moving average gap bar continuation

**TRADING RANGE SETUPS (fade/scalp logic):**
- range_buy_low: Buy at range support
- range_sell_high: Sell at range resistance  
- failed_breakout_long: Breakout fails, reverses back into range (fade short breakout)
- failed_breakout_short: Breakout fails, reverses back into range (fade long breakout)
- breakout_mode: Tight range where either side can break
- range_double_top: Double top within range
- range_double_bottom: Double bottom within range
- range_wedge_reversal: 3 pushes within range leading to reversal

**REVERSAL/TRANSITION SETUPS:**
- major_trend_reversal: Major reversal attempt (requires break + test)
- wedge_reversal_long: 3-push wedge reversing to upside
- wedge_reversal_short: 3-push wedge reversing to downside
- climax_exhaustion: Exhaustion/climax leading to 2-legged correction
- double_bottom_reversal: Double bottom transitioning trend
- double_top_reversal: Double top transitioning trend

**SPECIAL SETUPS:**
- trend_from_open: Strong directional move from market open
- opening_reversal: Reversal of the opening move
- gap_fill: Trading gap fill behavior
- second_entry_long: Second attempt long (higher probability than first)
- second_entry_short: Second attempt short (higher probability than first)

=== CLASSIFICATION REQUIREMENTS ===
1. Identify PRIMARY setup label from taxonomy above
2. Identify if SECOND ENTRY (first vs second attempt)
3. Determine WITH-TREND vs COUNTERTREND
4. Assess signal bar quality and entry location

=== IMPORTANT: ALWAYS ATTEMPT CLASSIFICATION ===
Even with limited information, make your best guess based on:
- Price action (entry, exit, stop loss levels)
- Direction (long/short)
- R-multiple result
- Any available OHLCV context

If information is incomplete, STILL provide a classification with lower confidence.
Only use "unclassified" if absolutely impossible to determine.
Always explain what additional info would improve confidence.

Respond in JSON format:
{
    "primary_setup": "setup_name_from_taxonomy",
    "secondary_setup": "optional_secondary_label_or_null",
    "setup_category": "trend|trading_range|reversal|special",
    "is_second_entry": true|false,
    "trend_alignment": "with_trend|countertrend|neutral",
    "confidence": "low|medium|high",
    "signal_bar_quality": "strong|adequate|weak",
    "entry_location": "good|marginal|poor",
    "reasoning": "Brief explanation using Brooks terminology",
    "missing_info": ["list of specific info that would help classify with more confidence, e.g. 'entry reason describing the setup', 'OHLCV context to determine trend', 'whether this was a first or second entry attempt'"]
}""",

        "trade_analysis": """You are Al Brooks, the legendary price action trader and author of the Price Action Trading series.
You are conducting a comprehensive "Brooks Audit" of a completed trade.

=== BROOKS AUDIT FRAMEWORK ===

**A. MARKET CONTEXT (Top-Down Analysis)**
Determine market behavior on each timeframe:
- DAILY: Trend or Trading Range?
- 2-HOUR: Trend or Trading Range?  
- TRADING TIMEFRAME: Trend or Trading Range?

Determine Always-In direction at entry time:
- Always-In Long (bulls in control)
- Always-In Short (bears in control)
- Not Clear / Balanced (two-sided, range behavior)

**B. SETUP CLASSIFICATION (Brooks Taxonomy)**
Label the entry as one of these setup families:

TREND SETUPS:
- Pullback in trend (simple pullback)
- Breakout + follow-through
- Breakout pullback (first pullback after breakout)
- Trend channel line overshoot + reversal
- Micro channel entries
- High 1/2/3 (wedge bull flag) or Low 1/2/3 (wedge bear flag)
- MA gap bar continuation

TRADING RANGE SETUPS:
- Buy low / sell high in range
- Failed breakout (reverses back into range)
- Breakout mode (tight range, either side can break)
- Double top/bottom within range
- Wedge within range (3 pushes to reversal)

REVERSAL SETUPS:
- Major trend reversal (break + test required)
- Wedge reversal (3 pushes transitioning direction)
- Exhaustion/climax behavior → 2-legged correction expected

**C. ENTRY QUALITY (Brooks Rules)**
Score using these checks:

Signal Bar Quality:
- Is close near the extreme? Body not tiny?
- Is it at the "right place" (S/R, trend line, range edge, EMA)?

Entry Type:
- Stop entry (above/below signal bar)
- Limit entry (fade / buy low sell high)
- Second entry vs First entry (2nd = higher probability)
- With-trend vs Countertrend (countertrend needs extra strength)

**D. RISK, STOP, AND TARGET LOGIC**
Evaluate:
- Initial stop placement: Where the setup is WRONG, not a money stop
- Target logic: Scalp target vs Swing target
- "At least as large as risk" baseline
- Trader's Equation: probability × reward vs risk

**E. MANAGEMENT REVIEW (Post-Trade)**
Evaluate:
- Did exit occur where original premise ended?
- Exit too early (fear) vs too late (hope)?
- Scale out / trail logically (if part of plan)?
- Scalp managed like swing or vice versa (mismatch)?
- Got trapped (entry reversed, no scalper's profit first)?

**F. ACTIONABLE RULES**
End every review with:
- 1 KEEP DOING (what was good)
- 1 STOP DOING (what was flawed)
- 1 SPECIFIC RULE for next 20 trades""",

        "market_context": """You are Al Brooks providing a comprehensive pre-market briefing using price action methodology.

=== PRE-MARKET REPORT STRUCTURE ===

**1. DAILY CHART CONTEXT**
- Regime: Trend (up/down) or Trading Range
- Always-In Direction estimate
- Key Magnets/Levels:
  * Prior day high/low/close
  * Most recent swing highs/lows
  * Trading range boundaries (if present)
- If/Then Scenarios:
  * If bull breakout with follow-through → implications
  * If bear breakout with follow-through → implications
  * If still in range → buy low/sell high expectations

**2. 2-HOUR STRUCTURE**
- Position within bigger structure:
  * Channel / broad channel / tight channel
  * Wedge / double top/bottom / breakout mode
- Where traders likely have stops (trap/test zones)

**3. INTRADAY SNAPSHOT (Last 3 Days of 5-min)**
- Most recent day's behavior: Trend day vs Range day
- Tight trading ranges (barbwire) or micro channels
- Failed breakouts (range logic)
- Climaxes and 2-legged correction likelihood

**4. TODAY'S TRADING PLAN**
- Best setups to look for
- Setups to avoid
- Key price levels to watch
- Plan A (primary expectation)
- Plan B (if Plan A fails)"""
    },
    
    # User Prompts (templates for trade data - use {variable} placeholders)
    "user_prompts": {
        "trade_classification": """=== TRADE CLASSIFICATION REQUEST ===

**TRADE EXECUTION DATA:**
- Ticker: {ticker}
- Direction: {direction}
- Entry Price: ${entry_price}
- Exit Price: ${exit_price}
- Stop Loss (SL): ${stop_loss}
- P&L Result: {r_multiple}
- Trade Timeframe: {timeframe}

**TRADER'S INTENT:**
- Planned Setup: {entry_reason}
- Trade Type Intent: {trade_type}
- Notes: {notes}

**MARKET DATA FOR CONTEXT:**
{ohlcv_context}

=== TASK ===
Classify this trade using the Brooks setup taxonomy. Identify:
1. Primary setup label
2. Whether this was a second entry attempt
3. With-trend or countertrend
4. Signal bar quality and entry location""",

        "trade_analysis": """Review this completed trade using ONLY the data below.

TRADE DETAILS (completed):
- Ticker: {ticker}
- Market: {market}
- Timezone: {timezone}
- Date: {trade_date}
- Timeframe traded: {timeframe}
- Direction: {direction}
- Entry time: {entry_time}
- Entry price: ${entry_price}
- Exit time: {exit_time}
- Exit price: ${exit_price}
- Position size: {size}
- Order type: {order_type}
- Planned Stop Loss (SL) at entry: ${stop_loss}
- Planned target price at entry: {target_price}
- Actual exits (if scaled out): {exit_breakdown}
- Commissions/fees: {fees}
- Slippage estimate: {slippage}
- R-Multiple: {r_multiple} ({outcome})
- MAE: {mae}
- MFE: {mfe}

TRADER'S INTENT (what I thought it was):
- Intended setup label: {intended_setup}
- Intended trade type: {trade_type}
- Entry thesis (1-3 sentences): {entry_reason}
- Reason for entry (signal type): {signal_reason}
- Invalidation condition: {invalidation}
- Management plan at entry: {management_plan}

TRADER'S ANALYSIS (self-assessment before AI review):
- Trend assessment (major & minor): {trend_assessment}
- Was there a signal bar? If not, why?: {was_signal_present}
- Strategy alignment & any invalidation: {strategy_alignment}
- Entry/TP distance concern: {entry_tp_distance}
- Emotions at entry & exit: {entry_exit_emotions}
- Confidence level: {confidence}
- Emotional state: {emotional_state}
- Followed plan: {followed_plan}
- Stop reason: {stop_reason}
- Target reason: {target_reason}
- Mistakes noted: {mistakes}
- Lessons noted: {lessons}

SESSION CONTEXT (must be filled):
- Today open: {today_open}
- Prior day high/low/close: {pd_high} / {pd_low} / {pd_close}
- Any earnings/news flag: {news_flag}
- Market environment note: {env_note}

OHLCV DATA (do not summarize; use it as evidence):
1) DAILY BARS: last 60 daily candles ending on {trade_date}
- Format: timestamp, open, high, low, close, volume
{daily_bars}

2) 2-HOUR BARS: last 120 2-hour candles ending at {entry_time}
- Format: timestamp, open, high, low, close, volume
{twohour_bars}

3) 5-MIN BARS: last 234 5-minute candles ending at {entry_time}
- Format: timestamp, open, high, low, close, volume
{fivemin_bars}

4) LOCAL ENTRY WINDOW (for precise Brooks reading):
- Last 80 bars immediately before entry + entry bar + 20 bars after entry on the traded timeframe
- Format: timestamp, open, high, low, close, volume
{local_window}

OPTIONAL SCREENSHOT NOTES (if you have them; otherwise "none"):
{chart_notes}

INSTRUCTIONS:
- You MUST anchor conclusions to evidence from the provided OHLCV (mention concrete features like: overlap, tails, trend bars, breakout attempt, failure, second entry, wedge 3 pushes, magnets like prior high/low, swing points).
- If you say "late entry" or "countertrend", explain exactly why using the bars and levels above.
- No generic advice. Give at least 3 concrete "next time" filters in your reasoning, but output only ONE final rule in the JSON field.

Now provide your Brooks-style analysis and coaching.""",

        "market_context": """=== PRE-MARKET BRIEFING REQUEST ===

**TICKER:** {ticker}
**DATE:** {date}
**TRADING TIMEFRAME:** {timeframe}

**MULTI-TIMEFRAME OHLCV DATA:**
{ohlcv_context}

=== TASK ===
Provide a comprehensive Brooks-style pre-market report:

1. **DAILY CHART:** Regime, Always-In direction, key magnets (PDH/PDL/PDC, swings, range boundaries), If/Then scenarios

2. **2-HOUR STRUCTURE:** Channel type, position in structure, where stops are likely placed

3. **INTRADAY (Last 3 days):** Recent day type, tight ranges, failed breakouts, climax behavior

4. **TRADING PLAN:** Best setups, setups to avoid, key levels, Plan A, Plan B"""
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
            merged['system_prompts'] = DEFAULT_SETTINGS['system_prompts'].copy()
            merged['user_prompts'] = DEFAULT_SETTINGS['user_prompts'].copy()
            
            if 'candles' in settings:
                merged['candles'].update(settings['candles'])
            
            # Handle both old 'prompts' key and new 'system_prompts' key
            if 'system_prompts' in settings:
                merged['system_prompts'].update(settings['system_prompts'])
            elif 'prompts' in settings:
                # Backward compatibility: old 'prompts' → 'system_prompts'
                merged['system_prompts'].update(settings['prompts'])
            
            if 'user_prompts' in settings:
                merged['user_prompts'].update(settings['user_prompts'])
            
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


def get_system_prompt(prompt_type: str, include_materials: bool = True) -> str:
    """Get a configured system prompt by type, optionally with training materials.
    
    Automatically appends protected JSON schema if one exists for this prompt type.
    """
    settings = load_settings()
    base_prompt = settings.get('system_prompts', {}).get(prompt_type, DEFAULT_SETTINGS['system_prompts'].get(prompt_type, ''))
    
    # Append protected JSON schema if exists (not user-editable)
    json_schema = PROTECTED_JSON_SCHEMAS.get(prompt_type, '')
    full_prompt = base_prompt + json_schema
    
    if include_materials:
        materials = get_materials_content()
        if materials:
            materials_section = f"""

=== TRAINING MATERIALS (Use for context) ===
{materials}
=== END TRAINING MATERIALS ===
"""
            return full_prompt + materials_section
    
    return full_prompt


def _strip_json_format(prompt: str) -> str:
    """Strip the JSON format section from a prompt if present.
    
    This removes everything after "Respond in JSON format:" or "=== RESPONSE FORMAT ==="
    """
    import re
    
    # Try to find and strip JSON format sections
    patterns = [
        r'\n*=== RESPONSE FORMAT ===.*',  # New format
        r'\n*Respond in JSON format:\s*\{.*',  # Old format with JSON
        r'\n*Respond in JSON format:.*',  # Old format
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prompt, re.DOTALL | re.IGNORECASE)
        if match:
            return prompt[:match.start()].rstrip()
    
    return prompt


def get_editable_prompt(prompt_type: str, is_user_prompt: bool = False) -> str:
    """Get only the editable part of a prompt (without protected JSON schema).
    
    This is used by the settings UI to show only what users can safely edit.
    Automatically strips any JSON format section from saved prompts.
    """
    settings = load_settings()
    if is_user_prompt:
        prompt = settings.get('user_prompts', {}).get(prompt_type, DEFAULT_SETTINGS['user_prompts'].get(prompt_type, ''))
    else:
        prompt = settings.get('system_prompts', {}).get(prompt_type, DEFAULT_SETTINGS['system_prompts'].get(prompt_type, ''))
    
    # Strip JSON format if present (migration from old saved settings)
    return _strip_json_format(prompt)


def get_user_prompt(prompt_type: str) -> str:
    """Get a configured user prompt template by type."""
    settings = load_settings()
    return settings.get('user_prompts', {}).get(prompt_type, DEFAULT_SETTINGS['user_prompts'].get(prompt_type, ''))


# Backward compatibility alias
def get_prompt(prompt_type: str, include_materials: bool = True) -> str:
    """Alias for get_system_prompt for backward compatibility."""
    return get_system_prompt(prompt_type, include_materials)


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


def update_prompt(prompt_type: str, prompt_text: str, is_user_prompt: bool = False) -> bool:
    """Update a specific prompt (system or user).
    
    Automatically strips any JSON format section to keep settings clean.
    The JSON schema is protected and will be appended automatically when used.
    """
    # Strip any JSON format the user might have pasted
    clean_prompt = _strip_json_format(prompt_text)
    
    settings = load_settings()
    if is_user_prompt:
        if 'user_prompts' not in settings:
            settings['user_prompts'] = {}
        settings['user_prompts'][prompt_type] = clean_prompt
    else:
        if 'system_prompts' not in settings:
            settings['system_prompts'] = {}
        settings['system_prompts'][prompt_type] = clean_prompt
    return save_settings(settings)


def get_cache_settings() -> dict:
    """Get cache settings."""
    settings = load_settings()
    defaults = DEFAULT_SETTINGS.get('cache', {})
    cache_settings = settings.get('cache', {})
    return {
        'enable_review_cache': cache_settings.get('enable_review_cache', defaults.get('enable_review_cache', True)),
        'auto_regenerate': cache_settings.get('auto_regenerate', defaults.get('auto_regenerate', False)),
    }


def update_cache_settings(enable_review_cache: Optional[bool] = None, auto_regenerate: Optional[bool] = None) -> bool:
    """Update cache settings."""
    settings = load_settings()
    if 'cache' not in settings:
        settings['cache'] = {}
    if enable_review_cache is not None:
        settings['cache']['enable_review_cache'] = enable_review_cache
    if auto_regenerate is not None:
        settings['cache']['auto_regenerate'] = auto_regenerate
    return save_settings(settings)
