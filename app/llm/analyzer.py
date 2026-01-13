"""
LLM-powered trade and market analysis.

Uses Claude via LiteLLM proxy (OpenAI-compatible API) to perform intelligent analysis:
- Setup/strategy classification
- Trade review and coaching
- Market regime analysis
- Pattern detection
"""

import json
import logging
from datetime import datetime
from typing import Optional, Literal

from app.config import settings, get_llm_api_key, get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """
    LLM-powered analyzer for trades and market context.
    
    Uses Claude via LiteLLM proxy - no hardcoded pattern matching.
    """

    def __init__(self):
        """Initialize LLM analyzer."""
        self.api_key = get_llm_api_key()
        self.base_url = get_llm_base_url()
        self.model = get_llm_model()
        self._client = None

    @property
    def is_available(self) -> bool:
        """Check if LLM is available."""
        return self.api_key is not None

    def _get_client(self):
        """Get or create OpenAI-compatible client for LiteLLM proxy."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError:
                logger.error("OpenAI package not installed. Run: pip install openai")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                return None
        return self._client

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> Optional[str]:
        """Make an LLM API call via LiteLLM proxy."""
        if not self.is_available:
            logger.error("LLM not available - check API key")
            return None

        client = self._get_client()
        if client is None:
            return None

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return None

    def classify_trade_setup(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_price: float,
        entry_reason: Optional[str] = None,
        notes: Optional[str] = None,
        ohlcv_context: Optional[str] = None,
    ) -> dict:
        """
        Use LLM to classify the trade setup/strategy.
        
        Returns dict with:
        - strategy_name: The classified strategy
        - strategy_category: with_trend, countertrend, trading_range, special
        - confidence: low, medium, high
        - reasoning: Why this classification
        """
        system_prompt = """You are an expert Al Brooks price action analyst. Your job is to classify trades into Brooks-style setups.

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
}"""

        user_prompt = f"""Classify this trade:

TRADE DETAILS:
- Ticker: {ticker}
- Direction: {direction.upper()}
- Entry: ${entry_price}
- Exit: ${exit_price}
- Stop: ${stop_price}
- P&L: {"+" if (exit_price - entry_price) * (1 if direction == "long" else -1) > 0 else ""}{((exit_price - entry_price) / (entry_price - stop_price) if direction == "long" else (entry_price - exit_price) / (stop_price - entry_price)):.2f}R

TRADER'S NOTES:
Entry Reason: {entry_reason or "Not provided"}
Notes: {notes or "Not provided"}

{f"MARKET CONTEXT (OHLCV):{chr(10)}{ohlcv_context}" if ohlcv_context else ""}

Classify this trade setup."""

        response = self._call_llm(system_prompt, user_prompt)
        
        if response:
            try:
                # Extract JSON from response
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse LLM response as JSON: {response}")
                return {
                    "strategy_name": "unclassified",
                    "strategy_category": "unknown",
                    "confidence": "low",
                    "reasoning": response,
                }
        
        return {
            "strategy_name": "unclassified",
            "strategy_category": "unknown", 
            "confidence": "low",
            "reasoning": "LLM analysis unavailable",
        }

    def analyze_trade(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_price: float,
        target_price: Optional[float] = None,
        entry_reason: Optional[str] = None,
        notes: Optional[str] = None,
        ohlcv_context: Optional[str] = None,
        mae: Optional[float] = None,
        mfe: Optional[float] = None,
    ) -> dict:
        """
        Comprehensive LLM analysis of a completed trade.
        
        Returns detailed Brooks-style review with coaching.
        """
        r_multiple = (
            (exit_price - entry_price) / (entry_price - stop_price)
            if direction == "long"
            else (entry_price - exit_price) / (stop_price - entry_price)
        )

        system_prompt = """You are Al Brooks, the legendary price action trader and author. 
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
    "errors": ["error1", "error2"],
    "what_was_good": ["good1", "good2"],
    "what_was_flawed": ["flaw1", "flaw2"],
    "rule_for_next_time": "specific actionable rule",
    "grade": "A|B|C|D|F",
    "grade_explanation": "why this grade",
    "overall_coaching": "2-3 sentence summary"
}"""

        user_prompt = f"""Review this completed trade:

TRADE DETAILS:
- Ticker: {ticker}
- Direction: {direction.upper()}
- Entry: ${entry_price:.2f}
- Exit: ${exit_price:.2f}
- Stop: ${stop_price:.2f}
{f"- Target: ${target_price:.2f}" if target_price else "- Target: Not set"}
- R-Multiple: {r_multiple:+.2f}R ({"WINNER" if r_multiple > 0 else "LOSER" if r_multiple < 0 else "BREAKEVEN"})
{f"- MAE: {mae:.2f}R (max adverse excursion)" if mae else ""}
{f"- MFE: {mfe:.2f}R (max favorable excursion)" if mfe else ""}

TRADER'S NOTES:
Entry Reason: {entry_reason or "Not provided"}
Notes: {notes or "Not provided"}

{f"MARKET CONTEXT (recent OHLCV data):{chr(10)}{ohlcv_context}" if ohlcv_context else "No market context data provided."}

Provide your Brooks-style analysis and coaching."""

        response = self._call_llm(system_prompt, user_prompt, max_tokens=2500)
        
        if response:
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse trade analysis as JSON")
                return {"raw_analysis": response}
        
        return {"error": "LLM analysis unavailable"}

    def analyze_market_context(
        self,
        ticker: str,
        daily_ohlcv: str,
        intraday_ohlcv: Optional[str] = None,
    ) -> dict:
        """
        LLM analysis of market context for premarket reports.
        
        Returns regime, always-in, key levels, and trading plan.
        """
        system_prompt = """You are an expert Al Brooks price action analyst preparing a premarket briefing.

Analyze the OHLCV data and provide:
1. REGIME: Is this a trend (up/down) or trading range?
2. ALWAYS-IN: If forced to have a position, which direction?
3. KEY LEVELS: Important support/resistance levels from the data
4. STRENGTH: Is the current move strong or weak?
5. PLAN A: Most likely scenario and setups to look for
6. PLAN B: What would change your mind? What's the alternative scenario?
7. AVOID: Conditions where trading would be low probability

Use Brooks terminology. Be specific about price levels.

Respond in JSON format:
{
    "regime": "trend_up|trend_down|trading_range",
    "regime_confidence": "high|medium|low",
    "always_in": "long|short|neutral",
    "key_levels": {
        "resistance": [{"price": 100.00, "description": "Prior high"}],
        "support": [{"price": 95.00, "description": "Prior low"}]
    },
    "strength": "strong|moderate|weak",
    "strength_reasoning": "explanation",
    "plan_a": {
        "scenario": "description",
        "bias": "LONG|SHORT|NEUTRAL",
        "setups": ["setup1", "setup2"],
        "entry_zones": "description",
        "targets": "description"
    },
    "plan_b": {
        "trigger": "what must happen to flip",
        "new_bias": "LONG|SHORT|NEUTRAL",
        "action": "what to do"
    },
    "avoid": ["condition1", "condition2"],
    "summary": "1-2 sentence executive summary"
}"""

        user_prompt = f"""Analyze this market data for {ticker}:

DAILY OHLCV (recent bars, newest last):
{daily_ohlcv}

{f"INTRADAY OHLCV:{chr(10)}{intraday_ohlcv}" if intraday_ohlcv else ""}

Provide your Brooks-style premarket analysis."""

        response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)
        
        if response:
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                return {"raw_analysis": response}
        
        return {"error": "LLM analysis unavailable"}

    def analyze_weekly_performance(
        self,
        trades_summary: str,
        strategy_stats: str,
    ) -> dict:
        """
        LLM analysis of weekly trading performance.
        
        Returns edge analysis, leaks, and coaching recommendations.
        """
        system_prompt = """You are a trading coach analyzing a trader's weekly performance using Al Brooks methodology.

Analyze their trades and provide:
1. OVERALL ASSESSMENT: How was the week?
2. EDGE: What's working for this trader?
3. LEAKS: What's costing them money?
4. PATTERNS: Any behavioral patterns you notice?
5. TOP 3 RULES: Specific rules for next week
6. STOP DOING: What should they eliminate?
7. DOUBLE DOWN: What should they do more of?

Be direct and specific. Reference the actual numbers.

Respond in JSON format:
{
    "week_grade": "A|B|C|D|F",
    "overall_assessment": "2-3 sentence summary",
    "edge": ["strength1", "strength2"],
    "leaks": ["leak1", "leak2"],
    "behavioral_patterns": ["pattern1", "pattern2"],
    "top_3_rules": ["rule1", "rule2", "rule3"],
    "stop_doing": ["stop1", "stop2"],
    "double_down": ["do_more1", "do_more2"],
    "focus_for_next_week": "single most important focus"
}"""

        user_prompt = f"""Analyze this trader's weekly performance:

TRADES SUMMARY:
{trades_summary}

STRATEGY STATISTICS:
{strategy_stats}

Provide your coaching analysis."""

        response = self._call_llm(system_prompt, user_prompt, max_tokens=1500)
        
        if response:
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                
                return json.loads(json_str.strip())
            except json.JSONDecodeError:
                return {"raw_analysis": response}
        
        return {"error": "LLM analysis unavailable"}

    def suggest_strategy_from_description(
        self,
        trade_description: str,
    ) -> dict:
        """
        Given a free-form trade description, suggest the strategy.
        
        Useful for bulk imports where trades don't have strategy tags.
        """
        system_prompt = """You are an Al Brooks price action expert. 
Given a trade description, identify the most likely Brooks-style setup.

Respond in JSON:
{
    "strategy_name": "the_strategy_name",
    "category": "with_trend|countertrend|trading_range|special",
    "confidence": "high|medium|low",
    "reasoning": "brief explanation"
}"""

        response = self._call_llm(system_prompt, trade_description, max_tokens=500)
        
        if response:
            try:
                json_str = response
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                return json.loads(json_str.strip())
            except:
                pass
        
        return {
            "strategy_name": "unclassified",
            "category": "unknown",
            "confidence": "low",
            "reasoning": "Could not analyze",
        }


# Singleton instance
_analyzer: Optional[LLMAnalyzer] = None


def get_analyzer() -> LLMAnalyzer:
    """Get the global LLM analyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = LLMAnalyzer()
    return _analyzer
