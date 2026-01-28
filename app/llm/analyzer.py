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
from typing import Optional

from app.config import get_llm_api_key, get_llm_base_url, get_llm_model

logger = logging.getLogger(__name__)


def _get_materials_for_trade(
    direction: str,
    timeframe: str = "5m",
    entry_reason: str = "",
    setup_type: str = "",
    ticker: str = "",
    cancellation_check: callable = None,
) -> str:
    """Get relevant training materials using RAG, with fallback to simple reader."""
    # Check for cancellation before expensive RAG operations
    if cancellation_check and cancellation_check():
        logger.debug("📚 Materials retrieval cancelled")
        return ""

    try:
        # Try RAG first (smarter retrieval)
        from app.materials_rag import get_relevant_materials, get_materials_rag

        rag = get_materials_rag()
        status = rag.get_status()

        if status.get("available") and status.get("total_chunks", 0) > 0:
            materials = get_relevant_materials(
                ticker=ticker,
                direction=direction,
                timeframe=timeframe,
                entry_reason=entry_reason,
                setup_type=setup_type,
                n_chunks=50,
                max_chars=50000,
            )
            if materials:
                logger.debug("📚 Retrieved relevant materials via RAG")
                return materials

        # If RAG not ready but materials exist, try to index them
        if status.get("needs_reindex") or status.get("total_chunks", 0) == 0:
            from app.materials_reader import has_materials

            if has_materials():
                logger.info("📚 Indexing materials for RAG...")
                rag.index_materials()
                # Try again
                materials = get_relevant_materials(
                    ticker=ticker,
                    direction=direction,
                    timeframe=timeframe,
                    entry_reason=entry_reason,
                    setup_type=setup_type,
                    n_chunks=50,
                    max_chars=50000,
                )
                if materials:
                    return materials

    except ImportError:
        logger.debug("RAG dependencies not available, using simple reader")
    except Exception as e:
        logger.warning(f"RAG retrieval failed: {e}, falling back to simple reader")

    # Fallback to simple materials reader
    try:
        from app.materials_reader import get_materials_context, has_materials

        if has_materials():
            return get_materials_context(max_chars=15000)
    except Exception as e:
        logger.warning(f"Simple materials reader failed: {e}")

    return ""


class SafeFormatDict(dict):
    """Dict that returns placeholder name for missing keys (avoids KeyError in format())."""

    def __missing__(self, key):
        return f"{{{key}}}"  # Return {key} for missing placeholders


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
        """Make an LLM API call via LiteLLM proxy (synchronous)."""
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
            error_str = str(e)
            if "401" in error_str or "Unauthorized" in error_str or "token" in error_str.lower():
                logger.error(f"LLM authentication failed - check your API key in .env file: {e}")
            elif "429" in error_str or "rate" in error_str.lower():
                logger.error(f"LLM rate limited - try again later: {e}")
            else:
                logger.error(f"LLM call failed: {e}")
            return None

    async def _call_llm_async(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.2,
    ) -> Optional[str]:
        """Make an LLM API call asynchronously (non-blocking)."""
        import asyncio

        return await asyncio.to_thread(
            self._call_llm, system_prompt, user_prompt, max_tokens, temperature
        )

    def classify_trade_setup(
        self,
        ticker: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        stop_price: Optional[float] = None,  # Renamed from stop_price, actually represents SL
        entry_reason: Optional[str] = None,
        notes: Optional[str] = None,
        ohlcv_context: Optional[str] = None,
        timeframe: Optional[str] = None,
        trade_type: Optional[str] = None,
    ) -> dict:
        """
        Use LLM to classify the trade setup/strategy.

        Args:
            stop_price: Stop Loss level (where trade is wrong). Legacy name, represents SL.

        Returns dict with:
        - strategy_name: The classified strategy (or 'needs_info' if more data needed)
        - strategy_category: with_trend, countertrend, trading_range, special
        - confidence: low, medium, high
        - reasoning: Why this classification
        - missing_info: List of what additional info would help (if applicable)
        """
        # Load configurable prompts
        from app.config_prompts import get_system_prompt, get_user_prompt

        system_prompt = get_system_prompt("trade_classification")

        # Get relevant training materials using RAG
        materials_context = _get_materials_for_trade(
            direction=direction,
            timeframe=timeframe or "5m",
            entry_reason=entry_reason or "",
            setup_type=trade_type or "",
            ticker=ticker,
        )
        if materials_context:
            system_prompt = f"{system_prompt}\n\n{materials_context}"

        # Calculate R-multiple safely (only if SL is provided)
        stop_loss = stop_price  # Use clearer name internally
        r_mult = 0.0
        r_mult_str = "N/A (no SL)"

        if stop_loss is not None:
            if direction == "long":
                risk = entry_price - stop_loss
                reward = exit_price - entry_price
            else:
                risk = stop_loss - entry_price
                reward = entry_price - exit_price

            if abs(risk) < 0.0001:
                r_mult = reward / (entry_price * 0.02) if entry_price > 0 else 0
            else:
                r_mult = reward / risk

            r_mult_str = f"{'+' if r_mult > 0 else ''}{r_mult:.2f}R"

        stop_loss_str = f"${stop_loss:.4f}" if stop_loss else "Not set"

        # Build user prompt from template (using SafeFormatDict to avoid KeyError)
        user_template = get_user_prompt("trade_classification")
        format_vars = SafeFormatDict(
            ticker=ticker,
            direction=direction.upper(),
            entry_price=f"{entry_price:.4f}",
            exit_price=f"{exit_price:.4f}",
            stop_price=stop_loss_str,
            stop_loss=stop_loss_str,
            r_multiple=r_mult_str,
            entry_reason=entry_reason or "Not provided",
            notes=notes or "Not provided",
            ohlcv_context=f"MARKET CONTEXT (OHLCV):\n{ohlcv_context}"
            if ohlcv_context
            else "No market context available.",
            timeframe=timeframe or "5m",
            trade_type=trade_type or "Not specified",
        )
        user_prompt = user_template.format_map(format_vars)

        response = self._call_llm(system_prompt, user_prompt)

        if response:
            parsed = self._parse_json_response(response) or {}

            # Normalize key names (older/newer prompt variants).
            if "strategy_name" not in parsed:
                parsed["strategy_name"] = (
                    parsed.get("primary_setup")
                    or parsed.get("primary_label")
                    or parsed.get("setup")
                    or "unclassified"
                )
            if "strategy_category" not in parsed:
                parsed["strategy_category"] = (
                    parsed.get("setup_category") or parsed.get("category") or "unknown"
                )
            if "confidence" not in parsed:
                parsed["confidence"] = "low"
            if "reasoning" not in parsed:
                parsed["reasoning"] = response

            return parsed

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
        stop_price: Optional[float] = None,  # SL - Stop Loss level (where trade is wrong)
        target_price: Optional[float] = None,  # TP - Take Profit level
        entry_reason: Optional[str] = None,
        exit_reason: Optional[str] = None,
        notes: Optional[str] = None,
        ohlcv_context: Optional[str] = None,
        r_multiple: Optional[float] = None,
        mae: Optional[float] = None,
        mfe: Optional[float] = None,
        # Brooks intent fields
        timeframe: Optional[str] = None,
        trade_type: Optional[str] = None,
        entry_time: Optional[str] = None,
        exit_time: Optional[str] = None,
        size: Optional[float] = None,
        pnl_dollars: Optional[float] = None,
        hold_time_minutes: Optional[int] = None,
        stop_reason: Optional[str] = None,
        target_reason: Optional[str] = None,
        invalidation_condition: Optional[str] = None,
        confidence_level: Optional[int] = None,
        emotional_state: Optional[str] = None,
        followed_plan: Optional[bool] = None,
        account_type: Optional[str] = None,
        mistakes: Optional[str] = None,  # Legacy
        lessons: Optional[str] = None,  # Legacy
        mistakes_and_lessons: Optional[str] = None,  # Combined field
        # New extended fields
        trade_date: Optional[str] = None,
        market: Optional[str] = None,
        timezone: Optional[str] = None,
        order_type: Optional[str] = None,
        fees: Optional[float] = None,
        slippage: Optional[float] = None,
        intended_setup: Optional[str] = None,
        management_plan: Optional[str] = None,
        today_open: Optional[float] = None,
        pd_high: Optional[float] = None,
        pd_low: Optional[float] = None,
        pd_close: Optional[float] = None,
        daily_bars: Optional[str] = None,
        twohour_bars: Optional[str] = None,
        fivemin_bars: Optional[str] = None,
        # Extended Brooks analysis fields
        trend_assessment: Optional[str] = None,
        signal_reason: Optional[str] = None,
        was_signal_present: Optional[str] = None,
        strategy_alignment: Optional[str] = None,
        entry_exit_emotions: Optional[str] = None,
        entry_tp_distance: Optional[str] = None,
        # Cancellation support
        cancellation_check: callable = None,
    ) -> dict:
        """
        Comprehensive Brooks Audit of a completed trade.

        Args:
            stop_price: Stop Loss (SL) level - where trade is considered wrong
            target_price: Take Profit (TP) level - intended target
            cancellation_check: Optional function that returns True if generation should stop

        Returns detailed Brooks-style review with coaching.
        """
        # Check for cancellation at the start
        if cancellation_check and cancellation_check():
            logger.info("⏹️ Trade analysis cancelled before start")
            return {"error": "Cancelled"}

        # Log received OHLCV data
        ohlcv_len = len(ohlcv_context) if ohlcv_context else 0
        logger.info(
            f"📊 analyze_trade received - OHLCV context: {ohlcv_len} chars, daily_bars: {len(daily_bars) if daily_bars else 0} chars"
        )
        if ohlcv_context and ohlcv_len > 0:
            # Log first 200 chars of OHLCV for debugging
            logger.debug(f"📊 OHLCV preview: {ohlcv_context[:200]}...")

        # Use clearer internal names
        stop_loss = stop_price
        take_profit = target_price

        # Use passed r_multiple if available, otherwise calculate
        if r_multiple is None:
            r_multiple = 0.0
            if stop_loss is not None:
                if direction == "long":
                    risk = entry_price - stop_loss
                    reward = exit_price - entry_price
                else:
                    risk = stop_loss - entry_price
                    reward = entry_price - exit_price

                # Handle edge case where stop == entry (zero risk)
                if abs(risk) < 0.0001:
                    r_multiple = (
                        reward / (entry_price * 0.02) if entry_price > 0 else 0
                    )  # Assume 2% risk
                else:
                    r_multiple = reward / risk
            else:
                # No SL set - calculate raw P&L change
                if direction == "long":
                    reward = exit_price - entry_price
                else:
                    reward = entry_price - exit_price

        # Check for cancellation before loading prompts and materials
        if cancellation_check and cancellation_check():
            return {"error": "Cancelled"}

        # Load configurable prompts
        from app.config_prompts import get_system_prompt, get_user_prompt

        system_prompt = get_system_prompt("trade_analysis")

        # Get relevant training materials using RAG (with cancellation support)
        materials_context = _get_materials_for_trade(
            direction=direction,
            timeframe=timeframe or "5m",
            entry_reason=entry_reason or "",
            setup_type=trade_type or "",
            ticker=ticker,
            cancellation_check=cancellation_check,
        )

        # Check for cancellation after RAG retrieval
        if cancellation_check and cancellation_check():
            return {"error": "Cancelled"}

        if materials_context:
            system_prompt = f"{system_prompt}\n\n{materials_context}"
            logger.debug(f"📚 Added {len(materials_context)} chars of relevant training materials")

        # Format hold time
        hold_time_str = "-"
        if hold_time_minutes:
            if hold_time_minutes >= 1440:
                days = hold_time_minutes // 1440
                hours = (hold_time_minutes % 1440) // 60
                hold_time_str = f"{days}d {hours}h"
            elif hold_time_minutes >= 60:
                hours = hold_time_minutes // 60
                mins = hold_time_minutes % 60
                hold_time_str = f"{hours}h {mins}m"
            else:
                hold_time_str = f"{hold_time_minutes}m"

        # Build user prompt from template
        outcome_str = "WINNER" if r_multiple > 0 else "LOSER" if r_multiple < 0 else "BREAKEVEN"
        user_template = get_user_prompt("trade_analysis")

        # Format entry/exit times
        entry_time_str = str(entry_time) if entry_time else "Not recorded"
        exit_time_str = str(exit_time) if exit_time else "Not recorded"

        # Parse OHLCV context into separate sections if provided as combined string
        daily_section = daily_bars or ""
        twohour_section = twohour_bars or ""
        fivemin_section = fivemin_bars or ""

        if ohlcv_context and not daily_bars:
            # Parse combined context into sections
            # Format is: "=== DAILY (...) ===\n...data...\n\n=== 2-HOUR (...) ===\n...data..."
            import re

            # Split by section headers (=== ... ===)
            section_pattern = r"(===\s*[^=]+\s*===)"
            parts = re.split(section_pattern, ohlcv_context)

            current_header = None
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if part.startswith("===") and part.endswith("==="):
                    current_header = part
                elif current_header:
                    # Combine header with content
                    full_section = f"{current_header}\n{part}"
                    if "DAILY" in current_header.upper() or "DAILY_" in current_header.upper():
                        daily_section = full_section
                    elif "2-HOUR" in current_header.upper() or "TWOHOUR" in current_header.upper():
                        twohour_section = full_section
                    elif (
                        "5-MINUTE" in current_header.upper()
                        or "5-MIN" in current_header.upper()
                        or "FIVEMIN" in current_header.upper()
                    ):
                        fivemin_section = full_section
                    current_header = None

            # Debug log what was parsed
            logger.debug(
                f"📊 OHLCV parsed - Daily: {len(daily_section)} chars, 2H: {len(twohour_section)} chars, 5M: {len(fivemin_section)} chars"
            )

        # Format MAE/MFE
        mae_str = f"{mae:.2f}R" if mae else "not recorded"
        mfe_str = f"{mfe:.2f}R" if mfe else "not recorded"

        # Build format variables (using SafeFormatDict to avoid KeyError for missing placeholders)
        format_vars = SafeFormatDict(
            # Trade identity
            ticker=ticker,
            market=market or "US stocks",
            timezone=timezone or "US/Eastern",
            trade_date=trade_date or str(entry_time)[:10] if entry_time else "unknown",
            timeframe=timeframe or "5m",
            tf_traded=timeframe or "5m",
            direction=direction.lower(),
            # Execution details
            entry_time=entry_time_str,
            entry_price=f"{entry_price:.4f}",
            exit_time=exit_time_str,
            exit_price=f"{exit_price:.4f}",
            size=f"{size:.0f}" if size else "1",
            order_type=order_type or "market",
            stop_price=f"${stop_loss:.4f}" if stop_loss else "not set",
            stop_loss=f"${stop_loss:.4f}" if stop_loss else "not set",
            target_price=f"${take_profit:.4f}" if take_profit else "not set",
            take_profit=f"${take_profit:.4f}" if take_profit else "not set",
            fees=f"${fees:.2f}" if fees else "N/A",
            slippage=f"${slippage:.4f}" if slippage else "N/A",
            # Performance
            r_multiple=f"{r_multiple:+.2f}R",
            outcome=outcome_str,
            mae=mae_str,
            mfe=mfe_str,
            # Trader intent
            intended_setup=intended_setup or entry_reason or "not specified",
            trade_type=trade_type or "not specified",
            entry_reason=entry_reason or "not provided",
            exit_reason=exit_reason or "not provided",
            invalidation=invalidation_condition or "not provided",
            management_plan=management_plan or "not specified",
            # Session context
            today_open=f"${today_open:.2f}" if today_open else "unknown",
            pd_high=f"${pd_high:.2f}" if pd_high else "unknown",
            pd_low=f"${pd_low:.2f}" if pd_low else "unknown",
            pd_close=f"${pd_close:.2f}" if pd_close else "unknown",
            # OHLCV data sections
            daily_bars=daily_section or "No daily data available",
            twohour_bars=twohour_section or "No 2-hour data available",
            fivemin_bars=fivemin_section or "No 5-min data available",
            # Aliases for compatibility with different naming conventions
            daily_60=daily_section or "No daily data available",
            twohour_120=twohour_section or "No 2-hour data available",
            fivemin_234=fivemin_section or "No 5-min data available",
            DAILY_60=daily_section or "No daily data available",
            TWOHOUR_120=twohour_section or "No 2-hour data available",
            FIVEMIN_234=fivemin_section or "No 5-min data available",
            position_size=f"{size:.0f}" if size else "1",
            intended_trade_type=trade_type or "not specified",
            # Legacy compatibility
            account_type=account_type or "paper",
            entry_order_type=order_type or "stop",
            exit_order_type="market",
            pnl_dollars=f"${pnl_dollars:.2f}" if pnl_dollars else "Not calculated",
            hold_time=hold_time_str,
            mae_line=f"- MAE: {mae:.2f}R" if mae else "",
            mfe_line=f"- MFE: {mfe:.2f}R" if mfe else "",
            stop_reason=stop_reason or "Not provided",
            target_reason=target_reason or "Not provided",
            confidence=f"{confidence_level}/5" if confidence_level else "Not rated",
            emotional_state=emotional_state or "Not recorded",
            followed_plan="Yes"
            if followed_plan
            else ("No" if followed_plan is False else "Not recorded"),
            notes=notes or "None",
            # Combined mistakes & lessons (prefer combined field, fall back to separate)
            mistakes_and_lessons=mistakes_and_lessons
            or (
                ((mistakes or "") + ("\n" if mistakes and lessons else "") + (lessons or ""))
                or "None noted"
            ),
            mistakes=mistakes or "None noted",  # Legacy compatibility
            lessons=lessons or "None noted",  # Legacy compatibility
            ohlcv_context=ohlcv_context if ohlcv_context else "No market context data provided.",
            # Extended Brooks analysis fields
            trend_assessment=trend_assessment or "Not provided",
            signal_reason=signal_reason or "Not provided",
            was_signal_present=was_signal_present or "Not provided",
            strategy_alignment=strategy_alignment or "Not provided",
            entry_exit_emotions=entry_exit_emotions or "Not provided",
            entry_tp_distance=entry_tp_distance or "Not provided",
        )
        user_prompt = user_template.format_map(format_vars)

        # Log OHLCV sections that will be sent to LLM
        logger.info(
            f"📊 LLM prompt sections - Daily: {len(daily_section)} chars, 2H: {len(twohour_section)} chars, 5M: {len(fivemin_section)} chars"
        )
        if daily_section and len(daily_section) > 50:
            logger.debug(f"📊 Daily bars preview: {daily_section[:150]}...")
        else:
            logger.warning(
                f"⚠️ Daily section is empty or too short: '{daily_section[:100] if daily_section else 'EMPTY'}'"
            )

        # Check for cancellation before expensive LLM call
        if cancellation_check and cancellation_check():
            return {"error": "Cancelled"}

        response = self._call_llm(system_prompt, user_prompt, max_tokens=20000)

        if response:
            try:
                result = self._parse_json_response(response)
                if result:
                    return result
                else:
                    logger.warning("Failed to parse trade analysis as JSON, returning raw")
                    return {"raw_analysis": response}
            except Exception as e:
                logger.warning(f"Error parsing trade analysis: {e}")
                return {"raw_analysis": response}

        return {"error": "LLM analysis unavailable"}

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Robustly parse JSON from LLM response, handling various formats."""
        import re

        if not response:
            logger.warning("Empty response from LLM")
            return None

        def try_parse(json_str: str) -> Optional[dict]:
            """Try to parse JSON with various fixes."""
            if not json_str or not json_str.strip():
                return None
            json_str = json_str.strip()

            # Try direct parse
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

            # Fix common issues: trailing commas before } or ]
            fixed = re.sub(r",\s*}", "}", json_str)
            fixed = re.sub(r",\s*]", "]", fixed)
            try:
                result = json.loads(fixed)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                pass

            # Try to fix truncated JSON by closing open braces/brackets
            # Count open vs close braces
            open_braces = json_str.count("{") - json_str.count("}")
            open_brackets = json_str.count("[") - json_str.count("]")

            if open_braces > 0 or open_brackets > 0:
                # Try to close the JSON - remove trailing comma first
                fixed = re.sub(r",\s*$", "", json_str)
                # Remove incomplete string values
                fixed = re.sub(r':\s*"[^"]*$', ': ""', fixed)
                # Add closing brackets and braces
                fixed += "]" * open_brackets + "}" * open_braces
                try:
                    result = json.loads(fixed)
                    if isinstance(result, dict):
                        logger.debug("Fixed truncated JSON by closing braces")
                        return result
                except json.JSONDecodeError:
                    pass

            return None

        # Try direct parse first
        result = try_parse(response)
        if result:
            return result

        # Try extracting from markdown code blocks
        patterns = [
            r"```json\s*([\s\S]*?)\s*```",  # ```json ... ```
            r"```\s*([\s\S]*?)\s*```",  # ``` ... ```
        ]

        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            for match in matches:
                result = try_parse(match)
                if result:
                    logger.debug("Parsed JSON from markdown code block")
                    return result

        # Try to find JSON object anywhere in the response
        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = response[start : end + 1]
            result = try_parse(json_str)
            if result:
                logger.debug("Parsed JSON by finding braces")
                return result

        logger.warning(f"Could not parse JSON from LLM response (length: {len(response)})")
        return None

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
        # Load configurable prompt
        from app.config_prompts import get_prompt

        system_prompt = get_prompt("market_context")

        user_prompt = f"""Analyze this market data for {ticker}:

DAILY OHLCV (recent bars, newest last):
{daily_ohlcv}

{f"INTRADAY OHLCV:{chr(10)}{intraday_ohlcv}" if intraday_ohlcv else ""}

Provide your Brooks-style premarket analysis."""

        response = self._call_llm(system_prompt, user_prompt, max_tokens=2000)

        if response:
            result = self._parse_json_response(response)
            if result:
                return result
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
            result = self._parse_json_response(response)
            if result:
                return result
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
            except (json.JSONDecodeError, IndexError, ValueError):
                pass  # Fall through to default response

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
