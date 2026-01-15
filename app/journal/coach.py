"""
Brooks-style Trade Coach for post-trade review.

Uses LLM (Claude/OpenAI) for intelligent trade analysis and coaching.
Falls back to rule-based analysis if LLM is unavailable.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Literal
import logging

import pandas as pd

from app.journal.models import Trade, TradeDirection, TradeOutcome, get_session
from app.journal.analytics import TradeAnalytics
from app.data.cache import get_cached_ohlcv
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TradeReview:
    """Complete Brooks Audit review of a trade."""

    # Required fields (no defaults) - must come first
    trade_id: int
    ticker: str

    # Context analysis (multi-timeframe)
    regime: str  # Daily regime
    always_in: str  # Always-in direction
    context_description: str  # Coaching summary

    # Setup classification (Brooks taxonomy)
    setup_classification: str  # Primary setup label
    setup_quality: Literal["good", "marginal", "poor"]

    # Trader's equation
    risk_reward_assessment: str
    probability_assessment: str

    # Errors/mistakes
    errors_detected: list[str]

    # Coaching output
    what_was_good: list[str]
    what_was_flawed: list[str]
    rule_for_next_time: str

    # Metrics
    r_multiple: float
    mae: Optional[float]
    mfe: Optional[float]

    # Overall grade
    grade: Literal["A", "B", "C", "D", "F"]
    grade_explanation: str

    # Optional fields (with defaults) - must come after required fields
    is_second_entry: bool = False
    with_trend_or_counter: str = "neutral"
    signal_bar_quality: str = "unknown"
    entry_location: str = "unknown"
    traders_equation: str = "unknown"
    exit_quality: str = "unknown"
    selection_vs_execution: str = "unknown"
    keep_doing: str = ""
    stop_doing: str = ""
    better_alternative: str = ""


class TradeCoach:
    """
    Review trades using Brooks price action principles.
    
    Uses LLM for intelligent analysis - no hardcoded pattern matching.
    Falls back to basic rule-based analysis if LLM unavailable.
    """

    def __init__(self):
        """Initialize coach."""
        self.analytics = TradeAnalytics()
        self._llm_analyzer = None

    @property
    def llm_analyzer(self):
        """Lazy load LLM analyzer."""
        if self._llm_analyzer is None:
            from app.llm.analyzer import get_analyzer
            self._llm_analyzer = get_analyzer()
        return self._llm_analyzer

    def review_trade(self, trade_id: int) -> Optional[TradeReview]:
        """
        Perform comprehensive review of a trade using LLM analysis.

        Args:
            trade_id: Trade ID to review

        Returns:
            TradeReview object or None if trade not found
        """
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                logger.warning(f"Trade {trade_id} not found")
                return None

            # Get market context data
            ohlcv_context = self._get_ohlcv_context_string(trade)

            # Use LLM for comprehensive Brooks Audit
            if self.llm_analyzer.is_available:
                llm_analysis = self.llm_analyzer.analyze_trade(
                    ticker=trade.ticker,
                    direction=trade.direction.value,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    stop_price=trade.stop_price,
                    target_price=trade.target_price,
                    entry_reason=trade.entry_reason,
                    notes=trade.notes,
                    ohlcv_context=ohlcv_context,
                    mae=trade.mae,
                    mfe=trade.mfe,
                    # Brooks intent fields
                    timeframe=trade.timeframe or "5m",
                    trade_type=trade.trade_type,
                    entry_time=trade.entry_time,
                    exit_time=trade.exit_time,
                    size=trade.size,
                    pnl_dollars=trade.pnl_dollars,
                    hold_time_minutes=trade.hold_time_minutes,
                    stop_reason=trade.stop_reason,
                    target_reason=trade.target_reason,
                    invalidation_condition=trade.invalidation_condition,
                    confidence_level=trade.confidence_level,
                    emotional_state=trade.emotional_state,
                    followed_plan=trade.followed_plan,
                    account_type=trade.account_type or "paper",
                    mistakes=trade.mistakes,
                    lessons=trade.lessons,
                )

                if "error" not in llm_analysis and "raw_analysis" not in llm_analysis:
                    # Extract nested fields from Brooks Audit response
                    context = llm_analysis.get("context", {})
                    setup = llm_analysis.get("setup", {})
                    entry_quality = llm_analysis.get("entry_quality", {})
                    risk_reward = llm_analysis.get("risk_reward", {})
                    management = llm_analysis.get("management", {})
                    coaching = llm_analysis.get("coaching", {})
                    
                    # Handle both flat and nested response formats
                    regime = context.get("daily_regime") or llm_analysis.get("regime", "unknown")
                    always_in = context.get("always_in_direction") or llm_analysis.get("always_in", "neutral")
                    
                    what_good = coaching.get("what_was_good") or llm_analysis.get("what_was_good", [])
                    what_flawed = coaching.get("what_was_flawed") or llm_analysis.get("what_was_flawed", [])
                    
                    return TradeReview(
                        trade_id=trade.id,
                        ticker=trade.ticker,
                        # Context
                        regime=regime,
                        always_in=always_in,
                        context_description=llm_analysis.get("coaching_summary", ""),
                        # Setup
                        setup_classification=setup.get("primary_label") or llm_analysis.get("setup_classification", "unclassified"),
                        setup_quality=entry_quality.get("entry_quality_score", "C")[0].lower() if entry_quality.get("entry_quality_score") else llm_analysis.get("setup_quality", "marginal"),
                        is_second_entry=setup.get("is_second_entry", False),
                        with_trend_or_counter=setup.get("with_trend_or_counter", "neutral"),
                        # Entry quality
                        signal_bar_quality=entry_quality.get("signal_bar_quality", "unknown"),
                        entry_location=entry_quality.get("entry_location", "unknown"),
                        # Risk/Reward
                        risk_reward_assessment=risk_reward.get("target_notes", "") or llm_analysis.get("risk_reward_assessment", ""),
                        probability_assessment=risk_reward.get("probability_estimate", "") or llm_analysis.get("probability_assessment", ""),
                        traders_equation=risk_reward.get("traders_equation", "unknown"),
                        # Management
                        exit_quality=management.get("exit_quality", "unknown"),
                        selection_vs_execution=coaching.get("selection_vs_execution", "unknown"),
                        # Errors
                        errors_detected=llm_analysis.get("errors", []),
                        # Coaching
                        what_was_good=what_good if isinstance(what_good, list) else [what_good] if what_good else [],
                        what_was_flawed=what_flawed if isinstance(what_flawed, list) else [what_flawed] if what_flawed else [],
                        keep_doing=coaching.get("keep_doing", ""),
                        stop_doing=coaching.get("stop_doing", ""),
                        rule_for_next_time=coaching.get("rule_for_next_20_trades", "") or llm_analysis.get("rule_for_next_time", ""),
                        better_alternative=coaching.get("better_alternative", ""),
                        # Metrics
                        r_multiple=trade.r_multiple or 0,
                        mae=trade.mae,
                        mfe=trade.mfe,
                        # Grade
                        grade=llm_analysis.get("grade", "C"),
                        grade_explanation=llm_analysis.get("grade_explanation", ""),
                    )

            # Fallback to basic rule-based analysis
            return self._fallback_review(trade, ohlcv_context)

        finally:
            session.close()

    def _get_ohlcv_context_string(self, trade: Trade) -> str:
        """
        Get OHLCV data as a string for LLM context.
        
        Fetches different data based on trade timeframe:
        - 5m (scalp): 30 daily bars, 60 hourly bars, 234 5-min bars
        - 2h (swing): 60 daily bars, 120 hourly bars
        - 1d (position): 100+ daily bars
        """
        try:
            end_date = datetime.combine(trade.trade_date, datetime.min.time())
            timeframe = trade.timeframe or "5m"
            
            logger.info(f"📊 Fetching multi-timeframe OHLCV for {trade.ticker} (trade timeframe: {timeframe})")
            
            all_context = []
            
            # Brooks-specified bar counts for comprehensive context:
            # - Daily: 60 bars (multi-week trend/range context)
            # - 2-Hour: 120 bars (structure, legs, wedges, tests)
            # - 5-Min: 234 bars (3 trading days = 78 bars/day × 3)
            
            if timeframe == "5m":
                # 5-min scalp/day trades: Full Brooks package
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "1d", end_date, 60, "DAILY"))
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "2h", end_date, 120, "2-HOUR"))
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "5m", end_date, 234, "5-MINUTE"))
            
            elif timeframe == "2h":
                # 2-hour swing trades: Daily + 2H context
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "1d", end_date, 60, "DAILY"))
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "2h", end_date, 120, "2-HOUR"))
            
            elif timeframe == "1d":
                # Daily position trades: Extended daily context
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "1d", end_date, 120, "DAILY"))
            
            else:
                # Default to 5m timeframe with full Brooks package
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "1d", end_date, 60, "DAILY"))
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "2h", end_date, 120, "2-HOUR"))
                all_context.append(self._fetch_ohlcv_section(trade.ticker, "5m", end_date, 234, "5-MINUTE"))
            
            result = "\n\n".join([ctx for ctx in all_context if ctx])
            if not result:
                return "No market data available"
            return result
            
        except Exception as e:
            logger.warning(f"Failed to get OHLCV context: {e}")
            return "Market data unavailable"
    
    def _fetch_ohlcv_section(self, ticker: str, interval: str, end_date: datetime, num_bars: int, label: str) -> str:
        """Fetch OHLCV data for a specific interval and format it."""
        try:
            # Calculate start date based on interval
            if interval == "1d":
                start_date = end_date - timedelta(days=num_bars + 10)  # Extra padding
            elif interval == "1h":
                start_date = end_date - timedelta(hours=num_bars + 10)
            elif interval == "5m":
                start_date = end_date - timedelta(minutes=num_bars * 5 + 60)
            else:
                start_date = end_date - timedelta(days=num_bars)
            
            logger.info(f"   📈 Fetching {label} ({interval}) bars for {ticker}: requesting {num_bars} bars")
            
            df = get_cached_ohlcv(ticker, interval, start_date, end_date)
            
            if df.empty:
                logger.warning(f"   ⚠️ No {label} data returned for {ticker}")
                return ""
            
            # Get the last N bars
            recent = df.tail(num_bars)
            logger.info(f"   ✅ Got {len(recent)} {label} candles for {ticker}")
            
            # Format for LLM
            lines = [f"=== {label} CANDLES ({interval}) - {len(recent)} bars ==="]
            lines.append("DateTime | Open | High | Low | Close | Volume")
            
            for _, row in recent.iterrows():
                if interval == "1d":
                    dt = row["datetime"].strftime("%Y-%m-%d") if hasattr(row["datetime"], "strftime") else str(row["datetime"])[:10]
                else:
                    dt = row["datetime"].strftime("%Y-%m-%d %H:%M") if hasattr(row["datetime"], "strftime") else str(row["datetime"])[:16]
                lines.append(f"{dt} | {row['open']:.2f} | {row['high']:.2f} | {row['low']:.2f} | {row['close']:.2f} | {int(row['volume'])}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.warning(f"Failed to fetch {label} data for {ticker}: {e}")
            return ""

    def _fallback_review(self, trade: Trade, ohlcv_context: str) -> TradeReview:
        """Basic rule-based review when LLM is unavailable."""
        # Simple analysis based on R-multiple
        r = trade.r_multiple or 0
        
        if r > 1:
            grade = "B"
            grade_exp = "Profitable trade"
        elif r > 0:
            grade = "C"
            grade_exp = "Small winner"
        elif r > -1:
            grade = "C"
            grade_exp = "Controlled loss"
        else:
            grade = "D"
            grade_exp = "Large loss - review stop management"

        errors = []
        if r < -1.5:
            errors.append("Loss exceeded 1.5R - consider tighter stops")
        if not trade.entry_reason:
            errors.append("No entry reason documented")

        goods = []
        if r > 0:
            goods.append(f"Profitable trade: +{r:.2f}R")
        if trade.entry_reason:
            goods.append("Entry reason documented")

        flaws = []
        if r < 0:
            flaws.append(f"Loss: {r:.2f}R")
        if not trade.stop_price:
            flaws.append("No stop defined")

        return TradeReview(
            trade_id=trade.id,
            ticker=trade.ticker,
            regime="unknown",
            always_in="neutral",
            context_description="LLM unavailable - basic analysis only. Set OPENAI_API_KEY for full analysis.",
            setup_classification=trade.strategy.name if trade.strategy else "unclassified",
            setup_quality="marginal",
            risk_reward_assessment="Unable to assess without LLM",
            probability_assessment="Unable to assess without LLM",
            errors_detected=errors,
            what_was_good=goods if goods else ["Trade was logged"],
            what_was_flawed=flaws if flaws else [],
            rule_for_next_time="Set OPENAI_API_KEY for detailed coaching",
            r_multiple=r,
            mae=trade.mae,
            mfe=trade.mfe,
            grade=grade,
            grade_explanation=grade_exp,
        )

    def _analyze_context(self, trade: Trade) -> dict:
        """Analyze market context at time of trade."""
        try:
            # Get daily data for context
            end_date = datetime.combine(trade.trade_date, datetime.min.time())
            start_date = end_date - timedelta(days=60)

            daily_df = get_cached_ohlcv(trade.ticker, "1d", start_date, end_date)

            if daily_df.empty:
                return {
                    "regime": "unknown",
                    "always_in": "neutral",
                    "description": "Unable to fetch market data for context analysis",
                }

            features = OHLCFeatures(daily_df)
            detector = BrooksPatternDetector(features)
            regime_analysis = detector.analyze_regime()

            return {
                "regime": regime_analysis.regime.value,
                "always_in": regime_analysis.always_in.value,
                "confidence": regime_analysis.confidence.value,
                "description": regime_analysis.description,
                "metrics": regime_analysis.metrics,
            }

        except Exception as e:
            logger.warning(f"Failed to analyze context: {e}")
            return {
                "regime": "unknown",
                "always_in": "neutral",
                "description": f"Context analysis failed: {e}",
            }

    def _classify_setup(self, trade: Trade, context: dict) -> tuple[str, str]:
        """Classify the trade setup and assess quality."""
        # Use existing strategy if set
        if trade.strategy:
            setup_class = trade.strategy.name
        elif trade.setup_type:
            setup_class = trade.setup_type
        else:
            setup_class = self._infer_setup_type(trade, context)

        # Assess quality based on context alignment
        quality = self._assess_setup_quality(trade, context, setup_class)

        return setup_class, quality

    def _infer_setup_type(self, trade: Trade, context: dict) -> str:
        """Infer setup type from trade characteristics and context."""
        regime = context.get("regime", "unknown")
        always_in = context.get("always_in", "neutral")
        direction = trade.direction

        # With-trend trades
        if regime == "trend_up" and direction == TradeDirection.LONG:
            return "with_trend_long"
        elif regime == "trend_down" and direction == TradeDirection.SHORT:
            return "with_trend_short"

        # Countertrend trades
        elif regime == "trend_up" and direction == TradeDirection.SHORT:
            return "countertrend_short"
        elif regime == "trend_down" and direction == TradeDirection.LONG:
            return "countertrend_long"

        # Trading range
        elif regime == "trading_range":
            return "range_trade"

        return "unclassified"

    def _assess_setup_quality(self, trade: Trade, context: dict, setup_class: str) -> str:
        """Assess setup quality based on context alignment."""
        regime = context.get("regime", "unknown")
        always_in = context.get("always_in", "neutral")
        direction = trade.direction

        # Good: trading with always-in direction
        if always_in == "long" and direction == TradeDirection.LONG:
            return "good"
        elif always_in == "short" and direction == TradeDirection.SHORT:
            return "good"

        # Poor: trading against strong always-in
        elif always_in == "long" and direction == TradeDirection.SHORT:
            if regime == "trend_up":
                return "poor"
            return "marginal"
        elif always_in == "short" and direction == TradeDirection.LONG:
            if regime == "trend_down":
                return "poor"
            return "marginal"

        # Neutral context
        return "marginal"

    def _assess_risk_reward(self, trade: Trade) -> str:
        """Assess risk/reward of the trade."""
        if not trade.entry_price or not trade.stop_price:
            return "Risk/reward could not be assessed (missing stop)"

        risk = abs(trade.entry_price - trade.stop_price)

        if trade.target_price:
            reward = abs(trade.target_price - trade.entry_price)
            rr_ratio = reward / risk if risk > 0 else 0

            if rr_ratio >= 2:
                return f"Good R:R of {rr_ratio:.1f}:1 - reward justifies risk"
            elif rr_ratio >= 1:
                return f"Marginal R:R of {rr_ratio:.1f}:1 - needs high win rate to be profitable"
            else:
                return f"Poor R:R of {rr_ratio:.1f}:1 - risking more than potential reward"
        else:
            return "No target set - consider defining exit targets before entry"

    def _assess_probability(self, trade: Trade, context: dict) -> str:
        """Assess probability of trade success based on context."""
        regime = context.get("regime", "unknown")
        always_in = context.get("always_in", "neutral")
        direction = trade.direction

        if regime == "trend_up" and direction == TradeDirection.LONG:
            return "HIGH probability - trading with the trend"
        elif regime == "trend_down" and direction == TradeDirection.SHORT:
            return "HIGH probability - trading with the trend"
        elif regime == "trading_range":
            return "MEDIUM probability - range-bound, need clear levels"
        elif (regime == "trend_up" and direction == TradeDirection.SHORT) or \
             (regime == "trend_down" and direction == TradeDirection.LONG):
            return "LOW probability - countertrend trade, needs strong reversal structure"
        else:
            return "UNKNOWN probability - context unclear"

    def _detect_errors(self, trade: Trade, context: dict) -> list[str]:
        """Detect common Brooks errors in the trade."""
        errors = []
        regime = context.get("regime", "unknown")
        always_in = context.get("always_in", "neutral")
        direction = trade.direction

        # Error 1: Countertrend without reversal structure
        if regime == "trend_up" and direction == TradeDirection.SHORT:
            if not trade.entry_reason or "reversal" not in trade.entry_reason.lower():
                errors.append(
                    "COUNTERTREND WITHOUT CLEAR REVERSAL: Faded an uptrend without "
                    "documented reversal structure (need strong bear bar, break of trendline + test)"
                )

        if regime == "trend_down" and direction == TradeDirection.LONG:
            if not trade.entry_reason or "reversal" not in trade.entry_reason.lower():
                errors.append(
                    "COUNTERTREND WITHOUT CLEAR REVERSAL: Bought in a downtrend without "
                    "documented reversal structure"
                )

        # Error 2: Scalp with poor R:R
        if trade.r_multiple and trade.r_multiple > 0 and trade.r_multiple < 0.5:
            if trade.target_price and trade.entry_price and trade.stop_price:
                risk = abs(trade.entry_price - trade.stop_price)
                target_r = abs(trade.target_price - trade.entry_price) / risk if risk > 0 else 0
                if target_r < 1:
                    errors.append(
                        "POOR SCALP MATH: Target was less than 1R - needs very high "
                        "win rate (>60%) to be profitable long-term"
                    )

        # Error 3: Trading tight range with stops
        if regime == "trading_range":
            if trade.setup_type and "stop" in trade.setup_type.lower():
                errors.append(
                    "STOP ENTRY IN TIGHT RANGE: Stop entries often fail in trading ranges "
                    "due to two-sided price action - consider limit entries or waiting for breakout"
                )

        # Error 4: Large MAE suggests poor entry
        if trade.mae and trade.mae > 1.5:
            errors.append(
                f"POOR ENTRY LOCATION: MAE of {trade.mae:.1f}R suggests entering at "
                "poor location or sizing too large - wait for better entry or reduce size"
            )

        # Error 5: Left money on table
        if trade.mfe and trade.r_multiple:
            if trade.mfe > trade.r_multiple + 1.5 and trade.r_multiple > 0:
                errors.append(
                    f"LEFT MONEY ON TABLE: MFE was {trade.mfe:.1f}R but only captured "
                    f"{trade.r_multiple:.1f}R - consider trailing stops or scaling out"
                )

        # Error 6: Holding loser too long
        if trade.r_multiple and trade.r_multiple < -1.5:
            errors.append(
                f"HELD LOSER TOO LONG: Lost {abs(trade.r_multiple):.1f}R - should have "
                "exited at initial stop. Honor your stops."
            )

        return errors

    def _find_what_was_good(self, trade: Trade, context: dict) -> list[str]:
        """Identify positive aspects of the trade."""
        goods = []

        # Won the trade
        if trade.r_multiple and trade.r_multiple > 0:
            goods.append(f"Profitable trade: +{trade.r_multiple:.2f}R")

        # Traded with trend
        if context.get("always_in") == "long" and trade.direction == TradeDirection.LONG:
            goods.append("Traded with the always-in direction (long)")
        elif context.get("always_in") == "short" and trade.direction == TradeDirection.SHORT:
            goods.append("Traded with the always-in direction (short)")

        # Had a documented plan
        if trade.entry_reason:
            goods.append("Had documented entry reason")
        if trade.stop_price and trade.target_price:
            goods.append("Defined stop and target before entry")

        # Controlled risk
        if trade.mae and trade.mae < 1:
            goods.append(f"Good entry location: MAE only {trade.mae:.1f}R")

        # Captured the move
        if trade.mfe and trade.r_multiple and trade.mfe > 0:
            capture_pct = trade.r_multiple / trade.mfe if trade.mfe > 0 else 0
            if capture_pct > 0.7:
                goods.append(f"Captured most of the move ({capture_pct:.0%} of MFE)")

        if not goods:
            goods.append("Trade was logged - tracking is the first step to improvement")

        return goods

    def _find_what_was_flawed(self, trade: Trade, context: dict, errors: list[str]) -> list[str]:
        """Identify flaws in the trade."""
        flaws = []

        # Lost the trade
        if trade.r_multiple and trade.r_multiple < 0:
            flaws.append(f"Losing trade: {trade.r_multiple:.2f}R")

        # Add detected errors as flaws
        for error in errors:
            # Extract the key issue (first part before colon)
            if ":" in error:
                flaw = error.split(":")[0]
                flaws.append(flaw)

        # Missing documentation
        if not trade.entry_reason:
            flaws.append("No documented entry reason")
        if not trade.stop_price:
            flaws.append("No stop loss defined")

        return flaws[:3]  # Top 3 flaws

    def _generate_rule(self, trade: Trade, errors: list[str], flaws: list[str]) -> str:
        """Generate a concrete rule for improvement."""
        # Prioritize based on most impactful error
        if any("COUNTERTREND" in e for e in errors):
            return (
                "RULE: Before taking countertrend trades, require: "
                "(1) strong reversal bar, (2) break of trendline, (3) successful test. "
                "If any missing, pass on the trade."
            )

        if any("SCALP MATH" in e for e in errors):
            return (
                "RULE: Scalps must target at least 1R reward. "
                "If target is less than 1R, either widen target or pass on trade."
            )

        if any("STOP ENTRY IN TIGHT RANGE" in e for e in errors):
            return (
                "RULE: In trading ranges, use limit orders at range extremes "
                "instead of stop entries. Wait for failed breakout for better entry."
            )

        if any("HELD LOSER" in e for e in errors):
            return (
                "RULE: Exit at initial stop, no exceptions. "
                "If stopped out, can re-enter with fresh stop - but honor the first stop."
            )

        if any("LEFT MONEY" in e for e in errors):
            return (
                "RULE: In trending markets, trail stop below prior swing instead of "
                "using fixed targets. Let winners run."
            )

        if "No documented entry reason" in flaws:
            return (
                "RULE: Write entry reason BEFORE clicking buy/sell. "
                "No reason = no trade."
            )

        # Default rule
        return "RULE: Review this trade in your weekly analysis to identify patterns."

    def _grade_trade(self, trade: Trade, context: dict, errors: list[str]) -> tuple[str, str]:
        """Grade the trade from A to F."""
        score = 100

        # Deductions for errors
        score -= len(errors) * 15

        # Deductions for going against trend
        if context.get("regime") == "trend_up" and trade.direction == TradeDirection.SHORT:
            score -= 20
        elif context.get("regime") == "trend_down" and trade.direction == TradeDirection.LONG:
            score -= 20

        # Deductions for missing documentation
        if not trade.entry_reason:
            score -= 10
        if not trade.stop_price:
            score -= 15

        # Bonus for winning
        if trade.r_multiple and trade.r_multiple > 0:
            score += 10

        # Bonus for good entry
        if trade.mae and trade.mae < 0.5:
            score += 10

        # Determine grade
        if score >= 90:
            grade = "A"
            explanation = "Excellent trade execution and context alignment"
        elif score >= 75:
            grade = "B"
            explanation = "Good trade with minor areas for improvement"
        elif score >= 60:
            grade = "C"
            explanation = "Average trade - review and learn from mistakes"
        elif score >= 40:
            grade = "D"
            explanation = "Below average - significant issues to address"
        else:
            grade = "F"
            explanation = "Poor trade - major errors, needs immediate attention"

        return grade, explanation

    def format_review(self, review: TradeReview) -> str:
        """Format review as markdown text."""
        lines = [
            f"# Trade Review: {review.ticker} (ID: {review.trade_id})",
            "",
            "## Context",
            f"- **Regime**: {review.regime}",
            f"- **Always-In**: {review.always_in}",
            f"- {review.context_description}",
            "",
            "## Setup",
            f"- **Classification**: {review.setup_classification}",
            f"- **Quality**: {review.setup_quality.upper()}",
            "",
            "## Trader's Equation",
            f"- **Risk/Reward**: {review.risk_reward_assessment}",
            f"- **Probability**: {review.probability_assessment}",
            "",
            "## Performance",
            f"- **R-Multiple**: {review.r_multiple:+.2f}R",
        ]

        if review.mae is not None:
            lines.append(f"- **MAE**: {review.mae:.2f}R")
        if review.mfe is not None:
            lines.append(f"- **MFE**: {review.mfe:.2f}R")

        lines.extend([
            "",
            f"## Grade: {review.grade}",
            f"*{review.grade_explanation}*",
            "",
        ])

        if review.errors_detected:
            lines.append("## Errors Detected")
            for error in review.errors_detected:
                lines.append(f"- ⚠️ {error}")
            lines.append("")

        lines.append("## Coaching")
        lines.append("")
        lines.append("### What Was Good")
        for good in review.what_was_good:
            lines.append(f"- ✅ {good}")

        lines.append("")
        lines.append("### What Was Flawed")
        for flaw in review.what_was_flawed:
            lines.append(f"- ❌ {flaw}")

        lines.extend([
            "",
            "### Rule for Next Time",
            f"📌 {review.rule_for_next_time}",
        ])

        return "\n".join(lines)

    def quick_review(self, trade_id: int) -> str:
        """Get a quick one-paragraph review of a trade."""
        review = self.review_trade(trade_id)
        if not review:
            return f"Trade {trade_id} not found."

        result = review.r_multiple
        result_str = f"+{result:.2f}R" if result >= 0 else f"{result:.2f}R"

        summary = (
            f"Trade {review.ticker}: {result_str}, Grade {review.grade}. "
            f"Context was {review.regime} with always-in {review.always_in}. "
            f"Setup quality: {review.setup_quality}. "
        )

        if review.errors_detected:
            summary += f"Main issue: {review.errors_detected[0].split(':')[0]}. "

        summary += f"Focus: {review.rule_for_next_time}"

        return summary
