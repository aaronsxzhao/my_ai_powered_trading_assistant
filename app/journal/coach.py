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
    """Complete review of a trade."""

    trade_id: int
    ticker: str

    # Context analysis
    regime: str
    always_in: str
    context_description: str

    # Setup classification
    setup_classification: str
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

            # Use LLM for comprehensive analysis
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
                )

                if "error" not in llm_analysis and "raw_analysis" not in llm_analysis:
                    return TradeReview(
                        trade_id=trade.id,
                        ticker=trade.ticker,
                        regime=llm_analysis.get("regime", "unknown"),
                        always_in=llm_analysis.get("always_in", "neutral"),
                        context_description=llm_analysis.get("overall_coaching", ""),
                        setup_classification=llm_analysis.get("setup_classification", "unclassified"),
                        setup_quality=llm_analysis.get("setup_quality", "marginal"),
                        risk_reward_assessment=llm_analysis.get("risk_reward_assessment", ""),
                        probability_assessment=llm_analysis.get("probability_assessment", ""),
                        errors_detected=llm_analysis.get("errors", []),
                        what_was_good=llm_analysis.get("what_was_good", []),
                        what_was_flawed=llm_analysis.get("what_was_flawed", []),
                        rule_for_next_time=llm_analysis.get("rule_for_next_time", ""),
                        r_multiple=trade.r_multiple or 0,
                        mae=trade.mae,
                        mfe=trade.mfe,
                        grade=llm_analysis.get("grade", "C"),
                        grade_explanation=llm_analysis.get("grade_explanation", ""),
                    )

            # Fallback to basic rule-based analysis
            return self._fallback_review(trade, ohlcv_context)

        finally:
            session.close()

    def _get_ohlcv_context_string(self, trade: Trade) -> str:
        """Get OHLCV data as a string for LLM context."""
        try:
            end_date = datetime.combine(trade.trade_date, datetime.min.time())
            start_date = end_date - timedelta(days=30)

            df = get_cached_ohlcv(trade.ticker, "1d", start_date, end_date)

            if df.empty:
                return "No market data available"

            # Format last 10 bars for context
            recent = df.tail(10)
            lines = ["Date | Open | High | Low | Close | Volume"]
            for _, row in recent.iterrows():
                dt = row["datetime"].strftime("%Y-%m-%d") if hasattr(row["datetime"], "strftime") else str(row["datetime"])[:10]
                lines.append(f"{dt} | {row['open']:.2f} | {row['high']:.2f} | {row['low']:.2f} | {row['close']:.2f} | {int(row['volume'])}")

            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"Failed to get OHLCV context: {e}")
            return "Market data unavailable"

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
