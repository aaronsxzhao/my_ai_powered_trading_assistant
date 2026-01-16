"""
Brooks-style Trade Coach for post-trade review.

Uses LLM (Claude/OpenAI) for intelligent trade analysis and coaching.
Falls back to rule-based analysis if LLM is unavailable.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Literal, Union
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
    rule_for_next_time: list[str]  # List of actionable rules

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
        self._data_provider = None

    @property
    def llm_analyzer(self):
        """Lazy load LLM analyzer."""
        if self._llm_analyzer is None:
            from app.llm.analyzer import get_analyzer
            self._llm_analyzer = get_analyzer()
        return self._llm_analyzer

    @property
    def data_provider(self):
        """Lazy load data provider."""
        if self._data_provider is None:
            from app.data.providers import get_provider
            self._data_provider = get_provider()
        return self._data_provider

    def _get_provider_for_ticker(self, ticker: str):
        """Get the best provider for a specific ticker (e.g., AllTick for HK)."""
        try:
            from app.data.providers import get_provider_for_ticker
            return get_provider_for_ticker(ticker)
        except ImportError:
            return self.data_provider
        return self._data_provider

    def review_trade(self, trade_id: int, cancellation_check: Optional[callable] = None) -> Optional[TradeReview]:
        """
        Perform comprehensive review of a trade using LLM analysis.

        Args:
            trade_id: Trade ID to review
            cancellation_check: Optional function that returns True if generation should stop

        Returns:
            TradeReview object or None if trade not found or cancelled
        """
        def is_cancelled():
            if cancellation_check and cancellation_check():
                logger.info(f"⏹️ Review cancelled for trade {trade_id}")
                return True
            return False
        
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                logger.warning(f"Trade {trade_id} not found")
                return None

            # Check for cancellation before expensive OHLCV fetch
            if is_cancelled():
                return None

            # Get market context data (separate sections) and daily DataFrame for session context
            ohlcv_context, daily_df = self._get_ohlcv_context_string(trade, cancellation_check=is_cancelled, return_daily_df=True)
            
            # Check for cancellation after OHLCV fetch
            if is_cancelled():
                return None
            
            # Get prior day levels and session context (reusing daily data to avoid extra API calls)
            pd_high, pd_low, pd_close, today_open = self._get_session_context(trade, daily_ohlcv=daily_df)

            # Use LLM for comprehensive Brooks Audit
            if self.llm_analyzer.is_available:
                llm_analysis = self.llm_analyzer.analyze_trade(
                    ticker=trade.ticker,
                    direction=trade.direction.value,
                    entry_price=trade.entry_price,
                    exit_price=trade.exit_price,
                    stop_price=trade.effective_stop_loss,
                    target_price=trade.effective_take_profit,
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
                    mistakes_and_lessons=trade.effective_mistakes_lessons,
                    # New extended fields
                    trade_date=str(trade.trade_date) if trade.trade_date else None,
                    market=getattr(trade, 'market', None) or "US stocks",
                    timezone=getattr(trade, 'market_timezone', None) or "America/New_York",
                    intended_setup=trade.entry_reason,
                    management_plan=trade.target_reason,
                    pd_high=pd_high,
                    pd_low=pd_low,
                    pd_close=pd_close,
                    today_open=today_open,
                    # Extended Brooks analysis fields
                    trend_assessment=getattr(trade, 'trend_assessment', None),
                    signal_reason=getattr(trade, 'signal_reason', None),
                    was_signal_present=getattr(trade, 'was_signal_present', None),
                    strategy_alignment=getattr(trade, 'strategy_alignment', None),
                    entry_exit_emotions=getattr(trade, 'entry_exit_emotions', None),
                    entry_tp_distance=getattr(trade, 'entry_tp_distance', None),
                    # Cancellation support
                    cancellation_check=is_cancelled,
                )

                # If cancelled during analysis, return None
                if llm_analysis.get("error") == "Cancelled":
                    return None

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
                        what_was_good=self._ensure_list(what_good),
                        what_was_flawed=self._ensure_list(what_flawed),
                        keep_doing=coaching.get("keep_doing", ""),
                        stop_doing=coaching.get("stop_doing", ""),
                        rule_for_next_time=self._ensure_list(
                            coaching.get("rules_for_next_20_trades") or 
                            coaching.get("rule_for_next_20_trades") or 
                            llm_analysis.get("rules_for_next_time") or
                            llm_analysis.get("rule_for_next_time")
                        ),
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

    def _get_ohlcv_context_string(self, trade: Trade, cancellation_check: Optional[callable] = None, return_daily_df: bool = False) -> Union[str, tuple]:
        """
        Get OHLCV data as a string for LLM context.

        IMPORTANT: Uses entry_time as the cutoff to avoid look-ahead bias.
        Only data that was available at entry time is included.

        Fetches different data based on trade timeframe:
        - 5m (scalp): 60 daily bars, 120 2-hour bars, 234 5-min bars
        - 2h (swing): 60 daily bars, 120 2-hour bars
        - 1d (position): 120 daily bars
        
        Args:
            trade: Trade object
            cancellation_check: Optional callable to check for cancellation
            return_daily_df: If True, returns (context_string, daily_dataframe) tuple
        
        Returns:
            String context, or tuple (context_string, daily_df) if return_daily_df=True
        
        Args:
            trade: Trade object
            cancellation_check: Optional function that returns True if should stop
        """
        try:
            # Use entry_time as cutoff to avoid look-ahead bias
            if trade.entry_time:
                entry_cutoff = trade.entry_time
            else:
                entry_cutoff = datetime.combine(trade.trade_date, datetime.max.time().replace(microsecond=0))

            timeframe = trade.timeframe or "5m"

            all_context = []
            daily_df = None  # Store daily DataFrame for session context reuse

            # Brooks-specified bar counts for comprehensive context:
            # - Daily: 60 bars (multi-week trend/range context) - up to prior day close
            # - 2-Hour: 120 bars (structure, legs, wedges, tests) - up to entry time
            # - 5-Min: 234 bars (3 trading days) - up to entry time

            # For daily bars, use the day BEFORE entry to avoid partial day data for LLM
            # But fetch up to trade date to get today's open for session context
            daily_cutoff = datetime.combine(trade.trade_date - timedelta(days=1), datetime.max.time())
            daily_fetch_end = datetime.combine(trade.trade_date, datetime.max.time())  # Include trade date for session context

            if timeframe == "5m":
                # 5-min scalp/day trades: Full Brooks package
                # Check cancellation between each fetch (rate-limited operations)
                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                daily_context, daily_df = self._fetch_ohlcv_section_with_cutoff_and_df(trade.ticker, "1d", daily_fetch_end, 65, "DAILY (up to prior day close)", daily_cutoff, cancellation_check)
                all_context.append(daily_context)

                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                all_context.append(self._fetch_ohlcv_section_with_cutoff(trade.ticker, "2h", entry_cutoff, 120, "2-HOUR (up to entry time)", cancellation_check))

                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                all_context.append(self._fetch_ohlcv_section_with_cutoff(trade.ticker, "5m", entry_cutoff, 234, "5-MINUTE (up to entry time)", cancellation_check))

            elif timeframe == "2h":
                # 2-hour swing trades: Daily + 2H context
                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                daily_context, daily_df = self._fetch_ohlcv_section_with_cutoff_and_df(trade.ticker, "1d", daily_fetch_end, 65, "DAILY (up to prior day close)", daily_cutoff, cancellation_check)
                all_context.append(daily_context)

                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                all_context.append(self._fetch_ohlcv_section_with_cutoff(trade.ticker, "2h", entry_cutoff, 120, "2-HOUR (up to entry time)", cancellation_check))

            elif timeframe == "1d":
                # Daily position trades: Extended daily context
                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                daily_context, daily_df = self._fetch_ohlcv_section_with_cutoff_and_df(trade.ticker, "1d", daily_fetch_end, 125, "DAILY (up to prior day close)", daily_cutoff, cancellation_check)
                all_context.append(daily_context)

            else:
                # Default to 5m timeframe with full Brooks package
                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                daily_context, daily_df = self._fetch_ohlcv_section_with_cutoff_and_df(trade.ticker, "1d", daily_fetch_end, 65, "DAILY (up to prior day close)", daily_cutoff, cancellation_check)
                all_context.append(daily_context)

                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                all_context.append(self._fetch_ohlcv_section_with_cutoff(trade.ticker, "2h", entry_cutoff, 120, "2-HOUR (up to entry time)", cancellation_check))

                if cancellation_check and cancellation_check():
                    return ("Cancelled", None) if return_daily_df else "Cancelled"
                all_context.append(self._fetch_ohlcv_section_with_cutoff(trade.ticker, "5m", entry_cutoff, 234, "5-MINUTE (up to entry time)", cancellation_check))

            result = "\n\n".join([ctx for ctx in all_context if ctx])
            if not result:
                logger.warning(f"⚠️ No OHLCV data collected for {trade.ticker}")
                return ("No market data available", daily_df) if return_daily_df else "No market data available"
            
            # Check for any sections that indicate no data or errors
            has_real_data = False
            for ctx in all_context:
                if ctx and "No data" not in ctx and "failed" not in ctx.lower() and len(ctx) > 100:
                    has_real_data = True
                    break
            
            if not has_real_data:
                logger.warning(f"⚠️ OHLCV sections contain no real data for {trade.ticker}")
            
            # Log summary of what was fetched
            logger.info(f"📊 OHLCV fetched for {trade.ticker}: {len(all_context)} sections, {len(result)} chars total, has_real_data={has_real_data}")
            return (result, daily_df) if return_daily_df else result

        except Exception as e:
            logger.warning(f"Failed to get OHLCV context: {e}")
            return ("Market data unavailable", None) if return_daily_df else "Market data unavailable"
    
    def _fetch_ohlcv_section_with_cutoff(self, ticker: str, interval: str, cutoff_time: datetime, num_bars: int, label: str, cancellation_check: callable = None) -> str:
        """Fetch OHLCV data up to a specific cutoff time (no future data).
        
        Args:
            ticker: Stock ticker
            interval: Time interval (1d, 2h, 5m, etc.)
            cutoff_time: Maximum timestamp - no data after this point
            num_bars: Number of bars to fetch
            label: Label for this section
        """
        try:
            # Calculate start date based on interval
            if interval == "1d":
                start_date = cutoff_time - timedelta(days=num_bars + 10)
            elif interval == "2h":
                start_date = cutoff_time - timedelta(hours=num_bars * 2 + 20)
            elif interval == "1h":
                start_date = cutoff_time - timedelta(hours=num_bars + 10)
            elif interval == "5m":
                start_date = cutoff_time - timedelta(minutes=num_bars * 5 + 100)
            else:
                start_date = cutoff_time - timedelta(days=num_bars + 10)
            
            # Check for cancellation before fetch
            if cancellation_check and cancellation_check():
                return f"=== {label} ===\nCancelled"
            
            # Use ticker-specific provider (e.g., AllTick for HK stocks)
            provider = self._get_provider_for_ticker(ticker)
            ohlcv = provider.get_ohlcv(
                ticker,
                interval,
                start_date,
                cutoff_time,  # Use cutoff_time as end date
                cancellation_check=cancellation_check
            )
            
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"No {interval} data returned for {ticker}")
                return f"=== {label} ===\nNo data available"
            
            # Filter to ensure no data after cutoff (extra safety)
            if 'timestamp' in ohlcv.columns:
                ohlcv = ohlcv[ohlcv['timestamp'] <= cutoff_time]
            
            # Limit to requested number of bars (most recent ones before cutoff)
            ohlcv = ohlcv.tail(num_bars)
            
            if ohlcv.empty:
                return f"=== {label} ===\nNo data available before cutoff"
            
            # Format as readable string
            lines = [f"=== {label} ==="]
            lines.append(f"Cutoff: {cutoff_time.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"Bars: {len(ohlcv)}")
            lines.append("timestamp, open, high, low, close, volume")
            
            for _, row in ohlcv.iterrows():
                ts = row.get('timestamp', row.name)
                if hasattr(ts, 'strftime'):
                    ts_str = ts.strftime('%Y-%m-%d %H:%M')
                else:
                    ts_str = str(ts)
                
                lines.append(
                    f"{ts_str}, {row['open']:.4f}, {row['high']:.4f}, "
                    f"{row['low']:.4f}, {row['close']:.4f}, {int(row.get('volume', 0))}"
                )
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.warning(f"Failed to fetch {interval} data for {ticker}: {e}")
            return f"=== {label} ===\nData fetch failed: {e}"

    def _fetch_ohlcv_section_with_cutoff_and_df(self, ticker: str, interval: str, fetch_end: datetime, num_bars: int, label: str, display_cutoff: datetime, cancellation_check: callable = None) -> tuple:
        """Fetch OHLCV data and return both formatted string and raw DataFrame.
        
        This is used for daily data to reuse it for session context.
        
        Args:
            ticker: Stock ticker
            interval: Time interval (1d, 2h, 5m, etc.)
            fetch_end: End time for data fetch (includes trade date for session context)
            num_bars: Number of bars to fetch
            label: Label for this section
            display_cutoff: Cutoff time for display (prior day close for LLM context)
            cancellation_check: Optional cancellation check callable
        
        Returns:
            tuple: (formatted_string, raw_dataframe)
        """
        try:
            # Calculate start date
            start_date = fetch_end - timedelta(days=num_bars + 10)
            
            # Check for cancellation before fetch
            if cancellation_check and cancellation_check():
                return f"=== {label} ===\nCancelled", None
            
            # Fetch data up to fetch_end (includes trade date)
            provider = self._get_provider_for_ticker(ticker)
            ohlcv = provider.get_ohlcv(
                ticker,
                interval,
                start_date,
                fetch_end,
                cancellation_check=cancellation_check
            )
            
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"No {interval} data returned for {ticker}")
                return f"=== {label} ===\nNo data available", None
            
            # Store full DataFrame for session context
            full_df = ohlcv.copy()
            
            # Filter for display (only up to display_cutoff for LLM context)
            if 'timestamp' in ohlcv.columns:
                ohlcv = ohlcv[ohlcv['timestamp'] <= display_cutoff]
            
            # Limit to requested number of bars
            ohlcv = ohlcv.tail(num_bars)
            
            if ohlcv.empty:
                return f"=== {label} ===\nNo data available before cutoff", full_df
            
            # Format as readable string
            lines = [f"=== {label} ==="]
            lines.append(f"Cutoff: {display_cutoff.strftime('%Y-%m-%d %H:%M')}")
            lines.append(f"Bars: {len(ohlcv)}")
            lines.append("timestamp, open, high, low, close, volume")
            
            for _, row in ohlcv.iterrows():
                ts = row.get('timestamp', row.name)
                if hasattr(ts, 'strftime'):
                    ts_str = ts.strftime('%Y-%m-%d %H:%M')
                else:
                    ts_str = str(ts)
                
                lines.append(
                    f"{ts_str}, {row['open']:.4f}, {row['high']:.4f}, "
                    f"{row['low']:.4f}, {row['close']:.4f}, {int(row.get('volume', 0))}"
                )
            
            return "\n".join(lines), full_df
            
        except Exception as e:
            logger.warning(f"Failed to fetch {interval} data for {ticker}: {e}")
            return f"=== {label} ===\nData fetch failed: {e}", None

    def _ensure_list(self, value) -> list:
        """Ensure value is a list (convert string to single-item list if needed)."""
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str) and value:
            return [value]
        return []

    def _get_session_context(self, trade: Trade, daily_ohlcv: Optional[pd.DataFrame] = None) -> tuple:
        """Get prior day levels and today's open for session context.

        Uses data available BEFORE entry (no look-ahead bias).
        Reuses already-fetched daily OHLCV data if provided to avoid extra API calls.
        
        Args:
            trade: Trade object
            daily_ohlcv: Optional pre-fetched daily OHLCV DataFrame
        
        Returns:
            tuple: (pd_high, pd_low, pd_close, today_open)
        """
        try:
            trade_date = trade.trade_date
            
            # If we have pre-fetched daily data, extract from it
            if daily_ohlcv is not None and not daily_ohlcv.empty:
                # Find prior day and trade day data from the existing DataFrame
                if 'timestamp' in daily_ohlcv.columns:
                    daily_ohlcv = daily_ohlcv.copy()
                    daily_ohlcv['date'] = pd.to_datetime(daily_ohlcv['timestamp']).dt.date
                elif 'datetime' in daily_ohlcv.columns:
                    daily_ohlcv = daily_ohlcv.copy()
                    daily_ohlcv['date'] = pd.to_datetime(daily_ohlcv['datetime']).dt.date
                else:
                    # Try index
                    daily_ohlcv = daily_ohlcv.copy()
                    daily_ohlcv['date'] = pd.to_datetime(daily_ohlcv.index).date
                
                # Prior day data
                prior_day_mask = daily_ohlcv['date'] < trade_date
                prior_days = daily_ohlcv[prior_day_mask]
                
                if not prior_days.empty:
                    prior_day = prior_days.iloc[-1]
                    pd_high = prior_day.get('high')
                    pd_low = prior_day.get('low')
                    pd_close = prior_day.get('close')
                else:
                    pd_high, pd_low, pd_close = None, None, None
                
                # Today's open
                trade_day_mask = daily_ohlcv['date'] == trade_date
                trade_day = daily_ohlcv[trade_day_mask]
                
                if not trade_day.empty:
                    today_open = trade_day.iloc[0].get('open')
                else:
                    today_open = None
                
                return (pd_high, pd_low, pd_close, today_open)
            
            # Fallback: fetch from API (only if no pre-fetched data)
            prior_day_date = trade_date - timedelta(days=1)
            start_date = prior_day_date - timedelta(days=5)
            end_date = datetime.combine(prior_day_date, datetime.max.time())
            
            provider = self._get_provider_for_ticker(trade.ticker)
            ohlcv = provider.get_ohlcv(
                trade.ticker,
                "1d",
                start_date,
                end_date
            )
            
            if ohlcv is None or ohlcv.empty:
                return None, None, None, None
            
            prior_day = ohlcv.iloc[-1]
            
            # For today's open, we need to fetch the trade date's data
            trade_day_start = datetime.combine(trade_date, datetime.min.time())
            trade_day_end = datetime.combine(trade_date, datetime.max.time())
            
            today_ohlcv = provider.get_ohlcv(
                trade.ticker,
                "1d",
                trade_day_start,
                trade_day_end
            )
            
            today_open = None
            if today_ohlcv is not None and not today_ohlcv.empty:
                today_open = today_ohlcv.iloc[-1]['open']
            
            return (
                prior_day['high'],
                prior_day['low'],
                prior_day['close'],
                today_open
            )
            
        except Exception as e:
            logger.warning(f"Failed to get session context: {e}")
            return None, None, None, None

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
            
            df = get_cached_ohlcv(ticker, interval, start_date, end_date)
            
            if df.empty:
                return ""
            
            # Get the last N bars
            recent = df.tail(num_bars)
            
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
        if not trade.effective_stop_loss:
            flaws.append("No stop defined")

        return TradeReview(
            trade_id=trade.id,
            ticker=trade.ticker,
            regime="unknown",
            always_in="neutral",
            context_description="LLM unavailable - basic analysis only. Check your API key in .env file (OPENAI_API_KEY).",
            setup_classification=trade.strategy.name if trade.strategy else "unclassified",
            setup_quality="marginal",
            risk_reward_assessment="Unable to assess without LLM",
            probability_assessment="Unable to assess without LLM",
            errors_detected=errors,
            what_was_good=goods if goods else ["Trade was logged"],
            what_was_flawed=flaws if flaws else [],
            rule_for_next_time=["Set OPENAI_API_KEY for detailed coaching"],
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
        if not trade.entry_price or not trade.effective_stop_loss:
            return "Risk/reward could not be assessed (missing stop)"

        risk = abs(trade.entry_price - trade.effective_stop_loss)

        if trade.effective_take_profit:
            reward = abs(trade.effective_take_profit - trade.entry_price)
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
            if trade.effective_take_profit and trade.entry_price and trade.effective_stop_loss:
                risk = abs(trade.entry_price - trade.effective_stop_loss)
                target_r = abs(trade.effective_take_profit - trade.entry_price) / risk if risk > 0 else 0
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
        if trade.effective_stop_loss and trade.effective_take_profit:
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
        if not trade.effective_stop_loss:
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
        if not trade.effective_stop_loss:
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
            "### Rules for Next Time",
        ])
        if isinstance(review.rule_for_next_time, list):
            for rule in review.rule_for_next_time:
                lines.append(f"📌 {rule}")
        else:
            lines.append(f"📌 {review.rule_for_next_time}")

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

        if isinstance(review.rule_for_next_time, list) and review.rule_for_next_time:
            summary += f"Focus: {review.rule_for_next_time[0]}"
        elif review.rule_for_next_time:
            summary += f"Focus: {review.rule_for_next_time}"

        return summary
