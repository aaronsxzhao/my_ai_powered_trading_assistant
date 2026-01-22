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

    def _save_llm_log(self, trade, ohlcv_context: str, llm_response: dict):
        """
        Save LLM request context and response to local file for debugging/review.
        
        Files are saved to data/llm_logs/ with format:
        trade_{id}_{ticker}_{timestamp}.json
        """
        import json
        from pathlib import Path
        from datetime import datetime
        
        try:
            # Create logs directory
            logs_dir = Path("data/llm_logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_ticker = trade.ticker.replace(":", "_").replace("/", "_")
            filename = f"trade_{trade.id}_{safe_ticker}_{timestamp}.json"
            filepath = logs_dir / filename
            
            # Build log data
            log_data = {
                "metadata": {
                    "trade_id": trade.id,
                    "ticker": trade.ticker,
                    "direction": trade.direction.value if trade.direction else None,
                    "trade_date": str(trade.trade_date) if trade.trade_date else None,
                    "entry_time": str(trade.entry_time) if trade.entry_time else None,
                    "exit_time": str(trade.exit_time) if trade.exit_time else None,
                    "entry_price": trade.entry_price,
                    "exit_price": trade.exit_price,
                    "r_multiple": trade.r_multiple,
                    "timeframe": trade.timeframe,
                    "generated_at": datetime.now().isoformat(),
                },
                "ohlcv_context": ohlcv_context,
                "llm_response": llm_response,
            }
            
            # Write to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"📝 Saved LLM log to {filepath}")
            
        except Exception as e:
            logger.warning(f"Failed to save LLM log: {e}")

    def _get_market_name(self, trade) -> str:
        """
        Derive market name from trade's ticker or timezone.
        
        Returns human-readable market name like "Hong Kong stocks", "US stocks", etc.
        """
        # Check for explicit market attribute first
        if hasattr(trade, 'market') and trade.market:
            return trade.market
        
        # Derive from ticker
        ticker = trade.ticker.upper() if trade.ticker else ""
        
        # Check exchange prefix (HKEX:0981, AMEX:SOXL)
        if ":" in ticker:
            exchange = ticker.split(":")[0]
            exchange_markets = {
                "HKEX": "Hong Kong stocks",
                "HKG": "Hong Kong stocks",
                "SSE": "China A-shares",
                "SZSE": "China A-shares",
                "TSE": "Japan stocks",
                "JPX": "Japan stocks",
                "LSE": "UK stocks",
            }
            if exchange in exchange_markets:
                return exchange_markets[exchange]
        
        # Check ticker suffix (.HK, .SS, etc.)
        if ".HK" in ticker:
            return "Hong Kong stocks"
        if ".SS" in ticker or ".SZ" in ticker:
            return "China A-shares"
        if ".T" in ticker:
            return "Japan stocks"
        if ".L" in ticker:
            return "UK stocks"
        
        # Check market_timezone
        tz = getattr(trade, 'market_timezone', None) or ""
        timezone_markets = {
            "Asia/Hong_Kong": "Hong Kong stocks",
            "Asia/Shanghai": "China A-shares",
            "Asia/Tokyo": "Japan stocks",
            "Europe/London": "UK stocks",
        }
        if tz in timezone_markets:
            return timezone_markets[tz]
        
        # Default to US
        return "US stocks"

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
        
        from app.journal.ingest import get_market_timezone
        
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
                # Calculate total slippage from entry and exit
                total_slippage = None
                if trade.slippage_entry or trade.slippage_exit:
                    total_slippage = (trade.slippage_entry or 0) + (trade.slippage_exit or 0)
                
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
                    r_multiple=trade.r_multiple,
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
                    market=self._get_market_name(trade),
                    timezone=get_market_timezone(trade.ticker),
                    intended_setup=trade.entry_reason,
                    management_plan=trade.target_reason,
                    pd_high=pd_high,
                    pd_low=pd_low,
                    pd_close=pd_close,
                    today_open=today_open,
                    # Order execution details
                    order_type=trade.entry_order_type,
                    slippage=total_slippage,
                    fees=getattr(trade, 'fees', None),
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

                # Save LLM response and context to local file for debugging/review
                self._save_llm_log(trade, ohlcv_context, llm_analysis)

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
            # Derive timezone from ticker for reliability (stored field may be wrong for old trades)
            from app.journal.ingest import get_market_timezone
            market_tz = get_market_timezone(trade.ticker)

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

            # Use parallel fetching for OHLCV data (significant speedup)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            if cancellation_check and cancellation_check():
                return ("Cancelled", None) if return_daily_df else "Cancelled"
            
            # Get entry_price for futures contract selection
            entry_price = trade.entry_price if trade.entry_price else None
            
            if timeframe == "5m" or timeframe not in ["2h", "1d"]:
                # 5-min scalp/day trades: Full Brooks package - fetch all 3 timeframes in parallel
                logger.info(f"📊 Fetching OHLCV data in parallel for {trade.ticker} (daily, 2h, 5m)")
                
                with ThreadPoolExecutor(max_workers=4) as executor:
                    # Submit all fetches in parallel
                    future_daily = executor.submit(
                        self._fetch_ohlcv_section_with_cutoff_and_df,
                        trade.ticker, "1d", daily_fetch_end, 60, 
                        "DAILY_60 (prior day close cutoff)", daily_cutoff, cancellation_check, market_tz, entry_price
                    )
                    future_2h = executor.submit(
                        self._fetch_ohlcv_section_with_cutoff,
                        trade.ticker, "2h", entry_cutoff, 120,
                        "TWOHOUR_120 (entry time cutoff)", cancellation_check, market_tz, entry_price
                    )
                    future_5m = executor.submit(
                        self._fetch_ohlcv_section_with_cutoff,
                        trade.ticker, "5m", entry_cutoff, 234,
                        "FIVEMIN_234 (entry time cutoff)", cancellation_check, market_tz, entry_price
                    )
                    future_pattern = executor.submit(
                        self._detect_patterns_before_entry,
                        trade, entry_cutoff
                    ) if trade.entry_time else None
                    
                    # Collect results (order matters for context)
                    try:
                        daily_context, daily_df = future_daily.result()
                        all_context.append(daily_context)
                    except Exception as e:
                        logger.warning(f"Daily fetch failed: {e}")
                        all_context.append(f"=== DAILY_60 ===\nFetch failed: {e}")
                    
                    try:
                        all_context.append(future_2h.result())
                    except Exception as e:
                        logger.warning(f"2-hour fetch failed: {e}")
                        all_context.append(f"=== TWOHOUR_120 ===\nFetch failed: {e}")
                    
                    try:
                        all_context.append(future_5m.result())
                    except Exception as e:
                        logger.warning(f"5-min fetch failed: {e}")
                        all_context.append(f"=== FIVEMIN_234 ===\nFetch failed: {e}")
                    
                    # Add pattern detection result
                    if future_pattern:
                        try:
                            pattern_context = future_pattern.result()
                            if pattern_context:
                                all_context.append(pattern_context)
                        except Exception as e:
                            logger.warning(f"Pattern detection failed: {e}")

            elif timeframe == "2h":
                # 2-hour swing trades: Daily + 2H context - fetch in parallel
                logger.info(f"📊 Fetching OHLCV data in parallel for {trade.ticker} (daily, 2h)")
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_daily = executor.submit(
                        self._fetch_ohlcv_section_with_cutoff_and_df,
                        trade.ticker, "1d", daily_fetch_end, 60,
                        "DAILY_60 (prior day close cutoff)", daily_cutoff, cancellation_check, market_tz, entry_price
                    )
                    future_2h = executor.submit(
                        self._fetch_ohlcv_section_with_cutoff,
                        trade.ticker, "2h", entry_cutoff, 120,
                        "TWOHOUR_120 (entry time cutoff)", cancellation_check, market_tz, entry_price
                    )
                    
                    try:
                        daily_context, daily_df = future_daily.result()
                        all_context.append(daily_context)
                    except Exception as e:
                        logger.warning(f"Daily fetch failed: {e}")
                    
                    try:
                        all_context.append(future_2h.result())
                    except Exception as e:
                        logger.warning(f"2-hour fetch failed: {e}")

            elif timeframe == "1d":
                # Daily position trades: Just daily context (no parallelization needed)
                daily_context, daily_df = self._fetch_ohlcv_section_with_cutoff_and_df(
                    trade.ticker, "1d", daily_fetch_end, 120,
                    "DAILY_120 (prior day close cutoff)", daily_cutoff, cancellation_check, market_tz, entry_price
                )
                all_context.append(daily_context)
            
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
    
    def _fetch_ohlcv_section_with_cutoff(self, ticker: str, interval: str, cutoff_time: datetime, num_bars: int, label: str, cancellation_check: callable = None, market_timezone: str = "America/New_York", target_price: float = None) -> str:
        """Fetch OHLCV data up to a specific cutoff time (no future data).
        
        IMPORTANT: Only includes COMPLETED bars. A bar is complete when:
        bar_start_time + interval <= cutoff_time
        
        For example, with entry at 11:36:
        - 5-min bar at 11:30 (ends 11:35) is INCLUDED (complete before entry)
        - 5-min bar at 11:35 (ends 11:40) is EXCLUDED (not complete at entry)
        
        Args:
            ticker: Stock ticker
            interval: Time interval (1d, 2h, 5m, etc.)
            cutoff_time: Entry time - only completed bars before this are included
            num_bars: Number of bars to fetch
            label: Label for this section
            cancellation_check: Optional cancellation callback
            market_timezone: Timezone of the market (e.g., "Asia/Hong_Kong" for HK stocks)
            target_price: Optional price to help select correct futures contract
        """
        try:
            from zoneinfo import ZoneInfo
            
            # Make cutoff_time timezone-aware if it's naive
            # This is critical for correct UTC conversion in the provider
            market_tz = ZoneInfo(market_timezone)
            if cutoff_time.tzinfo is None:
                cutoff_aware = cutoff_time.replace(tzinfo=market_tz)
            else:
                cutoff_aware = cutoff_time
            
            # Calculate interval duration for filtering complete bars
            if interval == "1d":
                interval_delta = timedelta(days=1)
                start_date = cutoff_aware - timedelta(days=num_bars + 10)
            elif interval == "2h":
                interval_delta = timedelta(hours=2)
                start_date = cutoff_aware - timedelta(hours=num_bars * 2 + 20)
            elif interval == "1h":
                interval_delta = timedelta(hours=1)
                start_date = cutoff_aware - timedelta(hours=num_bars + 10)
            elif interval == "5m":
                interval_delta = timedelta(minutes=5)
                start_date = cutoff_aware - timedelta(minutes=num_bars * 5 + 100)
            else:
                interval_delta = timedelta(days=1)
                start_date = cutoff_aware - timedelta(days=num_bars + 10)
            
            # Calculate the maximum bar start time for a COMPLETE bar
            # A bar starting at this time would end exactly at cutoff_time
            max_complete_bar_start = cutoff_aware - interval_delta
            
            # Check for cancellation before fetch
            if cancellation_check and cancellation_check():
                return f"=== {label} ===\nCancelled"
            
            # Use ticker-specific provider (e.g., AllTick for HK stocks, Databento for futures)
            provider = self._get_provider_for_ticker(ticker)
            
            # Check if provider supports target_price (for futures contract selection)
            import inspect
            if target_price and 'target_price' in inspect.signature(provider.get_ohlcv).parameters:
                ohlcv = provider.get_ohlcv(
                    ticker,
                    interval,
                    start_date,
                    cutoff_aware,  # Fetch up to cutoff_time, filter below
                    cancellation_check=cancellation_check,
                    target_price=target_price
                )
            else:
                ohlcv = provider.get_ohlcv(
                    ticker,
                    interval,
                    start_date,
                    cutoff_aware,  # Fetch up to cutoff_time, filter below
                    cancellation_check=cancellation_check
                )
            
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"No {interval} data returned for {ticker}")
                return f"=== {label} ===\nNo data available"
            
            # Filter to only include COMPLETED bars (bar must have ended before entry)
            # Bar timestamp is the START time, so bar ends at timestamp + interval
            # Include bar if: timestamp + interval <= cutoff_time
            # Which means: timestamp <= cutoff_time - interval = max_complete_bar_start
            
            # Find the datetime column (providers may return 'datetime' or 'timestamp')
            time_col = None
            if 'timestamp' in ohlcv.columns:
                time_col = 'timestamp'
            elif 'datetime' in ohlcv.columns:
                time_col = 'datetime'
            
            if time_col:
                # Handle timezone comparison for filtering
                filter_cutoff = max_complete_bar_start
                ohlcv_is_tz_aware = ohlcv[time_col].dt.tz is not None
                cutoff_is_tz_aware = filter_cutoff.tzinfo is not None
                
                if ohlcv_is_tz_aware and not cutoff_is_tz_aware:
                    # OHLCV is timezone-aware, cutoff is naive - make cutoff aware
                    filter_cutoff = max_complete_bar_start.replace(tzinfo=market_tz)
                elif not ohlcv_is_tz_aware and cutoff_is_tz_aware:
                    # OHLCV is naive (UTC from Yahoo API), cutoff is aware - convert cutoff to UTC naive
                    import pandas as pd
                    filter_cutoff = pd.Timestamp(filter_cutoff).tz_convert('UTC').tz_localize(None).to_pydatetime()
                
                before_filter_count = len(ohlcv)
                ohlcv = ohlcv[ohlcv[time_col] <= filter_cutoff]
                logger.info(f"📊 {label}: Filtered {before_filter_count} → {len(ohlcv)} bars (cutoff: {filter_cutoff})")
                
                if len(ohlcv) > 0:
                    last_ts = ohlcv[time_col].iloc[-1]
                    first_ts = ohlcv[time_col].iloc[0]
                    logger.info(f"📊 {label}: Data range {first_ts} to {last_ts}")
            
            # Limit to requested number of bars (most recent ones before cutoff)
            ohlcv = ohlcv.tail(num_bars)
            
            if ohlcv.empty:
                return f"=== {label} ===\nNo data available before cutoff"
            
            # Format as readable string
            # Convert times to market timezone for consistent display to LLM
            import pandas as pd
            from zoneinfo import ZoneInfo
            market_tz_obj = ZoneInfo(market_timezone)
            
            # Format entry time in market timezone
            if cutoff_time.tzinfo is None:
                entry_display = cutoff_time.strftime('%Y-%m-%d %H:%M')
            else:
                entry_display = cutoff_time.astimezone(market_tz_obj).strftime('%Y-%m-%d %H:%M')
            
            # Format signal bar time in market timezone  
            if max_complete_bar_start.tzinfo is None:
                signal_bar_display = max_complete_bar_start.strftime('%Y-%m-%d %H:%M')
            else:
                signal_bar_display = max_complete_bar_start.astimezone(market_tz_obj).strftime('%Y-%m-%d %H:%M')
            
            lines = [f"=== {label} ==="]
            lines.append(f"Entry time: {entry_display} ({market_timezone})")
            lines.append(f"Last complete bar (SIGNAL BAR): {signal_bar_display} ({market_timezone})")
            lines.append(f"Bars: {len(ohlcv)}")
            lines.append("timestamp, open, high, low, close, volume")
            
            # Track the last bar to explicitly label it as signal bar
            ohlcv_list = list(ohlcv.iterrows())
            for i, (_, row) in enumerate(ohlcv_list):
                # Get timestamp from available column and convert to market timezone
                ts = row.get('timestamp') or row.get('datetime') or row.name
                if hasattr(ts, 'strftime'):
                    # Convert UTC timestamp to market timezone
                    if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                        # Naive timestamp - assume UTC from Yahoo API
                        ts_utc = pd.Timestamp(ts).tz_localize('UTC')
                        ts_local = ts_utc.tz_convert(market_timezone)
                        ts_str = ts_local.strftime('%Y-%m-%d %H:%M')
                    elif hasattr(ts, 'tz_convert'):
                        ts_str = ts.tz_convert(market_timezone).strftime('%Y-%m-%d %H:%M')
                    else:
                        ts_str = ts.strftime('%Y-%m-%d %H:%M')
                else:
                    ts_str = str(ts)
                
                # Label the last bar as SIGNAL BAR for clarity
                is_signal_bar = (i == len(ohlcv_list) - 1) and interval in ['5m', '5min']
                signal_marker = " <<< SIGNAL BAR (bar before entry)" if is_signal_bar else ""
                
                lines.append(
                    f"{ts_str}, {row['open']:.4f}, {row['high']:.4f}, "
                    f"{row['low']:.4f}, {row['close']:.4f}, {int(row.get('volume', 0))}{signal_marker}"
                )
            
            # Add explicit signal bar analysis for 5-min data
            if interval in ['5m', '5min'] and len(ohlcv_list) > 0:
                _, signal_row = ohlcv_list[-1]
                signal_ts = signal_row.get('timestamp') or signal_row.get('datetime') or ohlcv.index[-1]
                signal_o = signal_row['open']
                signal_h = signal_row['high']
                signal_l = signal_row['low']
                signal_c = signal_row['close']
                
                # Classify the signal bar
                bar_range = signal_h - signal_l
                body = abs(signal_c - signal_o)
                body_pct = (body / bar_range * 100) if bar_range > 0 else 0
                is_bull = signal_c > signal_o
                is_bear = signal_c < signal_o
                is_doji = body_pct < 30
                
                close_position = ((signal_c - signal_l) / bar_range * 100) if bar_range > 0 else 50
                
                bar_type = "DOJI" if is_doji else ("BULL BAR" if is_bull else "BEAR BAR")
                close_location = "near HIGH" if close_position > 70 else ("near LOW" if close_position < 30 else "middle")
                
                if hasattr(signal_ts, 'strftime'):
                    signal_ts_str = signal_ts.strftime('%Y-%m-%d %H:%M')
                else:
                    signal_ts_str = str(signal_ts)
                
                lines.append("")
                lines.append("=== SIGNAL BAR ANALYSIS ===")
                lines.append(f"Signal Bar: {signal_ts_str}")
                lines.append(f"OHLC: O={signal_o:.4f}, H={signal_h:.4f}, L={signal_l:.4f}, C={signal_c:.4f}")
                lines.append(f"Bar Type: {bar_type} (body={body_pct:.0f}% of range)")
                lines.append(f"Close Location: {close_location} ({close_position:.0f}% from low)")
                
                # For short entries, analyze if this was a good signal bar
                lines.append(f"For SHORT entry: {'GOOD' if is_bear and close_position < 40 else 'WEAK'} signal bar")
                lines.append(f"For LONG entry: {'GOOD' if is_bull and close_position > 60 else 'WEAK'} signal bar")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.warning(f"Failed to fetch {interval} data for {ticker}: {e}")
            return f"=== {label} ===\nData fetch failed: {e}"

    def _fetch_ohlcv_section_with_cutoff_and_df(self, ticker: str, interval: str, fetch_end: datetime, num_bars: int, label: str, display_cutoff: datetime, cancellation_check: callable = None, market_timezone: str = "America/New_York", target_price: float = None) -> tuple:
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
            market_timezone: Timezone of the market (e.g., "Asia/Hong_Kong" for HK stocks)
        
        Returns:
            tuple: (formatted_string, raw_dataframe)
        """
        try:
            from zoneinfo import ZoneInfo
            import pandas as pd
            
            # Make fetch_end and display_cutoff timezone-aware if naive
            market_tz = ZoneInfo(market_timezone)
            if fetch_end.tzinfo is None:
                fetch_end_aware = fetch_end.replace(tzinfo=market_tz)
            else:
                fetch_end_aware = fetch_end
            
            if display_cutoff.tzinfo is None:
                display_cutoff_aware = display_cutoff.replace(tzinfo=market_tz)
            else:
                display_cutoff_aware = display_cutoff
            
            # Calculate start date (also timezone-aware)
            start_date = fetch_end_aware - timedelta(days=num_bars + 10)
            
            # Check for cancellation before fetch
            if cancellation_check and cancellation_check():
                return f"=== {label} ===\nCancelled", None
            
            # Fetch data up to fetch_end (includes trade date)
            provider = self._get_provider_for_ticker(ticker)
            
            # Check if provider supports target_price (for futures contract selection)
            import inspect
            if target_price and 'target_price' in inspect.signature(provider.get_ohlcv).parameters:
                ohlcv = provider.get_ohlcv(
                    ticker,
                    interval,
                    start_date,
                    fetch_end_aware,
                    cancellation_check=cancellation_check,
                    target_price=target_price
                )
            else:
                ohlcv = provider.get_ohlcv(
                    ticker,
                    interval,
                    start_date,
                    fetch_end_aware,
                    cancellation_check=cancellation_check
                )
            
            if ohlcv is None or ohlcv.empty:
                logger.warning(f"No {interval} data returned for {ticker}")
                return f"=== {label} ===\nNo data available", None
            
            # Store full DataFrame for session context
            full_df = ohlcv.copy()
            
            # Filter for display (only up to display_cutoff for LLM context)
            # Find the datetime column (providers may return 'datetime' or 'timestamp')
            time_col = None
            if 'timestamp' in ohlcv.columns:
                time_col = 'timestamp'
            elif 'datetime' in ohlcv.columns:
                time_col = 'datetime'
            
            if time_col:
                # Handle timezone comparison for filtering
                filter_cutoff = display_cutoff_aware
                ohlcv_is_tz_aware = ohlcv[time_col].dt.tz is not None
                
                if not ohlcv_is_tz_aware and filter_cutoff.tzinfo is not None:
                    # OHLCV is naive (UTC from Yahoo API), cutoff is aware - convert to UTC naive
                    filter_cutoff = pd.Timestamp(filter_cutoff).tz_convert('UTC').tz_localize(None).to_pydatetime()
                
                ohlcv = ohlcv[ohlcv[time_col] <= filter_cutoff]
            
            # Limit to requested number of bars
            ohlcv = ohlcv.tail(num_bars)
            
            if ohlcv.empty:
                return f"=== {label} ===\nNo data available before cutoff", full_df
            
            # Format as readable string
            # Convert times to market timezone for consistent display
            from zoneinfo import ZoneInfo
            market_tz_obj = ZoneInfo(market_timezone)
            
            # Format cutoff in market timezone
            if display_cutoff_aware.tzinfo is not None:
                cutoff_display = display_cutoff_aware.astimezone(market_tz_obj).strftime('%Y-%m-%d %H:%M')
            else:
                cutoff_display = display_cutoff.strftime('%Y-%m-%d %H:%M')
            
            lines = [f"=== {label} ==="]
            lines.append(f"Cutoff: {cutoff_display} ({market_timezone})")
            lines.append(f"Bars: {len(ohlcv)}")
            lines.append("timestamp, open, high, low, close, volume")
            
            for _, row in ohlcv.iterrows():
                # Get timestamp from available column and convert to market timezone
                ts = row.get('timestamp') or row.get('datetime') or row.name
                if hasattr(ts, 'strftime'):
                    # Convert UTC timestamp to market timezone
                    if hasattr(ts, 'tzinfo') and ts.tzinfo is None:
                        # Naive timestamp - assume UTC from Yahoo API
                        ts_utc = pd.Timestamp(ts).tz_localize('UTC')
                        ts_local = ts_utc.tz_convert(market_timezone)
                        ts_str = ts_local.strftime('%Y-%m-%d %H:%M')
                    elif hasattr(ts, 'tz_convert'):
                        ts_str = ts.tz_convert(market_timezone).strftime('%Y-%m-%d %H:%M')
                    else:
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
    
    def _detect_patterns_before_entry(self, trade: Trade, entry_cutoff: datetime) -> str:
        """
        Detect potential patterns in the 5-min data leading up to entry.
        
        This helps the LLM by pre-identifying potential wedges, channels, and other patterns
        rather than requiring it to parse raw OHLCV data.
        """
        try:
            from zoneinfo import ZoneInfo
            from app.journal.ingest import get_market_timezone
            
            # Make entry_cutoff timezone-aware
            market_tz_str = get_market_timezone(trade.ticker)
            market_tz = ZoneInfo(market_tz_str)
            
            if entry_cutoff.tzinfo is None:
                entry_cutoff_aware = entry_cutoff.replace(tzinfo=market_tz)
            else:
                entry_cutoff_aware = entry_cutoff
            
            # Fetch 5-min data for the trading session up to entry
            interval_delta = timedelta(minutes=5)
            start_date = entry_cutoff_aware - timedelta(hours=2)  # Look at last 2 hours before entry
            
            provider = self._get_provider_for_ticker(trade.ticker)
            ohlcv = provider.get_ohlcv(
                trade.ticker,
                "5m",
                start_date,
                entry_cutoff_aware
            )
            
            if ohlcv is None or ohlcv.empty or len(ohlcv) < 10:
                return ""
            
            # Find the datetime column
            time_col = 'timestamp' if 'timestamp' in ohlcv.columns else 'datetime'
            if time_col not in ohlcv.columns:
                return ""
            
            # Filter to only include complete bars before entry
            import pandas as pd
            max_complete_bar_start = entry_cutoff_aware - interval_delta
            
            # Handle timezone for filtering
            ohlcv_is_tz_aware = ohlcv[time_col].dt.tz is not None
            if not ohlcv_is_tz_aware and max_complete_bar_start.tzinfo is not None:
                # OHLCV is naive (UTC), cutoff is aware - convert to UTC naive
                filter_cutoff = pd.Timestamp(max_complete_bar_start).tz_convert('UTC').tz_localize(None).to_pydatetime()
            else:
                filter_cutoff = max_complete_bar_start
            
            ohlcv = ohlcv[ohlcv[time_col] <= filter_cutoff]
            
            if len(ohlcv) < 10:
                return ""
            
            lines = ["=== PATTERN DETECTION (last 2 hours before entry) ==="]
            lines.append(f"Analyzing bars from {ohlcv[time_col].iloc[0]} to {ohlcv[time_col].iloc[-1]}")
            lines.append("")
            
            # Calculate swing highs and lows
            highs = ohlcv['high'].values
            lows = ohlcv['low'].values
            closes = ohlcv['close'].values
            opens = ohlcv['open'].values
            timestamps = ohlcv[time_col].values
            
            # Find local swing highs (higher than 2 bars on each side)
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(highs) - 2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] >= highs[i+1] and highs[i] >= highs[i+2]:
                    swing_highs.append((i, highs[i], timestamps[i]))
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] <= lows[i+1] and lows[i] <= lows[i+2]:
                    swing_lows.append((i, lows[i], timestamps[i]))
            
            # Detect wedge patterns (3 consecutive higher lows / lower highs)
            wedge_detected = False
            wedge_type = None
            wedge_bars = []
            
            # Check for wedge top (3 higher lows with lower/equal highs - bearish)
            if len(swing_lows) >= 3:
                last_3_lows = swing_lows[-3:]
                if last_3_lows[0][1] < last_3_lows[1][1] < last_3_lows[2][1]:
                    # 3 consecutive higher lows
                    # Check if highs are relatively flat or declining
                    wedge_detected = True
                    wedge_type = "WEDGE TOP (3 higher lows - potential bearish reversal)"
                    wedge_bars = last_3_lows
            
            # Check for wedge bottom (3 lower highs with higher/equal lows - bullish)
            if not wedge_detected and len(swing_highs) >= 3:
                last_3_highs = swing_highs[-3:]
                if last_3_highs[0][1] > last_3_highs[1][1] > last_3_highs[2][1]:
                    # 3 consecutive lower highs
                    wedge_detected = True
                    wedge_type = "WEDGE BOTTOM (3 lower highs - potential bullish reversal)"
                    wedge_bars = last_3_highs
            
            # Report swing points
            lines.append("SWING HIGHS (last 2 hours):")
            for idx, price, ts in swing_highs[-5:]:
                ts_str = pd.Timestamp(ts).strftime('%H:%M') if hasattr(pd.Timestamp(ts), 'strftime') else str(ts)
                lines.append(f"  - {ts_str}: ${price:.4f}")
            
            lines.append("")
            lines.append("SWING LOWS (last 2 hours):")
            for idx, price, ts in swing_lows[-5:]:
                ts_str = pd.Timestamp(ts).strftime('%H:%M') if hasattr(pd.Timestamp(ts), 'strftime') else str(ts)
                lines.append(f"  - {ts_str}: ${price:.4f}")
            
            # Report wedge pattern if detected
            lines.append("")
            if wedge_detected:
                lines.append(f"*** PATTERN DETECTED: {wedge_type} ***")
                lines.append("Wedge pushes:")
                for idx, price, ts in wedge_bars:
                    ts_str = pd.Timestamp(ts).strftime('%H:%M') if hasattr(pd.Timestamp(ts), 'strftime') else str(ts)
                    lines.append(f"  Push at {ts_str}: ${price:.4f}")
                lines.append("")
                lines.append("NOTE: If trader shorted after this wedge top, this is a WITH-TREND reversal setup.")
            else:
                lines.append("No clear wedge pattern detected (3 pushes required)")
            
            # Detect trend direction in this 2-hour window
            first_close = closes[0]
            last_close = closes[-1]
            highest = max(highs)
            lowest = min(lows)
            
            trend_pct = ((last_close - first_close) / first_close * 100) if first_close > 0 else 0
            
            lines.append("")
            if trend_pct > 0.5:
                lines.append(f"2-HOUR TREND: BULLISH (+{trend_pct:.2f}%)")
            elif trend_pct < -0.5:
                lines.append(f"2-HOUR TREND: BEARISH ({trend_pct:.2f}%)")
            else:
                lines.append(f"2-HOUR TREND: SIDEWAYS ({trend_pct:+.2f}%)")
            
            lines.append(f"Range: ${lowest:.4f} - ${highest:.4f}")
            
            return "\n".join(lines)
            
        except Exception as e:
            logger.warning(f"Pattern detection failed: {e}")
            return ""

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
