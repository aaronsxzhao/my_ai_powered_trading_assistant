"""
Premarket / Before-Session Report Generator.

Uses LLM for intelligent market analysis:
- Daily chart: regime, key levels, magnets
- 2-hour chart: current leg, strength
- 5-minute (past 3 days): opening context, day type
- Plan A / Plan B with setups
- Avoid list
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
import logging

import pytz

from app.config import settings, OUTPUTS_DIR
from app.data.cache import get_cached_ohlcv
from app.features.ohlc_features import OHLCFeatures
from app.features.brooks_patterns import BrooksPatternDetector, RegimeAnalysis
from app.features.magnets import MagnetDetector

logger = logging.getLogger(__name__)


@dataclass
class TimeframeAnalysis:
    """Analysis for a single timeframe."""

    timeframe: str
    regime: str
    always_in: str
    confidence: str
    description: str
    key_levels: list[dict]
    patterns: list[str]
    strength: dict


@dataclass
class TickerReport:
    """Complete premarket report for a single ticker."""

    ticker: str
    report_date: date
    current_price: Optional[float]

    # Multi-timeframe analysis
    daily_analysis: TimeframeAnalysis
    two_hour_analysis: Optional[TimeframeAnalysis]
    five_min_analysis: Optional[TimeframeAnalysis]

    # Magnets
    magnets_above: list[dict]
    magnets_below: list[dict]
    measured_moves: list[dict]

    # Plans
    plan_a: dict  # Most likely scenario
    plan_b: dict  # If reversal

    # Avoid list
    avoid_conditions: list[str]

    # Overall bias
    overall_bias: str
    bias_confidence: str


class PremarketReport:
    """
    Generate premarket analysis reports.

    Uses LLM for intelligent analysis of daily, 2-hour, and 5-minute charts
    to provide Brooks-style context and trading plans.
    """

    def __init__(self):
        """Initialize report generator."""
        self.ny_tz = pytz.timezone("America/New_York")
        self._llm_analyzer = None

    @property
    def llm_analyzer(self):
        """Lazy load LLM analyzer."""
        if self._llm_analyzer is None:
            from app.llm.analyzer import get_analyzer
            self._llm_analyzer = get_analyzer()
        return self._llm_analyzer

    def generate_ticker_report(self, ticker: str, report_date: Optional[date] = None) -> TickerReport:
        """
        Generate premarket report for a single ticker using LLM analysis.

        Args:
            ticker: Stock symbol
            report_date: Date for report (defaults to today)

        Returns:
            TickerReport object
        """
        report_date = report_date or date.today()
        ticker = ticker.upper()

        logger.info(f"Generating premarket report for {ticker} on {report_date}")

        # Fetch data for each timeframe
        daily_df = self._fetch_daily_data(ticker, report_date)
        two_hour_df = self._fetch_2h_data(ticker, report_date)
        five_min_df = self._fetch_5m_data(ticker, report_date)

        # Get current price
        current_price = None
        if not daily_df.empty:
            current_price = float(daily_df.iloc[-1]["close"])

        # Try LLM analysis first
        if self.llm_analyzer.is_available and not daily_df.empty:
            llm_result = self._generate_llm_report(ticker, daily_df, two_hour_df)
            if llm_result:
                return llm_result

        # Fallback to rule-based analysis
        daily_analysis = self._analyze_timeframe(daily_df, "daily")
        two_hour_analysis = self._analyze_timeframe(two_hour_df, "2h") if not two_hour_df.empty else None
        five_min_analysis = self._analyze_timeframe(five_min_df, "5m") if not five_min_df.empty else None

        # Get magnets
        magnets_above, magnets_below, measured_moves = self._get_magnets(
            daily_df, current_price
        )

        # Generate plans
        plan_a, plan_b = self._generate_plans(
            daily_analysis, two_hour_analysis, five_min_analysis
        )

        # Generate avoid conditions
        avoid_conditions = self._generate_avoid_list(
            daily_analysis, two_hour_analysis
        )

        # Determine overall bias
        overall_bias, bias_confidence = self._determine_overall_bias(
            daily_analysis, two_hour_analysis, five_min_analysis
        )

        return TickerReport(
            ticker=ticker,
            report_date=report_date,
            current_price=current_price,
            daily_analysis=daily_analysis,
            two_hour_analysis=two_hour_analysis,
            five_min_analysis=five_min_analysis,
            magnets_above=magnets_above,
            magnets_below=magnets_below,
            measured_moves=measured_moves,
            plan_a=plan_a,
            plan_b=plan_b,
            avoid_conditions=avoid_conditions,
            overall_bias=overall_bias,
            bias_confidence=bias_confidence,
        )

    def _generate_llm_report(self, ticker: str, daily_df, two_hour_df) -> Optional[TickerReport]:
        """Generate report using LLM analysis."""
        try:
            # Format OHLCV data for LLM
            daily_str = self._format_ohlcv_for_llm(daily_df.tail(20))
            two_hour_str = self._format_ohlcv_for_llm(two_hour_df.tail(20)) if not two_hour_df.empty else None

            llm_analysis = self.llm_analyzer.analyze_market_context(
                ticker=ticker,
                daily_ohlcv=daily_str,
                intraday_ohlcv=two_hour_str,
            )

            if "error" in llm_analysis:
                return None

            current_price = float(daily_df.iloc[-1]["close"])
            report_date = date.today()

            # Extract key levels from LLM
            key_levels = llm_analysis.get("key_levels", {})
            resistance = key_levels.get("resistance", [])
            support = key_levels.get("support", [])

            # Build TimeframeAnalysis from LLM
            daily_analysis = TimeframeAnalysis(
                timeframe="daily",
                regime=llm_analysis.get("regime", "unknown"),
                always_in=llm_analysis.get("always_in", "neutral"),
                confidence=llm_analysis.get("regime_confidence", "medium"),
                description=llm_analysis.get("summary", ""),
                key_levels=[{"price": r.get("price"), "type": "resistance", "description": r.get("description", "")} for r in resistance] +
                           [{"price": s.get("price"), "type": "support", "description": s.get("description", "")} for s in support],
                patterns=[],
                strength={"strength": llm_analysis.get("strength", "unknown"), "reasoning": llm_analysis.get("strength_reasoning", "")},
            )

            # Extract plan A/B from LLM
            plan_a_data = llm_analysis.get("plan_a", {})
            plan_b_data = llm_analysis.get("plan_b", {})

            plan_a = {
                "scenario": plan_a_data.get("scenario", ""),
                "bias": plan_a_data.get("bias", "NEUTRAL"),
                "setups": plan_a_data.get("setups", []),
                "entry_zones": plan_a_data.get("entry_zones", ""),
                "targets": plan_a_data.get("targets", ""),
            }

            plan_b = {
                "scenario": "Alternative scenario",
                "trigger": plan_b_data.get("trigger", ""),
                "bias": plan_b_data.get("new_bias", "NEUTRAL"),
                "action": plan_b_data.get("action", ""),
            }

            # Format magnets from LLM analysis
            magnets_above = [{"price": r.get("price"), "type": "resistance", "description": r.get("description", "")} for r in resistance]
            magnets_below = [{"price": s.get("price"), "type": "support", "description": s.get("description", "")} for s in support]

            return TickerReport(
                ticker=ticker,
                report_date=report_date,
                current_price=current_price,
                daily_analysis=daily_analysis,
                two_hour_analysis=None,
                five_min_analysis=None,
                magnets_above=magnets_above,
                magnets_below=magnets_below,
                measured_moves=[],
                plan_a=plan_a,
                plan_b=plan_b,
                avoid_conditions=llm_analysis.get("avoid", []),
                overall_bias=plan_a.get("bias", "NEUTRAL"),
                bias_confidence=llm_analysis.get("regime_confidence", "medium"),
            )

        except Exception as e:
            logger.warning(f"LLM report generation failed: {e}")
            return None

    def _format_ohlcv_for_llm(self, df) -> str:
        """Format OHLCV DataFrame as string for LLM."""
        if df.empty:
            return "No data"

        lines = ["Date | Open | High | Low | Close | Volume"]
        for _, row in df.iterrows():
            dt = row["datetime"].strftime("%Y-%m-%d") if hasattr(row["datetime"], "strftime") else str(row["datetime"])[:10]
            lines.append(f"{dt} | {row['open']:.2f} | {row['high']:.2f} | {row['low']:.2f} | {row['close']:.2f} | {int(row['volume'])}")
        return "\n".join(lines)

    def _fetch_daily_data(self, ticker: str, report_date: date) -> 'pd.DataFrame':
        """Fetch daily OHLCV data."""
        import pandas as pd
        end = datetime.combine(report_date, datetime.min.time())
        start = end - timedelta(days=365)

        try:
            df = get_cached_ohlcv(ticker, "1d", start, end)
            return df
        except Exception as e:
            logger.error(f"Failed to fetch daily data for {ticker}: {e}")
            return pd.DataFrame()

    def _fetch_2h_data(self, ticker: str, report_date: date) -> 'pd.DataFrame':
        """Fetch 2-hour OHLCV data."""
        import pandas as pd
        end = datetime.combine(report_date, datetime.min.time())
        start = end - timedelta(days=60)

        try:
            df = get_cached_ohlcv(ticker, "2h", start, end)
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch 2h data for {ticker}: {e}")
            return pd.DataFrame()

    def _fetch_5m_data(self, ticker: str, report_date: date) -> 'pd.DataFrame':
        """Fetch 5-minute OHLCV data."""
        import pandas as pd
        end = datetime.combine(report_date, datetime.min.time())
        start = end - timedelta(days=5)

        try:
            df = get_cached_ohlcv(ticker, "5m", start, end)
            return df
        except Exception as e:
            logger.warning(f"Failed to fetch 5m data for {ticker}: {e}")
            return pd.DataFrame()

    def _analyze_timeframe(self, df: 'pd.DataFrame', timeframe: str) -> TimeframeAnalysis:
        """Analyze a single timeframe."""
        import pandas as pd

        if df.empty:
            return TimeframeAnalysis(
                timeframe=timeframe,
                regime="unknown",
                always_in="neutral",
                confidence="low",
                description="No data available",
                key_levels=[],
                patterns=[],
                strength={},
            )

        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)

        # Regime analysis
        regime = detector.analyze_regime()

        # Pattern detection
        patterns = detector.detect_all_patterns()
        pattern_names = [p.pattern for p in patterns]

        # Key levels from swings
        swing_highs = features.find_swing_highs()
        swing_lows = features.find_swing_lows()

        key_levels = []
        for sh in swing_highs[-3:]:  # Last 3 swing highs
            key_levels.append({
                "price": round(sh.price, 2),
                "type": "resistance",
                "description": "Swing High",
            })
        for sl in swing_lows[-3:]:
            key_levels.append({
                "price": round(sl.price, 2),
                "type": "support",
                "description": "Swing Low",
            })

        # Strength analysis
        strength = features.get_recent_strength()

        return TimeframeAnalysis(
            timeframe=timeframe,
            regime=regime.regime.value,
            always_in=regime.always_in.value,
            confidence=regime.confidence.value,
            description=regime.description,
            key_levels=key_levels,
            patterns=pattern_names,
            strength=strength,
        )

    def _get_magnets(
        self,
        daily_df: 'pd.DataFrame',
        current_price: Optional[float],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Get magnet levels."""
        if daily_df.empty or current_price is None:
            return [], [], []

        detector = MagnetDetector(daily_df)
        magnet_map = detector.get_magnet_map(current_price)

        above = [
            {"price": m.price, "type": m.type, "description": m.description}
            for m in magnet_map["magnets_above"]
        ]
        below = [
            {"price": m.price, "type": m.type, "description": m.description}
            for m in magnet_map["magnets_below"]
        ]
        moves = [
            {
                "target": mm.target_price,
                "type": mm.type,
                "size": mm.move_size,
            }
            for mm in magnet_map["measured_move_targets"]
        ]

        return above, below, moves

    def _generate_plans(
        self,
        daily: TimeframeAnalysis,
        two_hour: Optional[TimeframeAnalysis],
        five_min: Optional[TimeframeAnalysis],
    ) -> tuple[dict, dict]:
        """Generate Plan A and Plan B based on analysis."""

        # Plan A: Most likely scenario (with-trend)
        if daily.regime == "trend_up":
            plan_a = {
                "scenario": "Uptrend continuation",
                "bias": "LONG",
                "setups": [
                    "2nd entry buy on pullback",
                    "Breakout pullback after new high",
                    "Trend resumption after test of prior swing",
                ],
                "entry_zones": "Pullbacks to 20 EMA or prior swing highs",
                "targets": "Measured move targets, prior highs, round numbers",
                "stops": "Below prior swing low or 1 ATR below entry",
            }
            plan_b = {
                "scenario": "Trend reversal to range or downtrend",
                "trigger": "Strong break below 20 EMA + bear trend bar + test that fails",
                "bias": "NEUTRAL to SHORT",
                "action": "Stop looking for longs, wait for selling climax before new longs",
            }

        elif daily.regime == "trend_down":
            plan_a = {
                "scenario": "Downtrend continuation",
                "bias": "SHORT",
                "setups": [
                    "2nd entry sell on rally",
                    "Breakout pullback after new low",
                    "Trend resumption after test of prior swing",
                ],
                "entry_zones": "Rallies to 20 EMA or prior swing lows",
                "targets": "Measured move targets, prior lows, round numbers",
                "stops": "Above prior swing high or 1 ATR above entry",
            }
            plan_b = {
                "scenario": "Trend reversal to range or uptrend",
                "trigger": "Strong break above 20 EMA + bull trend bar + test that holds",
                "bias": "NEUTRAL to LONG",
                "action": "Stop looking for shorts, wait for buying climax before new shorts",
            }

        else:  # Trading range
            plan_a = {
                "scenario": "Trading range continuation",
                "bias": "NEUTRAL - fade extremes",
                "setups": [
                    "Fade range highs with limit orders",
                    "Fade range lows with limit orders",
                    "Wait for failed breakout to fade",
                ],
                "entry_zones": "Range extremes (top 20% for shorts, bottom 20% for longs)",
                "targets": "Opposite range extreme or middle of range",
                "stops": "Outside range by 0.5 ATR",
            }
            plan_b = {
                "scenario": "Breakout from range",
                "trigger": "Strong trend bar breaking range + follow-through + pullback that holds",
                "bias": "Direction of breakout",
                "action": "Switch to with-trend tactics, look for breakout pullbacks",
            }

        return plan_a, plan_b

    def _generate_avoid_list(
        self,
        daily: TimeframeAnalysis,
        two_hour: Optional[TimeframeAnalysis],
    ) -> list[str]:
        """Generate list of conditions to avoid trading."""
        avoid = []

        if daily.regime == "trend_up":
            avoid.extend([
                "Avoid shorting without clear reversal structure",
                "Avoid buying after extended move (3+ legs up without correction)",
                "Avoid chasing - wait for pullbacks",
            ])

        elif daily.regime == "trend_down":
            avoid.extend([
                "Avoid buying without clear reversal structure",
                "Avoid selling after extended move (3+ legs down)",
                "Avoid bottom-picking - wait for signs of strength",
            ])

        else:  # Range
            avoid.extend([
                "Avoid stop entries - they often fail in ranges",
                "Avoid trading middle of range",
                "Avoid expecting big moves - scale out early",
                "Avoid trading tight trading ranges (barbwire)",
            ])

        # Check for patterns that suggest caution
        if daily.patterns:
            if "wedge" in str(daily.patterns).lower():
                avoid.append("Wedge pattern detected - possible reversal, don't chase")
            if "climax" in str(daily.patterns).lower():
                avoid.append("Climax detected - expect correction, reduce size")

        # Low confidence context
        if daily.confidence == "low":
            avoid.append("Context unclear - reduce position size or wait for clarity")

        return avoid

    def _determine_overall_bias(
        self,
        daily: TimeframeAnalysis,
        two_hour: Optional[TimeframeAnalysis],
        five_min: Optional[TimeframeAnalysis],
    ) -> tuple[str, str]:
        """Determine overall bias across timeframes."""
        biases = []
        weights = []

        # Daily is most important
        if daily.always_in == "long":
            biases.append(1)
        elif daily.always_in == "short":
            biases.append(-1)
        else:
            biases.append(0)
        weights.append(3)

        # 2-hour
        if two_hour:
            if two_hour.always_in == "long":
                biases.append(1)
            elif two_hour.always_in == "short":
                biases.append(-1)
            else:
                biases.append(0)
            weights.append(2)

        # 5-min
        if five_min:
            if five_min.always_in == "long":
                biases.append(1)
            elif five_min.always_in == "short":
                biases.append(-1)
            else:
                biases.append(0)
            weights.append(1)

        # Weighted average
        if not biases:
            return "neutral", "low"

        weighted_avg = sum(b * w for b, w in zip(biases, weights)) / sum(weights)

        if weighted_avg > 0.5:
            bias = "LONG"
        elif weighted_avg < -0.5:
            bias = "SHORT"
        else:
            bias = "NEUTRAL"

        # Confidence based on alignment
        if all(b == biases[0] for b in biases) and biases[0] != 0:
            confidence = "high"
        elif abs(weighted_avg) > 0.3:
            confidence = "medium"
        else:
            confidence = "low"

        return bias, confidence

    def generate_all_reports(self, report_date: Optional[date] = None) -> list[TickerReport]:
        """
        Generate reports for all favorite tickers.

        Args:
            report_date: Date for reports

        Returns:
            List of TickerReport objects
        """
        tickers = settings.tickers
        reports = []

        for ticker in tickers:
            try:
                report = self.generate_ticker_report(ticker, report_date)
                reports.append(report)
            except Exception as e:
                logger.error(f"Failed to generate report for {ticker}: {e}")

        return reports

    def format_report(self, report: TickerReport) -> str:
        """Format ticker report as markdown."""
        lines = [
            f"# Premarket Report: {report.ticker}",
            f"**Date**: {report.report_date}",
            f"**Current Price**: ${report.current_price:.2f}" if report.current_price else "",
            f"**Overall Bias**: {report.overall_bias} ({report.bias_confidence} confidence)",
            "",
            "---",
            "",
            "## Daily Chart Analysis",
            f"- **Regime**: {report.daily_analysis.regime}",
            f"- **Always-In**: {report.daily_analysis.always_in}",
            f"- **Confidence**: {report.daily_analysis.confidence}",
            f"- {report.daily_analysis.description}",
            "",
        ]

        if report.daily_analysis.key_levels:
            lines.append("### Key Levels (Daily)")
            for level in report.daily_analysis.key_levels:
                lines.append(f"- ${level['price']}: {level['description']} ({level['type']})")
            lines.append("")

        if report.two_hour_analysis:
            lines.extend([
                "## 2-Hour Chart Analysis",
                f"- **Regime**: {report.two_hour_analysis.regime}",
                f"- **Always-In**: {report.two_hour_analysis.always_in}",
                f"- {report.two_hour_analysis.description}",
                "",
            ])

        if report.five_min_analysis:
            lines.extend([
                "## 5-Minute Context (Past 3 Days)",
                f"- **Recent Regime**: {report.five_min_analysis.regime}",
                f"- **Strength**: {report.five_min_analysis.strength.get('strength', 'unknown')}",
                "",
            ])

        # Magnets
        lines.extend([
            "## Magnet Map",
            "",
            "### Resistance (above current price)",
        ])
        for m in report.magnets_above[:5]:
            lines.append(f"- ${m['price']:.2f}: {m['description']}")

        lines.append("")
        lines.append("### Support (below current price)")
        for m in report.magnets_below[:5]:
            lines.append(f"- ${m['price']:.2f}: {m['description']}")

        if report.measured_moves:
            lines.append("")
            lines.append("### Measured Move Targets")
            for mm in report.measured_moves:
                direction = "↑" if mm["type"] == "up" else "↓"
                lines.append(f"- {direction} ${mm['target']:.2f}")

        # Plans
        lines.extend([
            "",
            "---",
            "",
            "## Plan A (Most Likely)",
            f"**Scenario**: {report.plan_a['scenario']}",
            f"**Bias**: {report.plan_a['bias']}",
            "",
            "**Setups to look for**:",
        ])
        for setup in report.plan_a.get("setups", []):
            lines.append(f"- {setup}")

        lines.extend([
            "",
            f"**Entry Zones**: {report.plan_a.get('entry_zones', 'N/A')}",
            f"**Targets**: {report.plan_a.get('targets', 'N/A')}",
            f"**Stops**: {report.plan_a.get('stops', 'N/A')}",
            "",
            "## Plan B (If Reversal)",
            f"**Scenario**: {report.plan_b['scenario']}",
            f"**Trigger**: {report.plan_b.get('trigger', 'N/A')}",
            f"**New Bias**: {report.plan_b.get('bias', 'N/A')}",
            f"**Action**: {report.plan_b.get('action', 'N/A')}",
            "",
            "---",
            "",
            "## ⚠️ Avoid List",
        ])
        for avoid in report.avoid_conditions:
            lines.append(f"- {avoid}")

        lines.append("")
        lines.append("---")
        lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def save_reports(self, reports: list[TickerReport], report_date: date) -> Path:
        """
        Save all reports to disk.

        Args:
            reports: List of ticker reports
            report_date: Date of reports

        Returns:
            Path to output directory
        """
        output_dir = OUTPUTS_DIR / report_date.strftime("%Y-%m-%d") / "premarket"
        output_dir.mkdir(parents=True, exist_ok=True)

        for report in reports:
            content = self.format_report(report)
            file_path = output_dir / f"{report.ticker}_premarket.md"

            with open(file_path, "w") as f:
                f.write(content)

            logger.info(f"Saved report: {file_path}")

        # Generate summary
        summary = self._generate_summary(reports)
        summary_path = output_dir / "SUMMARY.md"
        with open(summary_path, "w") as f:
            f.write(summary)

        return output_dir

    def _generate_summary(self, reports: list[TickerReport]) -> str:
        """Generate summary of all ticker reports."""
        lines = [
            "# Premarket Summary",
            f"**Date**: {date.today()}",
            f"**Tickers Analyzed**: {len(reports)}",
            "",
            "## Bias Overview",
            "",
            "| Ticker | Bias | Confidence | Daily Regime |",
            "|--------|------|------------|--------------|",
        ]

        for r in reports:
            lines.append(
                f"| {r.ticker} | {r.overall_bias} | {r.bias_confidence} | {r.daily_analysis.regime} |"
            )

        lines.extend([
            "",
            "## Best Setups Today",
            "",
        ])

        # Group by bias
        longs = [r for r in reports if r.overall_bias == "LONG"]
        shorts = [r for r in reports if r.overall_bias == "SHORT"]
        neutral = [r for r in reports if r.overall_bias == "NEUTRAL"]

        if longs:
            lines.append("### Long Candidates")
            for r in longs:
                lines.append(f"- **{r.ticker}**: {r.plan_a['setups'][0] if r.plan_a.get('setups') else 'N/A'}")

        if shorts:
            lines.append("")
            lines.append("### Short Candidates")
            for r in shorts:
                lines.append(f"- **{r.ticker}**: {r.plan_a['setups'][0] if r.plan_a.get('setups') else 'N/A'}")

        if neutral:
            lines.append("")
            lines.append("### Neutral / Range-Bound")
            for r in neutral:
                lines.append(f"- **{r.ticker}**: Wait for breakout or fade extremes")

        return "\n".join(lines)
