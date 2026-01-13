"""
Brooks Price Action Pattern Detection.

Heuristic detection of key Brooks concepts:
- Trend vs Trading Range
- Always-In direction
- Wedges / 3-push patterns
- 2nd entries
- Breakouts and pullbacks
- Climaxes
- Failed breakouts

All detections include confidence levels (low/medium/high).
These are approximations for coaching purposes, not trading signals.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from app.features.ohlc_features import OHLCFeatures, SwingPoint
from app.config import settings


class Regime(Enum):
    """Market regime classification."""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    TRADING_RANGE = "trading_range"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


class AlwaysIn(Enum):
    """Always-In direction."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Confidence(Enum):
    """Confidence level for pattern detection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PatternDetection:
    """Result of pattern detection."""

    pattern: str
    confidence: Confidence
    description: str
    index: int | None = None
    supporting_data: dict | None = None


@dataclass
class RegimeAnalysis:
    """Result of regime analysis."""

    regime: Regime
    always_in: AlwaysIn
    confidence: Confidence
    description: str
    metrics: dict


class BrooksPatternDetector:
    """
    Detect Brooks price action patterns from OHLCV data.

    All methods are heuristic approximations with confidence levels.
    """

    def __init__(self, features: OHLCFeatures):
        """
        Initialize with OHLCFeatures object.

        Args:
            features: OHLCFeatures instance with computed indicators
        """
        self.features = features
        self.df = features.df

        # Ensure required indicators are computed
        self.features.add_ema()
        self.features.add_atr()
        self.features.detect_trend_bars()
        self.features.detect_overlap()
        self.features.compute_ema_slope()
        self.features.closes_above_ema_pct()

    def analyze_regime(self, window: int = 20) -> RegimeAnalysis:
        """
        Determine current market regime (trend vs trading range).

        Uses:
        - EMA slope
        - Percentage of closes above/below EMA
        - Overlap ratio
        - Swing structure (higher highs/lows or lower highs/lows)

        Args:
            window: Lookback window for analysis

        Returns:
            RegimeAnalysis with regime, always-in, and confidence
        """
        if len(self.df) < window:
            return RegimeAnalysis(
                regime=Regime.UNKNOWN,
                always_in=AlwaysIn.NEUTRAL,
                confidence=Confidence.LOW,
                description="Insufficient data for regime analysis",
                metrics={},
            )

        recent = self.df.tail(window)

        # Metrics
        ema_slope = recent["ema_slope"].iloc[-1] if "ema_slope" in recent.columns else 0
        closes_above_pct = (
            recent["closes_above_ema_pct"].iloc[-1]
            if "closes_above_ema_pct" in recent.columns
            else 0.5
        )
        avg_overlap = (
            recent["overlap_ratio"].mean() if "overlap_ratio" in recent.columns else 0.5
        )

        # Swing structure
        swings = self.features.get_all_swings(lookback=3)
        recent_swings = [s for s in swings if s.index >= len(self.df) - window]

        hh_hl = self._check_higher_highs_lows(recent_swings)
        lh_ll = self._check_lower_highs_lows(recent_swings)

        # Determine regime
        slope_threshold = settings.get("analysis.trend_ema_slope_threshold", 0.001)
        closes_threshold = settings.get("analysis.trend_closes_above_ema_pct", 0.6)
        overlap_threshold = settings.get("analysis.overlap_ratio_range_threshold", 0.5)

        metrics = {
            "ema_slope": round(ema_slope, 4) if not np.isnan(ema_slope) else 0,
            "closes_above_ema_pct": round(closes_above_pct, 2) if not np.isnan(closes_above_pct) else 0.5,
            "avg_overlap_ratio": round(avg_overlap, 2) if not np.isnan(avg_overlap) else 0.5,
            "higher_highs_lows": hh_hl,
            "lower_highs_lows": lh_ll,
        }

        # Strong uptrend
        if (
            ema_slope > slope_threshold
            and closes_above_pct > closes_threshold
            and hh_hl
        ):
            regime = Regime.TREND_UP
            always_in = AlwaysIn.LONG
            confidence = Confidence.HIGH if avg_overlap < overlap_threshold else Confidence.MEDIUM
            description = "Strong uptrend with higher highs and higher lows"

        # Moderate uptrend
        elif ema_slope > 0 and closes_above_pct > 0.5:
            regime = Regime.TREND_UP
            always_in = AlwaysIn.LONG
            confidence = Confidence.MEDIUM if hh_hl else Confidence.LOW
            description = "Moderate uptrend, price above EMA"

        # Strong downtrend
        elif (
            ema_slope < -slope_threshold
            and closes_above_pct < (1 - closes_threshold)
            and lh_ll
        ):
            regime = Regime.TREND_DOWN
            always_in = AlwaysIn.SHORT
            confidence = Confidence.HIGH if avg_overlap < overlap_threshold else Confidence.MEDIUM
            description = "Strong downtrend with lower highs and lower lows"

        # Moderate downtrend
        elif ema_slope < 0 and closes_above_pct < 0.5:
            regime = Regime.TREND_DOWN
            always_in = AlwaysIn.SHORT
            confidence = Confidence.MEDIUM if lh_ll else Confidence.LOW
            description = "Moderate downtrend, price below EMA"

        # Trading range
        elif avg_overlap > overlap_threshold:
            regime = Regime.TRADING_RANGE
            always_in = AlwaysIn.NEUTRAL
            confidence = Confidence.MEDIUM
            description = "Trading range with high overlap and no clear direction"

        else:
            regime = Regime.TRADING_RANGE
            always_in = AlwaysIn.NEUTRAL
            confidence = Confidence.LOW
            description = "Mixed signals, treat as potential trading range"

        return RegimeAnalysis(
            regime=regime,
            always_in=always_in,
            confidence=confidence,
            description=description,
            metrics=metrics,
        )

    def _check_higher_highs_lows(self, swings: list[SwingPoint]) -> bool:
        """Check if recent swings show higher highs and higher lows."""
        if len(swings) < 4:
            return False

        highs = [s for s in swings if s.type == "high"]
        lows = [s for s in swings if s.type == "low"]

        if len(highs) < 2 or len(lows) < 2:
            return False

        # Check last 2 highs and lows
        hh = highs[-1].price > highs[-2].price
        hl = lows[-1].price > lows[-2].price

        return hh and hl

    def _check_lower_highs_lows(self, swings: list[SwingPoint]) -> bool:
        """Check if recent swings show lower highs and lower lows."""
        if len(swings) < 4:
            return False

        highs = [s for s in swings if s.type == "high"]
        lows = [s for s in swings if s.type == "low"]

        if len(highs) < 2 or len(lows) < 2:
            return False

        lh = highs[-1].price < highs[-2].price
        ll = lows[-1].price < lows[-2].price

        return lh and ll

    def detect_wedge(self, lookback: int = 20) -> PatternDetection | None:
        """
        Detect wedge / 3-push pattern.

        A wedge has 3 pushes with diminishing momentum into a trendline area.

        Args:
            lookback: Bars to analyze

        Returns:
            PatternDetection if wedge found, None otherwise
        """
        swings = self.features.get_all_swings(lookback=3)
        recent_swings = [s for s in swings if s.index >= len(self.df) - lookback]

        highs = [s for s in recent_swings if s.type == "high"]
        lows = [s for s in recent_swings if s.type == "low"]

        # Check for bull wedge (3 higher lows with diminishing momentum)
        if len(highs) >= 3:
            last_3_highs = highs[-3:]
            if all(
                last_3_highs[i].price > last_3_highs[i - 1].price
                for i in range(1, 3)
            ):
                # Check diminishing momentum (smaller range between pushes)
                push1 = last_3_highs[1].price - last_3_highs[0].price
                push2 = last_3_highs[2].price - last_3_highs[1].price

                if push2 < push1 * 0.7:  # Second push is weaker
                    return PatternDetection(
                        pattern="wedge_bull",
                        confidence=Confidence.MEDIUM,
                        description="3-push wedge up with diminishing momentum - potential reversal setup",
                        index=last_3_highs[-1].index,
                        supporting_data={
                            "push1_size": round(push1, 2),
                            "push2_size": round(push2, 2),
                            "highs": [h.price for h in last_3_highs],
                        },
                    )

        # Check for bear wedge (3 lower lows with diminishing momentum)
        if len(lows) >= 3:
            last_3_lows = lows[-3:]
            if all(
                last_3_lows[i].price < last_3_lows[i - 1].price
                for i in range(1, 3)
            ):
                push1 = last_3_lows[0].price - last_3_lows[1].price
                push2 = last_3_lows[1].price - last_3_lows[2].price

                if push2 < push1 * 0.7:
                    return PatternDetection(
                        pattern="wedge_bear",
                        confidence=Confidence.MEDIUM,
                        description="3-push wedge down with diminishing momentum - potential reversal setup",
                        index=last_3_lows[-1].index,
                        supporting_data={
                            "push1_size": round(push1, 2),
                            "push2_size": round(push2, 2),
                            "lows": [l.price for l in last_3_lows],
                        },
                    )

        return None

    def detect_second_entry(self, direction: Literal["long", "short"], lookback: int = 15) -> PatternDetection | None:
        """
        Detect 2nd entry pattern.

        In a pullback, after two failed attempts to resume countertrend,
        price attempts to resume the prior trend.

        Args:
            direction: Expected direction of 2nd entry
            lookback: Bars to analyze

        Returns:
            PatternDetection if pattern found, None otherwise
        """
        if len(self.df) < lookback:
            return None

        recent = self.df.tail(lookback)
        swings = self.features.get_all_swings(lookback=2)
        recent_swings = [s for s in swings if s.index >= len(self.df) - lookback]

        if direction == "long":
            # Look for 2 higher lows in a pullback
            lows = [s for s in recent_swings if s.type == "low"]
            if len(lows) >= 2:
                if lows[-1].price >= lows[-2].price * 0.995:  # Second low holds
                    # Check if we're in an uptrend context
                    regime = self.analyze_regime()
                    if regime.always_in == AlwaysIn.LONG:
                        return PatternDetection(
                            pattern="second_entry_long",
                            confidence=Confidence.MEDIUM,
                            description="2nd entry buy: second higher low in uptrend pullback",
                            index=lows[-1].index,
                            supporting_data={
                                "first_low": lows[-2].price,
                                "second_low": lows[-1].price,
                                "context": regime.regime.value,
                            },
                        )

        else:  # short
            highs = [s for s in recent_swings if s.type == "high"]
            if len(highs) >= 2:
                if highs[-1].price <= highs[-2].price * 1.005:  # Second high holds
                    regime = self.analyze_regime()
                    if regime.always_in == AlwaysIn.SHORT:
                        return PatternDetection(
                            pattern="second_entry_short",
                            confidence=Confidence.MEDIUM,
                            description="2nd entry sell: second lower high in downtrend pullback",
                            index=highs[-1].index,
                            supporting_data={
                                "first_high": highs[-2].price,
                                "second_high": highs[-1].price,
                                "context": regime.regime.value,
                            },
                        )

        return None

    def detect_climax(self, lookback: int = 10) -> PatternDetection | None:
        """
        Detect climax pattern.

        A climax has:
        - Large trend bar(s)
        - Extended move (multiple ATRs)
        - Channel overshoot
        - Increase in range/ATR

        Args:
            lookback: Bars to analyze

        Returns:
            PatternDetection if climax found, None otherwise
        """
        if len(self.df) < lookback + 10:
            return None

        recent = self.df.tail(lookback)
        prior = self.df.iloc[-(lookback + 10) : -lookback]

        # Check for large trend bars
        if "is_trend_bar" not in recent.columns:
            return None

        trend_bar_count = recent["is_trend_bar"].sum()
        if trend_bar_count < 2:
            return None

        # Check for range expansion
        recent_avg_range = recent["range"].mean()
        prior_avg_range = prior["range"].mean()

        if recent_avg_range < prior_avg_range * 1.5:
            return None

        # Check for directional bars
        bull_bars = recent["is_bull"].sum()
        bear_bars = recent["is_bear"].sum()

        if bull_bars >= lookback * 0.7:
            direction = "bull"
        elif bear_bars >= lookback * 0.7:
            direction = "bear"
        else:
            return None

        # Calculate extension from EMA
        ema_col = "ema_20" if "ema_20" in recent.columns else None
        if ema_col:
            last_close = recent["close"].iloc[-1]
            last_ema = recent[ema_col].iloc[-1]
            atr = recent["atr"].iloc[-1] if "atr" in recent.columns else recent["range"].mean()

            extension = abs(last_close - last_ema) / atr if atr > 0 else 0

            if extension > 2:  # Extended more than 2 ATR from EMA
                return PatternDetection(
                    pattern=f"climax_{direction}",
                    confidence=Confidence.MEDIUM,
                    description=f"Potential {direction} climax: extended move with large bars and range expansion",
                    index=len(self.df) - 1,
                    supporting_data={
                        "trend_bars": int(trend_bar_count),
                        "range_expansion": round(recent_avg_range / prior_avg_range, 2),
                        "ema_extension_atr": round(extension, 2),
                    },
                )

        return None

    def detect_failed_breakout(self, lookback: int = 10) -> PatternDetection | None:
        """
        Detect failed breakout pattern.

        Price breaks above/below a key level but quickly reverses.

        Args:
            lookback: Bars to analyze

        Returns:
            PatternDetection if pattern found, None otherwise
        """
        if len(self.df) < lookback + 20:
            return None

        recent = self.df.tail(lookback)
        prior = self.df.iloc[-(lookback + 20) : -lookback]

        prior_high = prior["high"].max()
        prior_low = prior["low"].min()

        recent_high = recent["high"].max()
        recent_low = recent["low"].min()
        last_close = recent["close"].iloc[-1]

        # Failed breakout above
        if recent_high > prior_high and last_close < prior_high:
            return PatternDetection(
                pattern="failed_breakout_high",
                confidence=Confidence.MEDIUM,
                description="Failed breakout above prior high - price rejected and closed back inside range",
                index=recent["high"].idxmax(),
                supporting_data={
                    "prior_high": round(prior_high, 2),
                    "breakout_high": round(recent_high, 2),
                    "close": round(last_close, 2),
                },
            )

        # Failed breakout below
        if recent_low < prior_low and last_close > prior_low:
            return PatternDetection(
                pattern="failed_breakout_low",
                confidence=Confidence.MEDIUM,
                description="Failed breakout below prior low - price rejected and closed back inside range",
                index=recent["low"].idxmin(),
                supporting_data={
                    "prior_low": round(prior_low, 2),
                    "breakout_low": round(recent_low, 2),
                    "close": round(last_close, 2),
                },
            )

        return None

    def detect_breakout_pullback(self, lookback: int = 15) -> PatternDetection | None:
        """
        Detect breakout pullback pattern.

        After a breakout, price pulls back to retest the breakout level.

        Args:
            lookback: Bars to analyze

        Returns:
            PatternDetection if pattern found, None otherwise
        """
        if len(self.df) < lookback + 30:
            return None

        # Get regime to understand context
        regime = self.analyze_regime()

        if regime.regime == Regime.TREND_UP:
            # Look for pullback to prior resistance (now support)
            recent = self.df.tail(lookback)
            prior = self.df.iloc[-(lookback + 30) : -(lookback + 10)]

            prior_high = prior["high"].max()
            recent_low = recent["low"].min()

            # Pullback to within 0.5% of prior high
            if abs(recent_low - prior_high) / prior_high < 0.005:
                return PatternDetection(
                    pattern="breakout_pullback_long",
                    confidence=Confidence.MEDIUM,
                    description="Breakout pullback: price testing prior resistance as support in uptrend",
                    index=recent["low"].idxmin(),
                    supporting_data={
                        "prior_high": round(prior_high, 2),
                        "pullback_low": round(recent_low, 2),
                    },
                )

        elif regime.regime == Regime.TREND_DOWN:
            recent = self.df.tail(lookback)
            prior = self.df.iloc[-(lookback + 30) : -(lookback + 10)]

            prior_low = prior["low"].min()
            recent_high = recent["high"].max()

            if abs(recent_high - prior_low) / prior_low < 0.005:
                return PatternDetection(
                    pattern="breakout_pullback_short",
                    confidence=Confidence.MEDIUM,
                    description="Breakout pullback: price testing prior support as resistance in downtrend",
                    index=recent["high"].idxmax(),
                    supporting_data={
                        "prior_low": round(prior_low, 2),
                        "pullback_high": round(recent_high, 2),
                    },
                )

        return None

    def detect_all_patterns(self) -> list[PatternDetection]:
        """
        Run all pattern detections and return findings.

        Returns:
            List of all detected patterns
        """
        patterns = []

        # Wedge
        wedge = self.detect_wedge()
        if wedge:
            patterns.append(wedge)

        # 2nd entries
        for direction in ["long", "short"]:
            second = self.detect_second_entry(direction)
            if second:
                patterns.append(second)

        # Climax
        climax = self.detect_climax()
        if climax:
            patterns.append(climax)

        # Failed breakout
        failed_bo = self.detect_failed_breakout()
        if failed_bo:
            patterns.append(failed_bo)

        # Breakout pullback
        bo_pb = self.detect_breakout_pullback()
        if bo_pb:
            patterns.append(bo_pb)

        return patterns

    def get_trading_context(self) -> dict:
        """
        Get comprehensive trading context for coaching.

        Returns:
            Dictionary with regime, patterns, and recommendations
        """
        regime = self.analyze_regime()
        patterns = self.detect_all_patterns()
        strength = self.features.get_recent_strength()

        # Determine best setups based on context
        best_setups = []
        avoid_setups = []

        if regime.regime == Regime.TREND_UP:
            best_setups = [
                "2nd entry buy",
                "Breakout pullback long",
                "Trend resumption long",
            ]
            avoid_setups = [
                "Fading strong moves",
                "Countertrend without reversal structure",
                "Trading range tactics in trend",
            ]

        elif regime.regime == Regime.TREND_DOWN:
            best_setups = [
                "2nd entry sell",
                "Breakout pullback short",
                "Trend resumption short",
            ]
            avoid_setups = [
                "Buying dips without reversal structure",
                "Countertrend longs without momentum shift",
            ]

        elif regime.regime == Regime.TRADING_RANGE:
            best_setups = [
                "Fade range extremes",
                "Wait for breakout with follow-through",
                "Reduce position size",
            ]
            avoid_setups = [
                "Stop entries in tight range",
                "Expecting big moves",
                "With-trend tactics",
            ]

        return {
            "regime": regime,
            "patterns": patterns,
            "strength": strength,
            "best_setups": best_setups,
            "avoid_setups": avoid_setups,
        }
