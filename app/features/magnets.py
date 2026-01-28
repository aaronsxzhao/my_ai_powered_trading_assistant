"""
Magnet Detection for Brooks Trading Coach.

Magnets are price levels that attract price:
- Prior day high/low/close
- Gaps
- Measured moves
- Major swing points
- Round numbers
- Moving averages
"""

from dataclasses import dataclass
from datetime import datetime as dt
from typing import Literal, Optional

import numpy as np
import pandas as pd

from app.features.ohlc_features import OHLCFeatures


@dataclass
class Magnet:
    """A price magnet / key level."""

    price: float
    type: str
    description: str
    strength: Literal["weak", "moderate", "strong"]
    timestamp: Optional[dt] = None


@dataclass
class MeasuredMove:
    """A measured move target."""

    start_price: float
    end_price: float
    target_price: float
    move_size: float
    type: Literal["up", "down"]


class MagnetDetector:
    """
    Detect key price levels (magnets) from OHLCV data.

    Magnets are levels where price is likely to travel to and
    where reactions often occur.
    """

    def __init__(self, daily_df: pd.DataFrame, intraday_df: Optional[pd.DataFrame] = None):
        """
        Initialize with OHLCV data.

        Args:
            daily_df: Daily OHLCV DataFrame
            intraday_df: Optional intraday OHLCV DataFrame
        """
        self.daily_df = daily_df.copy()
        self.intraday_df = intraday_df.copy() if intraday_df is not None else None

        # Compute features for daily data
        self.daily_features = OHLCFeatures(self.daily_df)
        self.daily_features.add_ema(20)
        self.daily_features.add_sma(50)
        self.daily_features.add_sma(200)

    def get_prior_day_levels(self) -> list[Magnet]:
        """
        Get prior day high, low, and close as magnets.

        Returns:
            List of Magnet objects for prior day levels
        """
        if len(self.daily_df) < 2:
            return []

        prior_day = self.daily_df.iloc[-2]
        magnets = []

        magnets.append(
            Magnet(
                price=prior_day["high"],
                type="prior_day_high",
                description="Prior Day High",
                strength="strong",
                timestamp=prior_day.get("datetime"),
            )
        )

        magnets.append(
            Magnet(
                price=prior_day["low"],
                type="prior_day_low",
                description="Prior Day Low",
                strength="strong",
                timestamp=prior_day.get("datetime"),
            )
        )

        magnets.append(
            Magnet(
                price=prior_day["close"],
                type="prior_day_close",
                description="Prior Day Close",
                strength="moderate",
                timestamp=prior_day.get("datetime"),
            )
        )

        return magnets

    def get_gap_levels(self, lookback_days: int = 10) -> list[Magnet]:
        """
        Find unfilled gaps.

        A gap exists when today's range doesn't overlap with prior day's range.

        Args:
            lookback_days: Days to look back for gaps

        Returns:
            List of Magnet objects for gap levels
        """
        if len(self.daily_df) < 2:
            return []

        magnets = []
        df = self.daily_df.tail(lookback_days + 1)

        for i in range(1, len(df)):
            current = df.iloc[i]
            prior = df.iloc[i - 1]

            # Gap up: current low > prior high
            if current["low"] > prior["high"]:
                gap_low = prior["high"]
                gap_high = current["low"]

                # Check if gap is still unfilled
                remaining_bars = df.iloc[i + 1 :] if i + 1 < len(df) else pd.DataFrame()
                if remaining_bars.empty or remaining_bars["low"].min() > gap_low:
                    magnets.append(
                        Magnet(
                            price=gap_low,
                            type="gap_up_bottom",
                            description="Gap Up Bottom (unfilled)",
                            strength="strong",
                            timestamp=current.get("datetime"),
                        )
                    )

            # Gap down: current high < prior low
            elif current["high"] < prior["low"]:
                gap_high = prior["low"]
                gap_low = current["high"]

                remaining_bars = df.iloc[i + 1 :] if i + 1 < len(df) else pd.DataFrame()
                if remaining_bars.empty or remaining_bars["high"].max() < gap_high:
                    magnets.append(
                        Magnet(
                            price=gap_high,
                            type="gap_down_top",
                            description="Gap Down Top (unfilled)",
                            strength="strong",
                            timestamp=current.get("datetime"),
                        )
                    )

        return magnets

    def get_swing_levels(self, lookback_days: int = 60) -> list[Magnet]:
        """
        Get major swing high/low levels.

        Args:
            lookback_days: Days to look back

        Returns:
            List of Magnet objects for swing levels
        """
        magnets = []

        # Get swings from daily features
        swing_highs = self.daily_features.find_swing_highs(lookback=5)
        swing_lows = self.daily_features.find_swing_lows(lookback=5)

        # Filter to recent swings
        min_index = max(0, len(self.daily_df) - lookback_days)

        for swing in swing_highs:
            if swing.index >= min_index:
                magnets.append(
                    Magnet(
                        price=swing.price,
                        type="swing_high",
                        description="Swing High",
                        strength="moderate",
                        timestamp=swing.datetime,
                    )
                )

        for swing in swing_lows:
            if swing.index >= min_index:
                magnets.append(
                    Magnet(
                        price=swing.price,
                        type="swing_low",
                        description="Swing Low",
                        strength="moderate",
                        timestamp=swing.datetime,
                    )
                )

        return magnets

    def get_moving_average_levels(self) -> list[Magnet]:
        """
        Get moving average levels as magnets.

        Returns:
            List of Magnet objects for MA levels
        """
        magnets = []
        df = self.daily_features.get_dataframe()

        if len(df) == 0:
            return magnets

        last_row = df.iloc[-1]

        # EMA 20
        if "ema_20" in last_row and not np.isnan(last_row["ema_20"]):
            magnets.append(
                Magnet(
                    price=round(last_row["ema_20"], 2),
                    type="ema_20",
                    description="20 EMA (Daily)",
                    strength="moderate",
                )
            )

        # SMA 50
        if "sma_50" in last_row and not np.isnan(last_row["sma_50"]):
            magnets.append(
                Magnet(
                    price=round(last_row["sma_50"], 2),
                    type="sma_50",
                    description="50 SMA (Daily)",
                    strength="moderate",
                )
            )

        # SMA 200
        if "sma_200" in last_row and not np.isnan(last_row["sma_200"]):
            magnets.append(
                Magnet(
                    price=round(last_row["sma_200"], 2),
                    type="sma_200",
                    description="200 SMA (Daily)",
                    strength="strong",
                )
            )

        return magnets

    def calculate_measured_moves(self, lookback: int = 30) -> list[MeasuredMove]:
        """
        Calculate measured move targets from recent legs.

        A measured move projects the size of a prior leg from a pullback.

        Args:
            lookback: Bars to analyze

        Returns:
            List of MeasuredMove objects
        """
        moves = []

        swing_highs = self.daily_features.find_swing_highs(lookback=3)
        swing_lows = self.daily_features.find_swing_lows(lookback=3)

        min_index = max(0, len(self.daily_df) - lookback)

        # Look for up moves (low to high to low pattern)
        recent_lows = [s for s in swing_lows if s.index >= min_index]
        recent_highs = [s for s in swing_highs if s.index >= min_index]

        if len(recent_lows) >= 2 and len(recent_highs) >= 1:
            # Find pattern: low1 -> high -> low2
            for i, low1 in enumerate(recent_lows[:-1]):
                # Find high after low1
                highs_after = [h for h in recent_highs if h.index > low1.index]
                if not highs_after:
                    continue

                high = highs_after[0]

                # Find low2 after high
                lows_after = [
                    low_swing for low_swing in recent_lows if low_swing.index > high.index
                ]
                if not lows_after:
                    continue

                low2 = lows_after[0]

                # Calculate measured move up from low2
                move_size = high.price - low1.price
                target = low2.price + move_size

                moves.append(
                    MeasuredMove(
                        start_price=low1.price,
                        end_price=high.price,
                        target_price=round(target, 2),
                        move_size=round(move_size, 2),
                        type="up",
                    )
                )

        # Look for down moves (high to low to high pattern)
        if len(recent_highs) >= 2 and len(recent_lows) >= 1:
            for i, high1 in enumerate(recent_highs[:-1]):
                lows_after = [
                    low_swing for low_swing in recent_lows if low_swing.index > high1.index
                ]
                if not lows_after:
                    continue

                low = lows_after[0]

                highs_after = [h for h in recent_highs if h.index > low.index]
                if not highs_after:
                    continue

                high2 = highs_after[0]

                move_size = high1.price - low.price
                target = high2.price - move_size

                moves.append(
                    MeasuredMove(
                        start_price=high1.price,
                        end_price=low.price,
                        target_price=round(target, 2),
                        move_size=round(move_size, 2),
                        type="down",
                    )
                )

        return moves

    def get_round_number_levels(
        self, current_price: float, range_pct: float = 0.05
    ) -> list[Magnet]:
        """
        Get round number levels near current price.

        Args:
            current_price: Current price
            range_pct: Percentage range to search

        Returns:
            List of Magnet objects for round numbers
        """
        magnets = []

        # Determine appropriate round number interval based on price
        if current_price >= 1000:
            intervals = [100, 50]
        elif current_price >= 100:
            intervals = [50, 25, 10]
        elif current_price >= 10:
            intervals = [10, 5, 1]
        else:
            intervals = [1, 0.5]

        price_range = current_price * range_pct

        for interval in intervals:
            # Find round numbers in range
            low = current_price - price_range
            high = current_price + price_range

            start = int(low / interval) * interval
            level = start

            while level <= high:
                if low <= level <= high and level != current_price:
                    strength = "strong" if interval == intervals[0] else "weak"
                    magnets.append(
                        Magnet(
                            price=level,
                            type="round_number",
                            description=f"Round Number ({interval} interval)",
                            strength=strength,
                        )
                    )
                level += interval

        return magnets

    def get_all_magnets(
        self,
        current_price: Optional[float] = None,
        include_round_numbers: bool = True,
    ) -> list[Magnet]:
        """
        Get all magnets/key levels.

        Args:
            current_price: Current price (for round numbers)
            include_round_numbers: Whether to include round number levels

        Returns:
            List of all Magnet objects, sorted by price
        """
        magnets = []

        # Prior day levels
        magnets.extend(self.get_prior_day_levels())

        # Gaps
        magnets.extend(self.get_gap_levels())

        # Swing levels
        magnets.extend(self.get_swing_levels())

        # Moving averages
        magnets.extend(self.get_moving_average_levels())

        # Round numbers
        if include_round_numbers and current_price:
            magnets.extend(self.get_round_number_levels(current_price))

        # Deduplicate close prices (within 0.1%)
        unique_magnets = []
        for mag in magnets:
            is_duplicate = False
            for existing in unique_magnets:
                if abs(mag.price - existing.price) / existing.price < 0.001:
                    # Keep the stronger one
                    if self._strength_rank(mag.strength) > self._strength_rank(existing.strength):
                        unique_magnets.remove(existing)
                        unique_magnets.append(mag)
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_magnets.append(mag)

        # Sort by price
        return sorted(unique_magnets, key=lambda m: m.price)

    def _strength_rank(self, strength: str) -> int:
        """Convert strength to numeric rank."""
        ranks = {"weak": 1, "moderate": 2, "strong": 3}
        return ranks.get(strength, 0)

    def get_magnet_map(self, current_price: float) -> dict:
        """
        Get a structured magnet map for coaching.

        Args:
            current_price: Current price

        Returns:
            Dictionary with magnets above/below price and measured move targets
        """
        all_magnets = self.get_all_magnets(current_price)
        measured_moves = self.calculate_measured_moves()

        above = [m for m in all_magnets if m.price > current_price]
        below = [m for m in all_magnets if m.price < current_price]

        return {
            "current_price": current_price,
            "magnets_above": above[:5],  # Top 5 above
            "magnets_below": below[-5:],  # Top 5 below (closest to current)
            "measured_move_targets": measured_moves,
            "nearest_resistance": above[0] if above else None,
            "nearest_support": below[-1] if below else None,
        }

    def format_magnet_summary(self, current_price: float) -> str:
        """
        Format magnet map as readable text.

        Args:
            current_price: Current price

        Returns:
            Formatted string summary
        """
        magnet_map = self.get_magnet_map(current_price)

        lines = [f"**Magnet Map** (Current: ${current_price:.2f})", ""]

        # Resistance levels
        lines.append("**Resistance (above price):**")
        for m in magnet_map["magnets_above"]:
            lines.append(f"  - ${m.price:.2f}: {m.description} [{m.strength}]")

        lines.append("")

        # Support levels
        lines.append("**Support (below price):**")
        for m in reversed(magnet_map["magnets_below"]):
            lines.append(f"  - ${m.price:.2f}: {m.description} [{m.strength}]")

        # Measured moves
        if magnet_map["measured_move_targets"]:
            lines.append("")
            lines.append("**Measured Move Targets:**")
            for mm in magnet_map["measured_move_targets"][:3]:
                direction = "↑" if mm.type == "up" else "↓"
                lines.append(
                    f"  - {direction} ${mm.target_price:.2f} (leg size: ${mm.move_size:.2f})"
                )

        return "\n".join(lines)
