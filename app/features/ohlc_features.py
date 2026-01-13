"""
OHLCV feature extraction for technical analysis.

Computes indicators needed for Brooks price action analysis:
- Moving averages (EMA, SMA)
- ATR (Average True Range)
- Swing points (highs/lows)
- Bar characteristics (trend bars, dojis, overlap)
- Trendline approximations
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from app.config import settings


@dataclass
class BarCharacteristics:
    """Characteristics of a single price bar."""

    is_bull: bool
    is_bear: bool
    is_doji: bool
    is_trend_bar: bool  # Strong directional bar
    body_size: float
    upper_wick: float
    lower_wick: float
    range_size: float
    close_location: float  # 0 = low, 1 = high


@dataclass
class SwingPoint:
    """A swing high or low point."""

    index: int
    datetime: pd.Timestamp
    price: float
    type: Literal["high", "low"]


class OHLCFeatures:
    """
    Compute technical features from OHLCV data.

    All methods expect a DataFrame with columns: datetime, open, high, low, close, volume
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        self._compute_basic_features()

    def _compute_basic_features(self) -> None:
        """Compute basic bar characteristics."""
        df = self.df

        # Bar range and body
        df["range"] = df["high"] - df["low"]
        df["body"] = abs(df["close"] - df["open"])
        df["body_pct"] = df["body"] / df["range"].replace(0, np.nan)

        # Directional
        df["is_bull"] = df["close"] > df["open"]
        df["is_bear"] = df["close"] < df["open"]

        # Doji detection (body < 30% of range)
        df["is_doji"] = df["body_pct"] < 0.3

        # Close location within bar (0 = low, 1 = high)
        df["close_location"] = (df["close"] - df["low"]) / df["range"].replace(0, np.nan)
        df["close_location"] = df["close_location"].fillna(0.5)

        # Wicks
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]

        self.df = df

    def add_ema(self, period: int | None = None, column: str = "close") -> pd.Series:
        """
        Add Exponential Moving Average.

        Args:
            period: EMA period (default from config)
            column: Column to compute EMA on

        Returns:
            EMA Series
        """
        period = period or settings.ema_period
        col_name = f"ema_{period}"

        if col_name not in self.df.columns:
            self.df[col_name] = self.df[column].ewm(span=period, adjust=False).mean()

        return self.df[col_name]

    def add_sma(self, period: int, column: str = "close") -> pd.Series:
        """Add Simple Moving Average."""
        col_name = f"sma_{period}"

        if col_name not in self.df.columns:
            self.df[col_name] = self.df[column].rolling(window=period).mean()

        return self.df[col_name]

    def add_atr(self, period: int | None = None) -> pd.Series:
        """
        Add Average True Range.

        Args:
            period: ATR period (default from config)

        Returns:
            ATR Series
        """
        period = period or settings.atr_period

        if "atr" not in self.df.columns:
            df = self.df
            high_low = df["high"] - df["low"]
            high_close = abs(df["high"] - df["close"].shift(1))
            low_close = abs(df["low"] - df["close"].shift(1))

            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            self.df["atr"] = true_range.rolling(window=period).mean()

        return self.df["atr"]

    def find_swing_highs(self, lookback: int | None = None) -> list[SwingPoint]:
        """
        Find swing high points.

        A swing high is a bar with highs lower on both sides.

        Args:
            lookback: Number of bars to look on each side

        Returns:
            List of SwingPoint objects
        """
        lookback = lookback or settings.swing_lookback
        df = self.df
        swing_highs = []

        for i in range(lookback, len(df) - lookback):
            is_swing_high = True
            current_high = df.iloc[i]["high"]

            # Check bars to the left
            for j in range(1, lookback + 1):
                if df.iloc[i - j]["high"] >= current_high:
                    is_swing_high = False
                    break

            # Check bars to the right
            if is_swing_high:
                for j in range(1, lookback + 1):
                    if df.iloc[i + j]["high"] >= current_high:
                        is_swing_high = False
                        break

            if is_swing_high:
                swing_highs.append(
                    SwingPoint(
                        index=i,
                        datetime=df.iloc[i]["datetime"],
                        price=current_high,
                        type="high",
                    )
                )

        return swing_highs

    def find_swing_lows(self, lookback: int | None = None) -> list[SwingPoint]:
        """Find swing low points."""
        lookback = lookback or settings.swing_lookback
        df = self.df
        swing_lows = []

        for i in range(lookback, len(df) - lookback):
            is_swing_low = True
            current_low = df.iloc[i]["low"]

            for j in range(1, lookback + 1):
                if df.iloc[i - j]["low"] <= current_low:
                    is_swing_low = False
                    break

            if is_swing_low:
                for j in range(1, lookback + 1):
                    if df.iloc[i + j]["low"] <= current_low:
                        is_swing_low = False
                        break

            if is_swing_low:
                swing_lows.append(
                    SwingPoint(
                        index=i,
                        datetime=df.iloc[i]["datetime"],
                        price=current_low,
                        type="low",
                    )
                )

        return swing_lows

    def get_all_swings(self, lookback: int | None = None) -> list[SwingPoint]:
        """Get all swing points sorted by index."""
        highs = self.find_swing_highs(lookback)
        lows = self.find_swing_lows(lookback)
        all_swings = highs + lows
        return sorted(all_swings, key=lambda x: x.index)

    def detect_trend_bars(self, threshold: float = 0.7) -> pd.Series:
        """
        Detect strong trend bars.

        A trend bar has body > threshold of range and closes near extreme.

        Args:
            threshold: Minimum body/range ratio

        Returns:
            Boolean Series indicating trend bars
        """
        df = self.df
        is_trend_bar = (df["body_pct"] > threshold) & (
            (df["close_location"] > 0.8) | (df["close_location"] < 0.2)
        )
        self.df["is_trend_bar"] = is_trend_bar
        return is_trend_bar

    def detect_overlap(self, window: int = 5) -> pd.Series:
        """
        Detect overlapping price action (trading range behavior).

        Overlap ratio = how much current bar overlaps with previous bars.

        Args:
            window: Number of bars to check for overlap

        Returns:
            Series of overlap ratios (0 = no overlap, 1 = full overlap)
        """
        df = self.df
        overlap_ratios = []

        for i in range(len(df)):
            if i < window:
                overlap_ratios.append(np.nan)
                continue

            current_high = df.iloc[i]["high"]
            current_low = df.iloc[i]["low"]
            current_range = current_high - current_low

            if current_range == 0:
                overlap_ratios.append(1.0)
                continue

            # Find overlap with previous bars
            prev_high = df.iloc[i - window : i]["high"].max()
            prev_low = df.iloc[i - window : i]["low"].min()

            overlap_high = min(current_high, prev_high)
            overlap_low = max(current_low, prev_low)
            overlap = max(0, overlap_high - overlap_low)

            overlap_ratio = overlap / current_range
            overlap_ratios.append(min(1.0, overlap_ratio))

        self.df["overlap_ratio"] = overlap_ratios
        return pd.Series(overlap_ratios, index=df.index)

    def compute_ema_slope(self, period: int | None = None, lookback: int = 5) -> pd.Series:
        """
        Compute slope of EMA as trend indicator.

        Args:
            period: EMA period
            lookback: Bars to compute slope over

        Returns:
            Series of slope values (positive = uptrend, negative = downtrend)
        """
        period = period or settings.ema_period
        ema = self.add_ema(period)

        # Compute slope as percentage change
        slope = (ema - ema.shift(lookback)) / ema.shift(lookback)
        self.df["ema_slope"] = slope
        return slope

    def closes_above_ema_pct(self, period: int | None = None, window: int = 20) -> pd.Series:
        """
        Percentage of closes above EMA in rolling window.

        Args:
            period: EMA period
            window: Rolling window size

        Returns:
            Series of percentages (0-1)
        """
        period = period or settings.ema_period
        ema = self.add_ema(period)

        above_ema = (self.df["close"] > ema).astype(int)
        pct_above = above_ema.rolling(window=window).mean()

        self.df["closes_above_ema_pct"] = pct_above
        return pct_above

    def get_bar_characteristics(self, index: int) -> BarCharacteristics:
        """Get characteristics for a specific bar."""
        row = self.df.iloc[index]

        return BarCharacteristics(
            is_bull=row.get("is_bull", row["close"] > row["open"]),
            is_bear=row.get("is_bear", row["close"] < row["open"]),
            is_doji=row.get("is_doji", False),
            is_trend_bar=row.get("is_trend_bar", False),
            body_size=row.get("body", abs(row["close"] - row["open"])),
            upper_wick=row.get("upper_wick", 0),
            lower_wick=row.get("lower_wick", 0),
            range_size=row.get("range", row["high"] - row["low"]),
            close_location=row.get("close_location", 0.5),
        )

    def get_recent_strength(self, window: int = 10) -> dict:
        """
        Analyze recent price action strength.

        Returns:
            Dictionary with strength metrics
        """
        if len(self.df) < window:
            return {"strength": "unknown", "confidence": "low"}

        recent = self.df.tail(window)

        # Count bull vs bear bars
        bull_count = recent["is_bull"].sum()
        bear_count = recent["is_bear"].sum()

        # Check trend bar presence
        if "is_trend_bar" not in self.df.columns:
            self.detect_trend_bars()
        trend_bars = recent["is_trend_bar"].sum()

        # Check closes relative to EMA
        if "ema_20" not in self.df.columns:
            self.add_ema(20)
        ema = self.df["ema_20"].iloc[-1]
        closes_above = (recent["close"] > ema).sum()

        # Determine strength
        if bull_count >= window * 0.7 and closes_above >= window * 0.8:
            strength = "strong_bull"
            confidence = "high" if trend_bars >= 2 else "medium"
        elif bear_count >= window * 0.7 and closes_above <= window * 0.2:
            strength = "strong_bear"
            confidence = "high" if trend_bars >= 2 else "medium"
        elif bull_count > bear_count and closes_above > window * 0.5:
            strength = "weak_bull"
            confidence = "medium"
        elif bear_count > bull_count and closes_above < window * 0.5:
            strength = "weak_bear"
            confidence = "medium"
        else:
            strength = "neutral"
            confidence = "medium"

        return {
            "strength": strength,
            "confidence": confidence,
            "bull_bars": int(bull_count),
            "bear_bars": int(bear_count),
            "trend_bars": int(trend_bars),
            "closes_above_ema": int(closes_above),
        }

    def get_dataframe(self) -> pd.DataFrame:
        """Get the DataFrame with all computed features."""
        return self.df


def compute_r_multiple(
    entry_price: float,
    exit_price: float,
    stop_price: float,
    direction: Literal["long", "short"],
) -> float:
    """
    Compute R-multiple for a trade.

    R = (Exit - Entry) / (Entry - Stop) for longs
    R = (Entry - Exit) / (Stop - Entry) for shorts

    Args:
        entry_price: Entry price
        exit_price: Exit price
        stop_price: Initial stop loss price
        direction: Trade direction

    Returns:
        R-multiple (positive = profit, negative = loss)
    """
    if direction == "long":
        risk = entry_price - stop_price
        if risk <= 0:
            return 0.0
        reward = exit_price - entry_price
    else:  # short
        risk = stop_price - entry_price
        if risk <= 0:
            return 0.0
        reward = entry_price - exit_price

    return reward / risk


def compute_mae_mfe(
    entry_price: float,
    high_during_trade: float,
    low_during_trade: float,
    direction: Literal["long", "short"],
    stop_price: float,
) -> tuple[float, float]:
    """
    Compute Maximum Adverse Excursion and Maximum Favorable Excursion.

    Args:
        entry_price: Entry price
        high_during_trade: Highest price during trade
        low_during_trade: Lowest price during trade
        direction: Trade direction
        stop_price: Initial stop price (for R calculation)

    Returns:
        Tuple of (MAE in R, MFE in R)
    """
    if direction == "long":
        risk = entry_price - stop_price
        if risk <= 0:
            return (0.0, 0.0)
        mae = (entry_price - low_during_trade) / risk  # How far against
        mfe = (high_during_trade - entry_price) / risk  # How far in favor
    else:
        risk = stop_price - entry_price
        if risk <= 0:
            return (0.0, 0.0)
        mae = (high_during_trade - entry_price) / risk
        mfe = (entry_price - low_during_trade) / risk

    return (mae, mfe)
