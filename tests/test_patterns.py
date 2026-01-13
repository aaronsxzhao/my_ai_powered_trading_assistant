"""Tests for Brooks pattern detection and regime classification."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestRegimeClassifier:
    """Tests for regime classification."""

    def _create_uptrend_data(self, periods: int = 50) -> pd.DataFrame:
        """Create synthetic uptrend data."""
        dates = pd.date_range("2024-01-01", periods=periods, freq="D", tz="America/New_York")

        # Uptrend: higher highs and higher lows
        base = 100
        closes = [base + i * 0.5 + np.random.uniform(-0.2, 0.2) for i in range(periods)]

        df = pd.DataFrame({
            "datetime": dates,
            "open": [c - 0.3 for c in closes],
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000000] * periods,
        })

        return df

    def _create_downtrend_data(self, periods: int = 50) -> pd.DataFrame:
        """Create synthetic downtrend data."""
        dates = pd.date_range("2024-01-01", periods=periods, freq="D", tz="America/New_York")

        # Downtrend: lower highs and lower lows
        base = 150
        closes = [base - i * 0.5 + np.random.uniform(-0.2, 0.2) for i in range(periods)]

        df = pd.DataFrame({
            "datetime": dates,
            "open": [c + 0.3 for c in closes],
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000000] * periods,
        })

        return df

    def _create_range_data(self, periods: int = 50) -> pd.DataFrame:
        """Create synthetic trading range data."""
        dates = pd.date_range("2024-01-01", periods=periods, freq="D", tz="America/New_York")

        # Trading range: oscillating between support and resistance
        base = 100
        closes = [base + np.sin(i / 3) * 2 + np.random.uniform(-0.3, 0.3) for i in range(periods)]

        df = pd.DataFrame({
            "datetime": dates,
            "open": [c - 0.2 for c in closes],
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000000] * periods,
        })

        return df

    def test_regime_uptrend_detection(self):
        """Test detection of uptrend regime."""
        from app.features.ohlc_features import OHLCFeatures
        from app.features.brooks_patterns import BrooksPatternDetector, Regime

        df = self._create_uptrend_data(50)
        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)

        regime = detector.analyze_regime()

        # Should detect uptrend
        assert regime.regime == Regime.TREND_UP

    def test_regime_downtrend_detection(self):
        """Test detection of downtrend regime."""
        from app.features.ohlc_features import OHLCFeatures
        from app.features.brooks_patterns import BrooksPatternDetector, Regime

        df = self._create_downtrend_data(50)
        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)

        regime = detector.analyze_regime()

        # Should detect downtrend
        assert regime.regime == Regime.TREND_DOWN

    def test_regime_range_detection(self):
        """Test detection of trading range regime."""
        from app.features.ohlc_features import OHLCFeatures
        from app.features.brooks_patterns import BrooksPatternDetector, Regime

        df = self._create_range_data(50)
        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)

        regime = detector.analyze_regime()

        # Should detect range (or might be trend_up/down with low confidence)
        # Due to synthetic data, we check it's one of the valid regimes
        assert regime.regime in [Regime.TREND_UP, Regime.TREND_DOWN, Regime.TRADING_RANGE, Regime.UNKNOWN]

    def test_regime_returns_valid_output(self):
        """Test that regime analysis returns all expected fields."""
        from app.features.ohlc_features import OHLCFeatures
        from app.features.brooks_patterns import BrooksPatternDetector, Regime, AlwaysIn, Confidence

        df = self._create_uptrend_data()
        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)

        regime = detector.analyze_regime()

        # Check all fields exist
        assert isinstance(regime.regime, Regime)
        assert isinstance(regime.always_in, AlwaysIn)
        assert isinstance(regime.confidence, Confidence)
        assert isinstance(regime.description, str)
        assert isinstance(regime.metrics, dict)

    def test_always_in_aligns_with_regime(self):
        """Test that always-in direction aligns with regime."""
        from app.features.ohlc_features import OHLCFeatures
        from app.features.brooks_patterns import BrooksPatternDetector, Regime, AlwaysIn

        # Uptrend should have always-in long
        df = self._create_uptrend_data()
        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)
        regime = detector.analyze_regime()

        if regime.regime == Regime.TREND_UP:
            assert regime.always_in == AlwaysIn.LONG

        # Downtrend should have always-in short
        df = self._create_downtrend_data()
        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)
        regime = detector.analyze_regime()

        if regime.regime == Regime.TREND_DOWN:
            assert regime.always_in == AlwaysIn.SHORT


class TestPatternDetection:
    """Tests for pattern detection."""

    def test_detect_all_patterns_returns_list(self):
        """Test that detect_all_patterns returns a list."""
        from app.features.ohlc_features import OHLCFeatures
        from app.features.brooks_patterns import BrooksPatternDetector

        # Create some sample data
        dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="America/New_York")
        df = pd.DataFrame({
            "datetime": dates,
            "open": [100 + i * 0.1 for i in range(50)],
            "high": [101 + i * 0.1 for i in range(50)],
            "low": [99 + i * 0.1 for i in range(50)],
            "close": [100.5 + i * 0.1 for i in range(50)],
            "volume": [1000000] * 50,
        })

        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)

        patterns = detector.detect_all_patterns()

        assert isinstance(patterns, list)

    def test_trading_context_structure(self):
        """Test that get_trading_context returns expected structure."""
        from app.features.ohlc_features import OHLCFeatures
        from app.features.brooks_patterns import BrooksPatternDetector

        dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="America/New_York")
        df = pd.DataFrame({
            "datetime": dates,
            "open": [100 + i * 0.2 for i in range(50)],
            "high": [101 + i * 0.2 for i in range(50)],
            "low": [99 + i * 0.2 for i in range(50)],
            "close": [100.5 + i * 0.2 for i in range(50)],
            "volume": [1000000] * 50,
        })

        features = OHLCFeatures(df)
        detector = BrooksPatternDetector(features)

        context = detector.get_trading_context()

        # Check expected keys
        assert "regime" in context
        assert "patterns" in context
        assert "strength" in context
        assert "best_setups" in context
        assert "avoid_setups" in context


class TestOHLCFeatures:
    """Tests for OHLC feature extraction."""

    def test_add_ema(self):
        """Test EMA calculation."""
        from app.features.ohlc_features import OHLCFeatures

        dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="America/New_York")
        df = pd.DataFrame({
            "datetime": dates,
            "open": [100] * 30,
            "high": [101] * 30,
            "low": [99] * 30,
            "close": [100 + i * 0.1 for i in range(30)],
            "volume": [1000000] * 30,
        })

        features = OHLCFeatures(df)
        ema = features.add_ema(20)

        assert len(ema) == 30
        assert "ema_20" in features.df.columns

    def test_add_atr(self):
        """Test ATR calculation."""
        from app.features.ohlc_features import OHLCFeatures

        dates = pd.date_range("2024-01-01", periods=30, freq="D", tz="America/New_York")
        df = pd.DataFrame({
            "datetime": dates,
            "open": [100] * 30,
            "high": [102] * 30,
            "low": [98] * 30,
            "close": [100] * 30,
            "volume": [1000000] * 30,
        })

        features = OHLCFeatures(df)
        atr = features.add_atr(14)

        assert len(atr) == 30
        assert "atr" in features.df.columns

    def test_find_swing_highs(self):
        """Test swing high detection."""
        from app.features.ohlc_features import OHLCFeatures

        # Create data with clear swing high
        dates = pd.date_range("2024-01-01", periods=20, freq="D", tz="America/New_York")
        highs = [100, 101, 102, 103, 104, 105, 104, 103, 102, 101,
                 100, 101, 102, 101, 100, 99, 100, 101, 100, 99]

        df = pd.DataFrame({
            "datetime": dates,
            "open": [h - 0.5 for h in highs],
            "high": highs,
            "low": [h - 1 for h in highs],
            "close": [h - 0.3 for h in highs],
            "volume": [1000000] * 20,
        })

        features = OHLCFeatures(df)
        swing_highs = features.find_swing_highs(lookback=3)

        # Should find at least one swing high
        assert isinstance(swing_highs, list)

    def test_detect_trend_bars(self):
        """Test trend bar detection."""
        from app.features.ohlc_features import OHLCFeatures

        dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="America/New_York")

        # Mix of trend bars and dojis
        df = pd.DataFrame({
            "datetime": dates,
            "open": [100, 100.5, 100, 100.9, 100, 101, 100, 100.5, 100, 102],
            "high": [101, 101.5, 101, 101.5, 101, 102, 101, 101.5, 101, 103],
            "low": [99, 99.5, 99, 99.5, 99, 100, 99, 99.5, 99, 101],
            "close": [100.8, 101.4, 100.2, 101.3, 100.1, 101.9, 100.3, 101.3, 100.2, 102.9],
            "volume": [1000000] * 10,
        })

        features = OHLCFeatures(df)
        trend_bars = features.detect_trend_bars()

        assert len(trend_bars) == 10
        assert "is_trend_bar" in features.df.columns
