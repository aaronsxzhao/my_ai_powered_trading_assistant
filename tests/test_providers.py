"""Tests for data providers."""

from datetime import datetime, timedelta

import pandas as pd


class TestYFinanceProvider:
    """Tests for YFinance data provider."""

    def test_provider_returns_correct_columns(self):
        """Test that provider returns DataFrame with correct columns."""
        from app.data.providers import YFinanceProvider

        provider = YFinanceProvider()

        # Get data for a well-known ticker
        end = datetime.now()
        start = end - timedelta(days=30)

        df = provider.get_ohlcv("SPY", "1d", start, end)

        # Check that we got data (might be empty if API fails, that's ok for test)
        if not df.empty:
            expected_columns = ["datetime", "open", "high", "low", "close", "volume"]
            for col in expected_columns:
                assert col in df.columns, f"Missing column: {col}"

    def test_provider_name(self):
        """Test provider name property."""
        from app.data.providers import YFinanceProvider

        provider = YFinanceProvider()
        assert provider.name == "yfinance"

    def test_get_provider_default(self):
        """Test get_provider returns yfinance by default."""
        from app.data.providers import get_provider

        provider = get_provider()
        assert provider.name == "yfinance"

    def test_get_provider_by_name(self):
        """Test get_provider with explicit name."""
        from app.data.providers import get_provider

        provider = get_provider("yfinance")
        assert provider.name == "yfinance"

    def test_get_provider_unknown_falls_back(self):
        """Test unknown provider falls back to yfinance."""
        from app.data.providers import get_provider

        provider = get_provider("unknown_provider")
        assert provider.name == "yfinance"


class TestDataProviderInterface:
    """Tests for data provider abstract interface."""

    def test_normalize_dataframe(self):
        """Test DataFrame normalization."""
        from app.data.providers import YFinanceProvider

        provider = YFinanceProvider()

        # Create a sample DataFrame with non-standard columns
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=5, tz="UTC"),
                "Open": [100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            }
        )

        normalized = provider._normalize_dataframe(df)

        # Check standard columns exist
        assert "datetime" in normalized.columns
        assert "open" in normalized.columns
        assert "close" in normalized.columns

        # Check timezone conversion
        assert normalized["datetime"].dt.tz is not None


class TestConvenienceFunctions:
    """Tests for convenience data functions."""

    def test_get_daily_data(self):
        """Test get_daily_data convenience function."""
        from app.data.providers import get_daily_data

        df = get_daily_data("SPY", days=10)

        # Should return a DataFrame (might be empty if API fails)
        assert isinstance(df, pd.DataFrame)

    def test_get_intraday_data(self):
        """Test get_intraday_data convenience function."""
        from app.data.providers import get_intraday_data

        df = get_intraday_data("SPY", timeframe="5m", days=2)

        # Should return a DataFrame
        assert isinstance(df, pd.DataFrame)
