"""Pytest configuration and fixtures."""

import pytest
import tempfile
import os
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment with temp directories."""
    # Create temp directories for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set environment variables for test database
        os.environ["DATABASE_URL"] = f"sqlite:///{tmpdir}/test_trades.db"
        os.environ["CACHE_DIR"] = f"{tmpdir}/cache"
        os.environ["OUTPUTS_DIR"] = f"{tmpdir}/outputs"

        # Create directories
        Path(f"{tmpdir}/cache").mkdir(exist_ok=True)
        Path(f"{tmpdir}/outputs").mkdir(exist_ok=True)

        yield

        # Cleanup happens automatically with tempfile


@pytest.fixture
def sample_ohlcv_df():
    """Create sample OHLCV DataFrame for testing."""
    import pandas as pd
    import numpy as np

    dates = pd.date_range("2024-01-01", periods=50, freq="D", tz="America/New_York")

    # Create uptrending data
    base = 100
    closes = [base + i * 0.3 + np.random.uniform(-0.1, 0.1) for i in range(50)]

    df = pd.DataFrame(
        {
            "datetime": dates,
            "open": [c - 0.2 for c in closes],
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000000 + i * 10000 for i in range(50)],
        }
    )

    return df


@pytest.fixture
def sample_trades():
    """Create sample trades for testing."""
    from datetime import date

    return [
        {
            "ticker": "SPY",
            "trade_date": date(2024, 1, 15),
            "direction": "long",
            "entry_price": 475.0,
            "exit_price": 478.0,
            "stop_price": 473.0,
            "size": 100,
            "r_multiple": 1.5,
        },
        {
            "ticker": "SPY",
            "trade_date": date(2024, 1, 15),
            "direction": "long",
            "entry_price": 476.0,
            "exit_price": 474.5,
            "stop_price": 474.0,
            "size": 50,
            "r_multiple": -0.75,
        },
        {
            "ticker": "AAPL",
            "trade_date": date(2024, 1, 15),
            "direction": "short",
            "entry_price": 185.0,
            "exit_price": 182.0,
            "stop_price": 187.0,
            "size": 100,
            "r_multiple": 1.5,
        },
    ]
