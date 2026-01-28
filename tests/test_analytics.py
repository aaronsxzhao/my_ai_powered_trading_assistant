"""Tests for trade analytics and R-multiple calculations."""

from app.features.ohlc_features import compute_r_multiple, compute_mae_mfe


class TestRMultiple:
    """Tests for R-multiple calculation."""

    def test_r_multiple_long_winner(self):
        """Test R-multiple for a winning long trade."""
        # Entry $100, Stop $98 (risk = $2), Exit $104 (reward = $4)
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=104.0,
            stop_price=98.0,
            direction="long",
        )
        assert r == 2.0  # Made 2R

    def test_r_multiple_long_loser(self):
        """Test R-multiple for a losing long trade."""
        # Entry $100, Stop $98 (risk = $2), Exit $97 (loss = $3)
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=97.0,
            stop_price=98.0,
            direction="long",
        )
        assert r == -1.5  # Lost 1.5R

    def test_r_multiple_short_winner(self):
        """Test R-multiple for a winning short trade."""
        # Entry $100, Stop $102 (risk = $2), Exit $96 (reward = $4)
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=96.0,
            stop_price=102.0,
            direction="short",
        )
        assert r == 2.0  # Made 2R

    def test_r_multiple_short_loser(self):
        """Test R-multiple for a losing short trade."""
        # Entry $100, Stop $102 (risk = $2), Exit $105 (loss = $5)
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=105.0,
            stop_price=102.0,
            direction="short",
        )
        assert r == -2.5  # Lost 2.5R

    def test_r_multiple_breakeven(self):
        """Test R-multiple for a breakeven trade."""
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=100.0,
            stop_price=98.0,
            direction="long",
        )
        assert r == 0.0

    def test_r_multiple_zero_risk_returns_zero(self):
        """Test that zero risk returns 0."""
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=105.0,
            stop_price=100.0,  # Same as entry = 0 risk
            direction="long",
        )
        assert r == 0.0

    def test_r_multiple_invalid_stop_long(self):
        """Test long with stop above entry returns 0."""
        r = compute_r_multiple(
            entry_price=100.0,
            exit_price=105.0,
            stop_price=102.0,  # Invalid stop above entry for long
            direction="long",
        )
        assert r == 0.0


class TestMAEMFE:
    """Tests for MAE/MFE calculation."""

    def test_mae_mfe_long_trade(self):
        """Test MAE/MFE for a long trade."""
        # Entry $100, Stop $98 (1R = $2)
        # High during trade = $106 (MFE = 3R)
        # Low during trade = $99 (MAE = 0.5R)
        mae, mfe = compute_mae_mfe(
            entry_price=100.0,
            high_during_trade=106.0,
            low_during_trade=99.0,
            direction="long",
            stop_price=98.0,
        )
        assert mae == 0.5  # Went $1 against, risk was $2
        assert mfe == 3.0  # Went $6 in favor, risk was $2

    def test_mae_mfe_short_trade(self):
        """Test MAE/MFE for a short trade."""
        # Entry $100, Stop $102 (1R = $2)
        # High during trade = $101 (MAE = 0.5R)
        # Low during trade = $94 (MFE = 3R)
        mae, mfe = compute_mae_mfe(
            entry_price=100.0,
            high_during_trade=101.0,
            low_during_trade=94.0,
            direction="short",
            stop_price=102.0,
        )
        assert mae == 0.5  # Went $1 against
        assert mfe == 3.0  # Went $6 in favor


class TestExpectancy:
    """Tests for expectancy calculation."""

    def test_expectancy_positive(self):
        """Test expectancy with positive edge."""
        from app.journal.analytics import TradeAnalytics
        from app.journal.models import TradeOutcome

        analytics = TradeAnalytics()

        # Create mock trades
        class MockTrade:
            def __init__(self, r_multiple: float):
                self.r_multiple = r_multiple
                self.outcome = TradeOutcome.WIN if r_multiple > 0 else TradeOutcome.LOSS

        # 60% win rate, avg winner 1.5R, avg loser -1R
        # Expectancy = 0.6 * 1.5 - 0.4 * 1 = 0.9 - 0.4 = 0.5R
        trades = [
            MockTrade(1.5),  # Win
            MockTrade(1.5),  # Win
            MockTrade(1.5),  # Win
            MockTrade(-1.0),  # Loss
            MockTrade(-1.0),  # Loss
        ]

        # 3 wins, 2 losses
        # Win rate = 0.6
        # Avg winner = 1.5R
        # Avg loser = 1.0R
        # Expectancy = 0.6 * 1.5 - 0.4 * 1.0 = 0.9 - 0.4 = 0.5R

        expectancy = analytics.compute_expectancy(trades)
        assert abs(expectancy - 0.5) < 0.01

    def test_expectancy_negative(self):
        """Test expectancy with negative edge."""
        from app.journal.analytics import TradeAnalytics

        analytics = TradeAnalytics()

        class MockTrade:
            def __init__(self, r_multiple: float):
                self.r_multiple = r_multiple

        # 40% win rate, avg winner 1R, avg loser -1.5R
        # Expectancy = 0.4 * 1 - 0.6 * 1.5 = 0.4 - 0.9 = -0.5R
        trades = [
            MockTrade(1.0),  # Win
            MockTrade(1.0),  # Win
            MockTrade(-1.5),  # Loss
            MockTrade(-1.5),  # Loss
            MockTrade(-1.5),  # Loss
        ]

        expectancy = analytics.compute_expectancy(trades)
        assert expectancy < 0  # Negative expectancy

    def test_expectancy_empty_trades(self):
        """Test expectancy with no trades."""
        from app.journal.analytics import TradeAnalytics

        analytics = TradeAnalytics()
        expectancy = analytics.compute_expectancy([])
        assert expectancy == 0.0


class TestProfitFactor:
    """Tests for profit factor calculation."""

    def test_profit_factor_positive(self):
        """Test profit factor > 1."""
        from app.journal.analytics import TradeAnalytics

        analytics = TradeAnalytics()

        class MockTrade:
            def __init__(self, r_multiple: float):
                self.r_multiple = r_multiple

        # Gross profit = 4R, Gross loss = 2R, PF = 2.0
        trades = [
            MockTrade(2.0),  # Win
            MockTrade(2.0),  # Win
            MockTrade(-1.0),  # Loss
            MockTrade(-1.0),  # Loss
        ]

        pf = analytics.compute_profit_factor(trades)
        assert pf == 2.0

    def test_profit_factor_breakeven(self):
        """Test profit factor = 1."""
        from app.journal.analytics import TradeAnalytics

        analytics = TradeAnalytics()

        class MockTrade:
            def __init__(self, r_multiple: float):
                self.r_multiple = r_multiple

        # Gross profit = 2R, Gross loss = 2R, PF = 1.0
        trades = [
            MockTrade(1.0),  # Win
            MockTrade(1.0),  # Win
            MockTrade(-1.0),  # Loss
            MockTrade(-1.0),  # Loss
        ]

        pf = analytics.compute_profit_factor(trades)
        assert pf == 1.0

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses returns inf."""
        from app.journal.analytics import TradeAnalytics

        analytics = TradeAnalytics()

        class MockTrade:
            def __init__(self, r_multiple: float):
                self.r_multiple = r_multiple

        trades = [
            MockTrade(1.0),
            MockTrade(2.0),
        ]

        pf = analytics.compute_profit_factor(trades)
        assert pf == float("inf")
