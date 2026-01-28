"""Tests for trade journal functionality."""

from datetime import date, datetime
import tempfile
import os


class TestTradeModel:
    """Tests for Trade model."""

    def test_trade_compute_metrics_long_winner(self):
        """Test metric computation for winning long trade."""
        from app.journal.models import Trade, TradeDirection, TradeOutcome

        trade = Trade(
            ticker="AAPL",
            trade_date=date.today(),
            direction=TradeDirection.LONG,
            entry_price=150.0,
            exit_price=155.0,
            stop_price=148.0,
            size=100,
        )

        trade.compute_metrics()

        # R-multiple: (155-150)/(150-148) = 5/2 = 2.5
        assert trade.r_multiple == 2.5
        # PnL: (155-150)*100 = 500
        assert trade.pnl_dollars == 500.0
        # PnL%: 5/150*100 = 3.33%
        assert abs(trade.pnl_percent - 3.333) < 0.01
        # Outcome
        assert trade.outcome == TradeOutcome.WIN

    def test_trade_compute_metrics_long_loser(self):
        """Test metric computation for losing long trade."""
        from app.journal.models import Trade, TradeDirection, TradeOutcome

        trade = Trade(
            ticker="AAPL",
            trade_date=date.today(),
            direction=TradeDirection.LONG,
            entry_price=150.0,
            exit_price=147.0,
            stop_price=148.0,
            size=100,
        )

        trade.compute_metrics()

        # R-multiple: (147-150)/(150-148) = -3/2 = -1.5
        assert trade.r_multiple == -1.5
        # PnL: (147-150)*100 = -300
        assert trade.pnl_dollars == -300.0
        # Outcome
        assert trade.outcome == TradeOutcome.LOSS

    def test_trade_compute_metrics_short_winner(self):
        """Test metric computation for winning short trade."""
        from app.journal.models import Trade, TradeDirection, TradeOutcome

        trade = Trade(
            ticker="SPY",
            trade_date=date.today(),
            direction=TradeDirection.SHORT,
            entry_price=400.0,
            exit_price=395.0,
            stop_price=402.0,
            size=50,
        )

        trade.compute_metrics()

        # R-multiple: (400-395)/(402-400) = 5/2 = 2.5
        assert trade.r_multiple == 2.5
        # PnL: (400-395)*50 = 250
        assert trade.pnl_dollars == 250.0
        # Outcome
        assert trade.outcome == TradeOutcome.WIN

    def test_trade_compute_metrics_breakeven(self):
        """Test metric computation for breakeven trade."""
        from app.journal.models import Trade, TradeDirection, TradeOutcome

        trade = Trade(
            ticker="SPY",
            trade_date=date.today(),
            direction=TradeDirection.LONG,
            entry_price=400.0,
            exit_price=400.0,
            stop_price=398.0,
            size=50,
        )

        trade.compute_metrics()

        assert trade.r_multiple == 0.0
        assert trade.pnl_dollars == 0.0
        assert trade.outcome == TradeOutcome.BREAKEVEN

    def test_trade_compute_mae_mfe(self):
        """Test MAE/MFE computation."""
        from app.journal.models import Trade, TradeDirection

        trade = Trade(
            ticker="AAPL",
            trade_date=date.today(),
            direction=TradeDirection.LONG,
            entry_price=100.0,
            exit_price=104.0,
            stop_price=98.0,  # 1R = $2
            size=100,
            high_during_trade=106.0,  # MFE = 3R
            low_during_trade=99.0,  # MAE = 0.5R
        )

        trade.compute_metrics()

        assert trade.mfe == 3.0
        assert trade.mae == 0.5

    def test_trade_compute_hold_time(self):
        """Test hold time computation."""
        from app.journal.models import Trade, TradeDirection

        trade = Trade(
            ticker="SPY",
            trade_date=date.today(),
            direction=TradeDirection.LONG,
            entry_price=400.0,
            exit_price=402.0,
            stop_price=398.0,
            entry_time=datetime(2024, 1, 15, 10, 0, 0),
            exit_time=datetime(2024, 1, 15, 11, 30, 0),
        )

        trade.compute_metrics()

        assert trade.hold_time_minutes == 90

    def test_trade_is_winner_property(self):
        """Test is_winner property."""
        from app.journal.models import Trade, TradeDirection

        winning_trade = Trade(
            ticker="SPY",
            trade_date=date.today(),
            direction=TradeDirection.LONG,
            entry_price=400.0,
            exit_price=405.0,
            stop_price=398.0,
        )
        winning_trade.compute_metrics()
        assert winning_trade.is_winner is True

        losing_trade = Trade(
            ticker="SPY",
            trade_date=date.today(),
            direction=TradeDirection.LONG,
            entry_price=400.0,
            exit_price=395.0,
            stop_price=398.0,
        )
        losing_trade.compute_metrics()
        assert losing_trade.is_winner is False

    def test_trade_initial_risk_dollars(self):
        """Test initial risk calculation."""
        from app.journal.models import Trade, TradeDirection

        trade = Trade(
            ticker="SPY",
            trade_date=date.today(),
            direction=TradeDirection.LONG,
            entry_price=400.0,
            exit_price=405.0,
            stop_price=398.0,
            size=100,
        )

        # Risk = |400-398| * 100 = $200
        assert trade.initial_risk_dollars == 200.0


class TestTradeIngester:
    """Tests for trade ingestion."""

    def test_add_trade_manual(self):
        """Test manual trade entry."""
        from app.journal.ingest import TradeIngester
        from app.journal.models import init_db

        # Use temp database

        # Initialize
        init_db()

        ingester = TradeIngester()

        trade = ingester.add_trade_manual(
            ticker="TEST",
            trade_date=date.today(),
            direction="long",
            entry_price=100.0,
            exit_price=102.0,
            stop_price=98.0,
            size=50,
        )

        assert trade.id is not None
        assert trade.ticker == "TEST"
        assert trade.r_multiple == 1.0  # (102-100)/(100-98)

    def test_csv_import(self):
        """Test CSV import."""
        import csv
        from app.journal.ingest import TradeIngester
        from app.journal.models import init_db

        init_db()

        # Create temp CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(
                ["ticker", "direction", "entry_price", "exit_price", "stop_price", "size"]
            )
            writer.writerow(["SPY", "long", "400", "405", "398", "100"])
            writer.writerow(["QQQ", "short", "350", "345", "352", "50"])
            csv_path = f.name

        try:
            ingester = TradeIngester()
            imported, errors, messages = ingester.import_csv(csv_path)

            assert imported == 2
            assert errors == 0
        finally:
            os.unlink(csv_path)


class TestDailySummary:
    """Tests for daily summary generation."""

    def test_daily_summary_generation(self):
        """Test daily summary is generated correctly."""
        from app.journal.models import DailySummary

        summary = DailySummary(
            summary_date=date.today(),
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            total_r=2.5,
            total_pnl=500.0,
        )

        assert summary.total_trades == 5
        assert summary.winning_trades == 3
        assert summary.total_r == 2.5
