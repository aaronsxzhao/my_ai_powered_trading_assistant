"""
Trade analytics for Brooks Trading Coach.

Computes:
- R-multiple statistics
- Expectancy
- Win rate
- Per-strategy performance
- Edge discovery
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
import logging

import numpy as np

from app.journal.models import (
    Trade,
    Strategy,
    DailySummary,
    TradeOutcome,
    get_session,
)

logger = logging.getLogger(__name__)


@dataclass
class StrategyStats:
    """Statistics for a single strategy."""

    strategy_name: str
    category: str
    trade_count: int
    win_count: int
    loss_count: int
    win_rate: float
    avg_r: float
    expectancy: float
    profit_factor: float
    avg_winner_r: float
    avg_loser_r: float
    avg_mae: float
    avg_mfe: float
    best_time_of_day: Optional[str]
    recent_20_performance: float  # Avg R of last 20 trades


@dataclass
class EdgeAnalysis:
    """Analysis of trading edge."""

    strengths: list[str]
    weaknesses: list[str]
    coaching_focus: list[str]
    stop_doing: list[str]
    double_down: list[str]


@dataclass
class PortfolioStats:
    """Overall portfolio statistics."""

    # Basic counts
    total_trades: int
    win_count: int
    loss_count: int
    breakeven_count: int
    win_rate: float

    # P&L metrics
    total_r: float
    total_pnl_dollars: float
    avg_r: float
    avg_pnl_dollars: float

    # Winner/Loser analysis
    avg_winner_r: float
    avg_loser_r: float
    avg_winner_dollars: float
    avg_loser_dollars: float
    largest_winner_r: float
    largest_loser_r: float
    largest_winner_dollars: float
    largest_loser_dollars: float

    # Risk metrics
    expectancy: float
    profit_factor: float
    max_drawdown_r: float
    max_drawdown_dollars: float
    max_drawdown_pct: float
    current_drawdown_r: float

    # Streak analysis
    current_streak: int  # Positive for wins, negative for losses
    max_win_streak: int
    max_loss_streak: int

    # Time analysis
    avg_trade_duration_minutes: float
    best_time_of_day: Optional[str]
    best_day_of_week: Optional[str]

    # Recent performance
    recent_20_avg_r: float

    # MAE/MFE
    avg_mae: float
    avg_mfe: float


class TradeAnalytics:
    """
    Compute analytics and statistics from trade journal.
    """

    def __init__(self):
        """Initialize analytics."""
        pass

    def get_all_trades(
        self,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        user_id: Optional[int] = None,
    ) -> list[Trade]:
        """Get all trades within date range, optionally filtered by user."""
        session = get_session()
        try:
            query = session.query(Trade)

            # Filter by user if provided
            if user_id is not None:
                query = query.filter(Trade.user_id == user_id)

            if start_date:
                query = query.filter(Trade.trade_date >= start_date)
            if end_date:
                query = query.filter(Trade.trade_date <= end_date)

            return query.order_by(Trade.trade_date.desc()).all()
        finally:
            session.close()

    def compute_expectancy(self, trades: list[Trade]) -> float:
        """
        Compute expectancy from a list of trades.

        Expectancy = (Win% × Avg Win) - (Loss% × Avg Loss)
        Returns expectancy in R

        Args:
            trades: List of Trade objects

        Returns:
            Expectancy in R
        """
        if not trades:
            return 0.0

        winners = [t for t in trades if t.r_multiple and t.r_multiple > 0]
        losers = [t for t in trades if t.r_multiple and t.r_multiple < 0]

        if not winners and not losers:
            return 0.0

        total = len(winners) + len(losers)
        win_rate = len(winners) / total if total > 0 else 0
        loss_rate = len(losers) / total if total > 0 else 0

        avg_winner = np.mean([t.r_multiple for t in winners]) if winners else 0
        avg_loser = abs(np.mean([t.r_multiple for t in losers])) if losers else 0

        expectancy = (win_rate * avg_winner) - (loss_rate * avg_loser)
        return round(expectancy, 3)

    def compute_profit_factor(self, trades: list[Trade]) -> float:
        """
        Compute profit factor.

        Profit Factor = Gross Profit / Gross Loss

        Args:
            trades: List of Trade objects

        Returns:
            Profit factor (> 1 is profitable)
        """
        if not trades:
            return 0.0

        gross_profit = sum(t.r_multiple for t in trades if t.r_multiple and t.r_multiple > 0)
        gross_loss = abs(sum(t.r_multiple for t in trades if t.r_multiple and t.r_multiple < 0))

        if gross_loss == 0:
            return float("inf") if gross_profit > 0 else 0.0

        return round(gross_profit / gross_loss, 2)

    def compute_strategy_stats(
        self, strategy_name: str, user_id: Optional[int] = None
    ) -> Optional[StrategyStats]:
        """
        Compute statistics for a specific strategy.

        Args:
            strategy_name: Strategy name
            user_id: Optional user ID to filter trades by

        Returns:
            StrategyStats object or None if no trades
        """
        session = get_session()
        try:
            query = session.query(Trade).join(Strategy).filter(Strategy.name == strategy_name)

            # Filter by user if provided
            if user_id is not None:
                query = query.filter(Trade.user_id == user_id)

            trades = query.all()

            if not trades:
                return None

            strategy = session.query(Strategy).filter(Strategy.name == strategy_name).first()

            return self._compute_stats_for_trades(
                trades, strategy_name, strategy.category if strategy else "unknown"
            )

        finally:
            session.close()

    def _compute_stats_for_trades(
        self,
        trades: list[Trade],
        strategy_name: str,
        category: str,
    ) -> StrategyStats:
        """Compute stats for a list of trades."""
        winners = [t for t in trades if t.r_multiple and t.r_multiple > 0]
        losers = [t for t in trades if t.r_multiple and t.r_multiple < 0]

        trade_count = len(trades)
        win_count = len(winners)
        loss_count = len(losers)

        win_rate = win_count / trade_count if trade_count > 0 else 0

        # R statistics
        r_multiples = [t.r_multiple for t in trades if t.r_multiple is not None]
        avg_r = np.mean(r_multiples) if r_multiples else 0
        avg_winner_r = np.mean([t.r_multiple for t in winners]) if winners else 0
        avg_loser_r = np.mean([t.r_multiple for t in losers]) if losers else 0

        # MAE/MFE
        maes = [t.mae for t in trades if t.mae is not None]
        mfes = [t.mfe for t in trades if t.mfe is not None]
        avg_mae = np.mean(maes) if maes else 0
        avg_mfe = np.mean(mfes) if mfes else 0

        # Best time of day
        best_time = self._find_best_time_of_day(trades)

        # Recent 20 performance
        recent_trades = sorted(trades, key=lambda t: t.trade_date, reverse=True)[:20]
        recent_r = [t.r_multiple for t in recent_trades if t.r_multiple is not None]
        recent_20_perf = np.mean(recent_r) if recent_r else 0

        return StrategyStats(
            strategy_name=strategy_name,
            category=category,
            trade_count=trade_count,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=round(win_rate, 3),
            avg_r=round(avg_r, 3),
            expectancy=self.compute_expectancy(trades),
            profit_factor=self.compute_profit_factor(trades),
            avg_winner_r=round(avg_winner_r, 3),
            avg_loser_r=round(avg_loser_r, 3),
            avg_mae=round(avg_mae, 3),
            avg_mfe=round(avg_mfe, 3),
            best_time_of_day=best_time,
            recent_20_performance=round(recent_20_perf, 3),
        )

    def _find_best_time_of_day(self, trades: list[Trade]) -> Optional[str]:
        """Find best performing time of day."""
        time_buckets = {
            "open_30min": (
                datetime.strptime("09:30", "%H:%M").time(),
                datetime.strptime("10:00", "%H:%M").time(),
            ),
            "morning": (
                datetime.strptime("10:00", "%H:%M").time(),
                datetime.strptime("12:00", "%H:%M").time(),
            ),
            "midday": (
                datetime.strptime("12:00", "%H:%M").time(),
                datetime.strptime("14:00", "%H:%M").time(),
            ),
            "afternoon": (
                datetime.strptime("14:00", "%H:%M").time(),
                datetime.strptime("15:30", "%H:%M").time(),
            ),
            "close_30min": (
                datetime.strptime("15:30", "%H:%M").time(),
                datetime.strptime("16:00", "%H:%M").time(),
            ),
        }

        bucket_stats = {}
        for bucket_name, (start, end) in time_buckets.items():
            bucket_trades = [
                t for t in trades if t.entry_time and start <= t.entry_time.time() < end
            ]
            if bucket_trades:
                r_vals = [t.r_multiple for t in bucket_trades if t.r_multiple]
                if r_vals:
                    bucket_stats[bucket_name] = np.mean(r_vals)

        if bucket_stats:
            best = max(bucket_stats, key=bucket_stats.get)
            return best if bucket_stats[best] > 0 else None

        return None

    def get_all_strategy_stats(self, user_id: Optional[int] = None) -> list[StrategyStats]:
        """Get statistics for all strategies with trades, optionally filtered by user."""
        session = get_session()
        try:
            # Build query - optionally filter by user_id
            query = session.query(Strategy).join(Trade)
            if user_id is not None:
                query = query.filter(Trade.user_id == user_id)

            strategies = query.distinct().all()

            stats = []
            for strat in strategies:
                s = self.compute_strategy_stats(strat.name, user_id=user_id)
                if s:
                    stats.append(s)

            return sorted(stats, key=lambda x: x.expectancy, reverse=True)

        finally:
            session.close()

    def get_portfolio_stats(self, user_id: Optional[int] = None) -> Optional[PortfolioStats]:
        """
        Calculate overall portfolio statistics including drawdown.

        Args:
            user_id: Optional user ID to filter trades by

        Returns:
            PortfolioStats object or None if no trades
        """
        trades = self.get_all_trades(user_id=user_id)

        if not trades:
            return None

        # Sort trades by date for sequential analysis
        trades_sorted = sorted(
            trades, key=lambda t: (t.trade_date, t.exit_time or t.entry_time or t.trade_date)
        )

        # Basic counts
        total_trades = len(trades)
        winners = [t for t in trades if t.outcome == TradeOutcome.WIN]
        losers = [t for t in trades if t.outcome == TradeOutcome.LOSS]
        breakevens = [t for t in trades if t.outcome == TradeOutcome.BREAKEVEN]

        win_count = len(winners)
        loss_count = len(losers)
        breakeven_count = len(breakevens)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # R-multiple metrics
        r_values = [t.r_multiple for t in trades if t.r_multiple is not None]
        total_r = sum(r_values) if r_values else 0
        avg_r = np.mean(r_values) if r_values else 0

        winner_r = [t.r_multiple for t in winners if t.r_multiple]
        loser_r = [t.r_multiple for t in losers if t.r_multiple]
        avg_winner_r = np.mean(winner_r) if winner_r else 0
        avg_loser_r = np.mean(loser_r) if loser_r else 0
        largest_winner_r = max(winner_r) if winner_r else 0
        largest_loser_r = min(loser_r) if loser_r else 0

        # Dollar metrics
        pnl_values = [t.pnl_dollars for t in trades if t.pnl_dollars is not None]
        total_pnl = sum(pnl_values) if pnl_values else 0
        avg_pnl = np.mean(pnl_values) if pnl_values else 0

        winner_pnl = [t.pnl_dollars for t in winners if t.pnl_dollars]
        loser_pnl = [t.pnl_dollars for t in losers if t.pnl_dollars]
        avg_winner_dollars = np.mean(winner_pnl) if winner_pnl else 0
        avg_loser_dollars = np.mean(loser_pnl) if loser_pnl else 0
        largest_winner_dollars = max(winner_pnl) if winner_pnl else 0
        largest_loser_dollars = min(loser_pnl) if loser_pnl else 0

        # Calculate drawdown (both R and dollars)
        cumulative_r = 0
        peak_r = 0
        max_dd_r = 0

        cumulative_pnl = 0
        peak_pnl = 0
        max_dd_pnl = 0

        for trade in trades_sorted:
            if trade.r_multiple:
                cumulative_r += trade.r_multiple
                if cumulative_r > peak_r:
                    peak_r = cumulative_r
                dd_r = peak_r - cumulative_r
                if dd_r > max_dd_r:
                    max_dd_r = dd_r

            if trade.pnl_dollars:
                cumulative_pnl += trade.pnl_dollars
                if cumulative_pnl > peak_pnl:
                    peak_pnl = cumulative_pnl
                dd_pnl = peak_pnl - cumulative_pnl
                if dd_pnl > max_dd_pnl:
                    max_dd_pnl = dd_pnl

        current_dd_r = peak_r - cumulative_r
        max_dd_pct = (max_dd_pnl / peak_pnl * 100) if peak_pnl > 0 else 0

        # Streak analysis
        current_streak = 0
        max_win_streak = 0
        max_loss_streak = 0
        win_streak = 0
        loss_streak = 0

        for trade in trades_sorted:
            if trade.outcome == TradeOutcome.WIN:
                win_streak += 1
                loss_streak = 0
                max_win_streak = max(max_win_streak, win_streak)
            elif trade.outcome == TradeOutcome.LOSS:
                loss_streak += 1
                win_streak = 0
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:  # Breakeven doesn't break streak
                pass

        # Current streak
        for trade in reversed(trades_sorted):
            if trade.outcome == TradeOutcome.WIN:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    break
            elif trade.outcome == TradeOutcome.LOSS:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    break
            else:
                break

        # Duration analysis
        durations = []
        for trade in trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 60
                if duration > 0:
                    durations.append(duration)
        avg_duration = np.mean(durations) if durations else 0

        # Time of day analysis
        best_time = self._find_best_time_of_day(trades)

        # Day of week analysis
        day_performance = {}
        for trade in trades:
            if trade.trade_date and trade.r_multiple:
                day = trade.trade_date.strftime("%A")
                if day not in day_performance:
                    day_performance[day] = []
                day_performance[day].append(trade.r_multiple)

        best_day = None
        if day_performance:
            day_avg = {day: np.mean(rs) for day, rs in day_performance.items() if len(rs) >= 3}
            if day_avg:
                best_day = max(day_avg.keys(), key=lambda d: day_avg[d])

        # Recent 20 performance
        recent_trades = trades_sorted[-20:] if len(trades_sorted) >= 20 else trades_sorted
        recent_r = [t.r_multiple for t in recent_trades if t.r_multiple]
        recent_20_avg = np.mean(recent_r) if recent_r else 0

        # MAE/MFE
        maes = [t.mae for t in trades if t.mae is not None]
        mfes = [t.mfe for t in trades if t.mfe is not None]
        avg_mae = np.mean(maes) if maes else 0
        avg_mfe = np.mean(mfes) if mfes else 0

        return PortfolioStats(
            total_trades=total_trades,
            win_count=win_count,
            loss_count=loss_count,
            breakeven_count=breakeven_count,
            win_rate=round(win_rate, 3),
            total_r=round(total_r, 2),
            total_pnl_dollars=round(total_pnl, 2),
            avg_r=round(avg_r, 3),
            avg_pnl_dollars=round(avg_pnl, 2),
            avg_winner_r=round(avg_winner_r, 3),
            avg_loser_r=round(avg_loser_r, 3),
            avg_winner_dollars=round(avg_winner_dollars, 2),
            avg_loser_dollars=round(avg_loser_dollars, 2),
            largest_winner_r=round(largest_winner_r, 2),
            largest_loser_r=round(largest_loser_r, 2),
            largest_winner_dollars=round(largest_winner_dollars, 2),
            largest_loser_dollars=round(largest_loser_dollars, 2),
            expectancy=self.compute_expectancy(trades),
            profit_factor=self.compute_profit_factor(trades),
            max_drawdown_r=round(max_dd_r, 2),
            max_drawdown_dollars=round(max_dd_pnl, 2),
            max_drawdown_pct=round(max_dd_pct, 1),
            current_drawdown_r=round(current_dd_r, 2),
            current_streak=current_streak,
            max_win_streak=max_win_streak,
            max_loss_streak=max_loss_streak,
            avg_trade_duration_minutes=round(avg_duration, 1),
            best_time_of_day=best_time,
            best_day_of_week=best_day,
            recent_20_avg_r=round(recent_20_avg, 3),
            avg_mae=round(avg_mae, 3),
            avg_mfe=round(avg_mfe, 3),
        )

    def analyze_edge(self) -> EdgeAnalysis:
        """
        Analyze trading edge and identify strengths/weaknesses.

        Returns:
            EdgeAnalysis with coaching recommendations
        """
        all_stats = self.get_all_strategy_stats()

        if not all_stats:
            return EdgeAnalysis(
                strengths=["No trading data yet"],
                weaknesses=[],
                coaching_focus=["Start logging trades to discover your edge"],
                stop_doing=[],
                double_down=[],
            )

        # Find strengths (positive expectancy, good win rate)
        strengths = []
        weaknesses = []
        stop_doing = []
        double_down = []

        for stat in all_stats:
            if stat.trade_count < 5:
                continue  # Not enough data

            if stat.expectancy > 0.3 and stat.win_rate > 0.5:
                strengths.append(
                    f"Strong at {stat.strategy_name}: {stat.win_rate:.0%} win rate, "
                    f"+{stat.expectancy:.2f}R expectancy"
                )
                double_down.append(f"Look for more {stat.strategy_name} setups")

            elif stat.expectancy < -0.2:
                weaknesses.append(
                    f"Losing money on {stat.strategy_name}: {stat.expectancy:.2f}R expectancy"
                )
                stop_doing.append(f"Stop trading {stat.strategy_name} until reviewed")

            # Check for poor risk management
            if stat.avg_loser_r < -1.5:
                weaknesses.append(
                    f"Losers too big in {stat.strategy_name}: avg loser is {stat.avg_loser_r:.2f}R"
                )
                stop_doing.append(f"Tighten stops on {stat.strategy_name}")

            # Check for leaving money on table
            if stat.avg_mfe > stat.avg_winner_r * 1.5:
                weaknesses.append(
                    f"Leaving money on table in {stat.strategy_name}: MFE {stat.avg_mfe:.2f}R but avg win only {stat.avg_winner_r:.2f}R"
                )

        # Overall metrics
        all_trades = self.get_all_trades()
        if all_trades:
            overall_exp = self.compute_expectancy(all_trades)
            if overall_exp > 0:
                strengths.insert(0, f"Overall edge is positive: {overall_exp:.2f}R per trade")
            else:
                weaknesses.insert(
                    0, f"Overall expectancy is negative: {overall_exp:.2f}R per trade"
                )

        coaching_focus = []
        if stop_doing:
            coaching_focus.append(f"Focus on eliminating: {stop_doing[0]}")
        if double_down:
            coaching_focus.append(f"Increase: {double_down[0]}")

        return EdgeAnalysis(
            strengths=strengths[:3],
            weaknesses=weaknesses[:3],
            coaching_focus=coaching_focus[:2],
            stop_doing=stop_doing[:2],
            double_down=double_down[:2],
        )

    def generate_daily_summary(self, summary_date: date) -> DailySummary:
        """
        Generate or update daily summary for a date.

        Args:
            summary_date: Date to summarize

        Returns:
            DailySummary object
        """
        session = get_session()
        try:
            trades = session.query(Trade).filter(Trade.trade_date == summary_date).all()

            # Check for existing summary
            summary = (
                session.query(DailySummary)
                .filter(DailySummary.summary_date == summary_date)
                .first()
            )

            if not summary:
                summary = DailySummary(summary_date=summary_date)
                session.add(summary)

            # Compute stats
            summary.total_trades = len(trades)
            summary.winning_trades = len([t for t in trades if t.outcome == TradeOutcome.WIN])
            summary.losing_trades = len([t for t in trades if t.outcome == TradeOutcome.LOSS])
            summary.breakeven_trades = len(
                [t for t in trades if t.outcome == TradeOutcome.BREAKEVEN]
            )

            # R and PnL
            r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]
            pnl_vals = [t.pnl_dollars for t in trades if t.pnl_dollars is not None]

            summary.total_r = round(sum(r_vals), 2) if r_vals else 0
            summary.total_pnl = round(sum(pnl_vals), 2) if pnl_vals else 0

            if summary.total_trades > 0:
                summary.win_rate = round(summary.winning_trades / summary.total_trades, 3)
            else:
                summary.win_rate = 0

            summary.profit_factor = self.compute_profit_factor(trades)

            # Winner/loser stats
            winners = [t.r_multiple for t in trades if t.r_multiple and t.r_multiple > 0]
            losers = [t.r_multiple for t in trades if t.r_multiple and t.r_multiple < 0]

            summary.avg_winner_r = round(np.mean(winners), 2) if winners else 0
            summary.avg_loser_r = round(np.mean(losers), 2) if losers else 0
            summary.largest_winner_r = round(max(winners), 2) if winners else 0
            summary.largest_loser_r = round(min(losers), 2) if losers else 0

            # Best/worst trades
            if trades:
                sorted_by_r = sorted(
                    [t for t in trades if t.r_multiple is not None],
                    key=lambda x: x.r_multiple,
                    reverse=True,
                )
                if sorted_by_r:
                    summary.best_trade_id = sorted_by_r[0].id
                    summary.worst_trade_id = sorted_by_r[-1].id

            # Consecutive losses
            consecutive = 0
            max_consecutive = 0
            for t in sorted(trades, key=lambda x: x.entry_time or x.created_at):
                if t.outcome == TradeOutcome.LOSS:
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            summary.consecutive_losses = max_consecutive

            session.commit()
            session.refresh(summary)
            return summary

        finally:
            session.close()

    def get_weekly_stats(self, year: int, week: int) -> dict:
        """
        Get statistics for a specific week.

        Args:
            year: Year (e.g., 2024)
            week: Week number (1-52)

        Returns:
            Dictionary with weekly statistics
        """
        # Calculate date range for the week
        first_day = datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w").date()
        last_day = first_day + timedelta(days=6)

        trades = self.get_all_trades(start_date=first_day, end_date=last_day)

        if not trades:
            return {
                "year": year,
                "week": week,
                "start_date": first_day,
                "end_date": last_day,
                "total_trades": 0,
                "message": "No trades this week",
            }

        return {
            "year": year,
            "week": week,
            "start_date": first_day,
            "end_date": last_day,
            "total_trades": len(trades),
            "winning_trades": len([t for t in trades if t.outcome == TradeOutcome.WIN]),
            "losing_trades": len([t for t in trades if t.outcome == TradeOutcome.LOSS]),
            "total_r": round(sum(t.r_multiple for t in trades if t.r_multiple) or 0, 2),
            "total_pnl": round(sum(t.pnl_dollars for t in trades if t.pnl_dollars) or 0, 2),
            "expectancy": self.compute_expectancy(trades),
            "profit_factor": self.compute_profit_factor(trades),
            "strategy_stats": self._get_strategy_breakdown(trades),
        }

    def _get_strategy_breakdown(self, trades: list[Trade]) -> list[dict]:
        """Get per-strategy breakdown for a set of trades."""
        from collections import defaultdict

        by_strategy = defaultdict(list)
        for t in trades:
            strat_name = t.strategy.name if t.strategy else "unclassified"
            by_strategy[strat_name].append(t)

        breakdown = []
        for strat_name, strat_trades in by_strategy.items():
            r_vals = [t.r_multiple for t in strat_trades if t.r_multiple]
            breakdown.append(
                {
                    "strategy": strat_name,
                    "count": len(strat_trades),
                    "total_r": round(sum(r_vals), 2) if r_vals else 0,
                    "win_rate": round(
                        len([t for t in strat_trades if t.outcome == TradeOutcome.WIN])
                        / len(strat_trades),
                        2,
                    )
                    if strat_trades
                    else 0,
                }
            )

        return sorted(breakdown, key=lambda x: x["total_r"], reverse=True)

    def format_strategy_leaderboard(self) -> str:
        """Format strategy stats as a leaderboard table."""
        stats = self.get_all_strategy_stats()

        if not stats:
            return "No strategy data available yet."

        lines = [
            "| Strategy | Trades | Win% | Avg R | Expectancy | PF |",
            "|----------|--------|------|-------|------------|-----|",
        ]

        for s in stats[:10]:  # Top 10
            lines.append(
                f"| {s.strategy_name[:20]} | {s.trade_count} | {s.win_rate:.0%} | "
                f"{s.avg_r:+.2f} | {s.expectancy:+.2f} | {s.profit_factor:.1f} |"
            )

        return "\n".join(lines)
