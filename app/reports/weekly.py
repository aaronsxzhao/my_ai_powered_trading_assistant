"""
Weekly Report Generator.

Generates weekly trading summary with:
- Strategy leaderboard
- Biggest leaks
- Top rules for next week
- Performance charts/tables
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
import logging

from app.config import OUTPUTS_DIR
from app.journal.models import Trade, TradeOutcome, get_session
from app.journal.analytics import TradeAnalytics, StrategyStats, EdgeAnalysis

logger = logging.getLogger(__name__)


@dataclass
class WeeklyReport:
    """Weekly report data."""

    year: int
    week: int
    start_date: date
    end_date: date

    # Trade summary
    total_trades: int
    winning_trades: int
    losing_trades: int
    trading_days: int

    # Performance
    total_r: float
    total_pnl: float
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_r_per_trade: float
    avg_r_per_day: float

    # Strategy performance
    strategy_leaderboard: list[dict]
    best_strategy: Optional[str]
    worst_strategy: Optional[str]

    # Edge analysis
    strengths: list[str]
    weaknesses: list[str]
    biggest_leaks: list[str]

    # Coaching
    top_rules: list[str]
    stop_doing: list[str]
    double_down: list[str]

    # Daily breakdown
    daily_breakdown: list[dict]


class WeeklyReport:
    """
    Generate weekly trading summary.
    """

    def __init__(self):
        """Initialize weekly report generator."""
        self.analytics = TradeAnalytics()

    def generate_report(self, year: int, week: int) -> 'WeeklyReportData':
        """
        Generate weekly report.

        Args:
            year: Year (e.g., 2024)
            week: ISO week number (1-52)

        Returns:
            WeeklyReportData object
        """
        logger.info(f"Generating weekly report for {year}-W{week:02d}")

        # Calculate week date range
        start_date = datetime.strptime(f"{year}-W{week:02d}-1", "%Y-W%W-%w").date()
        end_date = start_date + timedelta(days=6)

        session = get_session()
        try:
            # Get all trades for the week
            trades = (
                session.query(Trade)
                .filter(Trade.trade_date >= start_date)
                .filter(Trade.trade_date <= end_date)
                .order_by(Trade.trade_date, Trade.entry_time)
                .all()
            )

            if not trades:
                return self._empty_report(year, week, start_date, end_date)

            # Compute metrics
            total_trades = len(trades)
            winning = [t for t in trades if t.outcome == TradeOutcome.WIN]
            losing = [t for t in trades if t.outcome == TradeOutcome.LOSS]

            # Unique trading days
            trading_days = len(set(t.trade_date for t in trades))

            r_values = [t.r_multiple for t in trades if t.r_multiple is not None]
            pnl_values = [t.pnl_dollars for t in trades if t.pnl_dollars is not None]

            total_r = sum(r_values) if r_values else 0
            total_pnl = sum(pnl_values) if pnl_values else 0

            win_rate = len(winning) / total_trades if total_trades > 0 else 0
            profit_factor = self.analytics.compute_profit_factor(trades)
            expectancy = self.analytics.compute_expectancy(trades)

            avg_r_per_trade = total_r / total_trades if total_trades > 0 else 0
            avg_r_per_day = total_r / trading_days if trading_days > 0 else 0

            # Strategy performance
            strategy_stats = self._compute_strategy_breakdown(trades)
            best_strategy = strategy_stats[0]["strategy"] if strategy_stats else None
            worst_strategy = strategy_stats[-1]["strategy"] if strategy_stats else None

            # Edge analysis
            edge = self.analytics.analyze_edge()

            # Biggest leaks
            leaks = self._identify_leaks(trades, strategy_stats)

            # Coaching recommendations
            top_rules = self._generate_top_rules(trades, leaks)
            stop_doing = edge.stop_doing
            double_down = edge.double_down

            # Daily breakdown
            daily_breakdown = self._compute_daily_breakdown(trades, start_date, end_date)

            return WeeklyReportData(
                year=year,
                week=week,
                start_date=start_date,
                end_date=end_date,
                total_trades=total_trades,
                winning_trades=len(winning),
                losing_trades=len(losing),
                trading_days=trading_days,
                total_r=round(total_r, 2),
                total_pnl=round(total_pnl, 2),
                win_rate=round(win_rate, 3),
                profit_factor=profit_factor,
                expectancy=expectancy,
                avg_r_per_trade=round(avg_r_per_trade, 3),
                avg_r_per_day=round(avg_r_per_day, 2),
                strategy_leaderboard=strategy_stats,
                best_strategy=best_strategy,
                worst_strategy=worst_strategy,
                strengths=edge.strengths,
                weaknesses=edge.weaknesses,
                biggest_leaks=leaks,
                top_rules=top_rules,
                stop_doing=stop_doing,
                double_down=double_down,
                daily_breakdown=daily_breakdown,
            )

        finally:
            session.close()

    def _empty_report(
        self,
        year: int,
        week: int,
        start_date: date,
        end_date: date,
    ) -> 'WeeklyReportData':
        """Return empty report for weeks with no trades."""
        return WeeklyReportData(
            year=year,
            week=week,
            start_date=start_date,
            end_date=end_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            trading_days=0,
            total_r=0.0,
            total_pnl=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            avg_r_per_trade=0.0,
            avg_r_per_day=0.0,
            strategy_leaderboard=[],
            best_strategy=None,
            worst_strategy=None,
            strengths=["No trading data this week"],
            weaknesses=[],
            biggest_leaks=[],
            top_rules=["Start trading and logging to discover patterns"],
            stop_doing=[],
            double_down=[],
            daily_breakdown=[],
        )

    def _compute_strategy_breakdown(self, trades: list[Trade]) -> list[dict]:
        """Compute per-strategy breakdown."""
        from collections import defaultdict

        by_strategy: dict[str, list[Trade]] = defaultdict(list)
        for t in trades:
            strat_name = t.strategy.name if t.strategy else "unclassified"
            by_strategy[strat_name].append(t)

        breakdown = []
        for strat_name, strat_trades in by_strategy.items():
            r_vals = [t.r_multiple for t in strat_trades if t.r_multiple is not None]
            if not r_vals:
                continue

            winners = [t for t in strat_trades if t.outcome == TradeOutcome.WIN]

            breakdown.append({
                "strategy": strat_name,
                "count": len(strat_trades),
                "total_r": round(sum(r_vals), 2),
                "avg_r": round(sum(r_vals) / len(r_vals), 3),
                "win_rate": round(len(winners) / len(strat_trades), 3),
                "pf": self.analytics.compute_profit_factor(strat_trades),
            })

        return sorted(breakdown, key=lambda x: x["total_r"], reverse=True)

    def _identify_leaks(
        self,
        trades: list[Trade],
        strategy_stats: list[dict],
    ) -> list[str]:
        """Identify biggest performance leaks."""
        leaks = []

        # Leaking strategies
        for stat in strategy_stats:
            if stat["total_r"] < -1:
                leaks.append(
                    f"Strategy '{stat['strategy']}' lost {abs(stat['total_r']):.1f}R "
                    f"({stat['count']} trades)"
                )

        # Large losers
        big_losers = [t for t in trades if t.r_multiple and t.r_multiple < -1.5]
        if big_losers:
            total_big_loss = sum(t.r_multiple for t in big_losers)
            leaks.append(
                f"{len(big_losers)} trades with losses > 1.5R "
                f"(total: {total_big_loss:.1f}R) - tighten stops"
            )

        # Low win rate with negative expectancy
        winners = [t for t in trades if t.outcome == TradeOutcome.WIN]
        win_rate = len(winners) / len(trades) if trades else 0
        if win_rate < 0.4:
            leaks.append(
                f"Win rate only {win_rate:.0%} - be more selective or "
                "improve entry timing"
            )

        # Average winner smaller than average loser
        winner_r = [t.r_multiple for t in trades if t.r_multiple and t.r_multiple > 0]
        loser_r = [t.r_multiple for t in trades if t.r_multiple and t.r_multiple < 0]

        if winner_r and loser_r:
            avg_win = sum(winner_r) / len(winner_r)
            avg_loss = abs(sum(loser_r) / len(loser_r))

            if avg_win < avg_loss:
                leaks.append(
                    f"Average winner ({avg_win:.2f}R) smaller than average loser "
                    f"({avg_loss:.2f}R) - let winners run"
                )

        return leaks[:5]

    def _generate_top_rules(
        self,
        trades: list[Trade],
        leaks: list[str],
    ) -> list[str]:
        """Generate top 3 rules for next week."""
        rules = []

        # Based on leaks
        if any("tighten stops" in leak.lower() for leak in leaks):
            rules.append("RULE 1: Exit at initial stop - no exceptions")

        if any("let winners run" in leak.lower() for leak in leaks):
            rules.append("RULE 2: Trail stops instead of fixed targets in trends")

        if any("selective" in leak.lower() for leak in leaks):
            rules.append("RULE 3: Only take A+ setups - skip marginal trades")

        # Default rules if not enough
        if len(rules) < 3:
            default_rules = [
                "Write entry reason before every trade",
                "Define stop and target before entry",
                "Review all trades at end of day",
            ]
            for r in default_rules:
                if len(rules) >= 3:
                    break
                if r not in rules:
                    rules.append(f"RULE {len(rules) + 1}: {r}")

        return rules[:3]

    def _compute_daily_breakdown(
        self,
        trades: list[Trade],
        start_date: date,
        end_date: date,
    ) -> list[dict]:
        """Compute day-by-day performance."""
        from collections import defaultdict

        by_date: dict[date, list[Trade]] = defaultdict(list)
        for t in trades:
            by_date[t.trade_date].append(t)

        breakdown = []
        current = start_date
        while current <= end_date:
            day_trades = by_date.get(current, [])

            if day_trades:
                r_vals = [t.r_multiple for t in day_trades if t.r_multiple]
                total_r = sum(r_vals) if r_vals else 0
                winners = len([t for t in day_trades if t.outcome == TradeOutcome.WIN])

                breakdown.append({
                    "date": current.strftime("%Y-%m-%d"),
                    "day": current.strftime("%A"),
                    "trades": len(day_trades),
                    "winners": winners,
                    "total_r": round(total_r, 2),
                })
            else:
                breakdown.append({
                    "date": current.strftime("%Y-%m-%d"),
                    "day": current.strftime("%A"),
                    "trades": 0,
                    "winners": 0,
                    "total_r": 0,
                })

            current += timedelta(days=1)

        return breakdown

    def format_report(self, report: 'WeeklyReportData') -> str:
        """Format weekly report as markdown."""
        emoji = "🟢" if report.total_r >= 0 else "🔴"

        lines = [
            f"# Weekly Report: {report.year}-W{report.week:02d}",
            f"**Period**: {report.start_date} to {report.end_date}",
            "",
            f"## Performance Summary {emoji}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Trading Days | {report.trading_days} |",
            f"| Total Trades | {report.total_trades} |",
            f"| Winners / Losers | {report.winning_trades} / {report.losing_trades} |",
            f"| Win Rate | {report.win_rate:.1%} |",
            f"| **Total R** | **{report.total_r:+.2f}R** |",
            f"| Total PnL | ${report.total_pnl:+,.2f} |",
            f"| Expectancy | {report.expectancy:+.3f}R |",
            f"| Profit Factor | {report.profit_factor:.2f} |",
            f"| Avg R per Trade | {report.avg_r_per_trade:+.3f}R |",
            f"| Avg R per Day | {report.avg_r_per_day:+.2f}R |",
            "",
            "---",
            "",
            "## Strategy Leaderboard 📊",
            "",
            "| Rank | Strategy | Trades | Total R | Win% | PF |",
            "|------|----------|--------|---------|------|-----|",
        ]

        for i, s in enumerate(report.strategy_leaderboard[:10], 1):
            lines.append(
                f"| {i} | {s['strategy']} | {s['count']} | "
                f"{s['total_r']:+.2f}R | {s['win_rate']:.0%} | {s['pf']:.1f} |"
            )

        lines.extend([
            "",
            f"🏆 **Best Strategy**: {report.best_strategy}" if report.best_strategy else "",
            f"⚠️ **Worst Strategy**: {report.worst_strategy}" if report.worst_strategy else "",
            "",
            "---",
            "",
            "## Daily Breakdown",
            "",
            "| Date | Day | Trades | Winners | Total R |",
            "|------|-----|--------|---------|---------|",
        ])

        for d in report.daily_breakdown:
            if d["trades"] > 0:
                emoji = "🟢" if d["total_r"] >= 0 else "🔴"
                lines.append(
                    f"| {d['date']} | {d['day']} | {d['trades']} | "
                    f"{d['winners']} | {d['total_r']:+.2f}R {emoji} |"
                )
            else:
                lines.append(
                    f"| {d['date']} | {d['day']} | - | - | - |"
                )

        lines.extend([
            "",
            "---",
            "",
            "## Edge Analysis",
            "",
            "### Strengths 💪",
        ])
        for s in report.strengths:
            lines.append(f"- {s}")

        lines.extend([
            "",
            "### Weaknesses 📉",
        ])
        for w in report.weaknesses:
            lines.append(f"- {w}")

        if report.biggest_leaks:
            lines.extend([
                "",
                "### Biggest Leaks 🕳️",
            ])
            for leak in report.biggest_leaks:
                lines.append(f"- ❌ {leak}")

        lines.extend([
            "",
            "---",
            "",
            "## Coaching for Next Week",
            "",
            "### Top 3 Rules 📌",
        ])
        for rule in report.top_rules:
            lines.append(f"- {rule}")

        if report.stop_doing:
            lines.extend([
                "",
                "### Stop Doing 🛑",
            ])
            for item in report.stop_doing:
                lines.append(f"- {item}")

        if report.double_down:
            lines.extend([
                "",
                "### Double Down On ✅",
            ])
            for item in report.double_down:
                lines.append(f"- {item}")

        lines.append("")
        lines.append("---")
        lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def save_report(self, report: 'WeeklyReportData') -> Path:
        """
        Save weekly report to disk.

        Args:
            report: WeeklyReportData object

        Returns:
            Path to saved file
        """
        output_dir = OUTPUTS_DIR / f"{report.year}-W{report.week:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)

        content = self.format_report(report)
        file_path = output_dir / "weekly_report.md"

        with open(file_path, "w") as f:
            f.write(content)

        logger.info(f"Saved weekly report: {file_path}")

        # Save strategy stats as CSV
        if report.strategy_leaderboard:
            import pandas as pd
            stats_df = pd.DataFrame(report.strategy_leaderboard)
            csv_path = output_dir / "strategy_stats.csv"
            stats_df.to_csv(csv_path, index=False)

        return file_path


@dataclass
class WeeklyReportData:
    """Weekly report data."""

    year: int
    week: int
    start_date: date
    end_date: date

    # Trade summary
    total_trades: int
    winning_trades: int
    losing_trades: int
    trading_days: int

    # Performance
    total_r: float
    total_pnl: float
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_r_per_trade: float
    avg_r_per_day: float

    # Strategy performance
    strategy_leaderboard: list[dict]
    best_strategy: Optional[str]
    worst_strategy: Optional[str]

    # Edge analysis
    strengths: list[str]
    weaknesses: list[str]
    biggest_leaks: list[str]

    # Coaching
    top_rules: list[str]
    stop_doing: list[str]
    double_down: list[str]

    # Daily breakdown
    daily_breakdown: list[dict]
