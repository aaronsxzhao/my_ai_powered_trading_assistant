"""
End-of-Day Report Generator.

Generates daily trading summary with:
- Total PnL and R
- Best/worst trades
- Rule violations
- Improvement focus
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional
import logging

from app.config import settings, OUTPUTS_DIR
from app.journal.models import Trade, TradeOutcome, get_session
from app.journal.analytics import TradeAnalytics
from app.journal.coach import TradeCoach

logger = logging.getLogger(__name__)


@dataclass
class EODReport:
    """End-of-day report data."""

    report_date: date

    # Trade summary
    total_trades: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    # Performance
    total_r: float
    total_pnl: float
    win_rate: float
    profit_factor: float
    avg_r: float

    # Best/worst
    best_trade: Optional[dict]
    worst_trade: Optional[dict]

    # Risk metrics
    max_drawdown_r: float
    consecutive_losses: int
    daily_loss_limit_hit: bool

    # Rule violations
    rule_violations: list[str]

    # Coaching
    top_mistake: str
    improvement_focus: str

    # All trades
    trades: list[dict]


class EndOfDayReport:
    """
    Generate end-of-day trading summary.
    """

    def __init__(self):
        """Initialize EOD report generator."""
        self.analytics = TradeAnalytics()
        self.coach = TradeCoach()

    def generate_report(self, report_date: Optional[date] = None) -> EODReport:
        """
        Generate end-of-day report.

        Args:
            report_date: Date for report (defaults to today)

        Returns:
            EODReport object
        """
        report_date = report_date or date.today()

        logger.info(f"Generating EOD report for {report_date}")

        session = get_session()
        try:
            # Get all trades for the day
            trades = (
                session.query(Trade)
                .filter(Trade.trade_date == report_date)
                .order_by(Trade.entry_time)
                .all()
            )

            if not trades:
                return self._empty_report(report_date)

            # Compute metrics
            total_trades = len(trades)
            winning = [t for t in trades if t.outcome == TradeOutcome.WIN]
            losing = [t for t in trades if t.outcome == TradeOutcome.LOSS]
            breakeven = [t for t in trades if t.outcome == TradeOutcome.BREAKEVEN]

            r_values = [t.r_multiple for t in trades if t.r_multiple is not None]
            pnl_values = [t.pnl_dollars for t in trades if t.pnl_dollars is not None]

            total_r = sum(r_values) if r_values else 0
            total_pnl = sum(pnl_values) if pnl_values else 0
            avg_r = total_r / len(r_values) if r_values else 0

            win_rate = len(winning) / total_trades if total_trades > 0 else 0
            profit_factor = self.analytics.compute_profit_factor(trades)

            # Best/worst trades
            best_trade = None
            worst_trade = None

            if r_values:
                sorted_trades = sorted(
                    [t for t in trades if t.r_multiple is not None],
                    key=lambda x: x.r_multiple,
                    reverse=True,
                )
                if sorted_trades:
                    best = sorted_trades[0]
                    best_trade = {
                        "id": best.id,
                        "ticker": best.ticker,
                        "r_multiple": best.r_multiple,
                        "strategy": best.strategy.name if best.strategy else "unclassified",
                    }
                    worst = sorted_trades[-1]
                    worst_trade = {
                        "id": worst.id,
                        "ticker": worst.ticker,
                        "r_multiple": worst.r_multiple,
                        "strategy": worst.strategy.name if worst.strategy else "unclassified",
                    }

            # Risk metrics
            max_drawdown = self._compute_max_drawdown(trades)
            consecutive_losses = self._compute_consecutive_losses(trades)
            daily_loss_limit_hit = total_r < -settings.max_daily_loss_r

            # Get trade reviews once (parallel) and share across analysis functions
            trade_reviews = self._get_trade_reviews_parallel(trades)

            # Rule violations (use cached reviews)
            rule_violations = self._detect_rule_violations(trades, reviews=trade_reviews)

            # Coaching (use cached reviews)
            top_mistake = self._identify_top_mistake(trades, reviews=trade_reviews)
            improvement_focus = self._generate_improvement_focus(trades, rule_violations)

            # Format trades for output
            trades_data = [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "direction": t.direction.value,
                    "r_multiple": t.r_multiple,
                    "pnl": t.pnl_dollars,
                    "strategy": t.strategy.name if t.strategy else "unclassified",
                    "outcome": t.outcome.value if t.outcome else "unknown",
                }
                for t in trades
            ]

            # Update daily summary in database
            self._update_daily_summary(report_date, trades)

            return EODReport(
                report_date=report_date,
                total_trades=total_trades,
                winning_trades=len(winning),
                losing_trades=len(losing),
                breakeven_trades=len(breakeven),
                total_r=round(total_r, 2),
                total_pnl=round(total_pnl, 2),
                win_rate=round(win_rate, 3),
                profit_factor=profit_factor,
                avg_r=round(avg_r, 3),
                best_trade=best_trade,
                worst_trade=worst_trade,
                max_drawdown_r=round(max_drawdown, 2),
                consecutive_losses=consecutive_losses,
                daily_loss_limit_hit=daily_loss_limit_hit,
                rule_violations=rule_violations,
                top_mistake=top_mistake,
                improvement_focus=improvement_focus,
                trades=trades_data,
            )

        finally:
            session.close()

    def _empty_report(self, report_date: date) -> EODReport:
        """Return empty report for days with no trades."""
        return EODReport(
            report_date=report_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            breakeven_trades=0,
            total_r=0.0,
            total_pnl=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_r=0.0,
            best_trade=None,
            worst_trade=None,
            max_drawdown_r=0.0,
            consecutive_losses=0,
            daily_loss_limit_hit=False,
            rule_violations=[],
            top_mistake="No trades today",
            improvement_focus="Review watchlist for tomorrow",
            trades=[],
        )

    def _compute_max_drawdown(self, trades: list[Trade]) -> float:
        """Compute maximum drawdown in R during the day."""
        if not trades:
            return 0.0

        sorted_trades = sorted(trades, key=lambda t: t.entry_time or t.created_at)
        cumulative = 0.0
        peak = 0.0
        max_dd = 0.0

        for trade in sorted_trades:
            r = trade.r_multiple or 0
            cumulative += r
            peak = max(peak, cumulative)
            drawdown = peak - cumulative
            max_dd = max(max_dd, drawdown)

        return max_dd

    def _compute_consecutive_losses(self, trades: list[Trade]) -> int:
        """Compute maximum consecutive losses."""
        sorted_trades = sorted(trades, key=lambda t: t.entry_time or t.created_at)

        max_streak = 0
        current_streak = 0

        for trade in sorted_trades:
            if trade.outcome == TradeOutcome.LOSS:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    def _get_trade_reviews_parallel(self, trades: list[Trade]) -> dict:
        """Get reviews for all trades in parallel. Returns dict of trade_id -> review."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        reviews = {}
        max_workers = min(4, len(trades))  # Limit concurrency

        if not trades:
            return reviews

        logger.info(f"📊 Reviewing {len(trades)} trades in parallel ({max_workers} workers)")

        def review_trade_safe(trade_id):
            try:
                return trade_id, self.coach.review_trade(trade_id)
            except Exception as e:
                logger.warning(f"Failed to review trade {trade_id}: {e}")
                return trade_id, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(review_trade_safe, t.id) for t in trades]

            for future in as_completed(futures):
                try:
                    trade_id, review = future.result()
                    reviews[trade_id] = review
                except Exception as e:
                    logger.warning(f"Error getting review result: {e}")

        return reviews

    def _detect_rule_violations(self, trades: list[Trade], reviews: dict = None) -> list[str]:
        """Detect rule violations in trades."""
        violations = []

        # Use provided reviews or fetch them
        if reviews is None:
            reviews = self._get_trade_reviews_parallel(trades)

        for trade in trades:
            review = reviews.get(trade.id)
            if review and review.errors_detected:
                for error in review.errors_detected:
                    key = error.split(":")[0]
                    violations.append(f"{trade.ticker}: {key}")

        # Deduplicate
        return list(set(violations))[:5]

    def _identify_top_mistake(self, trades: list[Trade], reviews: dict = None) -> str:
        """Identify the biggest mistake of the day."""
        mistake_counts: dict[str, int] = {}

        # Use provided reviews or fetch them
        if reviews is None:
            reviews = self._get_trade_reviews_parallel(trades)

        for trade in trades:
            review = reviews.get(trade.id)
            if review and review.errors_detected:
                for error in review.errors_detected:
                    key = error.split(":")[0]
                    mistake_counts[key] = mistake_counts.get(key, 0) + 1

        if mistake_counts:
            top = max(mistake_counts, key=mistake_counts.get)
            return f"{top} ({mistake_counts[top]} occurrences)"

        # Check for losing trades without clear errors
        losers = [t for t in trades if t.outcome == TradeOutcome.LOSS]
        if losers:
            return "Losing trades - review setups and context alignment"

        return "No major mistakes identified"

    def _generate_improvement_focus(
        self,
        trades: list[Trade],
        violations: list[str],
    ) -> str:
        """Generate improvement focus for tomorrow."""
        if not trades:
            return "Start trading and logging to discover patterns"

        # Check for pattern in violations
        if violations:
            if any("COUNTERTREND" in v for v in violations):
                return "FOCUS: Only take countertrend trades with clear reversal structure"
            if any("SCALP" in v for v in violations):
                return "FOCUS: Increase target distances or improve win rate"
            if any("STOP" in v for v in violations):
                return "FOCUS: Use limit entries in ranges, avoid stop entries"

        # Check win rate
        winners = [t for t in trades if t.outcome == TradeOutcome.WIN]
        win_rate = len(winners) / len(trades) if trades else 0

        if win_rate < 0.4:
            return "FOCUS: Be more selective - only take A+ setups tomorrow"
        elif win_rate > 0.6:
            return "FOCUS: Great win rate - can you let winners run longer?"

        return "FOCUS: Review losing trades tonight, document lessons learned"

    def _update_daily_summary(self, report_date: date, trades: list[Trade]) -> None:
        """Update daily summary in database."""
        try:
            self.analytics.generate_daily_summary(report_date)
        except Exception as e:
            logger.warning(f"Failed to update daily summary: {e}")

    def format_report(self, report: EODReport) -> str:
        """Format EOD report as markdown."""
        emoji = "🟢" if report.total_r >= 0 else "🔴"

        lines = [
            f"# End of Day Report: {report.report_date}",
            "",
            f"## Performance Summary {emoji}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Trades | {report.total_trades} |",
            f"| Winners | {report.winning_trades} |",
            f"| Losers | {report.losing_trades} |",
            f"| Win Rate | {report.win_rate:.1%} |",
            f"| **Total R** | **{report.total_r:+.2f}R** |",
            f"| Total PnL | ${report.total_pnl:+,.2f} |",
            f"| Profit Factor | {report.profit_factor:.2f} |",
            f"| Max Drawdown | {report.max_drawdown_r:.2f}R |",
            "",
        ]

        if report.best_trade:
            lines.extend(
                [
                    "## Best Trade 🏆",
                    f"- **{report.best_trade['ticker']}**: +{report.best_trade['r_multiple']:.2f}R",
                    f"- Strategy: {report.best_trade['strategy']}",
                    "",
                ]
            )

        if report.worst_trade:
            lines.extend(
                [
                    "## Worst Trade ⚠️",
                    f"- **{report.worst_trade['ticker']}**: {report.worst_trade['r_multiple']:.2f}R",
                    f"- Strategy: {report.worst_trade['strategy']}",
                    "",
                ]
            )

        if report.daily_loss_limit_hit:
            lines.extend(
                [
                    "## ⛔ DAILY LOSS LIMIT HIT",
                    f"Lost more than {settings.max_daily_loss_r}R today. Stop trading.",
                    "",
                ]
            )

        if report.consecutive_losses >= settings.get("risk.warn_after_consecutive_losses", 2):
            lines.extend(
                [
                    f"## ⚠️ Consecutive Losses: {report.consecutive_losses}",
                    "Consider taking a break or reducing size.",
                    "",
                ]
            )

        if report.rule_violations:
            lines.extend(
                [
                    "## Rule Violations",
                ]
            )
            for v in report.rule_violations:
                lines.append(f"- ❌ {v}")
            lines.append("")

        lines.extend(
            [
                "## Top Mistake",
                f"📌 {report.top_mistake}",
                "",
                "## Tomorrow's Focus",
                f"🎯 {report.improvement_focus}",
                "",
                "---",
                "",
                "## All Trades",
                "",
                "| # | Ticker | Direction | R | PnL | Strategy | Outcome |",
                "|---|--------|-----------|---|-----|----------|---------|",
            ]
        )

        for i, t in enumerate(report.trades, 1):
            r_str = f"{t['r_multiple']:+.2f}" if t["r_multiple"] else "N/A"
            pnl_str = f"${t['pnl']:+.0f}" if t["pnl"] else "N/A"
            lines.append(
                f"| {i} | {t['ticker']} | {t['direction']} | {r_str} | {pnl_str} | {t['strategy']} | {t['outcome']} |"
            )

        lines.append("")
        lines.append("---")
        lines.append(f"*Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def save_report(self, report: EODReport) -> Path:
        """
        Save EOD report to disk.

        Args:
            report: EODReport object

        Returns:
            Path to saved file
        """
        output_dir = OUTPUTS_DIR / report.report_date.strftime("%Y-%m-%d")
        output_dir.mkdir(parents=True, exist_ok=True)

        content = self.format_report(report)
        file_path = output_dir / "eod_report.md"

        with open(file_path, "w") as f:
            f.write(content)

        logger.info(f"Saved EOD report: {file_path}")

        # Also save trades as CSV
        if report.trades:
            import pandas as pd

            trades_df = pd.DataFrame(report.trades)
            csv_path = output_dir / "trades.csv"
            trades_df.to_csv(csv_path, index=False)

        return file_path
