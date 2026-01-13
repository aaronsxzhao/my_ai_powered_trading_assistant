"""
Brooks Trading Coach CLI Application.

Command-line interface for the trading coach system.
"""

from datetime import date, datetime
from pathlib import Path
from typing import Optional
import logging

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print as rprint

from app.config import settings, save_config, load_config
from app.journal.models import init_db

# Initialize CLI app
app = typer.Typer(
    name="brooks",
    help="Brooks Price Action Trading Coach - Advisory system for discretionary day traders",
    add_completion=False,
)

# Sub-command groups
trade_app = typer.Typer(help="Trade journal commands")
report_app = typer.Typer(help="Report generation commands")
stats_app = typer.Typer(help="Statistics and analytics commands")
config_app = typer.Typer(help="Configuration commands")

app.add_typer(trade_app, name="trade")
app.add_typer(report_app, name="report")
app.add_typer(stats_app, name="stats")
app.add_typer(config_app, name="config")

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def init():
    """Initialize database and settings."""
    init_db()


# ==================== TRADE COMMANDS ====================


@trade_app.command("add")
def trade_add(
    ticker: str = typer.Option(..., "--ticker", "-t", help="Stock symbol"),
    direction: str = typer.Option(..., "--direction", "-d", help="Trade direction (long/short)"),
    entry: float = typer.Option(..., "--entry", "-e", help="Entry price"),
    exit_price: float = typer.Option(..., "--exit", "-x", help="Exit price"),
    stop: float = typer.Option(..., "--stop", "-s", help="Stop loss price"),
    size: float = typer.Option(1.0, "--size", help="Position size"),
    trade_date: str = typer.Option(None, "--date", help="Trade date (YYYY-MM-DD)"),
    timeframe: str = typer.Option("5m", "--timeframe", "-tf", help="Timeframe"),
    strategy: str = typer.Option(None, "--strategy", help="Strategy name"),
    notes: str = typer.Option(None, "--notes", "-n", help="Trade notes"),
):
    """Add a new trade to the journal."""
    init()

    from app.journal.ingest import TradeIngester

    ingester = TradeIngester()

    # Parse date
    if trade_date:
        parsed_date = datetime.strptime(trade_date, "%Y-%m-%d").date()
    else:
        parsed_date = date.today()

    try:
        trade = ingester.add_trade_manual(
            ticker=ticker,
            trade_date=parsed_date,
            direction=direction,
            entry_price=entry,
            exit_price=exit_price,
            stop_price=stop,
            size=size,
            timeframe=timeframe,
            strategy_name=strategy,
            notes=notes,
        )

        # Display result
        r = trade.r_multiple or 0
        emoji = "🟢" if r >= 0 else "🔴"

        console.print(
            Panel(
                f"[bold]Trade Added[/bold]\n\n"
                f"ID: {trade.id}\n"
                f"Ticker: {trade.ticker}\n"
                f"Direction: {direction.upper()}\n"
                f"Entry: ${entry:.2f} → Exit: ${exit_price:.2f}\n"
                f"R-Multiple: {r:+.2f}R {emoji}\n"
                f"PnL: ${trade.pnl_dollars:+.2f}" if trade.pnl_dollars else "",
                title="✅ Trade Logged",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(f"[red]Error adding trade: {e}[/red]")
        raise typer.Exit(1)


@trade_app.command("import")
def trade_import(
    csv_path: Path = typer.Argument(..., help="Path to CSV file"),
    broker: str = typer.Option("generic", "--broker", "-b", help="Broker format"),
):
    """Import trades from CSV file."""
    init()

    from app.journal.ingest import TradeIngester

    if not csv_path.exists():
        console.print(f"[red]File not found: {csv_path}[/red]")
        raise typer.Exit(1)

    ingester = TradeIngester()

    try:
        imported, errors, messages = ingester.import_csv(csv_path, broker=broker)

        console.print(
            Panel(
                f"[bold]Import Complete[/bold]\n\n"
                f"✅ Imported: {imported}\n"
                f"❌ Errors: {errors}",
                title="CSV Import",
                border_style="blue",
            )
        )

        if messages:
            console.print("\n[yellow]Errors:[/yellow]")
            for msg in messages[:10]:
                console.print(f"  - {msg}")

    except Exception as e:
        console.print(f"[red]Import failed: {e}[/red]")
        raise typer.Exit(1)


@trade_app.command("review")
def trade_review(
    trade_id: int = typer.Argument(..., help="Trade ID to review"),
):
    """Review a trade with Brooks-style coaching."""
    init()

    from app.journal.coach import TradeCoach

    coach = TradeCoach()
    review = coach.review_trade(trade_id)

    if not review:
        console.print(f"[red]Trade {trade_id} not found[/red]")
        raise typer.Exit(1)

    # Format and display
    markdown = coach.format_review(review)
    console.print(Markdown(markdown))


@trade_app.command("list")
def trade_list(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of trades to show"),
    ticker: str = typer.Option(None, "--ticker", "-t", help="Filter by ticker"),
):
    """List recent trades."""
    init()

    from app.journal.ingest import TradeIngester

    ingester = TradeIngester()

    if ticker:
        trades = ingester.get_trades_by_ticker(ticker)[:limit]
    else:
        trades = ingester.get_recent_trades(limit)

    if not trades:
        console.print("[yellow]No trades found[/yellow]")
        return

    table = Table(title="Recent Trades")
    table.add_column("ID", style="dim")
    table.add_column("Date")
    table.add_column("Ticker", style="cyan")
    table.add_column("Dir")
    table.add_column("R", justify="right")
    table.add_column("PnL", justify="right")
    table.add_column("Strategy")

    for t in trades:
        r_str = f"{t.r_multiple:+.2f}" if t.r_multiple else "N/A"
        pnl_str = f"${t.pnl_dollars:+.0f}" if t.pnl_dollars else "N/A"
        strat = t.strategy.name if t.strategy else "-"

        r_style = "green" if t.r_multiple and t.r_multiple > 0 else "red"

        table.add_row(
            str(t.id),
            str(t.trade_date),
            t.ticker,
            t.direction.value[:1].upper(),
            f"[{r_style}]{r_str}[/{r_style}]",
            pnl_str,
            strat[:20],
        )

    console.print(table)


# ==================== REPORT COMMANDS ====================


@report_app.command("premarket")
def report_premarket(
    report_date: str = typer.Option(None, "--date", "-d", help="Report date (YYYY-MM-DD)"),
    ticker: str = typer.Option(None, "--ticker", "-t", help="Single ticker (or all favorites)"),
):
    """Generate premarket analysis report."""
    init()

    from app.reports.premarket import PremarketReport

    # Parse date
    if report_date:
        parsed_date = datetime.strptime(report_date, "%Y-%m-%d").date()
    else:
        parsed_date = date.today()

    generator = PremarketReport()

    with console.status("[bold green]Generating premarket report..."):
        if ticker:
            reports = [generator.generate_ticker_report(ticker, parsed_date)]
        else:
            reports = generator.generate_all_reports(parsed_date)

    if not reports:
        console.print("[yellow]No reports generated[/yellow]")
        return

    # Save reports
    output_dir = generator.save_reports(reports, parsed_date)

    # Display first report
    console.print(Markdown(generator.format_report(reports[0])))

    if len(reports) > 1:
        console.print(f"\n[dim]+ {len(reports) - 1} more reports saved to {output_dir}[/dim]")

    console.print(f"\n[green]Reports saved to: {output_dir}[/green]")


@report_app.command("eod")
def report_eod(
    report_date: str = typer.Option(None, "--date", "-d", help="Report date (YYYY-MM-DD)"),
):
    """Generate end-of-day summary report."""
    init()

    from app.reports.eod import EndOfDayReport

    # Parse date
    if report_date:
        parsed_date = datetime.strptime(report_date, "%Y-%m-%d").date()
    else:
        parsed_date = date.today()

    generator = EndOfDayReport()

    with console.status("[bold green]Generating EOD report..."):
        report = generator.generate_report(parsed_date)

    # Display report
    console.print(Markdown(generator.format_report(report)))

    # Save report
    output_path = generator.save_report(report)
    console.print(f"\n[green]Report saved to: {output_path}[/green]")


@report_app.command("weekly")
def report_weekly(
    week: str = typer.Option(None, "--week", "-w", help="Week (YYYY-WW format)"),
):
    """Generate weekly summary report."""
    init()

    from app.reports.weekly import WeeklyReport as WeeklyReportGenerator

    # Parse week
    if week:
        year, week_num = week.split("-W")
        year = int(year)
        week_num = int(week_num)
    else:
        today = date.today()
        year = today.year
        week_num = today.isocalendar()[1]

    generator = WeeklyReportGenerator()

    with console.status("[bold green]Generating weekly report..."):
        report = generator.generate_report(year, week_num)

    # Display report
    console.print(Markdown(generator.format_report(report)))

    # Save report
    output_path = generator.save_report(report)
    console.print(f"\n[green]Report saved to: {output_path}[/green]")


# ==================== STATS COMMANDS ====================


@stats_app.command("strategies")
def stats_strategies():
    """Show strategy performance leaderboard."""
    init()

    from app.journal.analytics import TradeAnalytics

    analytics = TradeAnalytics()
    stats = analytics.get_all_strategy_stats()

    if not stats:
        console.print("[yellow]No strategy data available yet. Add some trades first.[/yellow]")
        return

    table = Table(title="Strategy Leaderboard")
    table.add_column("Rank", style="dim")
    table.add_column("Strategy", style="cyan")
    table.add_column("Trades", justify="right")
    table.add_column("Win%", justify="right")
    table.add_column("Avg R", justify="right")
    table.add_column("Expectancy", justify="right")
    table.add_column("PF", justify="right")

    for i, s in enumerate(stats, 1):
        exp_style = "green" if s.expectancy > 0 else "red"

        table.add_row(
            str(i),
            s.strategy_name,
            str(s.trade_count),
            f"{s.win_rate:.0%}",
            f"{s.avg_r:+.2f}",
            f"[{exp_style}]{s.expectancy:+.3f}[/{exp_style}]",
            f"{s.profit_factor:.1f}",
        )

    console.print(table)


@stats_app.command("edge")
def stats_edge():
    """Analyze trading edge and weaknesses."""
    init()

    from app.journal.analytics import TradeAnalytics

    analytics = TradeAnalytics()
    edge = analytics.analyze_edge()

    console.print(Panel("[bold]Edge Analysis[/bold]", border_style="blue"))

    console.print("\n[green bold]Strengths:[/green bold]")
    for s in edge.strengths:
        console.print(f"  ✅ {s}")

    console.print("\n[red bold]Weaknesses:[/red bold]")
    for w in edge.weaknesses:
        console.print(f"  ❌ {w}")

    if edge.stop_doing:
        console.print("\n[yellow bold]Stop Doing:[/yellow bold]")
        for item in edge.stop_doing:
            console.print(f"  🛑 {item}")

    if edge.double_down:
        console.print("\n[cyan bold]Double Down On:[/cyan bold]")
        for item in edge.double_down:
            console.print(f"  ✨ {item}")

    if edge.coaching_focus:
        console.print("\n[magenta bold]Coaching Focus:[/magenta bold]")
        for item in edge.coaching_focus:
            console.print(f"  🎯 {item}")


@stats_app.command("summary")
def stats_summary(
    days: int = typer.Option(30, "--days", "-d", help="Number of days to analyze"),
):
    """Show overall performance summary."""
    init()

    from app.journal.analytics import TradeAnalytics
    from datetime import timedelta

    analytics = TradeAnalytics()

    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    trades = analytics.get_all_trades(start_date, end_date)

    if not trades:
        console.print(f"[yellow]No trades in the last {days} days[/yellow]")
        return

    # Compute metrics
    from app.journal.models import TradeOutcome

    total = len(trades)
    winners = len([t for t in trades if t.outcome == TradeOutcome.WIN])
    losers = len([t for t in trades if t.outcome == TradeOutcome.LOSS])

    r_vals = [t.r_multiple for t in trades if t.r_multiple is not None]
    total_r = sum(r_vals) if r_vals else 0
    avg_r = total_r / len(r_vals) if r_vals else 0

    pnl_vals = [t.pnl_dollars for t in trades if t.pnl_dollars is not None]
    total_pnl = sum(pnl_vals) if pnl_vals else 0

    win_rate = winners / total if total > 0 else 0
    expectancy = analytics.compute_expectancy(trades)
    pf = analytics.compute_profit_factor(trades)

    # Display
    emoji = "🟢" if total_r >= 0 else "🔴"

    console.print(
        Panel(
            f"[bold]Last {days} Days Summary[/bold]\n\n"
            f"Total Trades: {total}\n"
            f"Winners: {winners} | Losers: {losers}\n"
            f"Win Rate: {win_rate:.1%}\n"
            f"\n"
            f"[bold]Total R: {total_r:+.2f}R {emoji}[/bold]\n"
            f"Total PnL: ${total_pnl:+,.2f}\n"
            f"Avg R per Trade: {avg_r:+.3f}R\n"
            f"Expectancy: {expectancy:+.3f}R\n"
            f"Profit Factor: {pf:.2f}",
            title="📊 Performance Summary",
            border_style="blue",
        )
    )


# ==================== CONFIG COMMANDS ====================


@config_app.command("tickers")
def config_tickers(
    action: str = typer.Argument("list", help="Action: list, add, remove"),
    ticker: str = typer.Argument(None, help="Ticker symbol (for add/remove)"),
):
    """Manage favorite tickers list."""
    init()

    if action == "list":
        tickers = settings.tickers
        console.print("[bold]Favorite Tickers:[/bold]")
        for t in tickers:
            console.print(f"  - {t}")

    elif action == "add":
        if not ticker:
            console.print("[red]Please provide a ticker symbol[/red]")
            raise typer.Exit(1)

        settings.add_ticker(ticker)
        console.print(f"[green]Added {ticker.upper()} to favorites[/green]")

    elif action == "remove":
        if not ticker:
            console.print("[red]Please provide a ticker symbol[/red]")
            raise typer.Exit(1)

        if settings.remove_ticker(ticker):
            console.print(f"[green]Removed {ticker.upper()} from favorites[/green]")
        else:
            console.print(f"[yellow]{ticker.upper()} not in favorites[/yellow]")

    else:
        console.print(f"[red]Unknown action: {action}. Use list, add, or remove.[/red]")


@config_app.command("show")
def config_show():
    """Show current configuration."""
    config = load_config()

    console.print(Panel("[bold]Current Configuration[/bold]", border_style="blue"))

    console.print(f"\n[cyan]Tickers:[/cyan] {', '.join(config.get('tickers', []))}")
    console.print(f"[cyan]Timezone:[/cyan] {config.get('timezone', 'America/New_York')}")
    console.print(f"[cyan]Data Provider:[/cyan] {config.get('data', {}).get('provider', 'yfinance')}")
    console.print(f"[cyan]LLM Enabled:[/cyan] {config.get('llm', {}).get('enabled', False)}")

    risk = config.get("risk", {})
    console.print(f"\n[yellow]Risk Controls:[/yellow]")
    console.print(f"  Max Daily Loss: {risk.get('max_daily_loss_r', 3.0)}R")
    console.print(f"  Max Losing Streak: {risk.get('max_losing_streak', 3)}")


@config_app.command("init")
def config_init():
    """Initialize database and create default config."""
    init()
    console.print("[green]✅ Database initialized[/green]")
    console.print("[green]✅ Configuration ready[/green]")
    console.print(f"\n[dim]Database: {settings.get('database_url', 'sqlite:///data/trades.db')}[/dim]")


# ==================== MAIN ====================


@app.callback()
def main():
    """
    Brooks Price Action Trading Coach

    Advisory system for discretionary day traders grounded in
    Al Brooks price action concepts.

    ⚠️  This is an ADVISORY ONLY system. It does NOT place trades.
    """
    pass


if __name__ == "__main__":
    app()
