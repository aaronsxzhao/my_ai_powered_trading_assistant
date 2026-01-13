"""
Report rendering utilities.

Handles:
- Markdown generation
- Chart creation (matplotlib)
- File output
"""

from datetime import date, datetime
from pathlib import Path
from typing import Optional
import logging

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from app.config import OUTPUTS_DIR

logger = logging.getLogger(__name__)


class ReportRenderer:
    """
    Render reports and charts to files.
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize renderer.

        Args:
            output_dir: Output directory (defaults to OUTPUTS_DIR)
        """
        self.output_dir = output_dir or OUTPUTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set matplotlib style
        plt.style.use("seaborn-v0_8-whitegrid")

    def create_price_chart(
        self,
        df: pd.DataFrame,
        ticker: str,
        title: str = "",
        levels: Optional[list[dict]] = None,
        save_path: Optional[Path] = None,
    ) -> Path:
        """
        Create a simple price chart with key levels.

        Args:
            df: OHLCV DataFrame
            ticker: Ticker symbol
            title: Chart title
            levels: List of key levels to mark
            save_path: Path to save chart

        Returns:
            Path to saved chart
        """
        if df.empty:
            logger.warning(f"Empty DataFrame for {ticker}, skipping chart")
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot close prices
        dates = pd.to_datetime(df["datetime"])
        closes = df["close"]

        ax.plot(dates, closes, linewidth=1.5, color="#2962FF", label="Close")

        # Add EMA if available
        if "ema_20" in df.columns:
            ax.plot(
                dates,
                df["ema_20"],
                linewidth=1,
                color="#FF6D00",
                linestyle="--",
                label="EMA 20",
            )

        # Mark key levels
        if levels:
            for level in levels:
                price = level.get("price")
                level_type = level.get("type", "")
                desc = level.get("description", "")

                if level_type == "resistance":
                    color = "#F44336"
                elif level_type == "support":
                    color = "#4CAF50"
                else:
                    color = "#9E9E9E"

                ax.axhline(
                    y=price,
                    color=color,
                    linestyle=":",
                    linewidth=1,
                    alpha=0.7,
                )
                ax.annotate(
                    f"${price:.2f} - {desc}",
                    xy=(dates.iloc[-1], price),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=8,
                    color=color,
                )

        # Formatting
        ax.set_title(title or f"{ticker} Price Chart", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price ($)")
        ax.legend(loc="upper left")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.xticks(rotation=45)

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f"{ticker}_chart.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved chart: {save_path}")
        return save_path

    def create_equity_curve(
        self,
        trades: list[dict],
        title: str = "Equity Curve (R)",
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Create equity curve chart from trades.

        Args:
            trades: List of trade dicts with 'date' and 'r_multiple'
            title: Chart title
            save_path: Path to save chart

        Returns:
            Path to saved chart
        """
        if not trades:
            return None

        # Compute cumulative R
        dates = []
        cumulative_r = []
        running_total = 0

        for trade in sorted(trades, key=lambda x: x.get("date", "")):
            r = trade.get("r_multiple", 0) or 0
            running_total += r
            dates.append(trade.get("date"))
            cumulative_r.append(running_total)

        fig, ax = plt.subplots(figsize=(12, 5))

        # Plot equity curve
        ax.plot(range(len(cumulative_r)), cumulative_r, linewidth=2, color="#2962FF")
        ax.fill_between(
            range(len(cumulative_r)),
            cumulative_r,
            alpha=0.3,
            color="#2962FF",
        )

        # Add zero line
        ax.axhline(y=0, color="#666666", linestyle="-", linewidth=1)

        # Formatting
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Cumulative R")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "equity_curve.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved equity curve: {save_path}")
        return save_path

    def create_strategy_bar_chart(
        self,
        strategy_stats: list[dict],
        title: str = "Strategy Performance (R)",
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Create bar chart of strategy performance.

        Args:
            strategy_stats: List of strategy stat dicts
            title: Chart title
            save_path: Path to save chart

        Returns:
            Path to saved chart
        """
        if not strategy_stats:
            return None

        fig, ax = plt.subplots(figsize=(12, 6))

        strategies = [s["strategy"][:15] for s in strategy_stats]
        total_r = [s["total_r"] for s in strategy_stats]
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in total_r]

        bars = ax.barh(strategies, total_r, color=colors)

        # Add value labels
        for bar, r in zip(bars, total_r):
            width = bar.get_width()
            ax.annotate(
                f"{r:+.1f}R",
                xy=(width, bar.get_y() + bar.get_height() / 2),
                xytext=(3, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )

        ax.axvline(x=0, color="#666666", linestyle="-", linewidth=1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Total R")
        ax.invert_yaxis()

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "strategy_performance.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved strategy chart: {save_path}")
        return save_path

    def create_daily_pnl_chart(
        self,
        daily_breakdown: list[dict],
        title: str = "Daily P&L (R)",
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Create daily P&L bar chart.

        Args:
            daily_breakdown: List of daily stat dicts
            title: Chart title
            save_path: Path to save chart

        Returns:
            Path to saved chart
        """
        if not daily_breakdown:
            return None

        fig, ax = plt.subplots(figsize=(10, 5))

        dates = [d["day"][:3] for d in daily_breakdown]  # Mon, Tue, etc.
        r_values = [d["total_r"] for d in daily_breakdown]
        colors = ["#4CAF50" if r >= 0 else "#F44336" for r in r_values]

        bars = ax.bar(dates, r_values, color=colors, edgecolor="white")

        # Add value labels
        for bar, r in zip(bars, r_values):
            if r != 0:
                height = bar.get_height()
                ax.annotate(
                    f"{r:+.1f}R",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -12),
                    textcoords="offset points",
                    ha="center",
                    fontsize=9,
                )

        ax.axhline(y=0, color="#666666", linestyle="-", linewidth=1)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel("R-Multiple")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "daily_pnl.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved daily P&L chart: {save_path}")
        return save_path

    def create_win_rate_pie(
        self,
        winners: int,
        losers: int,
        breakeven: int = 0,
        title: str = "Trade Outcomes",
        save_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Create pie chart of trade outcomes.

        Args:
            winners: Number of winning trades
            losers: Number of losing trades
            breakeven: Number of breakeven trades
            title: Chart title
            save_path: Path to save chart

        Returns:
            Path to saved chart
        """
        if winners + losers + breakeven == 0:
            return None

        fig, ax = plt.subplots(figsize=(8, 8))

        labels = []
        sizes = []
        colors = []

        if winners > 0:
            labels.append(f"Winners\n({winners})")
            sizes.append(winners)
            colors.append("#4CAF50")

        if losers > 0:
            labels.append(f"Losers\n({losers})")
            sizes.append(losers)
            colors.append("#F44336")

        if breakeven > 0:
            labels.append(f"Breakeven\n({breakeven})")
            sizes.append(breakeven)
            colors.append("#9E9E9E")

        ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            explode=[0.02] * len(sizes),
        )
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path is None:
            save_path = self.output_dir / "win_rate_pie.png"

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved win rate pie: {save_path}")
        return save_path

    def save_markdown(
        self,
        content: str,
        filename: str,
        date_folder: Optional[date] = None,
    ) -> Path:
        """
        Save markdown content to file.

        Args:
            content: Markdown content
            filename: File name (without path)
            date_folder: Optional date for subfolder

        Returns:
            Path to saved file
        """
        if date_folder:
            output_dir = self.output_dir / date_folder.strftime("%Y-%m-%d")
        else:
            output_dir = self.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / filename
        with open(file_path, "w") as f:
            f.write(content)

        logger.info(f"Saved markdown: {file_path}")
        return file_path

    def save_csv(
        self,
        df: pd.DataFrame,
        filename: str,
        date_folder: Optional[date] = None,
    ) -> Path:
        """
        Save DataFrame to CSV.

        Args:
            df: DataFrame to save
            filename: File name (without path)
            date_folder: Optional date for subfolder

        Returns:
            Path to saved file
        """
        if date_folder:
            output_dir = self.output_dir / date_folder.strftime("%Y-%m-%d")
        else:
            output_dir = self.output_dir

        output_dir.mkdir(parents=True, exist_ok=True)

        file_path = output_dir / filename
        df.to_csv(file_path, index=False)

        logger.info(f"Saved CSV: {file_path}")
        return file_path
