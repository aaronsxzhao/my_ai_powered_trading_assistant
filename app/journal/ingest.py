"""
Trade ingestion for Brooks Trading Coach.

Supports:
- Manual trade entry
- CSV import from brokers
"""

from datetime import datetime, date
from pathlib import Path
from typing import Optional, Literal
import logging

import pandas as pd

from app.journal.models import (
    Trade,
    TradeDirection,
    Strategy,
    get_session,
    get_strategy_by_name,
    init_db,
)

logger = logging.getLogger(__name__)


class TradeIngester:
    """Handle trade data ingestion from various sources."""

    def __init__(self):
        """Initialize ingester and ensure database exists."""
        init_db()

    def add_trade_manual(
        self,
        ticker: str,
        trade_date: date,
        direction: Literal["long", "short"],
        entry_price: float,
        exit_price: float,
        stop_price: float,
        size: float = 1.0,
        timeframe: str = "5m",
        entry_time: Optional[datetime] = None,
        exit_time: Optional[datetime] = None,
        target_price: Optional[float] = None,
        strategy_name: Optional[str] = None,
        setup_type: Optional[str] = None,
        notes: Optional[str] = None,
        entry_reason: Optional[str] = None,
        exit_reason: Optional[str] = None,
        high_during_trade: Optional[float] = None,
        low_during_trade: Optional[float] = None,
    ) -> Trade:
        """
        Add a trade manually.

        Args:
            ticker: Stock symbol
            trade_date: Date of trade
            direction: 'long' or 'short'
            entry_price: Entry price
            exit_price: Exit price
            stop_price: Initial stop loss price
            size: Position size (shares/contracts)
            timeframe: Trade timeframe
            entry_time: Entry datetime
            exit_time: Exit datetime
            target_price: Price target
            strategy_name: Strategy name from taxonomy
            setup_type: Quick setup classification
            notes: Trade notes
            entry_reason: Reason for entry
            exit_reason: Reason for exit
            high_during_trade: Highest price during trade
            low_during_trade: Lowest price during trade

        Returns:
            Created Trade object
        """
        session = get_session()

        try:
            # Get strategy if provided
            strategy = None
            if strategy_name:
                strategy = session.query(Strategy).filter(Strategy.name == strategy_name).first()

            # Create trade
            trade = Trade(
                ticker=ticker.upper(),
                trade_date=trade_date,
                timeframe=timeframe,
                direction=TradeDirection.LONG if direction == "long" else TradeDirection.SHORT,
                entry_price=entry_price,
                exit_price=exit_price,
                stop_price=stop_price,
                size=size,
                entry_time=entry_time,
                exit_time=exit_time,
                target_price=target_price,
                strategy=strategy,
                setup_type=setup_type,
                notes=notes,
                entry_reason=entry_reason,
                exit_reason=exit_reason,
                high_during_trade=high_during_trade,
                low_during_trade=low_during_trade,
            )

            # Compute metrics
            trade.compute_metrics()

            session.add(trade)
            session.commit()
            session.refresh(trade)

            logger.info(f"Added trade: {trade}")
            return trade

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to add trade: {e}")
            raise

        finally:
            session.close()

    def import_csv(
        self,
        file_path: str | Path,
        broker: str = "generic",
        skip_errors: bool = True,
    ) -> tuple[int, int, list[str]]:
        """
        Import trades from CSV file.

        Args:
            file_path: Path to CSV file
            broker: Broker format ('generic', 'thinkorswim', 'tradovate', etc.)
            skip_errors: Whether to skip rows with errors

        Returns:
            Tuple of (imported_count, error_count, error_messages)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()

        # Map columns based on broker format
        column_map = self._get_column_map(broker)
        df = df.rename(columns=column_map)

        imported = 0
        errors = 0
        error_messages = []

        for idx, row in df.iterrows():
            try:
                trade = self._row_to_trade(row)
                if trade:
                    imported += 1
            except Exception as e:
                errors += 1
                msg = f"Row {idx + 1}: {str(e)}"
                error_messages.append(msg)
                logger.warning(msg)
                if not skip_errors:
                    raise

        logger.info(f"Imported {imported} trades, {errors} errors")
        return imported, errors, error_messages

    def _get_column_map(self, broker: str) -> dict[str, str]:
        """Get column mapping for broker format."""
        generic_map = {
            "symbol": "ticker",
            "date": "trade_date",
            "side": "direction",
            "entry": "entry_price",
            "exit": "exit_price",
            "stop": "stop_price",
            "qty": "size",
            "quantity": "size",
            "shares": "size",
        }

        broker_maps = {
            "generic": generic_map,
            "thinkorswim": {
                **generic_map,
                "underlying": "ticker",
                "pos effect": "direction",
                "trade price": "entry_price",
            },
            "tradovate": {
                **generic_map,
                "contract": "ticker",
                "bought price": "entry_price",
                "sold price": "exit_price",
            },
        }

        return broker_maps.get(broker, generic_map)

    def _row_to_trade(self, row: pd.Series) -> Optional[Trade]:
        """Convert a DataFrame row to a Trade object."""
        # Check required fields
        required = ["ticker", "entry_price", "exit_price"]
        for field in required:
            if field not in row or pd.isna(row[field]):
                raise ValueError(f"Missing required field: {field}")

        # Parse direction
        direction_str = str(row.get("direction", "long")).lower()
        if direction_str in ["long", "buy", "b", "bot"]:
            direction = "long"
        elif direction_str in ["short", "sell", "s", "sld"]:
            direction = "short"
        else:
            direction = "long"

        # Parse date
        trade_date = row.get("trade_date", date.today())
        if isinstance(trade_date, str):
            trade_date = pd.to_datetime(trade_date).date()
        elif isinstance(trade_date, datetime):
            trade_date = trade_date.date()

        # Get stop price or estimate
        stop_price = row.get("stop_price")
        if pd.isna(stop_price) or stop_price is None:
            # Estimate stop as 1% from entry
            entry = float(row["entry_price"])
            if direction == "long":
                stop_price = entry * 0.99
            else:
                stop_price = entry * 1.01

        return self.add_trade_manual(
            ticker=str(row["ticker"]),
            trade_date=trade_date,
            direction=direction,
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            stop_price=float(stop_price),
            size=float(row.get("size", 1)),
            timeframe=str(row.get("timeframe", "5m")),
            strategy_name=row.get("strategy") if pd.notna(row.get("strategy")) else None,
            notes=row.get("notes") if pd.notna(row.get("notes")) else None,
        )

    def get_trade_by_id(self, trade_id: int) -> Optional[Trade]:
        """Get a trade by ID."""
        session = get_session()
        try:
            return session.query(Trade).filter(Trade.id == trade_id).first()
        finally:
            session.close()

    def get_trades_by_date(self, trade_date: date) -> list[Trade]:
        """Get all trades for a specific date."""
        session = get_session()
        try:
            return session.query(Trade).filter(Trade.trade_date == trade_date).all()
        finally:
            session.close()

    def get_trades_by_ticker(self, ticker: str) -> list[Trade]:
        """Get all trades for a specific ticker."""
        session = get_session()
        try:
            return (
                session.query(Trade)
                .filter(Trade.ticker == ticker.upper())
                .order_by(Trade.trade_date.desc())
                .all()
            )
        finally:
            session.close()

    def get_trades_by_strategy(self, strategy_name: str) -> list[Trade]:
        """Get all trades for a specific strategy."""
        session = get_session()
        try:
            return (
                session.query(Trade)
                .join(Strategy)
                .filter(Strategy.name == strategy_name)
                .order_by(Trade.trade_date.desc())
                .all()
            )
        finally:
            session.close()

    def get_recent_trades(self, limit: int = 20) -> list[Trade]:
        """Get most recent trades."""
        session = get_session()
        try:
            return (
                session.query(Trade)
                .order_by(Trade.trade_date.desc(), Trade.id.desc())
                .limit(limit)
                .all()
            )
        finally:
            session.close()

    def update_trade(self, trade_id: int, **kwargs) -> Optional[Trade]:
        """Update a trade."""
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                return None

            for key, value in kwargs.items():
                if hasattr(trade, key):
                    setattr(trade, key, value)

            trade.compute_metrics()
            session.commit()
            session.refresh(trade)
            return trade

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to update trade: {e}")
            raise

        finally:
            session.close()

    def delete_trade(self, trade_id: int) -> bool:
        """Delete a trade."""
        session = get_session()
        try:
            trade = session.query(Trade).filter(Trade.id == trade_id).first()
            if not trade:
                return False

            session.delete(trade)
            session.commit()
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete trade: {e}")
            raise

        finally:
            session.close()
