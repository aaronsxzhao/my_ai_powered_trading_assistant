"""
Trade ingestion for Brooks Trading Coach.

Supports:
- Manual trade entry
- CSV import from brokers
- Bulk import from imports/ folder
- LLM-powered strategy classification
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
from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Imports folder for bulk uploads
IMPORTS_DIR = PROJECT_ROOT / "imports"


class TradeIngester:
    """Handle trade data ingestion from various sources."""

    def __init__(self, use_llm_classification: bool = True):
        """
        Initialize ingester and ensure database exists.
        
        Args:
            use_llm_classification: Whether to use LLM to classify trades
        """
        init_db()
        IMPORTS_DIR.mkdir(exist_ok=True)
        self.use_llm_classification = use_llm_classification
        self._llm_analyzer = None

    @property
    def llm_analyzer(self):
        """Lazy load LLM analyzer."""
        if self._llm_analyzer is None:
            from app.llm.analyzer import get_analyzer
            self._llm_analyzer = get_analyzer()
        return self._llm_analyzer

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

    def bulk_import_from_folder(self) -> tuple[int, int, list[str]]:
        """
        Import all CSV files from the imports/ folder.
        
        Returns:
            Tuple of (total_imported, total_errors, error_messages)
        """
        if not IMPORTS_DIR.exists():
            IMPORTS_DIR.mkdir(exist_ok=True)
            return 0, 0, ["imports/ folder created. Add CSV files there."]

        csv_files = list(IMPORTS_DIR.glob("*.csv"))
        
        if not csv_files:
            return 0, 0, ["No CSV files found in imports/ folder"]

        total_imported = 0
        total_errors = 0
        all_messages = []

        for csv_file in csv_files:
            logger.info(f"Processing {csv_file.name}...")
            try:
                imported, errors, messages = self.import_csv(csv_file)
                total_imported += imported
                total_errors += errors
                all_messages.extend([f"{csv_file.name}: {m}" for m in messages])
                
                # Move processed file to processed subfolder
                processed_dir = IMPORTS_DIR / "processed"
                processed_dir.mkdir(exist_ok=True)
                csv_file.rename(processed_dir / csv_file.name)
                
            except Exception as e:
                total_errors += 1
                all_messages.append(f"{csv_file.name}: Failed - {e}")

        return total_imported, total_errors, all_messages

    def classify_trade_with_llm(self, trade: Trade) -> Optional[str]:
        """
        Use LLM to classify a trade's strategy.
        
        Args:
            trade: Trade object to classify
            
        Returns:
            Strategy name or None
        """
        if not self.use_llm_classification or not self.llm_analyzer.is_available:
            return None

        try:
            result = self.llm_analyzer.classify_trade_setup(
                ticker=trade.ticker,
                direction=trade.direction.value,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                stop_price=trade.stop_price,
                entry_reason=trade.entry_reason,
                notes=trade.notes,
            )
            
            strategy_name = result.get("strategy_name", "unclassified")
            logger.info(f"LLM classified trade as: {strategy_name} ({result.get('confidence', 'unknown')} confidence)")
            return strategy_name
            
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}")
            return None

    def reclassify_all_trades(self) -> tuple[int, int]:
        """
        Use LLM to reclassify all trades that are unclassified.
        
        Returns:
            Tuple of (reclassified_count, failed_count)
        """
        session = get_session()
        try:
            # Find unclassified trades
            unclassified_strategy = session.query(Strategy).filter(
                Strategy.name == "unclassified"
            ).first()
            
            if not unclassified_strategy:
                return 0, 0

            trades = session.query(Trade).filter(
                (Trade.strategy_id == unclassified_strategy.id) | 
                (Trade.strategy_id == None)
            ).all()

            reclassified = 0
            failed = 0

            for trade in trades:
                strategy_name = self.classify_trade_with_llm(trade)
                
                if strategy_name and strategy_name != "unclassified":
                    new_strategy = session.query(Strategy).filter(
                        Strategy.name == strategy_name
                    ).first()
                    
                    if new_strategy:
                        trade.strategy = new_strategy
                        trade.setup_type = strategy_name
                        reclassified += 1
                    else:
                        failed += 1
                else:
                    failed += 1

            session.commit()
            return reclassified, failed

        finally:
            session.close()

    @staticmethod
    def get_imports_folder_path() -> Path:
        """Get the path to the imports folder."""
        IMPORTS_DIR.mkdir(exist_ok=True)
        return IMPORTS_DIR
