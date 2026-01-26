"""
Trade ingestion for Brooks Trading Coach.

Supports:
- Manual trade entry
- CSV import from multiple formats (Generic, TradingView, ThinkOrSwim, etc.)
- Bulk import from imports/ folder
- LLM-powered strategy classification
"""

from datetime import datetime, date
from pathlib import Path
from typing import Optional, Literal
import logging
import re
from zoneinfo import ZoneInfo

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

# Common timezones for user selection
INPUT_TIMEZONES = {
    "America/New_York": "Eastern Time (US)",
    "America/Chicago": "Central Time (US)",
    "America/Denver": "Mountain Time (US)", 
    "America/Los_Angeles": "Pacific Time (US)",
    "Asia/Shanghai": "Beijing/Shanghai Time",
    "Asia/Hong_Kong": "Hong Kong Time",
    "Asia/Tokyo": "Tokyo Time",
    "Asia/Singapore": "Singapore Time",
    "Europe/London": "London Time",
    "Europe/Paris": "Central European Time",
    "UTC": "UTC",
}

# Exchange prefix to market timezone mapping
EXCHANGE_TIMEZONES = {
    # US Markets (Eastern Time)
    "AMEX": "America/New_York",
    "NYSE": "America/New_York",
    "NASDAQ": "America/New_York",
    "ARCA": "America/New_York",
    "BATS": "America/New_York",
    "OTC": "America/New_York",
    # Hong Kong
    "HKEX": "Asia/Hong_Kong",
    "HKG": "Asia/Hong_Kong",
    # China
    "SSE": "Asia/Shanghai",
    "SZSE": "Asia/Shanghai",
    # Japan
    "TSE": "Asia/Tokyo",
    "JPX": "Asia/Tokyo",
    # Europe
    "LSE": "Europe/London",
    "XETRA": "Europe/Berlin",
    "EURONEXT": "Europe/Paris",
    # Default
    "DEFAULT": "America/New_York",
}


def get_market_timezone(ticker: str) -> str:
    """
    Determine the market timezone based on ticker's exchange prefix or pattern.
    
    Args:
        ticker: Ticker symbol, possibly with exchange prefix (e.g., "AMEX:SOXL", "HKEX:0700")
        
    Returns:
        Timezone string (e.g., "America/New_York", "Asia/Hong_Kong")
    """
    ticker = ticker.upper().strip()
    
    # Detect futures (=F suffix or CME prefix) - CME is in Chicago but we use Eastern
    # for consistency with US equity hours and chart display
    if "=F" in ticker:
        return "America/New_York"
    
    # Check exchange prefix (e.g., "HKEX:0981", "CME_MINI:MES1!")
    if ":" in ticker:
        exchange = ticker.split(":")[0]
        # CME futures exchanges
        if exchange in ["CME", "CME_MINI", "CBOT", "NYMEX", "COMEX"]:
            return "America/New_York"  # Use Eastern for consistency
        return EXCHANGE_TIMEZONES.get(exchange, EXCHANGE_TIMEZONES["DEFAULT"])
    
    # Check suffix (e.g., "0981.HK")
    if ".HK" in ticker:
        return "Asia/Hong_Kong"
    if ".SS" in ticker or ".SZ" in ticker:
        return "Asia/Shanghai"
    if ".T" in ticker:
        return "Asia/Tokyo"
    if ".L" in ticker:
        return "Europe/London"
    
    # Detect HK stocks by numeric pattern (1-5 digits)
    clean_ticker = ticker.replace(".", "").replace("-", "")
    if clean_ticker.isdigit() and 1 <= len(clean_ticker) <= 5:
        return "Asia/Hong_Kong"
    
    return EXCHANGE_TIMEZONES["DEFAULT"]


def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """
    Convert datetime from one timezone to another.
    
    Args:
        dt: Datetime object (naive or aware)
        from_tz: Source timezone string (e.g., "Asia/Shanghai")
        to_tz: Target timezone string (e.g., "America/New_York")
        
    Returns:
        Datetime in target timezone (as naive datetime for DB storage)
    """
    if from_tz == to_tz:
        return dt
    
    try:
        from_zone = ZoneInfo(from_tz)
        to_zone = ZoneInfo(to_tz)
        
        # If datetime is naive, assume it's in from_tz
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=from_zone)
        
        # Convert to target timezone
        converted = dt.astimezone(to_zone)
        
        # Return as naive datetime (for SQLite compatibility)
        return converted.replace(tzinfo=None)
    except Exception as e:
        logger.warning(f"Timezone conversion failed ({from_tz} → {to_tz}): {e}")
        return dt


# Supported import formats
IMPORT_FORMATS = {
    "generic": "Generic CSV (ticker, direction, entry_price, exit_price, size, sl, tp)",
    "tv_order_history": "TradingView - Order History Export (RECOMMENDED)",
    "tv_balance_history": "TradingView - Balance History Export (simpler, less detail)",
    "robinhood": "Robinhood - Auto Import (requires login)",
    "thinkorswim": "TD Ameritrade ThinkOrSwim",
    "tradovate": "Tradovate",
    "interactive_brokers": "Interactive Brokers",
}


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

    def _check_duplicate_trade(
        self,
        session,
        ticker: str,
        direction: str,
        entry_price: float,
        trade_date: date,
        entry_time: Optional[datetime] = None,
    ) -> Optional[Trade]:
        """
        Check if a duplicate trade already exists.
        
        Duplicate detection logic:
        1. If entry_time is provided: match on ticker + entry_time + direction
        2. Fallback: match on ticker + trade_date + entry_price + direction
        
        Args:
            session: Database session
            ticker: Stock symbol
            direction: 'long' or 'short'
            entry_price: Entry price
            trade_date: Date of trade
            entry_time: Entry datetime (optional)
            
        Returns:
            Existing Trade if duplicate found, None otherwise
        """
        ticker_upper = ticker.upper()
        trade_direction = TradeDirection.LONG if direction == "long" else TradeDirection.SHORT
        
        # Primary check: ticker + entry_time + direction (most reliable for intraday)
        if entry_time:
            existing = session.query(Trade).filter(
                Trade.ticker == ticker_upper,
                Trade.entry_time == entry_time,
                Trade.direction == trade_direction,
            ).first()
            
            if existing:
                logger.info(f"Duplicate trade found (by entry_time): {ticker_upper} {direction} at {entry_time}")
                return existing
        
        # Fallback check: ticker + trade_date + entry_price + direction
        # Use a small tolerance for price matching (0.01% to handle floating point)
        price_tolerance = entry_price * 0.0001  # 0.01% tolerance
        
        existing = session.query(Trade).filter(
            Trade.ticker == ticker_upper,
            Trade.trade_date == trade_date,
            Trade.direction == trade_direction,
            Trade.entry_price.between(entry_price - price_tolerance, entry_price + price_tolerance),
        ).first()
        
        if existing:
            logger.info(f"Duplicate trade found (by date+price): {ticker_upper} {direction} @ {entry_price} on {trade_date}")
            return existing
        
        return None

    def add_trade_manual(
        self,
        ticker: str,
        trade_date: date,
        direction: Literal["long", "short"],
        entry_price: float,
        exit_price: float,
        size: float = 1.0,
        timeframe: str = "5m",
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
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
        currency: str = "USD",
        currency_rate: float = 1.0,
        market_timezone: str = "America/New_York",
        input_timezone: Optional[str] = None,
        skip_duplicates: bool = True,
        user_id: Optional[int] = None,
    ) -> Optional[Trade]:
        """
        Add a trade manually.

        Args:
            ticker: Stock symbol
            trade_date: Date of trade
            direction: 'long' or 'short'
            entry_price: Entry price
            exit_price: Exit price
            size: Position size (shares/contracts)
            timeframe: Trade timeframe
            stop_loss: Stop Loss level (optional, manually maintained)
            take_profit: Take Profit level (optional, manually maintained)
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
            currency: Currency code (USD, HKD, etc.)
            currency_rate: Exchange rate (1 USD = X currency)
            skip_duplicates: If True, skip duplicate trades instead of creating them

        Returns:
            Created Trade object, or None if duplicate detected and skipped
        """
        session = get_session()

        try:
            # Check for duplicate trade
            if skip_duplicates:
                existing = self._check_duplicate_trade(
                    session=session,
                    ticker=ticker,
                    direction=direction,
                    entry_price=entry_price,
                    trade_date=trade_date,
                    entry_time=entry_time,
                )
                if existing:
                    logger.info(f"Skipping duplicate trade: {ticker} {direction} @ {entry_price}")
                    return None
            
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
                stop_loss=stop_loss,
                take_profit=take_profit,
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
                currency=currency,
                currency_rate=currency_rate,
                market_timezone=market_timezone,
                input_timezone=input_timezone,
                user_id=user_id,
            )

            # If no entry_time provided, create one from trade_date (start of trading day)
            if not trade.entry_time and trade.trade_date:
                from datetime import time as dt_time
                # Default to market open time based on exchange
                # US/HK both open at 9:30 AM local time
                # Japan opens at 9:00 AM, UK at 8:00 AM
                market_open_times = {
                    "Asia/Hong_Kong": dt_time(9, 30, 0),
                    "Asia/Shanghai": dt_time(9, 30, 0),
                    "Asia/Tokyo": dt_time(9, 0, 0),
                    "Europe/London": dt_time(8, 0, 0),
                    "America/New_York": dt_time(9, 30, 0),
                }
                open_time = market_open_times.get(market_timezone, dt_time(9, 30, 0))
                trade.entry_time = datetime.combine(trade.trade_date, open_time)
            
            # If no exit_time, set it same as entry_time or later
            if not trade.exit_time and trade.entry_time:
                trade.exit_time = trade.entry_time

            # Compute metrics
            trade.compute_metrics()

            session.add(trade)
            session.commit()
            session.refresh(trade)

            # Recalculate trade numbers for all trades (so new trade gets correct number)
            from app.journal.models import recalculate_trade_numbers
            recalculate_trade_numbers()

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
        format: str = "generic",
        skip_errors: bool = True,
    ) -> tuple[int, int, list[str]]:
        """
        Import trades from CSV file.

        Args:
            file_path: Path to CSV file
            format: Import format ('generic', 'tradingview', 'thinkorswim', etc.)
            skip_errors: Whether to skip rows with errors

        Returns:
            Tuple of (imported_count, error_count, error_messages)
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        # Use format-specific parser
        if format == "tv_balance_history":
            return self._import_tv_balance_history(file_path, skip_errors)
        if format == "tv_order_history":
            return self._import_tv_order_history(file_path, skip_errors)
        
        # Generic/other formats use column mapping
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.strip()

        # Map columns based on format
        column_map = self._get_column_map(format)
        df = df.rename(columns=column_map)
        
        # Sort by date/time so oldest trade gets lowest ID
        date_cols = ['trade_date', 'date', 'entry_time', 'exit_time', 'time']
        for col in date_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df = df.sort_values(col, ascending=True)
                    break
                except (ValueError, TypeError, KeyError):
                    pass  # Try next date column

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
        
        # Recalculate trade numbers after import
        if imported > 0:
            from app.journal.models import recalculate_trade_numbers
            recalculate_trade_numbers()
        
        return imported, errors, error_messages

    def _import_tv_balance_history(
        self,
        file_path: Path,
        skip_errors: bool = True,
    ) -> tuple[int, int, list[str]]:
        """
        Import trades from TradingView Paper Trading BALANCE HISTORY export.
        
        This is the RECOMMENDED format because each row contains a complete trade
        with entry price, exit price, direction, and P&L already calculated.
        
        Columns: Time, Balance Before, Balance After, Realized P&L (value), 
                 Realized P&L (currency), Action
        
        The Action column contains:
        "Close long position for symbol AMEX:SOXL at price 26.73 for 370 units. 
         Position AVG Price was 26.590000..."
        
        Args:
            file_path: Path to CSV file
            skip_errors: Whether to skip errors
            
        Returns:
            Tuple of (imported_count, error_count, error_messages)
        """
        df = pd.read_csv(file_path)
        
        # Sort by Time so oldest trade gets lowest ID
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
            df = df.sort_values('Time', ascending=True)
        
        imported = 0
        errors = 0
        error_messages = []
        skipped = 0
        
        # Pattern to parse the Action column
        # Example: "Close long position for symbol AMEX:SOXL at price 160.7 for 400 units. Position AVG Price was 160.800000, currency: HKD..."
        action_pattern = re.compile(
            r'Close\s+(long|short)\s+position\s+for\s+symbol\s+(\S+)\s+at\s+price\s+([\d.]+)\s+for\s+([\d.]+)\s+units\.\s+Position\s+AVG\s+Price\s+was\s+([\d.]+)',
            re.IGNORECASE
        )
        
        for idx, row in df.iterrows():
            try:
                action = row.get('Action', '')
                match = action_pattern.search(action)
                
                if not match:
                    skipped += 1
                    continue
                
                direction = match.group(1).lower()  # 'long' or 'short'
                symbol = match.group(2)  # 'AMEX:SOXL'
                exit_price = float(match.group(3))
                size = float(match.group(4))
                entry_price = float(match.group(5))
                
                # Keep full symbol with exchange prefix for proper provider routing
                ticker = symbol  # e.g., 'AMEX:SOXL', 'HKEX:0981'
                
                # Parse time
                time_str = row.get('Time', '')
                try:
                    exit_time = pd.to_datetime(time_str)
                except (ValueError, TypeError):
                    exit_time = datetime.now()
                
                # Get P&L from the row
                pnl = row.get('Realized P&L (value)', 0)

                # SL/TP not available in balance history - user should set manually
                trade = self.add_trade_manual(
                    ticker=ticker,
                    trade_date=exit_time.date() if hasattr(exit_time, 'date') else date.today(),
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=size,
                    exit_time=exit_time,
                    notes=f"Imported from TradingView Balance History. P&L: {pnl:.2f}. Set SL/TP manually.",
                )
                if trade:
                    imported += 1
                else:
                    skipped += 1  # Count duplicates as skipped
                
            except Exception as e:
                errors += 1
                error_messages.append(f"Row {idx}: {str(e)}")
                logger.warning(f"Failed to import TradingView balance row {idx}: {e}")
                if not skip_errors:
                    raise
        
        summary = f"Imported {imported} trades from {len(df)} entries. Skipped {skipped} (non-trade or duplicate)."
        logger.info(summary)
        error_messages.insert(0, summary)
        
        # Recalculate trade numbers after import
        if imported > 0:
            from app.journal.models import recalculate_trade_numbers
            recalculate_trade_numbers()
        
        return imported, errors, error_messages

    def _import_tv_order_history(
        self,
        file_path: Path,
        skip_errors: bool = True,
        balance_file_path: Optional[Path] = None,
        input_timezone: str = "America/New_York",
    ) -> tuple[int, int, list[str]]:
        """
        Import trades from TradingView Paper Trading ORDER HISTORY export.
        
        Handles multi-leg trades with scaling in/out using average cost basis:
        - Buy 300 @ $10, Buy 200 @ $11 → avg entry = $10.40
        - Sell 400 @ $12 → creates trade: 400 shares, entry=$10.40, exit=$12
        - Sell 100 @ $13 → creates trade: 100 shares, entry=$10.40, exit=$13
        
        Position tracking:
        - Positive position = long
        - Negative position = short
        - Each exit (partial or full) creates a trade record
        
        Timezone handling:
        - User specifies their input timezone (what timezone the CSV times are in)
        - Market timezone is determined by exchange prefix (AMEX/NASDAQ → Eastern, HKEX → HK)
        - Times are converted from input timezone to market timezone for display/analysis
        
        Cross-validation:
        - If balance_file_path is provided, validates computed trades against balance history
        
        Columns: Symbol, Side, Type, Qty, Limit Price, Stop Price, Fill Price, 
                 Status, Closing Time, Level ID, Leverage, Margin
        
        Args:
            file_path: Path to CSV file
            skip_errors: Whether to skip errors
            balance_file_path: Optional path to balance history CSV for cross-validation
            input_timezone: Timezone of the times in the CSV (user's local timezone)
            
        Returns:
            Tuple of (imported_count, error_count, error_messages)
        """
        # Load balance history for cross-validation if provided
        balance_data = {}
        if balance_file_path and balance_file_path.exists():
            try:
                balance_df = pd.read_csv(balance_file_path)
                # Parse balance history into a lookup table
                # Key: (ticker, exit_price, qty) -> (avg_entry, direction)
                for _, row in balance_df.iterrows():
                    action = str(row.get('Action', ''))
                    if 'Close' not in action or 'position' not in action:
                        continue
                    
                    # Parse: "Close long/short position for symbol EXCHANGE:TICKER at price X for Y units. Position AVG Price was Z"
                    # Example: "Close short position for symbol AMEX:SOXL at price 39.67 for 200 units. Position AVG Price was 40.010000"
                    match = re.search(r'Close (\w+) position for symbol [\w:]+:(\w+) at price ([\d.]+) for (\d+) units.*AVG Price was ([\d.]+)', action, re.IGNORECASE)
                    if match:
                        direction = match.group(1)  # long or short
                        ticker = match.group(2)
                        exit_price = float(match.group(3))
                        qty = int(match.group(4))
                        avg_entry = float(match.group(5))
                        
                        key = (ticker, round(exit_price, 2), qty)
                        balance_data[key] = {
                            'direction': direction,
                            'avg_entry': avg_entry,
                            'exit_price': exit_price,
                        }
                logger.info(f"Loaded {len(balance_data)} trades from balance history for cross-validation")
            except Exception as e:
                logger.warning(f"Could not load balance history for cross-validation: {e}")
        
        df = pd.read_csv(file_path)
        
        total_orders = len(df)
        
        # Count orders by status
        status_counts = df['Status'].value_counts().to_dict()
        rejected_count = status_counts.get('Rejected', 0)
        cancelled_count = status_counts.get('Cancelled', 0)
        filled_count = status_counts.get('Filled', 0)
        
        logger.info(f"TradingView Order History: {total_orders} total orders")
        logger.info(f"  - Filled: {filled_count} (processing)")
        logger.info(f"  - Rejected: {rejected_count} (discarding)")
        logger.info(f"  - Cancelled: {cancelled_count} (discarding)")

        # ONLY process Filled orders - discard Rejected and Cancelled immediately
        df = df[df['Status'] == 'Filled'].copy()

        if df.empty:
            return 0, 0, [f"No filled orders found. Discarded {rejected_count} rejected, {cancelled_count} cancelled."]

        # Exchange to currency mapping
        EXCHANGE_CURRENCY = {
            'HKEX': ('HKD', 7.78),  # Hong Kong
            'TSE': ('JPY', 149.5),  # Tokyo
            'SSE': ('CNY', 7.24),  # Shanghai
            'SZSE': ('CNY', 7.24),  # Shenzhen
            'LSE': ('GBP', 0.79),  # London
            'EURONEXT': ('EUR', 0.92),
            'TSX': ('CAD', 1.36),  # Toronto
            'ASX': ('AUD', 1.53),  # Australia
            # US exchanges default to USD
            'NYSE': ('USD', 1.0),
            'NASDAQ': ('USD', 1.0),
            'AMEX': ('USD', 1.0),
        }
        
        # Parse the data
        df['parsed_exchange'] = df['Symbol'].apply(lambda x: x.split(':')[0] if ':' in str(x) else 'US')
        # Keep the full symbol with exchange prefix for proper provider routing
        # e.g., "HKEX:0981" stays as "HKEX:0981", "AMEX:SOXL" stays as "AMEX:SOXL"
        df['parsed_ticker'] = df['Symbol'].apply(lambda x: str(x).strip())
        df['parsed_datetime'] = pd.to_datetime(df['Closing Time'])
        df['parsed_price'] = df['Fill Price'].astype(float)
        df['parsed_qty'] = df['Qty'].astype(float)
        df['parsed_side'] = df['Side'].str.lower()
        df['stop_price_raw'] = pd.to_numeric(df['Stop Price'], errors='coerce')
        
        # Keep original CSV row order - DO NOT SORT!
        # TradingView exports newest first (row 1 = newest, row N = oldest)
        # Process exactly in CSV order: row 1 → row 2 → ... → row N
        df['csv_row_order'] = range(len(df))
        
        # ============================================================
        # POSITION ACCUMULATOR ALGORITHM
        # ============================================================
        # Process rows in EXACT CSV order (first row to last row)
        # Row 1 = newest order, Row N = oldest order
        # Track running position per ticker: sell → -qty, buy → +qty
        # When position reaches 0 → complete trade set
        # Direction: last order BUY → LONG, last order SELL → SHORT
        # Only dump unmatched orders at the END (last rows)
        # ============================================================
        
        trades_data = []
        unmatched_count = 0
        
        # Group orders by ticker - maintain EXACT CSV row order within each ticker
        for ticker in df['parsed_ticker'].unique():
            ticker_orders = df[df['parsed_ticker'] == ticker].copy()
            
            # Sort by csv_row_order to maintain exact CSV sequence
            ticker_orders = ticker_orders.sort_values('csv_row_order').reset_index(drop=True)
            
            # Get currency info from first order's exchange
            first_exchange = ticker_orders.iloc[0]['parsed_exchange']
            currency, currency_rate = EXCHANGE_CURRENCY.get(first_exchange, ('USD', 1.0))
            
            logger.info(f"Processing {len(ticker_orders)} orders for {ticker} (CSV row order, first→last)")
            
            # Position tracking
            position = 0  # Running position: positive = net long, negative = net short
            pending_orders = []  # Orders accumulated for current trade set
            last_stop_price = None
            
            for idx, order in ticker_orders.iterrows():
                side = order['parsed_side']  # 'buy' or 'sell'
                qty = float(order['parsed_qty'])
                price = float(order['parsed_price'])
                order_time = order['parsed_datetime']
                stop = order['stop_price_raw'] if pd.notna(order['stop_price_raw']) else None
                
                if stop:
                    last_stop_price = stop
                
                # Track position before update
                prev_position = position
                
                # Update position
                if side == 'buy':
                    position += qty
                else:  # sell
                    position -= qty
                
                logger.info(f"  [{idx}] {side.upper()} {int(qty)} @ {price:.4f} | Position: {int(prev_position)} → {int(position)}")
                
                # Add to pending orders
                pending_orders.append({
                    'side': side,
                    'qty': qty,
                    'price': price,
                    'time': order_time,
                    'stop': stop,
                })
                
                # Check if position reached 0 (or very close due to float rounding)
                # Trade set is complete!
                if abs(position) < 0.001 and len(pending_orders) > 0:
                    position = 0  # Normalize rounding errors
                    
                    # Determine direction from LAST order (the opening order)
                    last_order = pending_orders[-1]
                    if last_order['side'] == 'buy':
                        direction = 'long'
                        # Long: buys are entry, sells are exit
                        entry_orders = [o for o in pending_orders if o['side'] == 'buy']
                        exit_orders = [o for o in pending_orders if o['side'] == 'sell']
                    else:
                        direction = 'short'
                        # Short: sells are entry, buys are exit
                        entry_orders = [o for o in pending_orders if o['side'] == 'sell']
                        exit_orders = [o for o in pending_orders if o['side'] == 'buy']
                    
                    # Calculate weighted average entry price
                    entry_total_cost = sum(o['qty'] * o['price'] for o in entry_orders)
                    entry_total_qty = sum(o['qty'] for o in entry_orders)
                    avg_entry_price = entry_total_cost / entry_total_qty if entry_total_qty > 0 else 0
                    
                    # Calculate weighted average exit price
                    exit_total_cost = sum(o['qty'] * o['price'] for o in exit_orders)
                    exit_total_qty = sum(o['qty'] for o in exit_orders)
                    avg_exit_price = exit_total_cost / exit_total_qty if exit_total_qty > 0 else 0
                    
                    # VALIDATION: Entry and exit quantities should match
                    if abs(entry_total_qty - exit_total_qty) > 0.001:
                        logger.warning(f"    ⚠️ Quantity mismatch: entry={entry_total_qty}, exit={exit_total_qty}")
                    
                    # VALIDATION: Ensure we have valid prices
                    if avg_entry_price <= 0 or avg_exit_price <= 0:
                        logger.warning(f"    ⚠️ Invalid prices: entry={avg_entry_price}, exit={avg_exit_price}")
                        pending_orders = []
                        continue
                    
                    # Entry time = oldest entry order (last in list since we process newest first)
                    # Exit time = newest exit order (first in list)
                    entry_time = entry_orders[-1]['time'] if entry_orders else None
                    exit_time = exit_orders[0]['time'] if exit_orders else None
                    
                    # Note: Stop Price from TV order history is just the order type (stop order at X price)
                    # NOT the actual SL level. User should set SL/TP manually after import.
                    
                    trade_entry = {
                        'ticker': ticker,
                        'direction': direction,
                        'entry_price': avg_entry_price,
                        'exit_price': avg_exit_price,
                        'size': entry_total_qty,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'currency': currency,
                        'currency_rate': currency_rate,
                    }
                    
                    # Cross-validate against balance history if available
                    if balance_data:
                        # Try to find a matching entry in balance history
                        key = (ticker, round(avg_exit_price, 2), int(entry_total_qty))
                        if key in balance_data:
                            expected = balance_data[key]
                            expected_entry = expected['avg_entry']
                            
                            if abs(avg_entry_price - expected_entry) > 0.01:
                                logger.warning(f"    Cross-validation MISMATCH: "
                                             f"computed={avg_entry_price:.4f}, expected={expected_entry:.4f}")
                                trade_entry['entry_price'] = expected_entry
                                trade_entry['notes'] = "Entry corrected via balance history"
                            else:
                                logger.info(f"    Cross-validation OK ✓")
                    
                    logger.info(f"    ══► TRADE: {direction.upper()} entry={trade_entry['entry_price']:.4f} "
                              f"exit={avg_exit_price:.4f} qty={int(entry_total_qty)} "
                              f"({len(pending_orders)} orders matched)")
                    
                    trades_data.append(trade_entry)
                    
                    # Reset for next trade set
                    pending_orders = []
            
            # Log any unmatched orders at the END only (as user requested)
            if len(pending_orders) > 0:
                logger.info(f"  ⚠️ Discarding {len(pending_orders)} unmatched orders at end (position={int(position)}):")
                for o in pending_orders[-5:]:  # Only show last 5
                    logger.info(f"     - {o['side'].upper()} {int(o['qty'])} @ {o['price']:.4f}")
                unmatched_count += len(pending_orders)
        
        # Sort trades by entry time (oldest first) so IDs are in chronological order
        trades_data.sort(key=lambda x: x.get('entry_time') or x.get('exit_time') or datetime.min)
        
        # Import the matched trades
        imported = 0
        duplicates_skipped = 0
        errors = 0
        error_messages = []
        
        for trade_data in trades_data:
            try:
                ticker = trade_data['ticker']
                
                # Determine market timezone based on exchange prefix
                market_tz = get_market_timezone(ticker)
                
                # Convert times from user's input timezone to market timezone
                entry_time = trade_data.get('entry_time')
                exit_time = trade_data.get('exit_time')
                
                if entry_time and input_timezone != market_tz:
                    entry_time = convert_timezone(entry_time, input_timezone, market_tz)
                    logger.debug(f"Converted entry time: {input_timezone} → {market_tz}")
                
                if exit_time and input_timezone != market_tz:
                    exit_time = convert_timezone(exit_time, input_timezone, market_tz)
                    logger.debug(f"Converted exit time: {input_timezone} → {market_tz}")
                
                # Use converted exit time for trade_date
                trade_date = exit_time.date() if exit_time else date.today()
                
                trade = self.add_trade_manual(
                    ticker=ticker,
                    trade_date=trade_date,
                    direction=trade_data['direction'],
                    entry_price=trade_data['entry_price'],
                    exit_price=trade_data['exit_price'],
                    size=trade_data['size'],
                    entry_time=entry_time,
                    exit_time=exit_time,
                    notes=f"Imported from TradingView Order History (times in {market_tz}). Set SL/TP manually.",
                    currency=trade_data.get('currency', 'USD'),
                    currency_rate=trade_data.get('currency_rate', 1.0),
                    market_timezone=market_tz,
                    input_timezone=input_timezone if input_timezone != market_tz else None,
                )
                if trade:
                    imported += 1
                else:
                    duplicates_skipped += 1
                
            except Exception as e:
                errors += 1
                error_messages.append(f"Trade {trade_data.get('ticker')}: {str(e)}")
                logger.warning(f"Failed to import TradingView trade: {e}")
                if not skip_errors:
                    raise
        
        logger.info(f"TradingView import: {imported} trades created, {duplicates_skipped} duplicates skipped, {errors} errors, {unmatched_count} unmatched positions discarded")
        
        # Add summary message
        summary = f"Processed {filled_count} filled orders → {imported} trades."
        if duplicates_skipped:
            summary += f" {duplicates_skipped} duplicate(s) skipped."
        if rejected_count or cancelled_count:
            summary += f" Discarded: {rejected_count} rejected, {cancelled_count} cancelled."
        if unmatched_count:
            summary += f" {unmatched_count} open position(s) discarded (no matching close)."
        error_messages.insert(0, summary)
        
        # Recalculate trade numbers after import
        if imported > 0:
            from app.journal.models import recalculate_trade_numbers
            recalculate_trade_numbers()
        
        return imported, errors, error_messages

    def import_from_robinhood(
        self,
        username: str,
        password: str,
        mfa_code: Optional[str] = None,
        days_back: int = 30,
    ) -> dict:
        """
        Auto-import trades from Robinhood.
        
        Fetches filled orders from the last N days and creates trades.
        Note: Robinhood orders are individual buy/sell, not round trips,
        so each order becomes a separate trade entry.
        
        Args:
            username: Robinhood email/username
            password: Robinhood password
            mfa_code: Optional MFA code if 2FA is enabled
            days_back: Number of days to look back (default 30)
            
        Returns:
            Dict with: imported, errors, messages, needs_mfa, needs_device_approval
        """
        from app.data.robinhood import get_robinhood_client
        
        client = get_robinhood_client()
        
        # Login
        login_result = client.login(username, password, mfa_code)
        
        if not login_result.success:
            return {
                "imported": 0,
                "errors": 1,
                "messages": [login_result.message or login_result.error],
                "needs_mfa": login_result.needs_mfa,
                "needs_device_approval": login_result.needs_device_approval,
            }
        
        messages = [login_result.message]
        
        try:
            # Get stock orders
            orders = client.get_stock_orders(days_back=days_back)

            if not orders:
                return {
                    "imported": 0,
                    "errors": 0,
                    "messages": [f"No filled orders found in the last {days_back} days."],
                    "needs_mfa": False,
                    "needs_device_approval": False,
                }

            messages.append(f"Found {len(orders)} filled orders from Robinhood")

            imported = 0
            errors = 0

            # Group orders by ticker to match buy/sell pairs
            orders_by_ticker = {}
            for order in orders:
                parsed = client.parse_stock_order_to_trade(order)
                if parsed:
                    ticker = parsed['ticker']
                    if ticker not in orders_by_ticker:
                        orders_by_ticker[ticker] = []
                    orders_by_ticker[ticker].append(parsed)

            # Match buy/sell orders to create trades
            for ticker, ticker_orders in orders_by_ticker.items():
                # Sort by time
                ticker_orders.sort(key=lambda x: x['entry_time'])

                # Track open positions
                open_buys = []

                for order in ticker_orders:
                    if order['direction'] == 'long':  # Buy
                        open_buys.append(order)
                    else:  # Sell
                        # Try to match with a buy
                        if open_buys:
                            buy_order = open_buys.pop(0)

                            # Create a trade from matched buy/sell
                            try:
                                # SL/TP not available from Robinhood API - user should set manually
                                trade = self.add_trade_manual(
                                    ticker=ticker,
                                    trade_date=order['exit_time'].date(),
                                    direction='long',
                                    entry_price=buy_order['entry_price'],
                                    exit_price=order['entry_price'],  # Sell price
                                    size=min(buy_order['size'], order['size']),
                                    entry_time=buy_order['entry_time'],
                                    exit_time=order['exit_time'],
                                    notes=f"Imported from Robinhood. Set SL/TP manually.",
                                )
                                if trade:
                                    imported += 1
                                else:
                                    messages.append(f"{ticker}: Duplicate trade skipped")
                            except Exception as e:
                                errors += 1
                                messages.append(f"{ticker}: {str(e)}")

            messages.insert(0, f"Imported {imported} trades from Robinhood ({errors} errors)")

            return {
                "imported": imported,
                "errors": errors,
                "messages": messages,
                "needs_mfa": False,
                "needs_device_approval": False,
            }

        except Exception as e:
            return {
                "imported": 0,
                "errors": 1,
                "messages": [f"Error: {str(e)}"],
                "needs_mfa": False,
                "needs_device_approval": False,
            }

    def robinhood_login_with_session(self) -> bool:
        """
        Try to login to Robinhood using stored session.
        
        Returns:
            True if successful
        """
        from app.data.robinhood import get_robinhood_client
        client = get_robinhood_client()
        return client.login_with_stored_session()

    def robinhood_has_session(self) -> bool:
        """Check if Robinhood has a stored session."""
        from app.data.robinhood import get_robinhood_client
        return get_robinhood_client().has_stored_session()

    def robinhood_clear_session(self):
        """Clear stored Robinhood session."""
        from app.data.robinhood import get_robinhood_client
        get_robinhood_client().clear_stored_session()

    def _get_column_map(self, broker: str) -> dict[str, str]:
        """Get column mapping for broker format."""
        generic_map = {
            "symbol": "ticker",
            "date": "trade_date",
            "side": "direction",
            "entry": "entry_price",
            "exit": "exit_price",
            "stop": "stop_loss",
            "sl": "stop_loss",
            "tp": "take_profit",
            "target": "take_profit",
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

        # Get SL/TP if provided in CSV
        stop_loss = row.get("stop_loss") or row.get("sl")
        if pd.isna(stop_loss):
            stop_loss = None
        else:
            stop_loss = float(stop_loss)
        
        take_profit = row.get("take_profit") or row.get("tp")
        if pd.isna(take_profit):
            take_profit = None
        else:
            take_profit = float(take_profit)

        return self.add_trade_manual(
            ticker=str(row["ticker"]),
            trade_date=trade_date,
            direction=direction,
            entry_price=float(row["entry_price"]),
            exit_price=float(row["exit_price"]),
            size=float(row.get("size", 1)),
            timeframe=str(row.get("timeframe", "5m")),
            stop_loss=stop_loss,
            take_profit=take_profit,
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
            
            # Recalculate trade numbers after delete
            from app.journal.models import recalculate_trade_numbers
            recalculate_trade_numbers()
            
            return True

        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete trade: {e}")
            raise

    def delete_all_trades(self) -> int:
        """Delete all trades. Returns count of deleted trades."""
        session = get_session()
        try:
            count = session.query(Trade).count()
            session.query(Trade).delete()
            session.commit()
            logger.info(f"Deleted all {count} trades")
            return count
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to delete all trades: {e}")
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
                stop_price=trade.effective_stop_loss,  # Use effective stop loss
                entry_reason=trade.entry_reason,
                notes=trade.notes,
            )
            
            return result.get("strategy_name", "unclassified")
            
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
