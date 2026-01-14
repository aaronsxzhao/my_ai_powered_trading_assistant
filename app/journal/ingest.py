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

# Supported import formats
IMPORT_FORMATS = {
    "generic": "Generic CSV (ticker, direction, entry_price, exit_price, stop_price)",
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
        currency: str = "USD",
        currency_rate: float = 1.0,
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
            currency: Currency code (USD, HKD, etc.)
            currency_rate: Exchange rate (1 USD = X currency)

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
                currency=currency,
                currency_rate=currency_rate,
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
        
        logger.info(f"TradingView Balance History: {len(df)} entries")
        
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
                
                # Extract ticker from symbol (remove exchange prefix)
                ticker = symbol.split(':')[1] if ':' in symbol else symbol
                
                # Parse time
                time_str = row.get('Time', '')
                try:
                    exit_time = pd.to_datetime(time_str)
                except:
                    exit_time = datetime.now()
                
                # Get P&L from the row
                pnl = row.get('Realized P&L (value)', 0)
                
                # Estimate stop price (not in balance history)
                # Use 2% from entry as default
                if direction == 'long':
                    stop_price = entry_price * 0.98
                else:
                    stop_price = entry_price * 1.02
                
                trade = self.add_trade_manual(
                    ticker=ticker,
                    trade_date=exit_time.date() if hasattr(exit_time, 'date') else date.today(),
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    stop_price=stop_price,
                    size=size,
                    exit_time=exit_time,
                    notes=f"Imported from TradingView Balance History. P&L: {pnl:.2f}",
                )
                imported += 1
                
            except Exception as e:
                errors += 1
                error_messages.append(f"Row {idx}: {str(e)}")
                logger.warning(f"Failed to import TradingView balance row {idx}: {e}")
                if not skip_errors:
                    raise
        
        summary = f"Imported {imported} trades from {len(df)} entries. Skipped {skipped} non-trade entries."
        logger.info(summary)
        error_messages.insert(0, summary)
        
        return imported, errors, error_messages

    def _import_tv_order_history(
        self,
        file_path: Path,
        skip_errors: bool = True,
        balance_file_path: Optional[Path] = None,
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
        
        Cross-validation:
        - If balance_file_path is provided, validates computed trades against balance history
        
        Columns: Symbol, Side, Type, Qty, Limit Price, Stop Price, Fill Price, 
                 Status, Closing Time, Level ID, Leverage, Margin
        
        Args:
            file_path: Path to CSV file
            skip_errors: Whether to skip errors
            balance_file_path: Optional path to balance history CSV for cross-validation
            
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
        df['parsed_exchange'] = df['Symbol'].apply(lambda x: x.split(':')[0] if ':' in str(x) else 'USD')
        df['parsed_ticker'] = df['Symbol'].apply(lambda x: x.split(':')[1] if ':' in str(x) else x)
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
                    
                    # Get stop price
                    stop_price = last_stop_price
                    for o in pending_orders:
                        if o['stop']:
                            stop_price = o['stop']
                            break
                    if stop_price is None:
                        if direction == 'long':
                            stop_price = avg_entry_price * 0.98
                        else:
                            stop_price = avg_entry_price * 1.02
                    
                    trade_entry = {
                        'ticker': ticker,
                        'direction': direction,
                        'entry_price': avg_entry_price,
                        'exit_price': avg_exit_price,
                        'stop_price': stop_price,
                        'size': entry_total_qty,
                        'entry_time': entry_time,
                        'exit_time': exit_time,
                        'currency': currency,
                        'currency_rate': currency_rate,
                        'child_orders': pending_orders.copy(),  # Store child orders for drawer display
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
        
        # Import the matched trades
        from app.journal.models import Order
        
        imported = 0
        errors = 0
        error_messages = []
        
        for trade_data in trades_data:
            try:
                trade = self.add_trade_manual(
                    ticker=trade_data['ticker'],
                    trade_date=trade_data['exit_time'].date() if trade_data.get('exit_time') else date.today(),
                    direction=trade_data['direction'],
                    entry_price=trade_data['entry_price'],
                    exit_price=trade_data['exit_price'],
                    stop_price=float(trade_data['stop_price']),
                    size=trade_data['size'],
                    entry_time=trade_data.get('entry_time'),
                    exit_time=trade_data.get('exit_time'),
                    notes=f"Imported from TradingView Order History",
                    currency=trade_data.get('currency', 'USD'),
                    currency_rate=trade_data.get('currency_rate', 1.0),
                )
                
                # Save child orders if present (for multi-leg trades)
                if trade and trade_data.get('child_orders'):
                    session = get_session()
                    try:
                        for order_data in trade_data['child_orders']:
                            order = Order(
                                trade_id=trade.id,
                                side=order_data['side'],
                                quantity=order_data['qty'],
                                price=order_data['price'],
                                order_time=order_data.get('time'),
                                stop_price=order_data.get('stop'),
                                status='filled'
                            )
                            session.add(order)
                        session.commit()
                    except Exception as e:
                        logger.warning(f"Failed to save child orders: {e}")
                    finally:
                        session.close()
                
                imported += 1
                
            except Exception as e:
                errors += 1
                error_messages.append(f"Trade {trade_data.get('ticker')}: {str(e)}")
                logger.warning(f"Failed to import TradingView trade: {e}")
                if not skip_errors:
                    raise
        
        logger.info(f"TradingView import: {imported} trades created from {filled_count} orders, {errors} errors, {unmatched_count} unmatched positions discarded")
        
        # Add summary message
        summary = f"Processed {filled_count} filled orders → {imported} trades."
        if rejected_count or cancelled_count:
            summary += f" Discarded: {rejected_count} rejected, {cancelled_count} cancelled."
        if unmatched_count:
            summary += f" {unmatched_count} open position(s) discarded (no matching close)."
        error_messages.insert(0, summary)
        
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
                                # Estimate stop as 2% from entry
                                stop = buy_order['entry_price'] * 0.98

                                trade = self.add_trade_manual(
                                    ticker=ticker,
                                    trade_date=order['exit_time'].date(),
                                    direction='long',
                                    entry_price=buy_order['entry_price'],
                                    exit_price=order['entry_price'],  # Sell price
                                    stop_price=stop,
                                    size=min(buy_order['size'], order['size']),
                                    entry_time=buy_order['entry_time'],
                                    exit_time=order['exit_time'],
                                    notes=f"Imported from Robinhood",
                                )
                                imported += 1
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
