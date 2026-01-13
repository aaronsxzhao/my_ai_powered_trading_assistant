"""
Robinhood API integration for auto-importing trades.

Uses robin_stocks library to fetch order history.
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional
from pathlib import Path

from app.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Store credentials securely
ROBINHOOD_PICKLE_PATH = PROJECT_ROOT / "data" / ".robinhood.pickle"


class RobinhoodClient:
    """
    Client for fetching trade data from Robinhood.
    
    Handles authentication and order history retrieval.
    """

    def __init__(self):
        """Initialize Robinhood client."""
        self._logged_in = False
        self._rs = None

    def _get_robin_stocks(self):
        """Lazy import robin_stocks."""
        if self._rs is None:
            try:
                import robin_stocks.robinhood as rs
                self._rs = rs
            except ImportError:
                raise ImportError("robin_stocks not installed. Run: pip install robin_stocks")
        return self._rs

    def login(
        self,
        username: str,
        password: str,
        mfa_code: Optional[str] = None,
        store_session: bool = True,
    ) -> bool:
        """
        Login to Robinhood.
        
        Args:
            username: Robinhood email/username
            password: Robinhood password
            mfa_code: Optional MFA code if 2FA is enabled
            store_session: Whether to store session for future use
            
        Returns:
            True if login successful
        """
        rs = self._get_robin_stocks()
        
        try:
            # Try to login with stored session first
            if store_session and ROBINHOOD_PICKLE_PATH.exists():
                login_result = rs.login(
                    username=username,
                    password=password,
                    store_session=True,
                    pickle_name=str(ROBINHOOD_PICKLE_PATH),
                )
            else:
                login_result = rs.login(
                    username=username,
                    password=password,
                    mfa_code=mfa_code,
                    store_session=store_session,
                    pickle_name=str(ROBINHOOD_PICKLE_PATH) if store_session else None,
                )
            
            if login_result:
                self._logged_in = True
                logger.info("Successfully logged in to Robinhood")
                return True
            else:
                logger.error("Robinhood login failed")
                return False
                
        except Exception as e:
            logger.error(f"Robinhood login error: {e}")
            return False

    def login_with_stored_session(self) -> bool:
        """
        Try to login using stored session.
        
        Returns:
            True if login successful
        """
        if not ROBINHOOD_PICKLE_PATH.exists():
            return False
            
        rs = self._get_robin_stocks()
        
        try:
            # robin_stocks will use the pickle file automatically
            login_result = rs.login(
                pickle_name=str(ROBINHOOD_PICKLE_PATH),
            )
            
            if login_result:
                self._logged_in = True
                logger.info("Logged in to Robinhood using stored session")
                return True
            return False
            
        except Exception as e:
            logger.warning(f"Stored session login failed: {e}")
            return False

    def logout(self):
        """Logout from Robinhood."""
        if self._rs:
            self._rs.logout()
        self._logged_in = False

    def is_logged_in(self) -> bool:
        """Check if logged in."""
        return self._logged_in

    def has_stored_session(self) -> bool:
        """Check if there's a stored session."""
        return ROBINHOOD_PICKLE_PATH.exists()

    def clear_stored_session(self):
        """Clear stored session."""
        if ROBINHOOD_PICKLE_PATH.exists():
            ROBINHOOD_PICKLE_PATH.unlink()
            logger.info("Cleared stored Robinhood session")

    def get_stock_orders(
        self,
        days_back: int = 30,
    ) -> list[dict]:
        """
        Get stock order history.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of order dictionaries
        """
        if not self._logged_in:
            raise RuntimeError("Not logged in to Robinhood")
            
        rs = self._get_robin_stocks()
        
        try:
            # Get all stock orders
            all_orders = rs.orders.get_all_stock_orders()
            
            if not all_orders:
                return []
            
            # Filter by date and status
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_orders = []
            
            for order in all_orders:
                # Only include filled orders
                if order.get('state') != 'filled':
                    continue
                    
                # Parse order date
                created_at = order.get('created_at', '')
                if created_at:
                    order_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if order_date < cutoff_date:
                        continue
                
                filtered_orders.append(order)
            
            logger.info(f"Found {len(filtered_orders)} filled orders from last {days_back} days")
            return filtered_orders
            
        except Exception as e:
            logger.error(f"Error fetching Robinhood orders: {e}")
            return []

    def get_option_orders(
        self,
        days_back: int = 30,
    ) -> list[dict]:
        """
        Get options order history.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            List of order dictionaries
        """
        if not self._logged_in:
            raise RuntimeError("Not logged in to Robinhood")
            
        rs = self._get_robin_stocks()
        
        try:
            all_orders = rs.orders.get_all_option_orders()
            
            if not all_orders:
                return []
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_orders = []
            
            for order in all_orders:
                if order.get('state') != 'filled':
                    continue
                    
                created_at = order.get('created_at', '')
                if created_at:
                    order_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    if order_date < cutoff_date:
                        continue
                
                filtered_orders.append(order)
            
            logger.info(f"Found {len(filtered_orders)} filled option orders from last {days_back} days")
            return filtered_orders
            
        except Exception as e:
            logger.error(f"Error fetching Robinhood option orders: {e}")
            return []

    def parse_stock_order_to_trade(self, order: dict) -> Optional[dict]:
        """
        Parse a Robinhood stock order into trade format.
        
        Args:
            order: Robinhood order dictionary
            
        Returns:
            Trade data dictionary or None
        """
        try:
            rs = self._get_robin_stocks()
            
            # Get instrument details to get ticker symbol
            instrument_url = order.get('instrument')
            if instrument_url:
                instrument_data = rs.stocks.get_instrument_by_url(instrument_url)
                ticker = instrument_data.get('symbol', 'UNKNOWN')
            else:
                ticker = 'UNKNOWN'
            
            # Parse order details
            side = order.get('side', 'buy').lower()
            quantity = float(order.get('quantity', 0))
            avg_price = float(order.get('average_price', 0))
            
            # Parse timestamps
            created_at = order.get('created_at', '')
            updated_at = order.get('updated_at', '')
            
            if created_at:
                entry_time = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            else:
                entry_time = datetime.now()
                
            if updated_at:
                exit_time = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            else:
                exit_time = entry_time
            
            return {
                'ticker': ticker,
                'direction': 'long' if side == 'buy' else 'short',
                'entry_price': avg_price,
                'exit_price': avg_price,  # Same for single order
                'size': quantity,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'order_id': order.get('id'),
                'order_type': order.get('type'),
                'state': order.get('state'),
            }
            
        except Exception as e:
            logger.warning(f"Error parsing Robinhood order: {e}")
            return None


# Singleton instance
_client: Optional[RobinhoodClient] = None


def get_robinhood_client() -> RobinhoodClient:
    """Get the Robinhood client instance."""
    global _client
    if _client is None:
        _client = RobinhoodClient()
    return _client
