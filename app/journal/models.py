"""
SQLAlchemy models for trade journal.

Models:
- Trade: Individual trade records
- Strategy: Strategy taxonomy
- DailySummary: Daily performance summaries
- Tag: Trade tags for categorization
"""

from datetime import datetime, date
from typing import Optional, Literal
import enum

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Date,
    Boolean,
    Text,
    Enum,
    ForeignKey,
    Table,
)
from sqlalchemy.orm import (
    declarative_base,
    relationship,
    sessionmaker,
    Session,
)

from app.config import get_database_url, DATA_DIR

Base = declarative_base()


class TradeDirection(enum.Enum):
    """Trade direction enum."""

    LONG = "long"
    SHORT = "short"


class TradeOutcome(enum.Enum):
    """Trade outcome enum."""

    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


# Many-to-many relationship between trades and tags
trade_tags = Table(
    "trade_tags",
    Base.metadata,
    Column("trade_id", Integer, ForeignKey("trades.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)


class Strategy(Base):
    """Strategy taxonomy model."""

    __tablename__ = "strategies"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    category = Column(String(50))  # with_trend, countertrend, trading_range, special
    description = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship to trades
    trades = relationship("Trade", back_populates="strategy")

    def __repr__(self):
        return f"<Strategy(name='{self.name}', category='{self.category}')>"


class Tag(Base):
    """Trade tag for categorization."""

    __tablename__ = "tags"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    color = Column(String(20))  # For UI display

    # Relationship to trades
    trades = relationship("Trade", secondary=trade_tags, back_populates="tags")

    def __repr__(self):
        return f"<Tag(name='{self.name}')>"


class Trade(Base):
    """Individual trade record."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    
    # Display order (chronological: oldest = 1)
    # Separate from id so trades display in time order regardless of when added
    trade_number = Column(Integer, index=True)

    # Basic trade info
    ticker = Column(String(20), nullable=False, index=True)
    trade_date = Column(Date, nullable=False, index=True)
    timeframe = Column(String(10))  # 1d, 2h, 5m, etc.
    direction = Column(Enum(TradeDirection), nullable=False)

    # Entry details
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime)
    entry_reason = Column(Text)

    # Exit details
    exit_price = Column(Float)
    exit_time = Column(DateTime)
    exit_reason = Column(Text)

    # Position sizing
    size = Column(Float)  # Number of shares/contracts
    
    # Risk management (manually maintained by user)
    stop_loss = Column(Float)  # SL - Stop Loss level (where you would exit if wrong)
    take_profit = Column(Float)  # TP - Take Profit level (target exit)
    
    # Legacy - kept for backward compatibility, use stop_loss instead
    stop_price = Column(Float)  # Deprecated: use stop_loss
    target_price = Column(Float)  # Deprecated: use take_profit
    
    # Currency (for non-USD trades)
    currency = Column(String(10), default="USD")  # USD, HKD, EUR, etc.
    currency_rate = Column(Float, default=1.0)  # Rate to convert to USD (1 USD = X currency)
    
    # Timezone of the market where the trade took place
    market_timezone = Column(String(50), default="America/New_York")  # e.g., America/New_York, Asia/Hong_Kong
    input_timezone = Column(String(50))  # User's original timezone when importing (for display)

    # Computed metrics (populated by analytics)
    r_multiple = Column(Float)  # PnL / initial risk
    pnl_dollars = Column(Float)
    pnl_percent = Column(Float)
    mae = Column(Float)  # Maximum Adverse Excursion in R
    mfe = Column(Float)  # Maximum Favorable Excursion in R
    hold_time_minutes = Column(Integer)
    outcome = Column(Enum(TradeOutcome))

    # Trade quality (for coaching)
    high_during_trade = Column(Float)
    low_during_trade = Column(Float)
    slippage_entry = Column(Float)
    slippage_exit = Column(Float)

    # Strategy and setup
    strategy_id = Column(Integer, ForeignKey("strategies.id"))
    strategy = relationship("Strategy", back_populates="trades")
    setup_type = Column(String(50))  # Quick classification
    setup_notes = Column(Text)
    
    # Original AI classification (preserved even if strategy is manually changed)
    ai_setup_classification = Column(String(100))  # Original AI classification, never overwritten

    # Context at time of trade
    market_regime = Column(String(20))  # trend_up, trend_down, range
    always_in_direction = Column(String(10))  # long, short, neutral

    # User notes and coaching
    notes = Column(Text)
    mistakes = Column(Text)
    lessons = Column(Text)
    coach_feedback = Column(Text)
    
    # Extended Brooks-style trade analysis fields
    trend_assessment = Column(Text)  # My assessment of the current trend (major and minor)
    signal_reason = Column(Text)  # Reason for entry (Strong Signal Bar, BO & follow-thru, High 2, etc)
    was_signal_present = Column(Text)  # Was there a signal? If not, why?
    strategy_alignment = Column(Text)  # Does this entry align with all the criteria of the chosen strategy?
    entry_exit_emotions = Column(Text)  # Emotions when exit & entry, any hesitation? Why?
    entry_tp_distance = Column(Text)  # Are the entry and TP too far apart?
    
    # Brooks-style trade intent fields
    trade_type = Column(String(20))  # scalp, swing, position
    entry_order_type = Column(String(20))  # market, limit, stop, stop_limit
    exit_order_type = Column(String(20))  # market, limit, stop, stop_limit
    stop_reason = Column(Text)  # Why stop is placed there
    target_reason = Column(Text)  # Why target is placed there
    invalidation_condition = Column(Text)  # What would prove setup wrong
    confidence_level = Column(Integer)  # 1-5
    emotional_state = Column(String(50))  # calm, rushed, revenge, fomo, tired
    followed_plan = Column(Boolean)  # Did trader follow their plan?
    account_type = Column(String(20), default="paper")  # paper, live
    
    # Cached AI review (JSON string)
    cached_review = Column(Text)  # Stores JSON of TradeReview
    review_generated_at = Column(DateTime)  # When the review was generated
    review_in_progress = Column(Boolean, default=False)  # True while regenerating

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Tags relationship
    tags = relationship("Tag", secondary=trade_tags, back_populates="trades")

    def __repr__(self):
        return f"<Trade(ticker='{self.ticker}', date='{self.trade_date}', direction='{self.direction.value}')>"

    @property
    def duration_display(self) -> str:
        """Get human-readable duration (e.g., '2h 15m' or '1d 3h 20m')."""
        if not self.entry_time or not self.exit_time:
            return "-"
        
        delta = self.exit_time - self.entry_time
        total_seconds = int(delta.total_seconds())
        
        if total_seconds < 0:
            return "-"
        
        days = total_seconds // 86400
        hours = (total_seconds % 86400) // 3600
        minutes = (total_seconds % 3600) // 60
        
        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0 or days > 0:  # Show hours if there are days
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        
        # Simplify display
        if days == 0 and hours == 0:
            return f"{minutes}m"
        elif days == 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{days}d {hours}h {minutes}m"

    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable."""
        if self.r_multiple is not None:
            return self.r_multiple > 0
        if self.pnl_dollars is not None:
            return self.pnl_dollars > 0
        return False

    @property
    def pnl_usd(self) -> Optional[float]:
        """Get P&L converted to USD."""
        if self.pnl_dollars is None:
            return None
        if self.currency and self.currency != 'USD' and self.currency_rate and self.currency_rate > 0:
            return self.pnl_dollars / self.currency_rate
        return self.pnl_dollars

    @property
    def entry_time_local(self) -> Optional[datetime]:
        """Get entry time in the original input timezone."""
        return self._convert_to_input_timezone(self.entry_time)
    
    @property
    def exit_time_local(self) -> Optional[datetime]:
        """Get exit time in the original input timezone."""
        return self._convert_to_input_timezone(self.exit_time)
    
    def _convert_to_input_timezone(self, dt: Optional[datetime]) -> Optional[datetime]:
        """Convert a market timezone datetime back to input timezone."""
        if dt is None or not self.input_timezone or not self.market_timezone:
            return dt
        if self.input_timezone == self.market_timezone:
            return dt
        try:
            from zoneinfo import ZoneInfo
            market_tz = ZoneInfo(self.market_timezone)
            input_tz = ZoneInfo(self.input_timezone)
            # Assume dt is in market timezone (naive)
            dt_market = dt.replace(tzinfo=market_tz)
            # Convert to input timezone
            dt_input = dt_market.astimezone(input_tz)
            return dt_input.replace(tzinfo=None)
        except Exception:
            return dt

    @property
    def effective_stop_loss(self) -> Optional[float]:
        """Get the effective stop loss (prefer stop_loss, fallback to stop_price for legacy)."""
        return self.stop_loss or self.stop_price
    
    @property
    def effective_take_profit(self) -> Optional[float]:
        """Get the effective take profit (prefer take_profit, fallback to target_price for legacy)."""
        return self.take_profit or self.target_price
    
    @property
    def initial_risk_dollars(self) -> Optional[float]:
        """Calculate initial risk in dollars."""
        sl = self.effective_stop_loss
        if self.entry_price and sl and self.size:
            risk_per_share = abs(self.entry_price - sl)
            return risk_per_share * self.size
        return None

    def compute_metrics(self) -> None:
        """Compute derived metrics from trade data."""
        sl = self.effective_stop_loss
        
        # R-multiple
        if self.entry_price and self.exit_price and sl:
            if self.direction == TradeDirection.LONG:
                risk = self.entry_price - sl
                reward = self.exit_price - self.entry_price
            else:
                risk = sl - self.entry_price
                reward = self.entry_price - self.exit_price

            # Handle zero or negative risk (stop == entry)
            if risk > 0.0001:
                self.r_multiple = reward / risk
            elif self.entry_price > 0:
                # Fallback: assume 2% risk for R calculation
                assumed_risk = self.entry_price * 0.02
                self.r_multiple = reward / assumed_risk

        # PnL
        if self.entry_price and self.exit_price:
            if self.direction == TradeDirection.LONG:
                self.pnl_percent = (self.exit_price - self.entry_price) / self.entry_price * 100
            else:
                self.pnl_percent = (self.entry_price - self.exit_price) / self.entry_price * 100

            if self.size:
                if self.direction == TradeDirection.LONG:
                    self.pnl_dollars = (self.exit_price - self.entry_price) * self.size
                else:
                    self.pnl_dollars = (self.entry_price - self.exit_price) * self.size

        # MAE/MFE
        if self.high_during_trade and self.low_during_trade and sl:
            if self.direction == TradeDirection.LONG:
                risk = self.entry_price - sl
                if risk > 0:
                    self.mae = (self.entry_price - self.low_during_trade) / risk
                    self.mfe = (self.high_during_trade - self.entry_price) / risk
            else:
                risk = sl - self.entry_price
                if risk > 0:
                    self.mae = (self.high_during_trade - self.entry_price) / risk
                    self.mfe = (self.entry_price - self.low_during_trade) / risk

        # Hold time
        if self.entry_time and self.exit_time:
            delta = self.exit_time - self.entry_time
            self.hold_time_minutes = int(delta.total_seconds() / 60)

        # Outcome - based on P&L dollars rounded to 2 decimals
        # Only breakeven if P&L rounds to exactly $0.00
        if self.pnl_dollars is not None:
            pnl_rounded = round(self.pnl_dollars, 2)
            if pnl_rounded > 0:
                self.outcome = TradeOutcome.WIN
            elif pnl_rounded < 0:
                self.outcome = TradeOutcome.LOSS
            else:
                self.outcome = TradeOutcome.BREAKEVEN
        elif self.r_multiple is not None:
            # Fallback to R-multiple if no P&L
            if self.r_multiple > 0:
                self.outcome = TradeOutcome.WIN
            elif self.r_multiple < 0:
                self.outcome = TradeOutcome.LOSS
            else:
                self.outcome = TradeOutcome.BREAKEVEN


class DailySummary(Base):
    """Daily performance summary."""

    __tablename__ = "daily_summaries"

    id = Column(Integer, primary_key=True)
    summary_date = Column(Date, unique=True, nullable=False, index=True)

    # Trade counts
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    breakeven_trades = Column(Integer, default=0)

    # Performance metrics
    total_r = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    avg_winner_r = Column(Float)
    avg_loser_r = Column(Float)
    largest_winner_r = Column(Float)
    largest_loser_r = Column(Float)

    # Risk metrics
    max_drawdown_r = Column(Float)
    consecutive_losses = Column(Integer, default=0)
    daily_loss_limit_hit = Column(Boolean, default=False)

    # Best/worst trades
    best_trade_id = Column(Integer, ForeignKey("trades.id"))
    worst_trade_id = Column(Integer, ForeignKey("trades.id"))

    # Notes
    market_notes = Column(Text)
    performance_notes = Column(Text)
    improvement_focus = Column(Text)
    rule_violations = Column(Text)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<DailySummary(date='{self.summary_date}', trades={self.total_trades}, r={self.total_r})>"


# Database setup
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        db_url = get_database_url()
        _engine = create_engine(db_url, echo=False)
    return _engine


def init_db() -> None:
    """Initialize database and create tables."""
    engine = get_engine()
    Base.metadata.create_all(engine)
    
    # Add new columns to existing trades table if they don't exist
    # This handles migrations for existing databases
    from sqlalchemy import text
    with engine.connect() as conn:
        # Currency columns
        try:
            conn.execute(text("SELECT currency FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN currency VARCHAR(10) DEFAULT 'USD'"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN currency_rate FLOAT DEFAULT 1.0"))
                conn.commit()
            except Exception:
                pass
        
        # Trade timeframe column (5m, 2h, 1d)
        try:
            conn.execute(text("SELECT timeframe FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN timeframe VARCHAR(10) DEFAULT '5m'"))
                conn.commit()
            except Exception:
                pass
        
        # Market timezone column
        try:
            conn.execute(text("SELECT market_timezone FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN market_timezone VARCHAR(50) DEFAULT 'America/New_York'"))
                conn.commit()
            except Exception:
                pass
        
        # Input timezone column (user's original timezone)
        try:
            conn.execute(text("SELECT input_timezone FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN input_timezone VARCHAR(50)"))
                conn.commit()
            except Exception:
                pass
        
        # Stop Loss column (SL)
        try:
            conn.execute(text("SELECT stop_loss FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN stop_loss FLOAT"))
                conn.commit()
            except Exception:
                pass
        
        # Take Profit column (TP)
        try:
            conn.execute(text("SELECT take_profit FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN take_profit FLOAT"))
                conn.commit()
            except Exception:
                pass

        # Cached AI review columns
        try:
            conn.execute(text("SELECT cached_review FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN cached_review TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN review_generated_at DATETIME"))
                conn.commit()
            except Exception:
                pass
        
        # Review in progress flag
        try:
            conn.execute(text("SELECT review_in_progress FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN review_in_progress BOOLEAN DEFAULT 0"))
                conn.commit()
            except Exception:
                pass
        
        # Brooks-style trade intent columns
        try:
            conn.execute(text("SELECT trade_type FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN trade_type VARCHAR(20)"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN entry_order_type VARCHAR(20)"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN exit_order_type VARCHAR(20)"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN stop_reason TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN target_reason TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN invalidation_condition TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN confidence_level INTEGER"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN emotional_state VARCHAR(50)"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN followed_plan BOOLEAN"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN account_type VARCHAR(20) DEFAULT 'paper'"))
                conn.commit()
            except Exception:
                pass
        
        # Original AI classification (permanent, never overwritten)
        try:
            conn.execute(text("SELECT ai_setup_classification FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN ai_setup_classification VARCHAR(100)"))
                conn.commit()
            except Exception:
                pass
        
        # Extended Brooks-style trade analysis fields
        try:
            conn.execute(text("SELECT trend_assessment FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN trend_assessment TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN signal_reason TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN was_signal_present TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN strategy_alignment TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN entry_exit_emotions TEXT"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN entry_tp_distance TEXT"))
                conn.commit()
            except Exception:
                pass
        
        # Trade number for display order (chronological: oldest = 1)
        try:
            conn.execute(text("SELECT trade_number FROM trades LIMIT 1"))
        except Exception:
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN trade_number INTEGER"))
                conn.commit()
            except Exception:
                pass

    # Seed default strategies
    session = get_session()
    try:
        if session.query(Strategy).count() == 0:
            default_strategies = [
                # With-trend
                Strategy(name="breakout_pullback_long", category="with_trend", description="Long after pullback to breakout level"),
                Strategy(name="breakout_pullback_short", category="with_trend", description="Short after rally to breakdown level"),
                Strategy(name="second_entry_buy", category="with_trend", description="2nd entry long in uptrend pullback"),
                Strategy(name="second_entry_sell", category="with_trend", description="2nd entry short in downtrend rally"),
                Strategy(name="trend_resumption_long", category="with_trend", description="Long on trend resumption"),
                Strategy(name="trend_resumption_short", category="with_trend", description="Short on trend resumption"),
                # Countertrend
                Strategy(name="failed_breakout_long", category="countertrend", description="Long on failed breakdown"),
                Strategy(name="failed_breakout_short", category="countertrend", description="Short on failed breakout"),
                Strategy(name="wedge_reversal_long", category="countertrend", description="Long on wedge/3-push reversal"),
                Strategy(name="wedge_reversal_short", category="countertrend", description="Short on wedge/3-push reversal"),
                Strategy(name="double_bottom_long", category="countertrend", description="Long on double bottom"),
                Strategy(name="double_top_short", category="countertrend", description="Short on double top"),
                Strategy(name="climax_reversal_long", category="countertrend", description="Long on climax reversal"),
                Strategy(name="climax_reversal_short", category="countertrend", description="Short on climax reversal"),
                # Trading range
                Strategy(name="range_fade_high", category="trading_range", description="Short at range high"),
                Strategy(name="range_fade_low", category="trading_range", description="Long at range low"),
                Strategy(name="range_scalp_long", category="trading_range", description="Quick long scalp in range"),
                Strategy(name="range_scalp_short", category="trading_range", description="Quick short scalp in range"),
                # Special
                Strategy(name="trend_from_open_long", category="special", description="Trend from open - long"),
                Strategy(name="trend_from_open_short", category="special", description="Trend from open - short"),
                Strategy(name="opening_reversal_long", category="special", description="Opening reversal up"),
                Strategy(name="opening_reversal_short", category="special", description="Opening reversal down"),
                Strategy(name="gap_fill_long", category="special", description="Gap fill long"),
                Strategy(name="gap_fill_short", category="special", description="Gap fill short"),
                Strategy(name="unclassified", category="other", description="Unclassified trade"),
            ]
            session.add_all(default_strategies)
            session.commit()
    finally:
        session.close()


def get_session() -> Session:
    """Get a new database session."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(bind=engine)
    return _SessionLocal()


def get_strategy_by_name(name: str) -> Optional[Strategy]:
    """Get strategy by name."""
    session = get_session()
    try:
        return session.query(Strategy).filter(Strategy.name == name).first()
    finally:
        session.close()


def get_all_strategies() -> list[Strategy]:
    """Get all strategies."""
    session = get_session()
    try:
        return session.query(Strategy).filter(Strategy.is_active == True).all()
    finally:
        session.close()


def recalculate_trade_numbers() -> int:
    """
    Recalculate trade_number for all trades based on chronological order.
    
    Trade number 1 = oldest trade (by exit_time, then by id for same exit_time).
    
    Returns:
        Number of trades updated
    """
    session = get_session()
    try:
        # Get all trades sorted chronologically (oldest first)
        trades = session.query(Trade).all()
        
        # Sort by exit_time (primary), then by id (secondary - add order)
        def sort_key(t):
            # Use exit_time if available, otherwise fall back to entry_time, then trade_date
            if t.exit_time:
                return (t.exit_time, t.id)
            elif t.entry_time:
                return (t.entry_time, t.id)
            elif t.trade_date:
                return (datetime.combine(t.trade_date, datetime.min.time()), t.id)
            else:
                return (datetime.max, t.id)
        
        sorted_trades = sorted(trades, key=sort_key)
        
        # Assign trade numbers (oldest = 1)
        for i, trade in enumerate(sorted_trades, start=1):
            trade.trade_number = i
        
        session.commit()
        return len(sorted_trades)
    finally:
        session.close()
