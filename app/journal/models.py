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
    stop_price = Column(Float)
    target_price = Column(Float)
    
    # Currency (for non-USD trades)
    currency = Column(String(10), default="USD")  # USD, HKD, EUR, etc.
    currency_rate = Column(Float, default=1.0)  # Rate to convert to USD (1 USD = X currency)

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

    # Context at time of trade
    market_regime = Column(String(20))  # trend_up, trend_down, range
    always_in_direction = Column(String(10))  # long, short, neutral

    # User notes and coaching
    notes = Column(Text)
    mistakes = Column(Text)
    lessons = Column(Text)
    coach_feedback = Column(Text)

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
    def initial_risk_dollars(self) -> Optional[float]:
        """Calculate initial risk in dollars."""
        if self.entry_price and self.stop_price and self.size:
            risk_per_share = abs(self.entry_price - self.stop_price)
            return risk_per_share * self.size
        return None

    def compute_metrics(self) -> None:
        """Compute derived metrics from trade data."""
        # R-multiple
        if self.entry_price and self.exit_price and self.stop_price:
            if self.direction == TradeDirection.LONG:
                risk = self.entry_price - self.stop_price
                reward = self.exit_price - self.entry_price
            else:
                risk = self.stop_price - self.entry_price
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
        if self.high_during_trade and self.low_during_trade and self.stop_price:
            if self.direction == TradeDirection.LONG:
                risk = self.entry_price - self.stop_price
                if risk > 0:
                    self.mae = (self.entry_price - self.low_during_trade) / risk
                    self.mfe = (self.high_during_trade - self.entry_price) / risk
            else:
                risk = self.stop_price - self.entry_price
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
    
    # Add currency columns to existing trades table if they don't exist
    # This handles migrations for existing databases
    from sqlalchemy import text
    with engine.connect() as conn:
        try:
            conn.execute(text("SELECT currency FROM trades LIMIT 1"))
        except Exception:
            # Column doesn't exist, add it
            try:
                conn.execute(text("ALTER TABLE trades ADD COLUMN currency VARCHAR(10) DEFAULT 'USD'"))
                conn.execute(text("ALTER TABLE trades ADD COLUMN currency_rate FLOAT DEFAULT 1.0"))
                conn.commit()
            except Exception:
                pass  # Column might already exist or other error

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
