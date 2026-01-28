-- Brooks Trading Coach - Supabase Initial Schema Migration
-- Run this in Supabase SQL Editor after creating your project
-- ============================================================

-- 1. Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search

-- ============================================================
-- 2. Profiles table (extends auth.users)
-- ============================================================
CREATE TABLE IF NOT EXISTS profiles (
    id UUID REFERENCES auth.users(id) ON DELETE CASCADE PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    
    -- App-specific settings
    timezone TEXT DEFAULT 'America/New_York',
    default_currency TEXT DEFAULT 'USD'
);

-- Row Level Security for profiles
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile" ON profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can insert own profile" ON profiles
    FOR INSERT WITH CHECK (auth.uid() = id);

-- Trigger to auto-create profile on signup
CREATE OR REPLACE FUNCTION handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO profiles (id, email, name)
    VALUES (NEW.id, NEW.email, NEW.raw_user_meta_data->>'name')
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION handle_new_user();

-- ============================================================
-- 3. User Settings (per-user prompts and preferences)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_settings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    setting_key TEXT NOT NULL,
    setting_value JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, setting_key)
);

CREATE INDEX IF NOT EXISTS idx_user_settings_user_key ON user_settings(user_id, setting_key);

ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own settings" ON user_settings
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================
-- 4. User Materials (metadata, files in Storage)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_materials (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    file_size INTEGER,
    mime_type TEXT,
    indexed_at TIMESTAMPTZ,
    chunk_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_materials_user ON user_materials(user_id);

ALTER TABLE user_materials ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own materials" ON user_materials
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================
-- 5. Embeddings table (pgvector for RAG)
-- ============================================================
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    material_id UUID REFERENCES user_materials(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    section TEXT,
    embedding vector(384),  -- all-MiniLM-L6-v2 produces 384-dim vectors
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_embeddings_user ON embeddings(user_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_material ON embeddings(material_id);

-- IVFFlat index for fast similarity search (requires some data first)
-- Run this after inserting initial data:
-- CREATE INDEX embeddings_vector_idx ON embeddings 
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

ALTER TABLE embeddings ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own embeddings" ON embeddings
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================
-- 6. Strategies table
-- ============================================================
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    category TEXT,  -- with_trend, countertrend, trading_range, special
    description TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Strategies are global (shared across all users)
-- No RLS needed, all users can read

-- ============================================================
-- 7. Tags table
-- ============================================================
CREATE TABLE IF NOT EXISTS tags (
    id SERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    color TEXT
);

-- ============================================================
-- 8. Trades table (main trading journal)
-- ============================================================
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    trade_number INTEGER,
    
    -- Basic trade info
    ticker TEXT NOT NULL,
    trade_date DATE NOT NULL,
    timeframe TEXT,  -- 1d, 2h, 5m, etc.
    direction TEXT NOT NULL CHECK (direction IN ('long', 'short')),
    
    -- Entry details
    entry_price DOUBLE PRECISION NOT NULL,
    entry_time TIMESTAMPTZ,
    entry_reason TEXT,
    
    -- Exit details
    exit_price DOUBLE PRECISION,
    exit_time TIMESTAMPTZ,
    exit_reason TEXT,
    
    -- Position sizing
    size DOUBLE PRECISION,
    
    -- Risk management
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    stop_price DOUBLE PRECISION,  -- Legacy
    target_price DOUBLE PRECISION,  -- Legacy
    
    -- Currency
    currency TEXT DEFAULT 'USD',
    currency_rate DOUBLE PRECISION DEFAULT 1.0,
    
    -- Timezone
    market_timezone TEXT DEFAULT 'America/New_York',
    input_timezone TEXT,
    
    -- Computed metrics
    r_multiple DOUBLE PRECISION,
    pnl_dollars DOUBLE PRECISION,
    pnl_percent DOUBLE PRECISION,
    mae DOUBLE PRECISION,
    mfe DOUBLE PRECISION,
    hold_time_minutes INTEGER,
    outcome TEXT CHECK (outcome IN ('win', 'loss', 'breakeven')),
    
    -- Trade quality
    high_during_trade DOUBLE PRECISION,
    low_during_trade DOUBLE PRECISION,
    slippage_entry DOUBLE PRECISION,
    slippage_exit DOUBLE PRECISION,
    fees DOUBLE PRECISION,
    
    -- Strategy and setup
    strategy_id INTEGER REFERENCES strategies(id),
    setup_type TEXT,
    ai_setup_classification TEXT,
    
    -- Context
    market_regime TEXT,
    always_in_direction TEXT,
    
    -- Notes and coaching
    notes TEXT,
    mistakes TEXT,
    lessons TEXT,
    mistakes_and_lessons TEXT,
    coach_feedback TEXT,
    
    -- Extended Brooks-style fields
    trend_assessment TEXT,
    signal_reason TEXT,
    was_signal_present TEXT,
    strategy_alignment TEXT,
    entry_exit_emotions TEXT,
    entry_tp_distance TEXT,
    
    -- Trade intent
    trade_type TEXT,
    entry_order_type TEXT,
    exit_order_type TEXT,
    stop_reason TEXT,
    target_reason TEXT,
    invalidation_condition TEXT,
    confidence_level INTEGER,
    emotional_state TEXT,
    followed_plan BOOLEAN,
    account_type TEXT DEFAULT 'paper',
    
    -- Cached AI review
    cached_review TEXT,
    review_generated_at TIMESTAMPTZ,
    review_in_progress BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_trades_ticker ON trades(ticker);

ALTER TABLE trades ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own trades" ON trades
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================
-- 9. Trade Tags (many-to-many)
-- ============================================================
CREATE TABLE IF NOT EXISTS trade_tags (
    trade_id INTEGER REFERENCES trades(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    PRIMARY KEY (trade_id, tag_id)
);

-- ============================================================
-- 10. Daily Summaries
-- ============================================================
CREATE TABLE IF NOT EXISTS daily_summaries (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
    summary_date DATE NOT NULL,
    
    -- Trade counts
    total_trades INTEGER DEFAULT 0,
    winning_trades INTEGER DEFAULT 0,
    losing_trades INTEGER DEFAULT 0,
    breakeven_trades INTEGER DEFAULT 0,
    
    -- Performance metrics
    total_r DOUBLE PRECISION DEFAULT 0.0,
    total_pnl DOUBLE PRECISION DEFAULT 0.0,
    win_rate DOUBLE PRECISION,
    profit_factor DOUBLE PRECISION,
    avg_winner_r DOUBLE PRECISION,
    avg_loser_r DOUBLE PRECISION,
    largest_winner_r DOUBLE PRECISION,
    largest_loser_r DOUBLE PRECISION,
    
    -- Risk metrics
    max_drawdown_r DOUBLE PRECISION,
    consecutive_losses INTEGER DEFAULT 0,
    daily_loss_limit_hit BOOLEAN DEFAULT FALSE,
    
    -- Best/worst trades
    best_trade_id INTEGER REFERENCES trades(id),
    worst_trade_id INTEGER REFERENCES trades(id),
    
    -- Notes
    market_notes TEXT,
    performance_notes TEXT,
    improvement_focus TEXT,
    rule_violations TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    UNIQUE(user_id, summary_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_summaries_user_date ON daily_summaries(user_id, summary_date);

ALTER TABLE daily_summaries ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can manage own summaries" ON daily_summaries
    FOR ALL USING (auth.uid() = user_id);

-- ============================================================
-- 11. Functions for vector similarity search
-- ============================================================
CREATE OR REPLACE FUNCTION match_embeddings(
    query_embedding vector(384),
    match_count int DEFAULT 5,
    match_user_id uuid DEFAULT NULL
)
RETURNS TABLE (
    id uuid,
    chunk_text text,
    section text,
    similarity float
)
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    RETURN QUERY
    SELECT
        e.id,
        e.chunk_text,
        e.section,
        1 - (e.embedding <=> query_embedding) as similarity
    FROM embeddings e
    WHERE e.user_id = COALESCE(match_user_id, auth.uid())
    ORDER BY e.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================
-- 12. Seed default strategies
-- ============================================================
INSERT INTO strategies (name, category, description) VALUES
    ('breakout_pullback_long', 'with_trend', 'Long after pullback to breakout level'),
    ('breakout_pullback_short', 'with_trend', 'Short after rally to breakdown level'),
    ('second_entry_buy', 'with_trend', '2nd entry long in uptrend pullback'),
    ('second_entry_sell', 'with_trend', '2nd entry short in downtrend rally'),
    ('trend_resumption_long', 'with_trend', 'Long on trend resumption'),
    ('trend_resumption_short', 'with_trend', 'Short on trend resumption'),
    ('failed_breakout_long', 'countertrend', 'Long on failed breakdown'),
    ('failed_breakout_short', 'countertrend', 'Short on failed breakout'),
    ('wedge_reversal_long', 'countertrend', 'Long on wedge/3-push reversal'),
    ('wedge_reversal_short', 'countertrend', 'Short on wedge/3-push reversal'),
    ('double_bottom_long', 'countertrend', 'Long on double bottom'),
    ('double_top_short', 'countertrend', 'Short on double top'),
    ('climax_reversal_long', 'countertrend', 'Long on climax reversal'),
    ('climax_reversal_short', 'countertrend', 'Short on climax reversal'),
    ('range_fade_high', 'trading_range', 'Short at range high'),
    ('range_fade_low', 'trading_range', 'Long at range low'),
    ('range_scalp_long', 'trading_range', 'Quick long scalp in range'),
    ('range_scalp_short', 'trading_range', 'Quick short scalp in range'),
    ('trend_from_open_long', 'special', 'Trend from open - long'),
    ('trend_from_open_short', 'special', 'Trend from open - short'),
    ('opening_reversal_long', 'special', 'Opening reversal up'),
    ('opening_reversal_short', 'special', 'Opening reversal down'),
    ('gap_fill_long', 'special', 'Gap fill long'),
    ('gap_fill_short', 'special', 'Gap fill short'),
    ('unclassified', 'other', 'Unclassified trade')
ON CONFLICT (name) DO NOTHING;

-- ============================================================
-- 13. Helper function to update timestamps
-- ============================================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply to tables with updated_at
CREATE TRIGGER update_trades_updated_at
    BEFORE UPDATE ON trades
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_daily_summaries_updated_at
    BEFORE UPDATE ON daily_summaries
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_settings_updated_at
    BEFORE UPDATE ON user_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
