-- Migration: Add trade_fills table for storing individual executions
-- Each round-trip trade can consist of multiple fills (e.g., scaling in/out)

-- Create trade_fills table
CREATE TABLE IF NOT EXISTS trade_fills (
    id SERIAL PRIMARY KEY,
    trade_id INTEGER REFERENCES trades(id) ON DELETE CASCADE,
    symbol VARCHAR(50),
    fill_time TIMESTAMP,
    side VARCHAR(10),  -- 'buy' or 'sell'
    quantity FLOAT,
    price FLOAT,
    commission FLOAT,
    exchange VARCHAR(50),
    execution_id VARCHAR(100),  -- IBKR IBExecID for deduplication
    created_at TIMESTAMP DEFAULT NOW()
);

-- Index for fast lookups by trade
CREATE INDEX IF NOT EXISTS idx_trade_fills_trade_id ON trade_fills(trade_id);

-- Unique index on execution_id for deduplication (only where not null)
CREATE UNIQUE INDEX IF NOT EXISTS idx_trade_fills_execution_id 
ON trade_fills(execution_id) 
WHERE execution_id IS NOT NULL;

-- Enable Row Level Security
ALTER TABLE trade_fills ENABLE ROW LEVEL SECURITY;

-- RLS Policy: Users can only see fills for their own trades
CREATE POLICY "Users can view their own trade fills"
ON trade_fills
FOR SELECT
USING (
    EXISTS (
        SELECT 1 FROM trades 
        WHERE trades.id = trade_fills.trade_id 
        AND trades.user_id = auth.uid()::text
    )
);

-- RLS Policy: Users can insert fills for their own trades
CREATE POLICY "Users can insert fills for their own trades"
ON trade_fills
FOR INSERT
WITH CHECK (
    EXISTS (
        SELECT 1 FROM trades 
        WHERE trades.id = trade_fills.trade_id 
        AND trades.user_id = auth.uid()::text
    )
);

-- RLS Policy: Users can update their own trade fills
CREATE POLICY "Users can update their own trade fills"
ON trade_fills
FOR UPDATE
USING (
    EXISTS (
        SELECT 1 FROM trades 
        WHERE trades.id = trade_fills.trade_id 
        AND trades.user_id = auth.uid()::text
    )
);

-- RLS Policy: Users can delete their own trade fills
CREATE POLICY "Users can delete their own trade fills"
ON trade_fills
FOR DELETE
USING (
    EXISTS (
        SELECT 1 FROM trades 
        WHERE trades.id = trade_fills.trade_id 
        AND trades.user_id = auth.uid()::text
    )
);

-- Grant permissions to authenticated users
GRANT SELECT, INSERT, UPDATE, DELETE ON trade_fills TO authenticated;
GRANT USAGE, SELECT ON SEQUENCE trade_fills_id_seq TO authenticated;
