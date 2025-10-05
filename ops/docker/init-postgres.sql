-- ΨQRH PostgreSQL Initialization Script
-- Creates database schema for consciousness logs and metrics

-- Create main consciousness logs table
CREATE TABLE IF NOT EXISTS consciousness_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message TEXT NOT NULL,
    fci FLOAT,
    state VARCHAR(50),
    entropy FLOAT,
    fractal_dimension FLOAT,
    convergence_achieved BOOLEAN,
    processing_steps INTEGER,
    metadata JSONB
);

-- Create index for timestamp queries
CREATE INDEX IF NOT EXISTS idx_consciousness_logs_timestamp
ON consciousness_logs(timestamp DESC);

-- Create index for state queries
CREATE INDEX IF NOT EXISTS idx_consciousness_logs_state
ON consciousness_logs(state);

-- Create metrics aggregation table
CREATE TABLE IF NOT EXISTS consciousness_metrics_summary (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE,
    total_messages INTEGER DEFAULT 0,
    avg_fci FLOAT,
    avg_entropy FLOAT,
    state_distribution JSONB,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create unique constraint on date
CREATE UNIQUE INDEX IF NOT EXISTS idx_consciousness_metrics_date
ON consciousness_metrics_summary(date);

-- Create sessions table for tracking conversation contexts
CREATE TABLE IF NOT EXISTS consciousness_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    message_count INTEGER DEFAULT 0,
    avg_fci FLOAT,
    metadata JSONB
);

-- Grant permissions (for when psiqrh user is created)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_user WHERE usename = 'psiqrh') THEN
        CREATE USER psiqrh WITH PASSWORD 'psiqrh123';
    END IF;
END $$;

GRANT ALL PRIVILEGES ON DATABASE psiqrh_dev TO psiqrh;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO psiqrh;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO psiqrh;

-- Insert initial test data
INSERT INTO consciousness_logs (message, fci, state, entropy, fractal_dimension, convergence_achieved, processing_steps)
VALUES
    ('System initialization', 0.001, 'COMA', 4.5, 1.0, true, 5),
    ('First test message', 0.005, 'COMA', 5.2, 1.008, true, 12)
ON CONFLICT DO NOTHING;

-- Create function for automatic metrics aggregation
CREATE OR REPLACE FUNCTION update_metrics_summary()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO consciousness_metrics_summary (date, total_messages, avg_fci, avg_entropy)
    SELECT
        CURRENT_DATE,
        COUNT(*),
        AVG(fci),
        AVG(entropy)
    FROM consciousness_logs
    WHERE DATE(timestamp) = CURRENT_DATE
    ON CONFLICT (date) DO UPDATE SET
        total_messages = EXCLUDED.total_messages,
        avg_fci = EXCLUDED.avg_fci,
        avg_entropy = EXCLUDED.avg_entropy,
        updated_at = CURRENT_TIMESTAMP;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic aggregation
DROP TRIGGER IF EXISTS trigger_update_metrics ON consciousness_logs;
CREATE TRIGGER trigger_update_metrics
    AFTER INSERT ON consciousness_logs
    FOR EACH ROW
    EXECUTE FUNCTION update_metrics_summary();

-- Success message
DO $$
BEGIN
    RAISE NOTICE '✅ ΨQRH PostgreSQL schema initialized successfully';
END $$;