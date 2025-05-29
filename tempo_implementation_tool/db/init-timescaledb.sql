-- Enable TimescaleDB extension for time-series data
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Convert relevant tables to hypertables for time-series optimization

-- Convert time_entries table to a hypertable
SELECT create_hypertable('tempo.time_entries', 'start_time', 
                         chunk_time_interval => interval '1 day',
                         if_not_exists => TRUE);

-- Convert productivity_metrics table to a hypertable
SELECT create_hypertable('tempo.productivity_metrics', 'date', 
                         chunk_time_interval => interval '7 days',
                         if_not_exists => TRUE);

-- Create time-based continuous aggregates for analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS tempo.daily_productivity 
WITH (timescaledb.continuous) AS
SELECT 
    user_id,
    time_bucket('1 day', date) AS bucket,
    AVG(focus_score) AS avg_focus_score,
    SUM(focus_time) AS total_focus_time,
    SUM(tasks_completed) AS total_tasks_completed
FROM tempo.productivity_metrics
GROUP BY user_id, bucket;

-- Set retention policy (optional - uncomment to enable)
-- SELECT add_retention_policy('tempo.time_entries', INTERVAL '1 year');

-- Create additional TimescaleDB-specific indexes
CREATE INDEX IF NOT EXISTS idx_time_entries_user_time 
ON tempo.time_entries (user_id, start_time DESC);

-- Add compression policy (optional - uncomment to enable)
-- ALTER TABLE tempo.time_entries SET (
--     timescaledb.compress,
--     timescaledb.compress_segmentby = 'user_id'
-- );
-- SELECT add_compression_policy('tempo.time_entries', INTERVAL '7 days');

-- Create additional useful views for time analysis
CREATE OR REPLACE VIEW tempo.productivity_by_hour AS
SELECT 
    user_id,
    date_trunc('hour', start_time) AS hour,
    COUNT(*) AS tasks_worked,
    SUM(EXTRACT(EPOCH FROM (end_time - start_time))/60) AS minutes_worked
FROM tempo.time_entries
WHERE end_time IS NOT NULL
GROUP BY user_id, hour
ORDER BY user_id, hour;
