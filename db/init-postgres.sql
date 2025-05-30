-- Initialize PostgreSQL schema for Tempo

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS tempo;

-- Users table
CREATE TABLE IF NOT EXISTS tempo.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Tasks table
CREATE TABLE IF NOT EXISTS tempo.tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES tempo.users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    priority INTEGER NOT NULL DEFAULT 3,
    estimated_duration INTEGER, -- in minutes
    actual_duration INTEGER, -- in minutes
    due_date TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    tags TEXT[],
    context VARCHAR(100),
    energy_required INTEGER, -- 1-5 scale
    recurring BOOLEAN DEFAULT FALSE,
    recurrence_pattern JSONB
);

-- Calendar events table
CREATE TABLE IF NOT EXISTS tempo.calendar_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES tempo.users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    location VARCHAR(255),
    is_all_day BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    calendar_source VARCHAR(100), -- e.g., 'google', 'outlook', 'internal'
    external_id VARCHAR(255), -- ID from external calendar system
    attendees JSONB,
    recurring BOOLEAN DEFAULT FALSE,
    recurrence_pattern JSONB,
    reminder_minutes INTEGER[]
);

-- Time blocks table (for focused work, breaks, etc.)
CREATE TABLE IF NOT EXISTS tempo.time_blocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES tempo.users(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    block_type VARCHAR(50) NOT NULL, -- 'focus', 'break', 'meeting', etc.
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    task_id UUID REFERENCES tempo.tasks(id) ON DELETE SET NULL,
    productivity_rating INTEGER, -- 1-5 scale, filled after completion
    notes TEXT
);

-- Time tracking entries
CREATE TABLE IF NOT EXISTS tempo.time_entries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES tempo.users(id) ON DELETE CASCADE,
    task_id UUID REFERENCES tempo.tasks(id) ON DELETE SET NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration INTEGER, -- in seconds, calculated on completion
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    notes TEXT,
    tags TEXT[]
);

-- Productivity metrics
CREATE TABLE IF NOT EXISTS tempo.productivity_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES tempo.users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    focus_time INTEGER, -- in minutes
    tasks_completed INTEGER,
    tasks_created INTEGER,
    focus_score FLOAT, -- calculated metric
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- External integrations
CREATE TABLE IF NOT EXISTS tempo.integrations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES tempo.users(id) ON DELETE CASCADE,
    service_name VARCHAR(100) NOT NULL,
    access_token TEXT,
    refresh_token TEXT,
    token_expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    settings JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(user_id, service_name)
);

-- User preferences
CREATE TABLE IF NOT EXISTS tempo.user_preferences (
    user_id UUID PRIMARY KEY REFERENCES tempo.users(id) ON DELETE CASCADE,
    work_hours JSONB, -- e.g., {"monday": {"start": "09:00", "end": "17:00"}, ...}
    time_zone VARCHAR(50) NOT NULL DEFAULT 'UTC',
    focus_duration INTEGER DEFAULT 25, -- in minutes
    break_duration INTEGER DEFAULT 5, -- in minutes
    long_break_duration INTEGER DEFAULT 15, -- in minutes
    pomodoro_cycles INTEGER DEFAULT 4, -- number of focus sessions before long break
    notification_preferences JSONB,
    ui_preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tempo.tasks(user_id);
CREATE INDEX IF NOT EXISTS idx_tasks_due_date ON tempo.tasks(due_date);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tempo.tasks(status);
CREATE INDEX IF NOT EXISTS idx_calendar_events_user_id ON tempo.calendar_events(user_id);
CREATE INDEX IF NOT EXISTS idx_calendar_events_start_time ON tempo.calendar_events(start_time);
CREATE INDEX IF NOT EXISTS idx_time_blocks_user_id ON tempo.time_blocks(user_id);
CREATE INDEX IF NOT EXISTS idx_time_blocks_start_time ON tempo.time_blocks(start_time);
CREATE INDEX IF NOT EXISTS idx_time_entries_user_id ON tempo.time_entries(user_id);
CREATE INDEX IF NOT EXISTS idx_time_entries_task_id ON tempo.time_entries(task_id);

-- Create views
CREATE OR REPLACE VIEW tempo.upcoming_tasks AS
SELECT * FROM tempo.tasks
WHERE status = 'pending' AND due_date IS NOT NULL
ORDER BY due_date ASC;

CREATE OR REPLACE VIEW tempo.daily_schedule AS
SELECT 
    COALESCE(calendar_events.user_id, time_blocks.user_id) AS user_id,
    COALESCE(calendar_events.start_time, time_blocks.start_time) AS start_time,
    COALESCE(calendar_events.end_time, time_blocks.end_time) AS end_time,
    COALESCE(calendar_events.title, time_blocks.title) AS title,
    CASE 
        WHEN calendar_events.id IS NOT NULL THEN 'calendar_event'
        ELSE 'time_block'
    END AS entry_type,
    COALESCE(calendar_events.id, time_blocks.id) AS entry_id
FROM 
    tempo.calendar_events
FULL OUTER JOIN 
    tempo.time_blocks ON 
    calendar_events.user_id = time_blocks.user_id AND
    (calendar_events.start_time, calendar_events.end_time) OVERLAPS 
    (time_blocks.start_time, time_blocks.end_time)
ORDER BY 
    COALESCE(calendar_events.user_id, time_blocks.user_id), start_time;

-- Create functions
CREATE OR REPLACE FUNCTION tempo.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers
CREATE TRIGGER update_users_updated_at
BEFORE UPDATE ON tempo.users
FOR EACH ROW EXECUTE FUNCTION tempo.update_updated_at();

CREATE TRIGGER update_tasks_updated_at
BEFORE UPDATE ON tempo.tasks
FOR EACH ROW EXECUTE FUNCTION tempo.update_updated_at();

CREATE TRIGGER update_calendar_events_updated_at
BEFORE UPDATE ON tempo.calendar_events
FOR EACH ROW EXECUTE FUNCTION tempo.update_updated_at();

CREATE TRIGGER update_time_blocks_updated_at
BEFORE UPDATE ON tempo.time_blocks
FOR EACH ROW EXECUTE FUNCTION tempo.update_updated_at();

CREATE TRIGGER update_time_entries_updated_at
BEFORE UPDATE ON tempo.time_entries
FOR EACH ROW EXECUTE FUNCTION tempo.update_updated_at();

CREATE TRIGGER update_productivity_metrics_updated_at
BEFORE UPDATE ON tempo.productivity_metrics
FOR EACH ROW EXECUTE FUNCTION tempo.update_updated_at();

CREATE TRIGGER update_integrations_updated_at
BEFORE UPDATE ON tempo.integrations
FOR EACH ROW EXECUTE FUNCTION tempo.update_updated_at();

CREATE TRIGGER update_user_preferences_updated_at
BEFORE UPDATE ON tempo.user_preferences
FOR EACH ROW EXECUTE FUNCTION tempo.update_updated_at();

-- Insert sample user for testing
INSERT INTO tempo.users (email, hashed_password, full_name)
VALUES ('demo@example.com', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'Demo User')
ON CONFLICT DO NOTHING;
