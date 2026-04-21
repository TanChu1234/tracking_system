-- =============================================================================
-- MCT (Multi-Camera Tracking) Database Tables
-- Created: 2026-02-04
-- Purpose: Store tracking data for person movement analysis
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Table 1: mct_sessions - Manages tracking sessions
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mct_sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(50) UNIQUE NOT NULL,  -- UUID for each run of demo_mct.py
    
    -- Session Info
    started_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh'),
    ended_at TIMESTAMP WITH TIME ZONE,       -- NULL if still running
    status VARCHAR(20) DEFAULT 'active',     -- 'active', 'stopped', 'crashed'
    
    -- Stats
    total_tracks INTEGER DEFAULT 0,          -- Number of people tracked
    total_identified INTEGER DEFAULT 0       -- Number of people identified by face
);

CREATE INDEX IF NOT EXISTS idx_mct_sessions_status ON mct_sessions(status);
CREATE INDEX IF NOT EXISTS idx_mct_sessions_started_at ON mct_sessions(started_at);

-- -----------------------------------------------------------------------------
-- Table 2: mct_face_recognition - Stores face recognition events
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mct_face_recognition (
    id SERIAL PRIMARY KEY,
    
    -- Tracking Info
    session_id VARCHAR(50) NOT NULL,         -- Links to mct_sessions.session_id
    local_track_id INTEGER NOT NULL,         -- display_id within session (0, 1, 2...)
    
    -- Person Info (KEY FOR MERGING across sessions)
    usr_id VARCHAR(50) NOT NULL DEFAULT 'unknown',  -- Employee ID or 'unknown'
    
    -- Location & Time
    floor VARCHAR(10) NOT NULL,              -- "1F", "3F", "7F"
    camera_id VARCHAR(50),                   -- cam48, cam29...
    detected_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh'),
    
    -- Face Recognition Details
    confidence DOUBLE PRECISION,             -- Recognition confidence score
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_mct_face_usr_id ON mct_face_recognition(usr_id);
CREATE INDEX IF NOT EXISTS idx_mct_face_detected_at ON mct_face_recognition(detected_at);
CREATE INDEX IF NOT EXISTS idx_mct_face_session ON mct_face_recognition(session_id, local_track_id);
CREATE INDEX IF NOT EXISTS idx_mct_face_floor ON mct_face_recognition(floor);

-- -----------------------------------------------------------------------------
-- Table 3: mct_position_tracking - Stores position data (sampled every 5s)
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mct_position_tracking (
    id SERIAL PRIMARY KEY,
    
    -- Tracking Info
    session_id VARCHAR(50) NOT NULL,         -- Links to mct_sessions.session_id
    local_track_id INTEGER NOT NULL,         -- display_id within session
    
    -- Person Info (KEY FOR MERGING)
    usr_id VARCHAR(50) NOT NULL DEFAULT 'unknown',
    
    -- Position on Floor Map (after projection from camera -> map)
    floor VARCHAR(10) NOT NULL,
    x DOUBLE PRECISION NOT NULL,             -- X coordinate on floor map (mm)
    y DOUBLE PRECISION NOT NULL,             -- Y coordinate on floor map (mm)
    
    -- Original Bbox in Camera Frame (for debugging)
    camera_id VARCHAR(50),
    bbox_center_x INTEGER,                   -- Pixel X in frame
    bbox_center_y INTEGER,                   -- Pixel Y in frame
    
    -- Time
    tracked_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh'),
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_mct_pos_usr_id ON mct_position_tracking(usr_id);
CREATE INDEX IF NOT EXISTS idx_mct_pos_tracked_at ON mct_position_tracking(tracked_at);
CREATE INDEX IF NOT EXISTS idx_mct_pos_floor ON mct_position_tracking(floor);
CREATE INDEX IF NOT EXISTS idx_mct_pos_session ON mct_position_tracking(session_id, local_track_id);

-- Composite index for daily movement queries
CREATE INDEX IF NOT EXISTS idx_mct_pos_usr_date ON mct_position_tracking(usr_id, tracked_at);

-- -----------------------------------------------------------------------------
-- Table 4: mct_desk_presence - Tracks if employees are at their assigned desk
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS mct_desk_presence (
    id SERIAL PRIMARY KEY,

    -- Session Info
    session_id VARCHAR(50) NOT NULL,

    -- ROI (Desk) Info
    roi_id INTEGER NOT NULL,                 -- ROI ID from rois table
    roi_owner VARCHAR(50) NOT NULL,          -- Employee ID assigned to this desk (ROI name)

    -- Location
    floor VARCHAR(10) NOT NULL,

    -- Presence
    is_present BOOLEAN NOT NULL DEFAULT FALSE,  -- true = someone at desk, false = empty

    -- Time
    checked_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh'),

    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh')
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_mct_desk_roi_owner ON mct_desk_presence(roi_owner);
CREATE INDEX IF NOT EXISTS idx_mct_desk_checked_at ON mct_desk_presence(checked_at);
CREATE INDEX IF NOT EXISTS idx_mct_desk_session ON mct_desk_presence(session_id);
CREATE INDEX IF NOT EXISTS idx_mct_desk_floor ON mct_desk_presence(floor);
CREATE INDEX IF NOT EXISTS idx_mct_desk_roi_id ON mct_desk_presence(roi_id);

-- Composite index for desk presence queries by owner and time
CREATE INDEX IF NOT EXISTS idx_mct_desk_owner_date ON mct_desk_presence(roi_owner, checked_at);

-- =============================================================================
-- Sample Queries for Reference
-- =============================================================================
-- 
-- 1. View daily movement of a person:
-- SELECT floor, x, y, camera_id, tracked_at
-- FROM mct_position_tracking
-- WHERE usr_id = 'INF1901002'
--   AND DATE(tracked_at) = CURRENT_DATE
-- ORDER BY tracked_at;
--
-- 2. Merge data from multiple sessions:
-- SELECT session_id, local_track_id, floor,
--        MIN(tracked_at) as first_seen,
--        MAX(tracked_at) as last_seen,
--        COUNT(*) as position_count
-- FROM mct_position_tracking
-- WHERE usr_id = 'INF1901002'
--   AND DATE(tracked_at) = CURRENT_DATE
-- GROUP BY session_id, local_track_id, floor
-- ORDER BY first_seen;
--
-- 3. View all face recognitions today:
-- SELECT usr_id, floor, camera_id, detected_at, confidence
-- FROM mct_face_recognition
-- WHERE DATE(detected_at) = CURRENT_DATE
--   AND usr_id != 'unknown'
-- ORDER BY detected_at DESC;
-- =============================================================================
