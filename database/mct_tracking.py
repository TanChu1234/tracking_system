"""
MCT (Multi-Camera Tracking) Database Module

This module provides functions to save tracking data to PostgreSQL database:
- Session management
- Face recognition events
- Position tracking (sampled every 5 seconds)

Usage:
    from database.mct_tracking import MCTTracker
    
    tracker = MCTTracker()
    tracker.start_session()
    
    # When face is recognized
    tracker.save_face_recognition(local_track_id=5, usr_id="INF1901002", 
                                   floor="3F", camera_id="cam48", confidence=0.95)
    
    # When position is tracked (called every 5s)
    tracker.save_position(local_track_id=5, usr_id="INF1901002",
                          floor="3F", x=1500.0, y=2300.0, 
                          camera_id="cam48", bbox_center=(640, 480))
    
    # When stopping
    tracker.end_session()
"""

import os
import uuid
import time
import threading
import psycopg2
from psycopg2.extras import execute_batch
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

# Vietnam timezone
VN_TZ = timezone(timedelta(hours=7))


class MCTTracker:
    """
    Multi-Camera Tracking data persistence handler.
    
    Features:
    - Automatic session management
    - Batched inserts for performance
    - 5-second sampling for position tracking
    - Thread-safe operations
    """
    
    # Database connection parameters (same as db_config.py)
    DEFAULT_HOST = "10.29.8.49"
    DEFAULT_PORT = 5432
    DEFAULT_USER = "infiniq_user"
    DEFAULT_PASSWORD = "infiniq_pass"
    DEFAULT_DATABASE = "camera_ai_db"
    
    # Sampling interval for position tracking (seconds)
    POSITION_SAMPLE_INTERVAL = 10.0
    
    # Minimum spatial movement to trigger logging (in mm, e.g., 1000.0 = 1 meter)
    MIN_MOVEMENT_DISTANCE = 1000.0
    
    # Sampling interval for desk presence checking (seconds)
    DESK_PRESENCE_INTERVAL = 60.0
    
    # Batch size for inserts — increased to 50 for database optimization
    BATCH_SIZE = 50
    
    # Periodic flush interval (seconds) — backup safety flush
    FLUSH_INTERVAL = 30.0
    
    def __init__(self):
        self.session_id: Optional[str] = None
        self.conn = None
        self._lock = threading.Lock()
        
        # Position sampling: track_id -> last_save_time
        self._last_position_time: Dict[int, float] = {}
        # Spatial sampling: track_id -> (last_saved_x, last_saved_y)
        self._last_position_coords: Dict[int, Tuple[float, float]] = {}
        
        # Desk presence sampling: last check time
        self._last_desk_presence_time: float = 0
        
        # Batch buffers
        self._position_buffer: List[dict] = []
        self._face_buffer: List[dict] = []
        self._desk_buffer: List[dict] = []
        
        # Stats
        self.total_tracks = 0
        self.total_identified = 0
        self._known_tracks: set = set()
        self._identified_tracks: set = set()
        
        # Start background flush thread
        self._stop_flush = threading.Event()
        self._flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self._flush_thread.start()
        
    def _periodic_flush(self):
        """Background thread: flush buffers every FLUSH_INTERVAL seconds"""
        last_cleanup = time.time()
        while not self._stop_flush.wait(self.FLUSH_INTERVAL):
            self._flush_buffers()
            
            # Daily cleanup: delete 'unknown' tracking data older than 12 hours once an hour
            if time.time() - last_cleanup >= 3600:
                self.cleanup_expired_unknowns(hours_old=12)
                last_cleanup = time.time()

    def _get_connection(self):
        """Get database connection with auto-reconnect"""
        if self.conn is None or self.conn.closed:
            self.conn = psycopg2.connect(
                host=os.environ.get('DB_HOST', self.DEFAULT_HOST),
                port=int(os.environ.get('DB_PORT', self.DEFAULT_PORT)),
                user=os.environ.get('DB_USER', self.DEFAULT_USER),
                password=os.environ.get('DB_PASSWORD', self.DEFAULT_PASSWORD),
                dbname=os.environ.get('DB_NAME', self.DEFAULT_DATABASE)
            )
            self.conn.autocommit = True
        return self.conn
    
    def _get_vn_time(self) -> datetime:
        """Get current time in Vietnam timezone"""
        return datetime.now(VN_TZ)
    
    def start_session(self) -> str:
        """
        Start a new tracking session.
        
        Returns:
            session_id (str): UUID of the new session
        """
        self.session_id = str(uuid.uuid4())[:8]  # Short UUID for readability
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO mct_sessions (session_id, started_at, status)
                VALUES (%s, %s, 'active')
            """, (self.session_id, self._get_vn_time()))
            
            print(f"✅ MCT Session started: {self.session_id}")
            
        except Exception as e:
            print(f"⚠️ Failed to start MCT session in DB: {e}")
            # Continue anyway - session_id is still valid for in-memory tracking
        
        return self.session_id
    
    def end_session(self):
        """End the current tracking session"""
        if not self.session_id:
            return
        
        # Stop background flush thread
        self._stop_flush.set()
        
        # Flush any remaining buffered data
        self._flush_buffers()
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE mct_sessions 
                SET ended_at = %s, 
                    status = 'stopped',
                    total_tracks = %s,
                    total_identified = %s
                WHERE session_id = %s
            """, (self._get_vn_time(), self.total_tracks, self.total_identified, self.session_id))
            
            print(f"✅ MCT Session ended: {self.session_id}")
            print(f"   Total tracks: {self.total_tracks}, Identified: {self.total_identified}")
            
        except Exception as e:
            print(f"⚠️ Failed to end MCT session in DB: {e}")
        
        finally:
            if self.conn:
                self.conn.close()
                self.conn = None
    
    def save_face_recognition(self, 
                               local_track_id: int,
                               usr_id: str,
                               floor: str,
                               camera_id: str = None,
                               confidence: float = None) -> bool:
        """
        Save a face recognition event.
        
        Args:
            local_track_id: The display_id within current session
            usr_id: Employee ID (e.g., 'INF1901002') or 'unknown'
            floor: Floor name (e.g., '3F', '1F')
            camera_id: Camera identifier
            confidence: Face recognition confidence score
            
        Returns:
            bool: True if saved (new recognition), False if duplicate
        """
        if not self.session_id:
            return False
        
        # Track stats
        if local_track_id not in self._known_tracks:
            self._known_tracks.add(local_track_id)
            self.total_tracks += 1
        
        if usr_id != 'unknown' and local_track_id not in self._identified_tracks:
            self._identified_tracks.add(local_track_id)
            self.total_identified += 1
        
        # Add to buffer
        record = {
            'session_id': self.session_id,
            'local_track_id': local_track_id,
            'usr_id': usr_id,
            'floor': floor,
            'camera_id': camera_id,
            'confidence': confidence,
            'detected_at': self._get_vn_time()
        }
        
        with self._lock:
            self._face_buffer.append(record)
            
            # Flush if buffer is full
            if len(self._face_buffer) >= self.BATCH_SIZE:
                self._flush_face_buffer()
        
        return True
    
    def save_position(self,
                      local_track_id: int,
                      usr_id: str,
                      floor: str,
                      x: float,
                      y: float,
                      camera_id: str = None,
                      bbox_center: Tuple[int, int] = None) -> bool:
        """
        Save position tracking data (with 5-second sampling).
        
        Args:
            local_track_id: The display_id within current session
            usr_id: Employee ID or 'unknown'
            floor: Floor name
            x: X coordinate on floor map (in mm)
            y: Y coordinate on floor map (in mm)
            camera_id: Camera identifier
            bbox_center: (x, y) pixel coordinates in camera frame
            
        Returns:
            bool: True if saved, False if skipped (sampling)
        """
        if not self.session_id:
            return False
        
        current_time = time.time()
        
        # Check sampling interval
        last_time = self._last_position_time.get(local_track_id, 0)
        if current_time - last_time < self.POSITION_SAMPLE_INTERVAL:
            return False  # Skip - not yet time
            
        # Check spatial movement (Phương án A - Spatial Filtering)
        last_coords = self._last_position_coords.get(local_track_id)
        if last_coords:
            last_x, last_y = last_coords
            dist = ((x - last_x)**2 + (y - last_y)**2)**0.5
            if dist < self.MIN_MOVEMENT_DISTANCE:
                return False  # Skip - moved less than threshold (e.g. 1 meter)
        
        # Update last save states
        self._last_position_time[local_track_id] = current_time
        self._last_position_coords[local_track_id] = (x, y)
        
        # Prepare record
        record = {
            'session_id': self.session_id,
            'local_track_id': local_track_id,
            'usr_id': usr_id,
            'floor': floor,
            'x': x,
            'y': y,
            'camera_id': camera_id,
            'bbox_center_x': bbox_center[0] if bbox_center else None,
            'bbox_center_y': bbox_center[1] if bbox_center else None,
            'tracked_at': self._get_vn_time()
        }
        
        with self._lock:
            self._position_buffer.append(record)
            
            # Flush if buffer is full
            if len(self._position_buffer) >= self.BATCH_SIZE:
                self._flush_position_buffer()
        
        return True
    
    def _flush_face_buffer(self):
        """Flush face recognition buffer to database"""
        if not self._face_buffer:
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            execute_batch(cursor, """
                INSERT INTO mct_face_recognition 
                (session_id, local_track_id, usr_id, floor, camera_id, confidence, detected_at)
                VALUES (%(session_id)s, %(local_track_id)s, %(usr_id)s, %(floor)s, 
                        %(camera_id)s, %(confidence)s, %(detected_at)s)
            """, self._face_buffer)
            
            count = len(self._face_buffer)
            self._face_buffer.clear()
            # print(f"💾 Flushed {count} face recognition records")
            
        except Exception as e:
            print(f"⚠️ Failed to flush face buffer: {e}")
    
    def _flush_position_buffer(self):
        """Flush position tracking buffer to database"""
        if not self._position_buffer:
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            execute_batch(cursor, """
                INSERT INTO mct_position_tracking 
                (session_id, local_track_id, usr_id, floor, x, y, 
                 camera_id, bbox_center_x, bbox_center_y, tracked_at)
                VALUES (%(session_id)s, %(local_track_id)s, %(usr_id)s, %(floor)s, 
                        %(x)s, %(y)s, %(camera_id)s, %(bbox_center_x)s, 
                        %(bbox_center_y)s, %(tracked_at)s)
            """, self._position_buffer)
            
            count = len(self._position_buffer)
            self._position_buffer.clear()
            # print(f"💾 Flushed {count} position records")
            
        except Exception as e:
            print(f"⚠️ Failed to flush position buffer: {e}")
    
    def save_desk_presence(self,
                           roi_id: int,
                           roi_owner: str,
                           floor: str,
                           is_present: bool) -> bool:
        """
        Save desk presence record.
        
        Args:
            roi_id: ROI ID from rois table
            roi_owner: Employee ID assigned to this desk (ROI name)
            floor: Floor name (e.g., '3F')
            is_present: True if someone is at the desk
            
        Returns:
            bool: True if saved
        """
        if not self.session_id:
            return False
        
        record = {
            'session_id': self.session_id,
            'roi_id': roi_id,
            'roi_owner': roi_owner,
            'floor': floor,
            'is_present': is_present,
            'checked_at': self._get_vn_time()
        }
        
        with self._lock:
            self._desk_buffer.append(record)
            
            if len(self._desk_buffer) >= self.BATCH_SIZE:
                self._flush_desk_buffer()
        
        return True
    
    def should_check_desk_presence(self) -> bool:
        """
        Check if enough time has passed for desk presence sampling (60s).
        Returns True if should check, and updates the timer.
        """
        current_time = time.time()
        if current_time - self._last_desk_presence_time >= self.DESK_PRESENCE_INTERVAL:
            self._last_desk_presence_time = current_time
            return True
        return False
    
    def _flush_desk_buffer(self):
        """Flush desk presence buffer to database"""
        if not self._desk_buffer:
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            execute_batch(cursor, """
                INSERT INTO mct_desk_presence 
                (session_id, roi_id, roi_owner, floor, is_present, checked_at)
                VALUES (%(session_id)s, %(roi_id)s, %(roi_owner)s, %(floor)s,
                        %(is_present)s, %(checked_at)s)
            """, self._desk_buffer)
            
            count = len(self._desk_buffer)
            self._desk_buffer.clear()
            # print(f"💾 Flushed {count} desk presence records")
            
        except Exception as e:
            print(f"⚠️ Failed to flush desk presence buffer: {e}")
    
    def _flush_buffers(self):
        """Flush all buffers"""
        with self._lock:
            self._flush_face_buffer()
            self._flush_position_buffer()
            self._flush_desk_buffer()
    
    def update_track_usr_id(self, local_track_id: int, usr_id: str):
        """
        Update usr_id for a track when face is recognized later.
        This updates all previous position records for this track.
        
        Note: This is optional - positions are already saved with current usr_id.
        Call this only if you want to retroactively update old 'unknown' records.
        """
        if not self.session_id or usr_id == 'unknown':
            return
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Update positions
            cursor.execute("""
                UPDATE mct_position_tracking 
                SET usr_id = %s
                WHERE session_id = %s 
                  AND local_track_id = %s 
                  AND usr_id = 'unknown'
            """, (usr_id, self.session_id, local_track_id))
            
            updated = cursor.rowcount
            if updated > 0:
                print(f"📝 Updated {updated} position records for track {local_track_id} -> {usr_id}")
                
        except Exception as e:
            print(f"⚠️ Failed to update track usr_id: {e}")

    def cleanup_expired_unknowns(self, hours_old: int = 12):
        """
        Delete 'unknown' position tracking records that are older than X hours.
        This keeps the database clean of unassigned tracking data from past periods.
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Delete records older than 'hours_old' that are still unknown
            cursor.execute("""
                DELETE FROM mct_position_tracking 
                WHERE usr_id = 'unknown' 
                  AND tracked_at < (NOW() AT TIME ZONE 'Asia/Ho_Chi_Minh' - INTERVAL '%s hours')
            """, (hours_old,))
            
            deleted = cursor.rowcount
            if deleted > 0:
                print(f"🧹 Cleaned up {deleted} expired 'unknown' position records (older than {hours_old}h)")
                
        except Exception as e:
            print(f"⚠️ Failed to cleanup expired unknowns: {e}")

# Singleton instance for easy import
_tracker_instance: Optional[MCTTracker] = None


def get_mct_tracker() -> MCTTracker:
    """Get the global MCT tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = MCTTracker()
    return _tracker_instance
