"""
Logging utility functions for MCT system.
Includes floor-based file logging and face detection logging.
"""
import os
import logging
from logging.handlers import RotatingFileHandler

from .time_utils import get_vn_time
from .file_utils import load_json_file, save_json_file

# --- Custom Logger Setup ---
_loggers = {}
_yolo_logger = None


def get_yolo_logger():
    """
    Get or create a logger dedicated to YOLO detection events.
    Logs to both console (WARNING+) and file (DEBUG+) at ./logs/yolo_detection.log.
    
    Returns:
        logging.Logger instance
    """
    global _yolo_logger
    if _yolo_logger is not None:
        return _yolo_logger

    logger = logging.getLogger("yolo_detection")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.handlers:
        os.makedirs("./logs", exist_ok=True)

        # File handler — captures everything (DEBUG+)
        file_handler = RotatingFileHandler(
            "./logs/yolo_detection.log",
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )

        # Console handler — show WARNING+ so terminal is not flooded
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(
            logging.Formatter("[YOLO] %(asctime)s %(message)s", datefmt="%H:%M:%S")
        )

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    _yolo_logger = logger
    return _yolo_logger


def get_floor_logger(floor_id):
    """
    Get or create a rotating file logger for a specific floor.
    
    Args:
        floor_id: Floor identifier (e.g. 1, 3)
    
    Returns:
        logging.Logger instance
    """
    if floor_id not in _loggers:
        logger = logging.getLogger(f"Floor_{floor_id}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            log_file = f"./logs/floor_{floor_id}.log"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            handler = RotatingFileHandler(
                log_file, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8'
            )
            formatter = logging.Formatter('%(asctime)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.propagate = False
            
        _loggers[floor_id] = logger
    return _loggers[floor_id]


def remove_expired_names(log_dir, expire_seconds=60):
    """Remove entries older than expire_seconds from face_logs.json."""
    face_logs_path = os.path.join(log_dir, "face_logs.json")
    face_logs = load_json_file(face_logs_path)
    
    current_time = get_vn_time()
    updated_logs = {}
    
    for name, data in face_logs.items():
        try:
            if isinstance(data, str):
                timestamp_str = data
            elif isinstance(data, dict):
                timestamp_str = data.get('timestamp', data.get('time', ''))
            else:
                continue
                
            from datetime import datetime
            log_time = datetime.fromisoformat(timestamp_str)
            if (current_time - log_time).total_seconds() < expire_seconds:
                if isinstance(data, dict):
                    updated_logs[name] = data
                else:
                    updated_logs[name] = {"timestamp": timestamp_str, "reid_id": None}
        except Exception:
            pass
    
    save_json_file(face_logs_path, updated_logs)
    return updated_logs


def log_face_detection(name, cam_name, log_dir, person_id=None, floor=None):
    """
    Log face detection to face_logs.json and backup_db_*.json
    
    Args:
        name: Person name (e.g., 'nv001')
        cam_name: Camera name (e.g., 'cam01')
        log_dir: Directory to store logs (e.g., './logs/cam01')
        person_id: ReID person ID (e.g., 5)
    
    Returns:
        bool: True if logged (new detection), False if already logged recently
    """
    os.makedirs(log_dir, exist_ok=True)
    
    face_logs = remove_expired_names(log_dir, expire_seconds=60)
    
    if name in face_logs:
        return False
    
    current_time = get_vn_time()
    time_str = current_time.isoformat()
    
    face_logs[name] = {
        "timestamp": time_str,
        "reid_id": person_id
    }
    face_logs_path = os.path.join(log_dir, "face_logs.json")
    save_json_file(face_logs_path, face_logs)
    
    backup_filename = f"backup_db_{current_time.strftime('%m%Y')}.json"
    backup_path = os.path.join(log_dir, backup_filename)
    
    backup_data = load_json_file(backup_path)
    if not isinstance(backup_data, list):
        backup_data = []
    
    entry = {
        "WriteDate": time_str,
        "SN": cam_name,
        "Pin": name,
        "AttTime": time_str
    }
    
    if person_id is not None:
        entry["ReID_ID"] = person_id
    
    backup_data.append(entry)
    save_json_file(backup_path, backup_data)
    
    # --- MCT DB LOGGING: Face Recognition ---
    if person_id is not None:
        print(f"📝 Logged: {name} (ID:{person_id}) at {cam_name} - {current_time.strftime('%H:%M:%S')}")
        
        try:
            from database.mct_tracking import get_mct_tracker
            tracker = get_mct_tracker()
            tracker.save_face_recognition(
                local_track_id=person_id,
                usr_id=name,
                floor=floor or "Unknown",
                camera_id=cam_name,
                confidence=1.0
            )
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Failed to log face to DB: {e}")
    else:
        print(f"📝 Logged: {name} at {cam_name} - {current_time.strftime('%H:%M:%S')}")
    
    return True
