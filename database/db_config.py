"""
Database Configuration Loader for Multi-Camera Tracking System

This module provides functions to load camera, floor, and ROI configurations
from a PostgreSQL database instead of hardcoding them in the application.

Usage:
    from database.db_config import load_all_config
    
    config = load_all_config()
    cameras = config['cameras']
    floors = config['floors']
    rois = config['rois']
"""

import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# =============================================================================
# Configuration DataClasses
# =============================================================================

@dataclass
class CameraConfig:
    """Camera configuration from database"""
    camera_id: str
    name: str
    ip: str
    port: str
    username: str
    password: str
    floor: str          # e.g., "3F", "1F"
    floor_num: int      # Numeric floor: 3, 1, etc.
    region_id: str
    inout: str          # "IN" or "OUT"
    resolution: str = "1920x1080"
    fps: int = 30
    
    @property
    def rtsp_url(self) -> str:
        """Generate RTSP URL for this camera"""
        return f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}/cam/realmonitor?channel=1&subtype=00"
    
    @property
    def cam_id_str(self) -> str:
        """Generate camera ID string for internal use (uses IP for easy mapping)"""
        return self.ip


@dataclass
class FloorConfig:
    """Floor configuration"""
    floor_id: int
    name: str           # e.g., "3F", "7F"
    floor_num: int      # Numeric: 3, 7
    map_img_path: Optional[str] = None
    mm_per_pixel_x: float = 23.8
    mm_per_pixel_y: float = 23.3
    cameras: List[CameraConfig] = field(default_factory=list)
    rois: List[dict] = field(default_factory=list)


@dataclass
class ROIConfig:
    """ROI (Region of Interest) configuration"""
    roi_id: int
    name: str
    floor_id: int
    points_px: List[List[float]]
    angle: float = 0.0


# =============================================================================
# Database Connection
# =============================================================================

class DatabaseConfig:
    """Database connection configuration"""
    
    # Default connection parameters (can be overridden by environment variables)
    DEFAULT_HOST = "10.29.8.49"
    DEFAULT_PORT = 5432
    DEFAULT_USER = "infiniq_user"
    DEFAULT_PASSWORD = "infiniq_pass"
    DEFAULT_DATABASE = "camera_ai_db"
    
    @classmethod
    def get_connection_params(cls) -> dict:
        """Get database connection parameters from environment or defaults"""
        return {
            'host': os.environ.get('DB_HOST', cls.DEFAULT_HOST),
            'port': int(os.environ.get('DB_PORT', cls.DEFAULT_PORT)),
            'user': os.environ.get('DB_USER', cls.DEFAULT_USER),
            'password': os.environ.get('DB_PASSWORD', cls.DEFAULT_PASSWORD),
            'database': os.environ.get('DB_NAME', cls.DEFAULT_DATABASE),
        }
    
    @classmethod
    def get_connection(cls):
        """Get a database connection"""
        params = cls.get_connection_params()
        return psycopg2.connect(
            host=params['host'],
            port=params['port'],
            user=params['user'],
            password=params['password'],
            dbname=params['database'],
            cursor_factory=RealDictCursor
        )


# =============================================================================
# Data Loaders
# =============================================================================

def parse_floor_number(floor_str: str) -> int:
    """
    Parse floor string to numeric value.
    
    Examples:
        "3F" -> 3
        "1F" -> 1
        "7F" -> 7
        "LG" -> 0
        "B1" -> -1
    """
    if not floor_str:
        return 0
    
    floor_str = floor_str.strip().upper()
    
    # Handle special cases
    if floor_str in ("LG", "GF", "G"):
        return 0
    if floor_str.startswith("B"):
        # Basement floors
        try:
            return -int(floor_str[1:])
        except ValueError:
            return -1
    
    # Standard floors: "1F", "2F", etc.
    try:
        return int(floor_str.replace("F", "").strip())
    except ValueError:
        return 0


def load_cameras(conn=None, floor_filter: Optional[List[str]] = None) -> List[CameraConfig]:
    """
    Load camera configurations from database.
    
    Args:
        conn: Database connection (optional, will create one if not provided)
        floor_filter: Optional list of floor names to filter (e.g., ["3F", "1F"])
    
    Returns:
        List of CameraConfig objects
    """
    close_conn = False
    if conn is None:
        conn = DatabaseConfig.get_connection()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        query = """
            SELECT 
                c.camera_id,
                c.cam_nm,
                c.cam_ip,
                c.cam_port,
                c.cam_usr,
                c.cam_pass,
                c.inout,
                c.region_id,
                c.resolution,
                c.fts,
                r.floor
            FROM t_cma_cam c
            JOIN t_cma_region r ON c.region_id = r.region_id
            WHERE c.del_yn = 'N'
        """
        
        if floor_filter:
            placeholders = ','.join(['%s'] * len(floor_filter))
            query += f" AND r.floor IN ({placeholders})"
            cursor.execute(query, floor_filter)
        else:
            cursor.execute(query)
        
        rows = cursor.fetchall()
        
        cameras = []
        for row in rows:
            camera = CameraConfig(
                camera_id=str(row['camera_id']),
                name=row['cam_nm'] or f"Camera {row['camera_id']}",
                ip=row['cam_ip'],
                port=str(row['cam_port']),
                username=row['cam_usr'],
                password=row['cam_pass'],
                floor=row['floor'],
                floor_num=parse_floor_number(row['floor']),
                region_id=str(row['region_id']),
                inout=row['inout'] or 'IN',
                resolution=row['resolution'] or '1920x1080',
                fps=int(row['fts']) if row['fts'] else 30
            )
            cameras.append(camera)
        
        print(f"✅ Loaded {len(cameras)} cameras from database")
        return cameras
        
    finally:
        if close_conn:
            conn.close()


def load_floors(conn=None) -> Dict[int, FloorConfig]:
    """
    Load floor configurations from database.
    
    Returns:
        Dict mapping floor_id to FloorConfig
    """
    close_conn = False
    if conn is None:
        conn = DatabaseConfig.get_connection()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name
            FROM floors
            ORDER BY id
        """)
        
        rows = cursor.fetchall()
        
        floors = {}
        for row in rows:
            floor_id = int(row['id'])
            name = row['name']
            floor_num = parse_floor_number(name)
            
            # Simple default mapping for map images
            # This matches the directory structure: image2map/F{id}/{id}f.png
            map_img_path = None
            if floor_num > 0:
                 map_img_path = f"image2map/F{floor_num}/{floor_num}f.png"
            
            floor = FloorConfig(
                floor_id=floor_id,
                name=name,
                floor_num=floor_num,
                map_img_path=map_img_path
            )
            floors[floor.floor_id] = floor
        
        print(f"✅ Loaded {len(floors)} floors from database")
        return floors
        
    finally:
        if close_conn:
            conn.close()


def load_rois(conn=None, floor_id: Optional[int] = None) -> List[ROIConfig]:
    """
    Load ROI configurations from database.
    
    Args:
        conn: Database connection
        floor_id: Optional floor ID to filter by
    
    Returns:
        List of ROIConfig objects
    """
    close_conn = False
    if conn is None:
        conn = DatabaseConfig.get_connection()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        query = "SELECT id, name, floor_id, points_px FROM rois"
        if floor_id is not None:
            query += " WHERE floor_id = %s"
            cursor.execute(query, (str(floor_id),))
        else:
            cursor.execute(query)
        
        rows = cursor.fetchall()
        
        rois = []
        for row in rows:
            # Parse points_px (stored as JSON string)
            points_str = row['points_px']
            if isinstance(points_str, str):
                points = json.loads(points_str)
            else:
                points = points_str
            
            roi = ROIConfig(
                roi_id=int(row['id']),
                name=row['name'] or f"ROI {row['id']}",
                floor_id=int(row['floor_id']),
                points_px=points
            )
            rois.append(roi)
        
        print(f"✅ Loaded {len(rois)} ROIs from database")
        return rois
        
    finally:
        if close_conn:
            conn.close()


def load_zones(conn=None, floor_id: Optional[int] = None) -> List[dict]:
    """
    Load zone configurations from database.
    
    Zones are larger areas (teams, departments) compared to ROIs (individual desks).
    
    Returns:
        List of zone dictionaries
    """
    close_conn = False
    if conn is None:
        conn = DatabaseConfig.get_connection()
        close_conn = True
    
    try:
        cursor = conn.cursor()
        
        query = "SELECT id, name, floor_id, coords FROM zones"
        if floor_id is not None:
            query += " WHERE floor_id = %s"
            cursor.execute(query, (str(floor_id),))
        else:
            cursor.execute(query)
        
        rows = cursor.fetchall()
        
        zones = []
        for row in rows:
            coords = row['coords']
            if isinstance(coords, str):
                coords = json.loads(coords)
            
            zone = {
                'id': int(row['id']),
                'name': row['name'],
                'floor_id': int(row['floor_id']),
                'coords': coords
            }
            zones.append(zone)
        
        print(f"✅ Loaded {len(zones)} zones from database")
        return zones
        
    finally:
        if close_conn:
            conn.close()


# =============================================================================
# Main Config Loader
# =============================================================================

def load_all_config(floor_filter: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load all configurations from database.
    
    Args:
        floor_filter: Optional list of floor names to load (e.g., ["3F", "1F"])
    
    Returns:
        Dictionary containing:
            - cameras: List[CameraConfig]
            - floors: Dict[int, FloorConfig]
            - rois: List[ROIConfig]
            - zones: List[dict]
            - cameras_by_floor: Dict[str, List[CameraConfig]] - grouped by floor name
    """
    print("\n" + "="*60)
    print("📦 Loading Configuration from Database")
    print("="*60)
    
    conn = DatabaseConfig.get_connection()
    
    try:
        cameras = load_cameras(conn, floor_filter)
        floors = load_floors(conn)
        rois = load_rois(conn)
        zones = load_zones(conn)
        
        # Group cameras by floor
        cameras_by_floor: Dict[str, List[CameraConfig]] = {}
        for cam in cameras:
            if cam.floor not in cameras_by_floor:
                cameras_by_floor[cam.floor] = []
            cameras_by_floor[cam.floor].append(cam)
        
        # Associate cameras and ROIs with floors
        for cam in cameras:
            for floor in floors.values():
                if floor.name == cam.floor:
                    floor.cameras.append(cam)
                    break
        
        for roi in rois:
            if roi.floor_id in floors:
                floors[roi.floor_id].rois.append({
                    'id': roi.roi_id,
                    'label': roi.name,
                    'points_px': roi.points_px,
                    'angle': roi.angle
                })
        
        print("="*60 + "\n")
        
        return {
            'cameras': cameras,
            'floors': floors,
            'rois': rois,
            'zones': zones,
            'cameras_by_floor': cameras_by_floor
        }
        
    finally:
        conn.close()


def generate_active_cameras_list(cameras: List[CameraConfig]) -> List[dict]:
    """
    Generate active_cameras list compatible with MCT system format.
    
    Camera ID uses IP address for easy mapping to calibration folders.
    
    Returns:
        List of dicts with format:
        {'id': '10.29.98.52', 'url': 'rtsp://...', 'floor': 3, ...}
    """
    active_cameras = []
    
    for cam in cameras:
        active_cameras.append({
            'id': cam.cam_id_str,   # IP address (e.g., '10.29.98.52')
            'url': cam.rtsp_url,
            'floor': cam.floor_num,
            'floor_name': cam.floor,
            'name': cam.name,
            'inout': cam.inout,
            'region_id': cam.region_id,
            'ip': cam.ip
        })
    
    return active_cameras


# =============================================================================
# Test / CLI
# =============================================================================

if __name__ == "__main__":
    # Test loading configuration
    try:
        config = load_all_config()
        
        print("\n📷 Cameras by Floor:")
        for floor, cams in config['cameras_by_floor'].items():
            print(f"\n  {floor}:")
            for cam in cams:
                print(f"    - {cam.name} ({cam.ip}) [{cam.inout}]")
                print(f"      RTSP: {cam.rtsp_url}")
        
        print("\n🏢 Floors:")
        for floor_id, floor in config['floors'].items():
            print(f"  - {floor.name} (ID: {floor_id})")
            print(f"    Cameras: {len(floor.cameras)}, ROIs: {len(floor.rois)}")
        
        # Generate active_cameras list
        print("\n🎯 Generated active_cameras list:")
        active_cams = generate_active_cameras_list(config['cameras'])
        for cam in active_cams[:5]:  # Show first 5
            print(f"  - {cam}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
