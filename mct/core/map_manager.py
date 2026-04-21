import traceback
import time
from datetime import datetime


def update_maps_and_broadcast(mappers, confirmed_tracks, stream_to_cam_id,
                              cam_id_to_floor, reid_index):
    """
    Update floor maps with tracking data and broadcast via WebSocket.
    Also logs positions to MCT database.
    
    Args:
        mappers: Dict of {floor_num: Map}
        confirmed_tracks: Dict of {cam_idx: [ConfirmedTrack, ...]}
        stream_to_cam_id: Dict of {stream_idx: cam_id}
        cam_id_to_floor: Dict of {cam_id: floor_num}
        reid_index: ReIDIndex instance for face name lookup
    """
    import api_server
    
    if not mappers:
        return
    
    # Collect tracking points grouped by floor
    camera_points_by_floor = {floor_num: {} for floor_num in mappers}
    
    for cam_idx, tracks in confirmed_tracks.items():
        cam_idx_int = int(cam_idx)
        cam_id = stream_to_cam_id.get(cam_idx_int)
        if not cam_id:
            continue
        
        # Skip disabled cameras
        if not api_server.is_camera_enabled(cam_id):
            continue
        
        points = []
        for track in tracks:
            if track.miss_count > 0:
                continue
            
            face_name = reid_index.get_face_name(track.display_id) or "Unknown"
            
            x1, y1, x2, y2 = track.box
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            
            # map.py expects: (pid, u, v) or dict with face_name
            points.append({
                "id": track.display_id,
                "u": center[0],
                "v": center[1],
                "face_name": face_name
            })
        
        if points:
            floor = cam_id_to_floor.get(cam_id)
            if floor is not None and floor in camera_points_by_floor:
                camera_points_by_floor[floor][cam_id] = points
            else:
                # Debug logging for missing floor mapping
                if time.time() % 30 < 0.1:
                    print(f"⚠️ Track found but no floor mapping for cam {cam_id} (Floor: {floor})")

    # Check desk presence timing once (applies to all floors)
    try:
        from database.mct_tracking import get_mct_tracker
        should_check_desk = get_mct_tracker().should_check_desk_presence()
    except ImportError:
        should_check_desk = False

    # Update each floor's map and broadcast
    for floor_num, mapper in mappers.items():
        # Skip disabled floors
        if not api_server.is_floor_enabled(floor_num):
            continue
        
        camera_points = camera_points_by_floor.get(floor_num, {})
        
        try:
            # 1. Clear & Project
            mapper.projected = []
            mapper.image_to_map(camera_points)
            
            # 2. Merge
            merged_points = mapper.merge_points(distance_mm=130)
            
            # 3. Check ROIs
            roi_status = mapper.find_points_in_roi(merged_points)
            
            # 4. Construct Payload
            payload = {
                "points": merged_points,
                "rois": roi_status,
                "timestamp": datetime.now().isoformat()
            }
            
            # 5. Broadcast via WebSocket
            api_server.send_update(floor_num, payload)
            
            # 6. MCT DB Logging
            _log_positions_to_db(floor_num, merged_points, reid_index)
            
            # 7. Desk Presence Logging (every 1 minute)
            if should_check_desk:
                _log_desk_presence_to_db(floor_num, roi_status, mapper)
            
        except Exception as e:
            print(f"❌ Map Floor {floor_num} Update Error: {e}")
            traceback.print_exc()

    # Debug: Print summary of broadcast
    active_floors = [f for f in mappers if api_server.is_floor_enabled(f)]
    points_found = sum(len(camera_points_by_floor.get(f, {})) for f in mappers)
    if points_found > 0 or time.time() % 30 < 0.1:
        fps_info = "" # Can add FPS if needed
        print(f"📡 Broadcast: {len(active_floors)} floors active. Tracks found on {points_found} floor-cam pairs.")


def _log_positions_to_db(floor_num, merged_points, reid_index):
    """
    Log merged tracking positions to MCT database.
    
    Args:
        floor_num: Floor number
        merged_points: List of merged point dicts from map
        reid_index: ReIDIndex instance for face name lookup
    """
    try:
        from database.mct_tracking import get_mct_tracker
    except ImportError:
        return
    
    try:
        tracker = get_mct_tracker()
        floor_name = f"{floor_num}F"
        
        for p in merged_points:
            pid = p.get('point_id')
            wx, wy = p.get('world_mm', (0, 0))
            
            # Resolve usr_id
            usr_id = 'unknown'
            face_name = p.get('face_name')
            if face_name and face_name != 'Unknown':
                usr_id = face_name
            else:
                known = reid_index.get_face_name(pid)
                if known:
                    usr_id = known
            
            # Save Position
            tracker.save_position(
                local_track_id=pid,
                usr_id=usr_id,
                floor=floor_name,
                x=float(wx),
                y=float(wy),
                camera_id=p.get('cameras', [None])[0],
                bbox_center=p.get('map_px')
            )
            
            # Retroactively update usr_id if identified
            if usr_id != 'unknown':
                tracker.update_track_usr_id(pid, usr_id)
    except Exception as e:
        print(f"Warning: Failed to log positions to DB: {e}")


def _log_desk_presence_to_db(floor_num, roi_status, mapper):
    """
    Log desk presence status for named ROIs to MCT database.
    Called only when the 60-second sampling interval has passed.
    
    Args:
        floor_num: Floor number
        roi_status: Dict {roi_id: True/False} from find_points_in_roi()
        mapper: Map instance with roi_id_map containing ROI labels
    """
    try:
        from database.mct_tracking import get_mct_tracker
    except ImportError:
        return
    
    try:
        tracker = get_mct_tracker()
        floor_name = f"{floor_num}F"
        
        for roi_id, is_present in roi_status.items():
            # Get ROI info from mapper
            roi_info = mapper.roi_id_map.get(roi_id)
            if not roi_info:
                continue
            
            roi_label = roi_info.get('label', '')
            
            # Determine owner:
            # - Named ROIs (e.g., INF2503004) → use label directly
            # - Unnamed ROIs → use "ROI_{id}" so 2F data is not lost
            if roi_label and roi_label.startswith('INF'):
                roi_owner = roi_label
            elif not roi_label or roi_label in ('Unnamed Region', ''):
                # Log unnamed ROIs with generic ID so 2F presence is tracked
                roi_owner = f"ROI_{roi_id}"
            else:
                # Named but not INF-format (e.g. zone names) — skip
                continue
            
            tracker.save_desk_presence(
                roi_id=roi_id,
                roi_owner=roi_owner,
                floor=floor_name,
                is_present=is_present
            )
    except Exception as e:
        print(f"Warning: Failed to log desk presence to DB: {e}")
