"""
MCT Engine — Main demo loop for Multi-Camera Tracking.
Orchestrates model initialization, camera streams, frame processing,
map updates, and WebSocket broadcasting.
"""
import os
import sys
import time
import queue
import threading
import traceback

import cv2
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.stream import ThreadedStream
from ..core.frame_processor import FrameProcessor
from ..core.camera_config import (
    initialize_maps,
    load_cameras_from_db,
    load_cameras_from_args,
    register_cameras_to_maps,
)
from ..core.map_manager import update_maps_and_broadcast
from ..reid.model import setup_transreid, get_transforms
from ..reid.reid_index import ReIDIndex
from ..face.detector import setup_face_api
from ..face.indexer import rebuild_face_index
from ..utils.file_utils import monitor_face_directory


def run_demo(args):
    """
    Main entry point for the Multi-Camera Tracking system.
    
    Args:
        args: Parsed argparse Namespace with config_file, weights,
              use_db, floors, rtsp URLs, etc.
    """
    import api_server
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- MCT DB LOGGING: Start Session ---
    mct_tracker = None
    try:
        from database.mct_tracking import get_mct_tracker
        mct_tracker = get_mct_tracker()
        session_id = mct_tracker.start_session()
        print(f"🚀 MCT Session Started: {session_id}")
    except ImportError:
        print("⚠️ MCT Tracking module not available")
    except Exception as e:
        print(f"⚠️ MCT Session start failed: {e}")

    # 1. Initialize YOLO Model Pool (Dynamic based on camera count)
    # We will instantiate them later after counting cameras.


    # 2. Initialize TransReID
    print("Loading TransReID...")
    reid_model = setup_transreid(args.config_file, args.weights, device)
    reid_transform = get_transforms()

    # 3. Initialize Face API with FAISS index
    face_detector, face_recognizer, face_targets, face_index, face_id_to_name = setup_face_api()
    
    # 3.1 Setup Face Directory Monitoring
    faces_dir_path = os.path.abspath("./API_Face/faces/")
    face_update_queue = queue.Queue()
    face_resources_lock = threading.Lock()
    
    monitor_thread = threading.Thread(
        target=monitor_face_directory,
        args=(faces_dir_path, face_update_queue, 10),
        daemon=True
    )
    monitor_thread.start()

    # 3.2 Initialize Maps for ALL Floors
    mappers = initialize_maps("image2map")
        
    # 4. Load Camera Configuration
    active_cameras = []
    
    if getattr(args, 'use_db', False):
        active_cameras = load_cameras_from_db(args)
    
    if not active_cameras:
        active_cameras = load_cameras_from_args(args)
    
    # 5. Register Cameras to Maps
    cam_id_to_floor = register_cameras_to_maps(active_cameras, mappers, "image2map")

    # 6. Open RTSP Streams
    streams = []
    stream_to_cam_id = {}
    connected_count = 0
    failed_count = 0
    
    for idx, cam in enumerate(active_cameras):
        cam_id = cam['id']  # IP address
        cam_name = cam.get('name', cam_id)
        url = cam['url']
        stream = ThreadedStream(url)
        if stream.status:
            print(f"   ✅ {cam_id} ({cam_name}, Floor {cam['floor']}) — Connected")
            connected_count += 1
        else:
            print(f"   ❌ {cam_id} ({cam_name}, Floor {cam['floor']}) — FAILED to connect")
            print(f"      URL: {url}")
            failed_count += 1
        streams.append(stream)
        stream_to_cam_id[idx] = cam_id
    
    print(f"\n📊 Stream Summary: {connected_count} connected, {failed_count} failed, {len(active_cameras)} total")
    
    stream_to_cam_info = {idx: cam for idx, cam in enumerate(active_cameras)}

    # --- DYNAMIC YOLO POOL ---
    # Create N YOLO models based on connected cameras to distribute load without GIL bottleneck
    from ultralytics import YOLO
    connected = sum(1 for cam in active_cameras)
    num_yolo_models = max(1, connected // 3)  # 1 model per 3 cameras
    print(f"Loading {num_yolo_models} YOLO instances to distribute load...")
    yolo_pool = []
    _dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    for i in range(num_yolo_models):
        y_model = YOLO('yolo26s_person_train16_v3.pt')
        y_model.to(device)
        y_model.predict(_dummy, classes=[3], verbose=False, device=device)
        yolo_pool.append(y_model)

    # Detect the correct class index for 'person' in this model
    _sample_model = yolo_pool[0]
    _person_class_id = 3  # Hardcoded to class 3 as requested
    print(f"✅ YOLO pool warmup complete — model classes: {_sample_model.names}")
    print(f"   Person class index: {_person_class_id} (Forced)")

    # 7. Initialize ReID Index & Frame Processor
    reid_index = ReIDIndex()

    processor = FrameProcessor(
        yolo=yolo_pool,
        yolo_person_class=_person_class_id,
        reid_model=reid_model,
        reid_transform=reid_transform,
        device=device,
        face_detector=face_detector,
        face_recognizer=face_recognizer,
        face_resources_lock=face_resources_lock,
        reid_index=reid_index,
        cam_id_to_floor=cam_id_to_floor,
        stream_to_cam_id=stream_to_cam_id,
        stream_to_cam_info=stream_to_cam_info,
    )
    processor.set_face_resources(face_index, face_id_to_name)

    # 8. Start Main Loop
    print("\n🚀 Starting Dual-Recognition System (Face + Body ReID) - Multi-threaded")
    print("Press Ctrl+C to exit.")
    
    max_workers = len(streams)
    executor = ThreadPoolExecutor(max_workers=max_workers)
    
    # Initialize floor/camera status for API management
    api_server.init_floor_camera_status(active_cameras)
    
    # Start API Server
    try:
        api_thread = threading.Thread(
            target=api_server.start_api_server,
            kwargs={'host': '0.0.0.0', 'port': 8068}
        )
        api_thread.daemon = True
        api_thread.start()
        print("✅ WebSocket API Server started on port 8068")
        print("   📡 REST API: http://0.0.0.0:8068/api/status")
        print("   📡 WebSocket: ws://0.0.0.0:8068/ws/{floor_id}")
    except Exception as e:
        print(f"❌ Failed to start API Server: {e}")

    # --- MAIN PROCESSING LOOP ---
    try:
        # --- Track Loop Stats ---
        stats = {'f_count': 0, 's_time': time.time()}

        while True:
            # Check for face directory updates
            if not face_update_queue.empty():
                try:
                    signal = face_update_queue.get_nowait()
                    if signal == 'reload_faces':
                        print("\n🔄 Reloading face embeddings...")
                        new_targets, new_index, new_id_to_name = rebuild_face_index(
                            face_detector, face_recognizer, faces_dir_path
                        )
                        with face_resources_lock:
                            face_targets = new_targets
                            face_index = new_index
                            face_id_to_name = new_id_to_name
                        processor.set_face_resources(face_index, face_id_to_name)
                        print("✅ Face embeddings reloaded successfully!\n")
                except queue.Empty:
                    pass
            
            # --- MAIN PROCESSING LOOP ---
            
            # Read frames from all cameras
            frames = []
            for i, stream in enumerate(streams):
                ret, frame = stream.read()
                if not ret or frame is None:
                    frames.append(None)
                    continue
                frames.append(frame.copy())

            # Process cameras in parallel with distributed YOLO models
            futures = {}
            for cam_idx, frame in enumerate(frames):
                if frame is None:
                    continue
                cam_id = stream_to_cam_id.get(cam_idx)
                if cam_id and not api_server.is_camera_enabled(cam_id):
                    continue
                future = executor.submit(processor.process_camera_frame, cam_idx, frame)
                futures[future] = cam_idx
            
            # Collect results
            for future in as_completed(futures):
                cam_idx = futures[future]
                try:
                    processed_frame = future.result()
                except Exception as e:
                    print(f"Error processing camera {cam_idx+1}: {e}")
                    traceback.print_exc()

            # Update Maps & Broadcast WebSocket
            update_maps_and_broadcast(
                mappers, processor.confirmed_tracks,
                stream_to_cam_id, cam_id_to_floor, reid_index
            )

            # --- Update Stats ---
            stats['f_count'] += 1
            if stats['f_count'] % 30 == 0:
                elapsed = time.time() - stats['s_time']
                fps = stats['f_count'] / elapsed
                print(f"⏱️ Heartbeat: {stats['f_count']} frames, {fps:.2f} FPS")
                
                # --- NEW: Print detection summary per camera ---
                summary_strs = []
                # ensure we print them in order of cam_idx
                for c_idx in sorted(processor.confirmed_tracks.keys()):
                    tracks = processor.confirmed_tracks[c_idx]
                    c_id = stream_to_cam_id.get(c_idx, f"cam{c_idx}")
                    active_count = sum(1 for t in tracks if t.miss_count == 0)
                    summary_strs.append(f"{c_id}: {active_count} người")
                if summary_strs:
                    print(f"   👥 Tracking: " + " | ".join(summary_strs))

            time.sleep(0.03)  # Cap at ~30 FPS
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        # Cleanup
        if mct_tracker:
            try:
                mct_tracker.end_session()
            except Exception:
                pass
        
        executor.shutdown(wait=True)
        
        for stream in streams:
            stream.stop()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete.")
