"""
Frame processing logic for MCT system.
Contains the FrameProcessor class that handles per-camera-frame
Face Recognition + Person Detection + ReID matching + Drawing.
"""
import cv2
import numpy as np
import threading
from PIL import Image

from ..utils.geometry import compute_iou, is_face_inside_body
from ..utils.logging_utils import get_floor_logger, get_yolo_logger, log_face_detection
from ..face.detector import run_face_recognition
from ..reid.features import extract_features_batch
from ..core.tracker import PendingTrack, ConfirmedTrack

        # Constants
CONFIRM_FRAMES = 2        # Frames before assigning new ID
MAX_MISS_FRAMES = 1000    # Delete track if lost for N frames

# Log YOLO detection summary every N seconds per camera
_YOLO_LOG_INTERVAL = 5   # seconds


class FrameProcessor:
    """
    Processes individual camera frames with Face Recognition + ReID.
    
    Holds references to all shared resources (models, indexes, locks)
    and provides a thread-safe process_camera_frame() method.
    """
    
    def __init__(self, yolo, reid_model, reid_transform, device,
                 face_detector, face_recognizer, face_resources_lock,
                 reid_index, cam_id_to_floor, stream_to_cam_id,
                 stream_to_cam_info, yolo_person_class=3):
        """
        Args:
            yolo: YOLO model (or list of models) for person detection
            yolo_person_class: Class index that corresponds to 'person' in the YOLO model
            reid_model: TransReID model
            reid_transform: Image transform for ReID
            device: torch device string
            face_detector: SCRFD face detector
            face_recognizer: ArcFace recognizer
            face_resources_lock: threading.Lock for face_index/face_id_to_name
            reid_index: ReIDIndex instance
            cam_id_to_floor: Dict {cam_id: floor_num}
            stream_to_cam_id: Dict {stream_idx: cam_id}
            stream_to_cam_info: Dict {stream_idx: camera_info_dict}
        """
        self.yolo = yolo
        self.yolo_person_class = yolo_person_class
        self.reid_model = reid_model
        self.reid_transform = reid_transform
        self.device = device
        self.face_detector = face_detector
        self.face_recognizer = face_recognizer
        self.face_resources_lock = face_resources_lock
        self.reid_index = reid_index
        self.cam_id_to_floor = cam_id_to_floor
        self.stream_to_cam_id = stream_to_cam_id
        self.stream_to_cam_info = stream_to_cam_info
        
        # Per-camera tracking state
        self.pending_tracks = {}      # {cam_idx: [PendingTrack, ...]}
        self.confirmed_tracks = {}    # {cam_idx: [ConfirmedTrack, ...]}

        # YOLO detection logging state per camera
        self._yolo_last_log = {}      # {cam_idx: last_log_timestamp}
        self._yolo_frame_count = {}   # {cam_idx: total frames processed}
        self._yolo_det_count = {}     # {cam_idx: total detections since last log}
        self._yolo_zero_frames = {}   # {cam_idx: consecutive frames with 0 detections}
        
        # Current face resources (updated atomically via lock)
        self.face_index = None
        self.face_id_to_name = {}

    def set_face_resources(self, face_index, face_id_to_name):
        """Update face index and name mapping (called after reload)."""
        self.face_index = face_index
        self.face_id_to_name = face_id_to_name

    def process_camera_frame(self, cam_idx, frame):
        """
        Process a single camera frame with Face Recognition + ReID.
        Thread-safe.
        
        Args:
            cam_idx: Camera index
            frame: BGR numpy array
        
        Returns:
            Processed frame with annotations drawn
        """
        # Determine Floor and Logger
        cam_id_str = self.stream_to_cam_id.get(cam_idx, f"cam{cam_idx+1}")
        floor_id = self.cam_id_to_floor.get(cam_id_str, 3)
        data_logger = get_floor_logger(floor_id)

        # --- STEP A: Run Face Recognition ---
        with self.face_resources_lock:
            current_face_index = self.face_index
            current_face_id_to_name = self.face_id_to_name
        
        face_results = run_face_recognition(
            frame, self.face_detector, self.face_recognizer,
            current_face_index, current_face_id_to_name
        )
        
        # --- STEP B: Person Detection & ReID ---
        if cam_idx not in self.pending_tracks:
            self.pending_tracks[cam_idx] = []
        
        cam_info = self.stream_to_cam_info.get(cam_idx, {})
        cam_name = cam_info.get('name') or cam_info.get('ip') or f"cam{cam_idx+1:02d}"
        cam_name_safe = cam_name.replace(' ', '_').replace('/', '_')
        log_dir = f"./logs/{cam_name_safe}"

        assigned_ids = set()
        resolved_boxes = []
        detections = []

        # Detect persons using a YOLO instance assigned to this camera
        # self.yolo is now a list of YOLO models
        import time as _time
        yolo_model = self.yolo[cam_idx % len(self.yolo)]
        yolo_results = yolo_model(frame, classes=[self.yolo_person_class], verbose=False, device=self.device, conf=0.35, iou=0.4)

        # --- YOLO Detection Logging ---
        yolo_log = get_yolo_logger()

        num_det = sum(len(r.boxes) for r in yolo_results)
        confs = [
            float(conf)
            for r in yolo_results
            for conf in r.boxes.conf.cpu().numpy()
        ] if num_det > 0 else []

        # Per-camera counters
        self._yolo_frame_count.setdefault(cam_idx, 0)
        self._yolo_det_count.setdefault(cam_idx, 0)
        self._yolo_zero_frames.setdefault(cam_idx, 0)
        self._yolo_last_log.setdefault(cam_idx, 0.0)

        self._yolo_frame_count[cam_idx] += 1
        self._yolo_det_count[cam_idx] += num_det

        if num_det == 0:
            self._yolo_zero_frames[cam_idx] += 1
        else:
            self._yolo_zero_frames[cam_idx] = 0

        # Warn when YOLO has not detected anyone for many consecutive frames
        zero_streak = self._yolo_zero_frames[cam_idx]
        if zero_streak in (30, 100, 300) or (zero_streak > 300 and zero_streak % 300 == 0):
            yolo_log.warning(
                f"Cam {cam_idx} ({cam_id_str}): NO person detected for {zero_streak} consecutive frames"
            )

        # Periodic summary log
        now = _time.time()
        elapsed = now - self._yolo_last_log[cam_idx]
        if elapsed >= _YOLO_LOG_INTERVAL:
            frames = self._yolo_frame_count[cam_idx]
            total_dets = self._yolo_det_count[cam_idx]
            avg_dets = total_dets / frames if frames else 0.0

            if num_det > 0:
                conf_str = ", ".join(f"{c:.2f}" for c in confs)
                yolo_log.info(
                    f"Cam {cam_idx} ({cam_id_str}) | frame#{frames} | "
                    f"detected={num_det} person(s) | confs=[{conf_str}] | "
                    f"avg_per_frame={avg_dets:.2f} (last {elapsed:.1f}s)"
                )
            else:
                yolo_log.info(
                    f"Cam {cam_idx} ({cam_id_str}) | frame#{frames} | "
                    f"detected=0 persons | "
                    f"avg_per_frame={avg_dets:.2f} (last {elapsed:.1f}s) | "
                    f"zero_streak={zero_streak}"
                )

            # Reset counters for next window
            self._yolo_frame_count[cam_idx] = 0
            self._yolo_det_count[cam_idx] = 0
            self._yolo_last_log[cam_idx] = now
        
        # Collect all person crops for batch processing
        person_crops = []
        person_boxes = []
        blur_scores = []
        
        for r in yolo_results:
            boxes = r.boxes.xyxy.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                
                gray_crop = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray_crop, cv2.CV_64F).var()
                
                person_pil = Image.fromarray(cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB))
                
                person_crops.append(person_pil)
                person_boxes.append((x1, y1, x2, y2))
                blur_scores.append(blur_score)
        
        # Batch extract ReID features
        if len(person_crops) > 0:
            import faiss
            feats = extract_features_batch(self.reid_model, person_crops, self.reid_transform, self.device)
            
            self.reid_index.initialize_if_needed(feats)
            
            faiss.normalize_L2(feats)
            
            for i, (box, feat, blur_score) in enumerate(zip(person_boxes, feats, blur_scores)):
                is_sharp = blur_score > 100.0
                detections.append({'box': box, 'feat': feat.reshape(1, -1), 'is_sharp': is_sharp})

        # --- STEP B.1: Track Association (Temporal Consistency) ---
        if cam_idx not in self.confirmed_tracks:
            self.confirmed_tracks[cam_idx] = []
            
        active_confirmed = []
        unmatched_detections = []
        
        matched_track_indices = set()
        matched_det_indices = set()
        
        for t_idx, track in enumerate(self.confirmed_tracks[cam_idx]):
            best_iou = 0.5
            best_d_idx = -1
            
            for d_idx, det in enumerate(detections):
                if d_idx in matched_det_indices:
                    continue
                
                iou = compute_iou(track.box, det['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_d_idx = d_idx
            
            if best_d_idx != -1:
                matched_track_indices.add(t_idx)
                matched_det_indices.add(best_d_idx)
                
                det = detections[best_d_idx]
                feat = det['feat']
                
                # ReID voting for existing track
                raw_voting_id, sim_score = self.reid_index.search(feat)
                
                if raw_voting_id == -1:
                    raw_voting_id = track.display_id
                
                track.update(det['box'], feat, raw_voting_id)
                track.current_sim_score = sim_score
                track.current_is_sharp = det['is_sharp']
                active_confirmed.append(track)
            else:
                track.miss_count += 1
                if track.miss_count <= MAX_MISS_FRAMES:
                    active_confirmed.append(track)
        
        # --- CONFLICT RESOLUTION: Unique IDs ---
        id_usage = {}
        for track in active_confirmed:
            if track.miss_count == 0:
                pid = track.display_id
                if pid not in id_usage:
                    id_usage[pid] = []
                id_usage[pid].append(track)
        
        for pid, tracks in id_usage.items():
            if len(tracks) > 1:
                tracks.sort(key=lambda t: t.vote_history.count(pid), reverse=True)
                for loser in tracks[1:]:
                    self.reid_index.force_new_id_for_track(loser)

        # --- FACE NAME CONFLICT RESOLUTION ---
        track_face_matches = {}
        face_name_usage = {}

        for idx, track in enumerate(active_confirmed):
            if track.miss_count > 0:
                continue
            
            best_face_name = None
            best_face_score = 0.0
            
            for face in face_results:
                if is_face_inside_body(face['box'], track.box):
                    if face['name'] != "Unknown":
                        if face['sim_score'] > best_face_score:
                            best_face_score = face['sim_score']
                            best_face_name = face['name']
            
            if best_face_name:
                track_face_matches[idx] = (best_face_name, best_face_score)
                if best_face_name not in face_name_usage:
                    face_name_usage[best_face_name] = []
                face_name_usage[best_face_name].append((best_face_score, idx))

        final_track_faces = {}
        for name, matches in face_name_usage.items():
            if len(matches) > 1:
                matches.sort(key=lambda x: x[0], reverse=True)
                winner_score, winner_idx = matches[0]
                final_track_faces[winner_idx] = name
            else:
                final_track_faces[matches[0][1]] = name

        # --- DRAWING & FINALIZING CONFIRMED TRACKS ---
        for idx, track in enumerate(active_confirmed):
            if track.miss_count > 0:
                continue
            
            assigned_ids.add(track.display_id)
            resolved_boxes.append(track.box)
            
            # Update Feature Bank
            if (track.display_id == track.vote_history[-1]
                    and track.current_sim_score < 0.95
                    and track.current_is_sharp):
                self.reid_index.add_vector(track.feat, track.display_id)

            x1, y1, x2, y2 = track.box
            
            matched_face_name = final_track_faces.get(idx)
            
            if matched_face_name:
                self.reid_index.set_face_name(track.display_id, matched_face_name)
            
            known_face_name = self.reid_index.get_face_name(track.display_id)
            
            bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            mnv_info = matched_face_name if matched_face_name else (known_face_name if known_face_name else "Unknown")
            data_logger.info(f"({cam_id_str}, {track.display_id}, {mnv_info}, {bbox_center})")
            
            color = (0, 255, 0)
            label = f"ID:{track.display_id} ({track.current_sim_score:.2f})"
            
            if matched_face_name:
                label = f"{matched_face_name} | {label}"
                color = (0, 255, 255)
                log_face_detection(matched_face_name, cam_name, log_dir, person_id=track.display_id, floor=f"{floor_id}F")
            elif known_face_name:
                label = f"{known_face_name} | {label}"
                color = (0, 200, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Collect unmatched detections
        for i, det in enumerate(detections):
            if i not in matched_det_indices:
                unmatched_detections.append(det)

        self.confirmed_tracks[cam_idx] = active_confirmed

        # --- ReID Phase 1: Match unmatched detections to known IDs ---
        unknown_detections = []
        
        for det in unmatched_detections:
            feat = det['feat']
            box = det['box']
            
            matched_face_name = None
            matched_face_score = 0.0
            
            for face in face_results:
                if is_face_inside_body(face['box'], box):
                    if face['name'] != "Unknown":
                        matched_face_name = face['name']
                        matched_face_score = face['sim_score']
                        break
            
            found_person_id, sim_score = self.reid_index.search(feat)
            
            if found_person_id != -1 and found_person_id in assigned_ids:
                found_person_id = -1
            
            if found_person_id != -1:
                assigned_ids.add(found_person_id)
                resolved_boxes.append(box)
                
                new_track = ConfirmedTrack(found_person_id, box, feat)
                self.confirmed_tracks[cam_idx].append(new_track)
                
                if matched_face_name:
                    self.reid_index.set_face_name(found_person_id, matched_face_name)
                
                if sim_score < 0.95 and det['is_sharp']:
                    self.reid_index.add_vector(feat, found_person_id)

                x1, y1, x2, y2 = box
                color = (0, 255, 0)
                
                known_face_name = self.reid_index.get_face_name(found_person_id)
                
                label = f"ID:{found_person_id} ({sim_score:.2f})"
                if matched_face_name:
                    label = f"{matched_face_name} | {label}"
                    color = (0, 255, 255)
                    log_face_detection(matched_face_name, cam_name, log_dir, person_id=found_person_id, floor=f"{floor_id}F")
                elif known_face_name:
                    label = f"{known_face_name} | {label}"
                    color = (0, 200, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            else:
                det['matched_face'] = matched_face_name
                unknown_detections.append(det)

        # --- ReID Phase 2: Process truly unknown detections (pending tracks) ---
        current_pending = []
        
        active_tracks = []
        for track in self.pending_tracks[cam_idx]:
            is_resolved = False
            for rbox in resolved_boxes:
                if compute_iou(track.box, rbox) > 0.5:
                    is_resolved = True
                    break
            if not is_resolved:
                active_tracks.append(track)
        
        for det in unknown_detections:
            box = det['box']
            feat = det['feat']
            matched_face_name = det.get('matched_face')
            
            best_iou = 0
            best_track = None
            
            for track in active_tracks:
                iou = compute_iou(track.box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track
            
            if best_iou > 0.5:
                best_track.box = box
                best_track.feat = feat
                best_track.features.append(feat)
                best_track.count += 1
                
                if best_track.count >= CONFIRM_FRAMES:
                    new_id = self.reid_index.allocate_person_id()
                    
                    all_feats = np.vstack(best_track.features)
                    self.reid_index.add_vectors_batch(all_feats, new_id)
                    
                    if matched_face_name:
                        self.reid_index.set_face_name(new_id, matched_face_name)
                    
                    x1, y1, x2, y2 = box
                    color = (0, 0, 255)
                    label = f"ID:{new_id} (New)"

                    bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    mnv_info = matched_face_name if matched_face_name else "Unknown"
                    data_logger.info(f"({cam_id_str}, {new_id}, {mnv_info}, {bbox_center})")
                    
                    if matched_face_name:
                        label = f"{matched_face_name} | {label}"
                        color = (0, 255, 255)
                        log_face_detection(matched_face_name, cam_name, log_dir, person_id=new_id, floor=f"{floor_id}F")

                    new_track = ConfirmedTrack(new_id, box, feat)
                    self.confirmed_tracks[cam_idx].append(new_track)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    active_tracks.remove(best_track)
                else:
                    current_pending.append(best_track)
                    active_tracks.remove(best_track)
                    
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"Identifying... {best_track.count}/{CONFIRM_FRAMES}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            else:
                new_track = PendingTrack(box, feat)
                current_pending.append(new_track)
                
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, f"Identifying... 1/{CONFIRM_FRAMES}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            self.pending_tracks[cam_idx] = current_pending

        # --- STEP C: Draw ALL Face Boxes ---
        for face in face_results:
            fx1, fy1, fx2, fy2 = face['box']
            face_name = face['name']
            face_sim = face['sim_score']
            
            if face_name == "Unknown":
                face_color = (0, 0, 255)
                face_label = "Unknown"
            else:
                face_color = (0, 255, 0)
                face_label = f"{face_name} ({face_sim:.2f})"
            
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), face_color, 2)
            
            label_y = fy1 - 5 if fy1 > 20 else fy2 + 20
            cv2.putText(frame, face_label, (fx1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, face_color, 2)
        
        return frame
