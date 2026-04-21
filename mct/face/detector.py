"""
Face detection and recognition using SCRFD + ArcFace + FAISS.
"""
import os
import numpy as np


def setup_face_api():
    """
    Initialize Face API models and targets with FAISS index.
    Adjusts paths since we are running from project root.
    
    Returns:
        detector: SCRFD face detector
        recognizer: ArcFace recognizer
        face_targets: List of (embedding, name) tuples
        face_index: FAISS index for fast face matching
        face_id_to_name: Dict mapping FAISS ID to name
    """
    import faiss
    from API_Face.load_model import load_model
    from API_Face.c.cConst import Const
    from API_Face.service.processing import build_targets
    
    print("Setting up Face API...")
    
    det_weight_path = os.path.abspath("./API_Face/weights/det_10g.onnx")
    rec_weight_path = os.path.abspath("./API_Face/weights/w600k_r50.onnx")
    faces_dir_path = os.path.abspath("./API_Face/faces/")
    
    if not os.path.exists(det_weight_path):
        print(f"❌ Error: Face detection weights not found at {det_weight_path}")
        raise FileNotFoundError(f"Missing weights: {det_weight_path}")
    if not os.path.exists(rec_weight_path):
        print(f"❌ Error: Face recognition weights not found at {rec_weight_path}")
        raise FileNotFoundError(f"Missing weights: {rec_weight_path}")
    
    print(f"✅ Found Face weights at: {det_weight_path}")
    
    # Update Const class with correct paths
    Const.det_weight = det_weight_path
    Const.rec_weight = rec_weight_path
    Const.faces_dir = faces_dir_path

    # Load models (GPU with auto-fallback to CPU)
    detector, recognizer = load_model(
        det_weight=det_weight_path,
        rec_weight=rec_weight_path,
        conf_thres=Const.confidence_thresh,
        use_gpu=True
    )
    
    # Load targets (known faces)
    print(f"Loading known faces from {faces_dir_path}...")
    face_targets = build_targets(detector, recognizer, faces_dir_path)
    print(f"✅ Loaded {len(face_targets)} known faces.")
    
    # Build FAISS index for fast face matching
    if len(face_targets) == 0:
        print("⚠️ No faces loaded, skipping FAISS index creation")
        return detector, recognizer, face_targets, None, {}
    
    print("Building FAISS index for face recognition...")
    
    embeddings = []
    face_id_to_name = {}
    
    for idx, (embedding, name) in enumerate(face_targets):
        embeddings.append(embedding)
        face_id_to_name[idx] = name
    
    embeddings_np = np.vstack(embeddings)
    faiss.normalize_L2(embeddings_np)
    
    embedding_dim = embeddings_np.shape[1]
    face_index = faiss.IndexFlatIP(embedding_dim)
    face_index.add(embeddings_np)
    
    print(f"✅ FAISS face index built with {len(face_targets)} faces (dim={embedding_dim})")
    
    return detector, recognizer, face_targets, face_index, face_id_to_name


def run_face_recognition(frame, detector, recognizer, face_index, face_id_to_name):
    """
    Detect and recognize faces using FAISS for fast matching.
    
    Args:
        frame: Input frame (BGR numpy array)
        detector: SCRFD detector
        recognizer: ArcFace recognizer
        face_index: FAISS index for face embeddings
        face_id_to_name: Dict mapping FAISS ID to name
    
    Returns: 
        List of dicts {'box': [x1,y1,x2,y2], 'name': str, 'sim_score': float, 'det_score': float}
    """
    from API_Face.c.cConst import Const
    
    results = []
    
    bboxes, kpss = detector.detect(frame, max_num=0)
    
    if bboxes is None or len(bboxes) == 0:
        return results

    for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
        box = bbox[:4].astype(np.int32)
        det_score = bbox[4]
        
        embedding = recognizer(frame, kps)
        
        max_similarity = 0
        best_match_name = "Unknown"
        
        if face_index is not None and len(face_id_to_name) > 0:
            import faiss
            
            embedding_normalized = embedding.reshape(1, -1).astype(np.float32)
            faiss.normalize_L2(embedding_normalized)
            
            D, I = face_index.search(embedding_normalized, k=1)
            
            max_similarity = float(D[0][0])
            
            if max_similarity > Const.similarity_thresh:
                face_id = int(I[0][0])
                best_match_name = face_id_to_name.get(face_id, "Unknown")
                
        results.append({
            'box': box,
            'name': best_match_name,
            'sim_score': max_similarity,
            'det_score': det_score
        })
        
    return results
