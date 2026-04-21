"""
Face index rebuilding utility.
Used when the faces directory changes at runtime.
"""
import numpy as np


def rebuild_face_index(detector, recognizer, faces_dir):
    """
    Rebuild face targets and FAISS index.
    Called when faces directory changes.
    
    Args:
        detector: SCRFD face detector
        recognizer: ArcFace recognizer
        faces_dir: Path to faces directory
    
    Returns:
        face_targets: List of (embedding, name) tuples
        face_index: New FAISS index
        face_id_to_name: New ID mapping
    """
    import faiss
    from API_Face.service.processing import build_targets
    
    print("🔄 Rebuilding face targets and FAISS index...")
    
    face_targets = build_targets(detector, recognizer, faces_dir)
    print(f"   ✅ Loaded {len(face_targets)} known faces")
    
    if len(face_targets) == 0:
        print("   ⚠️ No faces loaded")
        return face_targets, None, {}
    
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
    
    print(f"   ✅ FAISS index rebuilt (dim={embedding_dim})")
    
    return face_targets, face_index, face_id_to_name
