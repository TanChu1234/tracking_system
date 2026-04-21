"""
ReID FAISS Index Manager.
Encapsulates the FAISS index, vector-to-person mappings, and thread-safe operations
for person re-identification across cameras.
"""
import threading
from collections import deque

import numpy as np


class ReIDIndex:
    """
    Thread-safe FAISS index for person re-identification.
    
    Manages:
    - FAISS IndexIDMap for fast nearest-neighbor search
    - vector_id -> person_id mapping
    - person_id -> list of vector_ids (for eviction)
    - person_id -> face_name mapping
    - Auto-incrementing person_id and vector_id counters
    """
    
    # Configuration constants
    MATCH_THRESH = 0.7        # Cosine similarity threshold
    MAX_VECTORS_PER_ID = 200  # Max feature vectors per person
    TOP_K_VOTING = 1          # Top-K nearest vectors for voting

    def __init__(self):
        self.index = None  # faiss.IndexIDMap, initialized on first feature
        
        self.vector_id_to_person_id = {}
        self.person_vector_ids = {}       # person_id -> deque of vector_ids
        self.person_id_to_face_name = {}  # person_id -> face name string
        
        self._next_person_id = 0
        self._next_vector_id = 0
        
        # Thread-safe locks
        self.faiss_lock = threading.Lock()
        self.person_id_lock = threading.Lock()
        self.vector_id_lock = threading.Lock()

    def _ensure_index(self, feature_dim):
        """Initialize FAISS index if not yet created."""
        if self.index is None:
            import faiss
            print(f"Initializing FAISS index with dim {feature_dim}")
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(feature_dim))

    def allocate_person_id(self):
        """Allocate and return a new unique person ID (thread-safe)."""
        with self.person_id_lock:
            pid = self._next_person_id
            self._next_person_id += 1
            return pid

    def _allocate_vector_ids(self, count):
        """Allocate multiple consecutive vector IDs (thread-safe)."""
        with self.vector_id_lock:
            start = self._next_vector_id
            self._next_vector_id += count
            return list(range(start, start + count))

    def initialize_if_needed(self, feats):
        """
        Ensure the index is initialized using the feature dimensionality.
        Call this after the first batch of features is extracted.
        
        Args:
            feats: numpy array [N, feat_dim]
        """
        with self.faiss_lock:
            self._ensure_index(feats.shape[1])

    def search(self, feat):
        """
        Search for the best matching person ID using adaptive top-K voting.
        
        Args:
            feat: numpy array [1, feat_dim], must be L2-normalized
        
        Returns:
            (person_id, similarity) or (-1, 0.0) if no match
        """
        with self.faiss_lock:
            if self.index is None or self.index.ntotal == 0:
                return -1, 0.0
            
            k = min(self.TOP_K_VOTING, self.index.ntotal)
            D, I = self.index.search(feat, k)
            
            votes = {}
            for i in range(k):
                similarity = float(D[0][i])
                if similarity > self.MATCH_THRESH:
                    vec_id = int(I[0][i])
                    if vec_id in self.vector_id_to_person_id:
                        pid = self.vector_id_to_person_id[vec_id]
                        if pid not in votes:
                            votes[pid] = {'count': 0, 'max_sim': 0.0}
                        votes[pid]['count'] += 1
                        votes[pid]['max_sim'] = max(votes[pid]['max_sim'], similarity)
            
            if not votes:
                return -1, 0.0
            
            sorted_votes = sorted(
                votes.items(),
                key=lambda x: (x[1]['count'], x[1]['max_sim']),
                reverse=True
            )
            winner_pid, winner_data = sorted_votes[0]
            required_votes = max(1, (k + 1) // 2)
            
            if winner_data['count'] >= required_votes:
                return winner_pid, winner_data['max_sim']
            
            return -1, 0.0

    def add_vector(self, feat, person_id):
        """
        Add a single feature vector for a person, evicting oldest if over limit.
        
        Args:
            feat: numpy array [1, feat_dim]
            person_id: The person ID to associate with
        """
        new_vid = self._allocate_vector_ids(1)[0]
        
        with self.faiss_lock:
            self.index.add_with_ids(feat, np.array([new_vid], dtype=np.int64))
            self.vector_id_to_person_id[new_vid] = person_id
            
            if person_id not in self.person_vector_ids:
                self.person_vector_ids[person_id] = deque()
            self.person_vector_ids[person_id].append(new_vid)
            
            # Evict oldest if over limit
            if len(self.person_vector_ids[person_id]) > self.MAX_VECTORS_PER_ID:
                old_vid = self.person_vector_ids[person_id].popleft()
                self.index.remove_ids(np.array([old_vid], dtype=np.int64))
                if old_vid in self.vector_id_to_person_id:
                    del self.vector_id_to_person_id[old_vid]

    def add_vectors_batch(self, feats, person_id):
        """
        Add multiple feature vectors for a person at once.
        
        Args:
            feats: numpy array [N, feat_dim]
            person_id: The person ID to associate with
        """
        num_feats = feats.shape[0] if len(feats.shape) > 1 else 1
        new_vids = self._allocate_vector_ids(num_feats)
        
        with self.faiss_lock:
            self.index.add_with_ids(feats, np.array(new_vids, dtype=np.int64))
            
            if person_id not in self.person_vector_ids:
                self.person_vector_ids[person_id] = deque()
            
            for vid in new_vids:
                self.vector_id_to_person_id[vid] = person_id
                self.person_vector_ids[person_id].append(vid)

    def set_face_name(self, person_id, face_name):
        """Associate a face name with a person ID (thread-safe)."""
        with self.person_id_lock:
            self.person_id_to_face_name[person_id] = face_name

    def get_face_name(self, person_id):
        """Get the face name for a person ID, or None (thread-safe)."""
        with self.person_id_lock:
            return self.person_id_to_face_name.get(person_id)

    def force_new_id_for_track(self, track):
        """
        Force a new ID for a track (used in conflict resolution).
        
        Args:
            track: ConfirmedTrack instance
        """
        new_id = self.allocate_person_id()
        track.person_id = new_id
        track.display_id = new_id
        track.vote_history.clear()
        track.vote_history.append(new_id)
