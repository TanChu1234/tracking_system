"""
Track data classes for MCT system.
PendingTrack: tracks waiting for confirmation (accumulating features).
ConfirmedTrack: confirmed tracks with temporal ID voting.
"""
from collections import deque


class PendingTrack:
    """A detection that needs N consecutive frames before being assigned an ID."""
    
    def __init__(self, box, feat):
        self.box = box
        self.feat = feat
        self.count = 1
        self.features = [feat]  # Store all features for later batch add


class ConfirmedTrack:
    """
    A confirmed track with a stable person ID.
    Uses temporal voting to prevent ID flickering.
    """
    
    def __init__(self, person_id, box, feat):
        self.person_id = person_id
        self.box = box
        self.feat = feat
        self.miss_count = 0
        self.vote_history = deque(maxlen=15)
        self.vote_history.append(person_id)
        self.display_id = person_id
        self.consecutive_matches = 0
        # Runtime attributes set during processing
        self.current_sim_score = 0.0
        self.current_is_sharp = False

    def update(self, box, feat, raw_voting_id):
        """
        Update track with new detection and perform temporal ID voting.
        
        Args:
            box: New bounding box
            feat: New feature vector
            raw_voting_id: The ID suggested by ReID search this frame
        """
        self.box = box
        self.feat = feat
        self.miss_count = 0
        self.vote_history.append(raw_voting_id)
        
        # Temporal Consistency Logic
        if len(self.vote_history) > 0:
            counts = {}
            for vid in self.vote_history:
                counts[vid] = counts.get(vid, 0) + 1
            
            # Find winner
            best_id, count = sorted(counts.items(), key=lambda x: x[1], reverse=True)[0]
            
            # Only switch ID if the new ID is dominant in recent history
            HISTORY_LEN = len(self.vote_history)
            THRESHOLD = max(3, HISTORY_LEN // 2 + 1)  # Majority vote
            
            if best_id != self.display_id:
                if count >= THRESHOLD:
                    self.display_id = best_id
