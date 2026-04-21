# MCT ReID Components
# Heavy components are imported on demand to avoid pulling in
# torch/faiss at package import time.
# Use: from mct.reid.model import setup_transreid, get_transforms
# Use: from mct.reid.features import extract_feature, extract_features_batch
# Use: from mct.reid.reid_index import ReIDIndex

__all__ = [
    'setup_transreid', 'get_transforms',
    'extract_feature', 'extract_features_batch',
    'ReIDIndex',
]
