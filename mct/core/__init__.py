# MCT Core Components
# Import lightweight components directly
from .stream import ThreadedStream
from .tracker import PendingTrack, ConfirmedTrack

# Heavy components (FrameProcessor, engine) are imported on demand
# to avoid pulling in torch/faiss/ultralytics at package import time.
# Use: from mct.core.engine import run_demo
# Use: from mct.core.frame_processor import FrameProcessor

__all__ = [
    'ThreadedStream', 'PendingTrack', 'ConfirmedTrack',
]
