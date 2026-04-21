# MCT Face Recognition Components
# Heavy components are imported on demand to avoid pulling in
# onnxruntime/faiss at package import time.
# Use: from mct.face.detector import setup_face_api, run_face_recognition
# Use: from mct.face.indexer import rebuild_face_index

__all__ = ['setup_face_api', 'run_face_recognition', 'rebuild_face_index']
