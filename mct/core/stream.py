"""
Threaded RTSP stream reader for MCT system.
"""
import cv2
import time
import threading


class ThreadedStream:
    """Thread-safe RTSP/video stream reader."""
    
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.lock = threading.Lock()
        self.frame = None
        self.status = False
        self.stopped = False
        
        # Check if opened
        if self.capture.isOpened():
            self.status = True
            self.status, self.frame = self.capture.read()
        
        # Start reader thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        """Continuously read frames in background thread."""
        while not self.stopped:
            if self.capture.isOpened():
                status, frame = self.capture.read()
                with self.lock:
                    self.status = status
                    self.frame = frame
                if not status:
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def read(self):
        """Return the latest frame (thread-safe)."""
        with self.lock:
            return self.status, self.frame if self.frame is not None else None

    def stop(self):
        """Stop the stream and release resources."""
        self.stopped = True
        self.thread.join()
        self.capture.release()
