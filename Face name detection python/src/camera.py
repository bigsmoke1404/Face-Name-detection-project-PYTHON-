"""
camera.py
---------
Background webcam capture thread.

Provides a non-blocking video feed via a callback and/or a shared
latest-frame buffer.  The recognition pipeline reads frames from here
without blocking the GUI thread.
"""

import logging
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

import settings

logger = logging.getLogger(__name__)


class Camera:
    """
    Manages a webcam in a dedicated background thread.

    Usage
    -----
        cam = Camera()
        cam.start()
        ...
        frame = cam.get_frame()  # latest BGR frame or None
        ...
        cam.stop()
    """

    def __init__(self, on_frame: Optional[Callable[[np.ndarray], None]] = None):
        """
        Parameters
        ----------
        on_frame : callable, optional
            Called from the capture thread with each new BGR frame.
            Keep this callback fast (or hand off to a queue).
        """
        self._on_frame    = on_frame
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event  = threading.Event()
        self._frame_lock  = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running     = False
        self._camera_index = settings.get("camera_index", 0)
        self._width  = settings.get("resolution_width",  1280)
        self._height = settings.get("resolution_height", 720)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> bool:
        """
        Open the webcam and begin capturing.
        Returns True on success, False if the camera couldn't be opened.
        """
        if self._running:
            logger.warning("Camera already running.")
            return True

        self._cap = cv2.VideoCapture(self._camera_index, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            # Fallback: try without DSHOW
            self._cap = cv2.VideoCapture(self._camera_index)

        if not self._cap.isOpened():
            logger.error("Cannot open camera index %d.", self._camera_index)
            return False

        # Apply resolution
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize buffer lag

        self._stop_event.clear()
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True, name="CameraThread")
        self._thread.start()
        logger.info("Camera started (index=%d, %dx%d).", self._camera_index, self._width, self._height)
        return True

    def stop(self) -> None:
        """Stop capturing and release the webcam."""
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._running = False
        with self._frame_lock:
            self._latest_frame = None
        logger.info("Camera stopped.")

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame (BGR), or None if no frame available."""
        with self._frame_lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def is_running(self) -> bool:
        return self._running

    def set_camera_index(self, index: int) -> None:
        """Switch to a different camera (restarts capture if already running)."""
        was_running = self._running
        if was_running:
            self.stop()
        self._camera_index = index
        settings.set("camera_index", index)
        if was_running:
            self.start()

    def set_resolution(self, width: int, height: int) -> None:
        """Change capture resolution (restarts capture if already running)."""
        was_running = self._running
        if was_running:
            self.stop()
        self._width, self._height = width, height
        settings.set("resolution_width",  width)
        settings.set("resolution_height", height)
        if was_running:
            self.start()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self._width, self._height)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """Main loop: read frames and store / dispatch them."""
        consecutive_failures = 0
        max_failures = 30  # ~1 second at 30fps

        while not self._stop_event.is_set():
            if self._cap is None:
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error("Camera feed lost after %d consecutive failures.", max_failures)
                    self._running = False
                    break
                time.sleep(0.033)
                continue

            consecutive_failures = 0
            frame = cv2.flip(frame, 1)  # mirror so it feels like a selfie-cam

            with self._frame_lock:
                self._latest_frame = frame

            if self._on_frame:
                try:
                    self._on_frame(frame)
                except Exception as e:
                    logger.error("Error in on_frame callback: %s", e)

            # ~30 FPS cap to avoid burning CPU
            time.sleep(0.01)


# ---------------------------------------------------------------------------
# Helper: list available camera indices
# ---------------------------------------------------------------------------

def list_cameras(max_check: int = 5) -> list[int]:
    """
    Probe camera indices 0..max_check-1 and return those that open.
    Useful for populating a camera-selection dropdown.
    """
    available = []
    for idx in range(max_check):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available
