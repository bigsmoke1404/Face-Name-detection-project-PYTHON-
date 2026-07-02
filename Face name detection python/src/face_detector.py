"""
face_detector.py
----------------
Detects face regions in BGR frames using OpenCV's DNN face detector
(deep-learning based, more accurate than Haar cascades on modern hardware)
with a Haar cascade fallback.

Returns bounding box tuples in (top, right, bottom, left) format to match
the face_recognition library convention.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

import settings
import utils

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DNN model paths (shipped with opencv-python ≥ 4.x via the data module)
# We try to locate the model files shipped with OpenCV; if not found we
# fall back to Haar cascades.
# ---------------------------------------------------------------------------

def _find_opencv_data(*names: str) -> Optional[str]:
    """Search common locations for an OpenCV data file."""
    try:
        import cv2
        cv2_data = Path(cv2.__file__).parent / "data"
        for name in names:
            p = cv2_data / name
            if p.exists():
                return str(p)
    except Exception:
        pass
    # Try system paths
    for base in ["/usr/share/opencv4", "/usr/share/opencv", "/usr/local/share/opencv4"]:
        for name in names:
            p = Path(base) / name
            if p.exists():
                return str(p)
    return None


# We will use face_recognition (dlib) for face detection.
# OpenCV Haar cascades were causing `cv2.error: invalid vector subscript`
# and are generally less accurate than HOG/CNN based methods.


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_faces(
    frame: np.ndarray,
    scale_factor: float = 0.5,
    apply_lighting: bool = True,
) -> list[tuple[int, int, int, int]]:
    """
    Detect faces in *frame* and return bounding boxes as
    ``[(top, right, bottom, left), ...]`` — matching face_recognition order.

    Parameters
    ----------
    frame         : BGR frame from OpenCV
    scale_factor  : Downsample frame before detection for speed (0 < x ≤ 1)
    apply_lighting: If True, run CLAHE normalisation on the detection copy
    """
    if frame is None or frame.size == 0:
        return []

    # Work on a smaller copy for speed
    small = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
    if apply_lighting:
        small = utils.normalize_lighting(small)

    # Ensure proper type and channels
    if small.ndim == 3 and small.shape[2] == 4:
        small = cv2.cvtColor(small, cv2.COLOR_BGRA2BGR)
    if small.dtype != np.uint8:
        small = np.clip(small, 0, 255).astype(np.uint8)

    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    
    # Use face_recognition (dlib HOG) instead of Haar cascades
    import face_recognition as fr
    boxes = fr.face_locations(rgb, model="hog")

    # Scale boxes back to original resolution
    # face_locations returns (top, right, bottom, left)
    inv = 1.0 / scale_factor
    result = []
    for (top_s, right_s, bottom_s, left_s) in boxes:
        top    = int(top_s * inv)
        right  = int(right_s * inv)
        bottom = int(bottom_s * inv)
        left   = int(left_s * inv)
        result.append((top, right, bottom, left))

    return result


def detect_faces_rgb(
    rgb_frame: np.ndarray,
    scale_factor: float = 0.5,
) -> list[tuple[int, int, int, int]]:
    """
    Same as detect_faces but accepts an RGB frame (as expected by
    face_recognition library internals).
    """
    bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    return detect_faces(bgr, scale_factor=scale_factor)


# ---------------------------------------------------------------------------
# Crop helper
# ---------------------------------------------------------------------------

def crop_face(
    frame: np.ndarray,
    top: int, right: int, bottom: int, left: int,
    padding: float = 0.15,
) -> np.ndarray:
    """
    Crop a face region from *frame* with optional proportional padding.
    Returns the cropped BGR image.
    """
    h, w = frame.shape[:2]
    pad_y = int((bottom - top) * padding)
    pad_x = int((right  - left) * padding)
    t = max(0, top    - pad_y)
    b = min(h, bottom + pad_y)
    l = max(0, left   - pad_x)
    r = min(w, right  + pad_x)
    return frame[t:b, l:r]
