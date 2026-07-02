"""
utils.py
--------
Shared utility functions:
  - Image quality checks (blur, size, lighting)
  - Lighting normalisation via CLAHE
  - FPS counter
  - Text-to-speech voice greeting
  - CSV export helper
"""

import csv
import logging
import time
import threading
from collections import deque
from pathlib import Path

import cv2
import numpy as np

import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image Quality
# ---------------------------------------------------------------------------

def check_image_quality(face_img: np.ndarray) -> tuple[bool, str]:
    """
    Validate that a face crop is suitable for encoding.

    Returns
    -------
    (ok: bool, reason: str)
      ok=True means the image passed all quality checks.
    """
    if face_img is None or face_img.size == 0:
        return False, "Empty image"

    h, w = face_img.shape[:2]
    min_size = settings.get("min_face_size", 80)
    if w < min_size or h < min_size:
        return False, f"Face too small ({w}×{h} < {min_size})"

    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) if len(face_img.shape) == 3 else face_img

    # ---- Blur check (Laplacian variance) ----
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_thresh = settings.get("blur_threshold", 80.0)
    if blur_score < blur_thresh:
        return False, f"Image too blurry (score={blur_score:.1f} < {blur_thresh})"

    # ---- Lighting check ----
    mean_brightness = gray.mean()
    if mean_brightness < 25:
        return False, f"Too dark (brightness={mean_brightness:.1f})"
    if mean_brightness > 240:
        return False, f"Overexposed (brightness={mean_brightness:.1f})"

    return True, "OK"


def normalize_lighting(frame: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to the luminance channel to compensate for poor lighting.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# FPS Counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling-window FPS counter."""

    def __init__(self, window: int = 30):
        self._times: deque = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.monotonic())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Text-to-Speech
# ---------------------------------------------------------------------------

_tts_lock = threading.Lock()
_tts_engine = None


def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        try:
            import pyttsx3
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", 160)
        except Exception as e:
            logger.warning("TTS engine unavailable: %s", e)
    return _tts_engine


def speak(text: str) -> None:
    """Speak *text* asynchronously if voice is enabled in settings."""
    if not settings.get("voice_enabled", True):
        return

    def _run():
        with _tts_lock:
            engine = _get_tts_engine()
            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    logger.warning("TTS error: %s", e)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------

def export_to_csv(data: list[dict], filepath: str | Path) -> bool:
    """
    Write a list-of-dicts to a CSV file.

    Returns True on success, False on error.
    """
    if not data:
        logger.warning("export_to_csv called with empty data.")
        return False
    filepath = Path(filepath)
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        logger.info("Exported %d rows to %s", len(data), filepath)
        return True
    except OSError as e:
        logger.error("CSV export failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Draw Utilities
# ---------------------------------------------------------------------------

def draw_face_box(
    frame: np.ndarray,
    top: int, right: int, bottom: int, left: int,
    label: str,
    confidence: float,
    color: tuple[int, int, int] = (0, 212, 255),
    unknown: bool = False,
) -> None:
    """
    Draw a stylised bounding box with label and confidence bar on *frame*.
    Modifies frame in-place.
    """
    box_color  = (60, 60, 200) if unknown else color
    text_color = (255, 255, 255)

    # ---- Bounding box ----
    cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

    # ---- Corner accents ----
    corner_len = 18
    thickness  = 3
    for x, y, dx, dy in [
        (left,  top,    1,  1),
        (right, top,   -1,  1),
        (left,  bottom, 1, -1),
        (right, bottom,-1, -1),
    ]:
        cv2.line(frame, (x, y), (x + dx * corner_len, y),          box_color, thickness)
        cv2.line(frame, (x, y), (x,                   y + dy * corner_len), box_color, thickness)

    # ---- Label background ----
    conf_text = f"{label}  {confidence*100:.0f}%" if not unknown else "Unknown"
    (tw, th), baseline = cv2.getTextSize(conf_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    label_y = max(top - 10, th + 10)
    cv2.rectangle(frame,
                  (left, label_y - th - baseline - 4),
                  (left + tw + 8, label_y + baseline),
                  box_color, cv2.FILLED)
    cv2.putText(frame, conf_text,
                (left + 4, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # ---- Confidence bar ----
    if not unknown:
        bar_x1, bar_y = left, bottom + 6
        bar_w = right - left
        bar_h = 5
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_w, bar_y + bar_h), (50, 50, 50), cv2.FILLED)
        filled_w = int(bar_w * confidence)
        bar_fill = (0, 200, 80) if confidence > 0.75 else (0, 180, 220) if confidence > 0.5 else (0, 80, 220)
        cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + filled_w, bar_y + bar_h), bar_fill, cv2.FILLED)


def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Overlay FPS counter on frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 212, 255), 2)


def draw_status_overlay(frame: np.ndarray, status: str) -> None:
    """Draw a status string in the bottom-left corner."""
    h = frame.shape[0]
    cv2.putText(frame, status, (10, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
