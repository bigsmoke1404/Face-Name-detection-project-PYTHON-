"""
face_recognition_engine.py
---------------------------
Wraps the `face_recognition` library to provide:
  - Encoding of face crops
  - Identification against the known-faces database
  - In-memory encoding cache (reloaded when DB changes)
  - Duplicate-registration detection
  - Per-face debounce (prevents repeated recognition events)
"""

import logging
import time
import threading
from typing import Optional

import numpy as np

import settings
import database as db

logger = logging.getLogger(__name__)

# Lazy import — face_recognition loads slowly; we import on first use
_fr = None

def _face_recognition():
    global _fr
    if _fr is None:
        import face_recognition as fr
        _fr = fr
    return _fr


# ---------------------------------------------------------------------------
# Encoding cache
# ---------------------------------------------------------------------------

class _EncodingCache:
    """Thread-safe in-memory cache of (person_id, name, encoding) triples."""

    def __init__(self):
        self._lock     = threading.RLock()
        self._data: list[tuple[int, str, np.ndarray]] = []
        self._loaded   = False

    def load(self) -> None:
        """Load (or reload) all encodings from the database."""
        records = db.load_all_encodings()
        with self._lock:
            self._data   = records
            self._loaded = True
        logger.info("Encoding cache loaded: %d vectors.", len(records))

    def add(self, person_id: int, name: str, encodings: list[np.ndarray]) -> None:
        """Append new encodings for a freshly registered person."""
        with self._lock:
            for enc in encodings:
                self._data.append((person_id, name, enc))

    def all(self) -> list[tuple[int, str, np.ndarray]]:
        with self._lock:
            return list(self._data)

    def remove(self, person_id: int) -> None:
        with self._lock:
            self._data = [(pid, name, enc) for pid, name, enc in self._data if pid != person_id]

    def rename(self, person_id: int, new_name: str) -> None:
        with self._lock:
            self._data = [
                (pid, new_name if pid == person_id else name, enc)
                for pid, name, enc in self._data
            ]

    def is_loaded(self) -> bool:
        return self._loaded

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


_cache = _EncodingCache()


def load_known_faces() -> None:
    """Load all known face encodings from the database into memory."""
    _cache.load()


def reload_known_faces() -> None:
    """Alias for load_known_faces — use after DB changes."""
    _cache.load()


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_face(rgb_frame: np.ndarray, locations: Optional[list] = None) -> list[np.ndarray]:
    """
    Compute 128-d face encodings from an RGB frame.

    Parameters
    ----------
    rgb_frame : np.ndarray  — full RGB frame (not a crop)
    locations : list of (top, right, bottom, left) — pre-computed face locations
                If None, face_recognition will detect them internally.

    Returns
    -------
    List of 128-d numpy arrays (one per detected face).
    """
    fr = _face_recognition()
    model = settings.get("recognition_model", "hog")

    # Enforce uint8 RGB — dlib requires exactly 8-bit 3-channel images
    if rgb_frame is None:
        return []
    if rgb_frame.dtype != np.uint8:
        rgb_frame = np.clip(rgb_frame, 0, 255).astype(np.uint8)
    # Drop alpha channel if present (BGRA/RGBA from some webcam drivers)
    if rgb_frame.ndim == 3 and rgb_frame.shape[2] == 4:
        rgb_frame = rgb_frame[:, :, :3]
    # Ensure contiguous memory layout
    if not rgb_frame.flags['C_CONTIGUOUS']:
        rgb_frame = np.ascontiguousarray(rgb_frame)

    try:
        if locations:
            encodings = fr.face_encodings(rgb_frame, known_face_locations=locations)
        else:
            encodings = fr.face_encodings(rgb_frame, model=model)
        return encodings
    except Exception as e:
        logger.error("Encoding error: %s", e)
        return []


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------

def identify_face(encoding: np.ndarray) -> tuple[Optional[int], str, float]:
    """
    Compare *encoding* against all known faces.

    Returns
    -------
    (person_id, name, confidence)
      person_id = None and name = "Unknown" if no match found.
      confidence is in [0, 1] where 1 = perfect match.
    """
    if not _cache.is_loaded():
        _cache.load()

    known = _cache.all()
    if not known:
        return None, "Unknown", 0.0

    known_encodings  = [e for _, _, e in known]
    person_ids       = [pid for pid, _, _ in known]
    names            = [n for _, n, _ in known]

    fr         = _face_recognition()
    threshold  = settings.get("confidence_threshold", 0.50)

    # face_distance returns values in [0, ∞); lower = better match
    distances  = fr.face_distance(known_encodings, encoding)
    best_idx   = int(np.argmin(distances))
    best_dist  = float(distances[best_idx])

    if best_dist <= threshold:
        # Convert distance to a confidence percentage (0–1, 1=perfect)
        confidence = max(0.0, 1.0 - best_dist)
        return person_ids[best_idx], names[best_idx], round(confidence, 4)

    return None, "Unknown", round(max(0.0, 1.0 - best_dist), 4)


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def is_duplicate(encoding: np.ndarray, strict_threshold: float = 0.40) -> tuple[bool, Optional[str]]:
    """
    Check if *encoding* is already in the database (for registration guard).

    Returns (is_dup: bool, matched_name: str | None)
    """
    known = _cache.all()
    if not known:
        return False, None

    fr = _face_recognition()
    known_encodings = [e for _, _, e in known]
    distances = fr.face_distance(known_encodings, encoding)
    best_idx  = int(np.argmin(distances))
    best_dist = float(distances[best_idx])

    if best_dist <= strict_threshold:
        _, name, _ = known[best_idx]
        return True, name
    return False, None


# ---------------------------------------------------------------------------
# Debounce tracker
# ---------------------------------------------------------------------------

class DebounceTracker:
    """
    Prevents repeated recognition events for the same person.
    Uses the face bounding box hash as a soft "track id".
    """

    def __init__(self):
        self._last: dict[int, tuple[Optional[int], float]] = {}
        # Maps person_id → last event time (for voice greeting dedup)
        self._greeted: dict[int, float] = {}
        self._lock = threading.Lock()

    def should_fire(self, person_id: Optional[int], box: tuple) -> bool:
        """
        Returns True if a recognition event should fire for this person/box.
        Respects the debounce_seconds setting.
        """
        debounce = settings.get("debounce_seconds", 2.5)
        box_key  = hash(box)
        now      = time.monotonic()

        with self._lock:
            if box_key in self._last:
                last_pid, last_t = self._last[box_key]
                if last_pid == person_id and (now - last_t) < debounce:
                    return False
            self._last[box_key] = (person_id, now)
            return True

    def should_greet(self, person_id: int, greet_cooldown: float = 30.0) -> bool:
        """
        Return True if a voice greeting should be played for this person.
        Greeted at most once per greet_cooldown seconds per person.
        """
        now = time.monotonic()
        with self._lock:
            last_t = self._greeted.get(person_id, 0.0)
            if (now - last_t) >= greet_cooldown:
                self._greeted[person_id] = now
                return True
        return False

    def reset(self) -> None:
        with self._lock:
            self._last.clear()
            self._greeted.clear()


# ---------------------------------------------------------------------------
# Registration helper
# ---------------------------------------------------------------------------

def compute_average_encoding(encodings: list[np.ndarray]) -> np.ndarray:
    """Average a list of 128-d encodings into one representative vector."""
    return np.mean(np.stack(encodings), axis=0)


# Module-level singleton debouncer
debouncer = DebounceTracker()
