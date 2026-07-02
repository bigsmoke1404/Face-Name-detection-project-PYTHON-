"""
settings.py
-----------
Persistent application settings stored as JSON.
Provides get/set interface with sensible defaults.
"""

import json
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------
DEFAULTS: dict = {
    "camera_index": 0,
    "resolution_width": 1280,
    "resolution_height": 720,
    "confidence_threshold": 0.50,       # face_recognition distance ≤ this → match
    "recognition_sensitivity": 0.55,    # looser bound for "possible match" display
    "admin_password_hash": hashlib.sha256(b"admin123").hexdigest(),
    "voice_enabled": True,
    "attendance_mode": True,
    "fps_display": True,
    "num_samples": 40,                  # frames captured during registration
    "blur_threshold": 80.0,             # Laplacian variance minimum
    "min_face_size": 80,                # minimum face bounding box width/height (px)
    "recognition_model": "hog",         # "hog" (CPU) or "cnn" (GPU)
    "debounce_seconds": 2.5,            # min seconds between recognition events
    "theme": "dark",
    "color_accent": "#00d4ff",
}

_DATA_DIR = Path(__file__).parent / "data"
_SETTINGS_FILE = _DATA_DIR / "settings.json"

# In-memory store
_store: dict = {}


def load() -> None:
    """Load settings from disk; missing keys filled from DEFAULTS."""
    global _store
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _SETTINGS_FILE.exists():
        try:
            with open(_SETTINGS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge: defaults first, then override with saved values
            _store = {**DEFAULTS, **saved}
            logger.info("Settings loaded from %s", _SETTINGS_FILE)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read settings (%s); using defaults.", e)
            _store = dict(DEFAULTS)
    else:
        _store = dict(DEFAULTS)
        save()
        logger.info("Settings initialised with defaults.")


def save() -> None:
    """Persist current settings to disk."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(_SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(_store, f, indent=4)
    except OSError as e:
        logger.error("Failed to save settings: %s", e)


def get(key: str, default=None):
    """Return a setting value."""
    if not _store:
        load()
    return _store.get(key, default)


def set(key: str, value) -> None:
    """Update a setting and persist immediately."""
    if not _store:
        load()
    _store[key] = value
    save()


def verify_admin_password(password: str) -> bool:
    """Return True if the supplied password matches the stored hash."""
    stored_hash = get("admin_password_hash")
    return hashlib.sha256(password.encode()).hexdigest() == stored_hash


def change_admin_password(new_password: str) -> None:
    """Hash and store a new admin password."""
    set("admin_password_hash", hashlib.sha256(new_password.encode()).hexdigest())


def all_settings() -> dict:
    """Return a copy of all current settings (without password hash)."""
    if not _store:
        load()
    result = dict(_store)
    result.pop("admin_password_hash", None)
    return result


# Auto-load when module is imported
load()
