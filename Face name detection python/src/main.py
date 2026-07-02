"""
main.py
-------
Entry point for the AI-Based Facial Recognition Attendance & Identity System.

Responsibilities
----------------
  1. Configure logging
  2. Initialise the database
  3. Load settings
  4. Launch the GUI
"""

import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging configuration (must happen before other imports)
# ---------------------------------------------------------------------------

_LOG_DIR = Path(__file__).parent / "data" / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(_LOG_DIR / "app.log", encoding="utf-8"),
    ],
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

def _check_dependencies() -> bool:
    """
    Verify that the critical external packages are importable.
    Prints a helpful message and returns False if any are missing.
    """
    missing = []
    packages = {
        "cv2":           "opencv-python",
        "face_recognition": "face_recognition",
        "customtkinter": "customtkinter",
        "PIL":           "Pillow",
        "numpy":         "numpy",
    }
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print("\n" + "=" * 60)
        print("❌  MISSING DEPENDENCIES")
        print("=" * 60)
        print("The following packages are not installed:")
        for p in missing:
            print(f"   • {p}")
        print()
        print("Install them with:")
        print("   pip install " + " ".join(missing))
        print()
        print("NOTE: face_recognition requires dlib.")
        print("On Windows, install a pre-built dlib wheel first:")
        print("   https://github.com/z-mahmud22/Dlib_Windows_Python3.x")
        print("=" * 60 + "\n")
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 50)
    logger.info("FaceID — AI Facial Recognition System — Starting")
    logger.info("=" * 50)

    # 1. Dependency check
    if not _check_dependencies():
        sys.exit(1)

    # 2. Settings (auto-loaded on import)
    import settings
    logger.info("Settings loaded. Camera index: %d", settings.get("camera_index", 0))

    # 3. Database
    import database as db
    db.initialize()
    stats = db.get_stats()
    logger.info(
        "Database ready — %d registered persons, %d recognitions total.",
        stats["total_persons"], stats["total_recognitions"]
    )

    # 4. Pre-load face encodings in background (lazy-loaded on first recognition anyway)
    import face_recognition_engine as fre
    import threading
    threading.Thread(target=fre.load_known_faces, daemon=True, name="EncodingPreload").start()

    # 5. Launch GUI
    logger.info("Launching GUI...")
    import gui
    gui.run()

    logger.info("FaceID exited cleanly.")


if __name__ == "__main__":
    main()
