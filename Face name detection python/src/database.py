"""
database.py
-----------
SQLite data layer for the facial recognition system.

Tables
------
  persons            — registered individuals
  face_encodings     — face encoding blobs per person
  attendance_log     — daily attendance records
  recognition_history — timestamped recognition events
"""

import sqlite3
import pickle
import logging
import numpy as np
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"
_DB_PATH   = _DATA_DIR / "faces.db"
_SAMPLES_DIR = _DATA_DIR / "face_samples"


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS persons (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT    NOT NULL,
    registration_date TEXT    NOT NULL,
    recognition_count INTEGER NOT NULL DEFAULT 0,
    last_seen         TEXT
);

CREATE TABLE IF NOT EXISTS face_encodings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    encoding    BLOB    NOT NULL,
    sample_idx  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS attendance_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    entry_time  TEXT    NOT NULL,
    date        TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS recognition_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER NOT NULL REFERENCES persons(id) ON DELETE CASCADE,
    name        TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    confidence  REAL    NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance_log(date, person_id);
CREATE INDEX IF NOT EXISTS idx_history_person  ON recognition_history(person_id);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------
def _connect() -> sqlite3.Connection:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def initialize() -> None:
    """Create tables and directories if they don't exist."""
    _SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    with _connect() as conn:
        conn.executescript(_SCHEMA_SQL)
    logger.info("Database initialised at %s", _DB_PATH)


# ---------------------------------------------------------------------------
# Person CRUD
# ---------------------------------------------------------------------------
def register_person(name: str, encodings: list[np.ndarray]) -> int:
    """
    Insert a new person and their face encodings.
    Returns the new person_id.
    """
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO persons (name, registration_date) VALUES (?, ?)",
            (name, now),
        )
        person_id = cur.lastrowid
        for idx, enc in enumerate(encodings):
            blob = pickle.dumps(enc)
            conn.execute(
                "INSERT INTO face_encodings (person_id, encoding, sample_idx) VALUES (?, ?, ?)",
                (person_id, blob, idx),
            )
    logger.info("Registered person '%s' (id=%d) with %d encodings.", name, person_id, len(encodings))
    return person_id


def get_all_persons() -> list[dict]:
    """Return all persons as a list of dicts."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, registration_date, recognition_count, last_seen FROM persons ORDER BY name"
        ).fetchall()
    return [dict(r) for r in rows]


def get_person_by_id(person_id: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, registration_date, recognition_count, last_seen FROM persons WHERE id=?",
            (person_id,),
        ).fetchone()
    return dict(row) if row else None


def rename_person(person_id: int, new_name: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE persons SET name=? WHERE id=?", (new_name, person_id))
    logger.info("Renamed person id=%d to '%s'.", person_id, new_name)


def delete_person(person_id: int) -> None:
    """Delete a person and all their encodings/logs (cascade)."""
    # Also remove face sample images
    sample_dir = _SAMPLES_DIR / str(person_id)
    if sample_dir.exists():
        import shutil
        shutil.rmtree(sample_dir, ignore_errors=True)
    with _connect() as conn:
        conn.execute("DELETE FROM persons WHERE id=?", (person_id,))
    logger.info("Deleted person id=%d.", person_id)


def update_recognition_stats(person_id: int, confidence: float) -> None:
    """Increment recognition_count and update last_seen + history."""
    now = datetime.now().isoformat(sep=" ", timespec="seconds")
    with _connect() as conn:
        conn.execute(
            "UPDATE persons SET recognition_count=recognition_count+1, last_seen=? WHERE id=?",
            (now, person_id),
        )
        # Fetch name for history
        name_row = conn.execute("SELECT name FROM persons WHERE id=?", (person_id,)).fetchone()
        if name_row:
            conn.execute(
                "INSERT INTO recognition_history (person_id, name, timestamp, confidence) VALUES (?,?,?,?)",
                (person_id, name_row["name"], now, round(confidence, 4)),
            )


def search_persons(query: str) -> list[dict]:
    """Case-insensitive search by name."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, registration_date, recognition_count, last_seen FROM persons WHERE LOWER(name) LIKE ?",
            (f"%{query.lower()}%",),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------
def load_all_encodings() -> list[tuple[int, str, np.ndarray]]:
    """
    Load all face encodings from DB.
    Returns list of (person_id, name, encoding_array).
    """
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT fe.person_id, p.name, fe.encoding
            FROM face_encodings fe
            JOIN persons p ON p.id = fe.person_id
            """
        ).fetchall()
    result = []
    for row in rows:
        try:
            enc = pickle.loads(row["encoding"])
            result.append((row["person_id"], row["name"], enc))
        except Exception as e:
            logger.warning("Failed to decode encoding for person_id=%d: %s", row["person_id"], e)
    return result


def get_encoding_count(person_id: int) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM face_encodings WHERE person_id=?", (person_id,)
        ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Attendance
# ---------------------------------------------------------------------------
def log_attendance(person_id: int, name: str) -> bool:
    """
    Record attendance for today.
    Returns True if this is a new entry (first time today), False if already recorded.
    """
    today = date.today().isoformat()
    now   = datetime.now().isoformat(sep=" ", timespec="seconds")
    with _connect() as conn:
        existing = conn.execute(
            "SELECT id FROM attendance_log WHERE person_id=? AND date=?",
            (person_id, today),
        ).fetchone()
        if existing:
            return False
        conn.execute(
            "INSERT INTO attendance_log (person_id, name, entry_time, date) VALUES (?,?,?,?)",
            (person_id, name, now, today),
        )
    return True


def get_attendance(filter_date: Optional[str] = None) -> list[dict]:
    """Return attendance records; optionally filter by date string 'YYYY-MM-DD'."""
    with _connect() as conn:
        if filter_date:
            rows = conn.execute(
                "SELECT * FROM attendance_log WHERE date=? ORDER BY entry_time",
                (filter_date,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM attendance_log ORDER BY date DESC, entry_time DESC"
            ).fetchall()
    return [dict(r) for r in rows]


def get_today_attendance() -> list[dict]:
    return get_attendance(date.today().isoformat())


# ---------------------------------------------------------------------------
# Recognition History
# ---------------------------------------------------------------------------
def get_recognition_history(limit: int = 500) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM recognition_history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_recognition_history_by_person(person_id: int, limit: int = 100) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM recognition_history WHERE person_id=? ORDER BY timestamp DESC LIMIT ?",
            (person_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def get_stats() -> dict:
    """Return summary statistics for the status bar."""
    with _connect() as conn:
        total_persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        today_count   = conn.execute(
            "SELECT COUNT(*) FROM attendance_log WHERE date=?",
            (date.today().isoformat(),),
        ).fetchone()[0]
        total_recog   = conn.execute("SELECT SUM(recognition_count) FROM persons").fetchone()[0] or 0
    return {
        "total_persons": total_persons,
        "today_attendance": today_count,
        "total_recognitions": total_recog,
    }
