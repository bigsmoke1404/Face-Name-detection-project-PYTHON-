"""
attendance.py
-------------
Manages the once-per-day attendance feature.

Wraps database attendance functions with an in-memory fast-path to
avoid repeated DB lookups for faces that have already been logged today.
"""

import logging
from datetime import date
from typing import Optional

import database as db
import utils

logger = logging.getLogger(__name__)

# In-memory set of (person_id, date_str) already logged this session
_logged_today: set[tuple[int, str]] = set()


def record_attendance(person_id: int, name: str) -> bool:
    """
    Record attendance for *person_id* if they haven't been logged today.

    Returns True on first entry (new log created), False if already recorded.
    """
    today = date.today().isoformat()
    key   = (person_id, today)

    # Fast in-memory check
    if key in _logged_today:
        return False

    # Delegate to database (which also deduplicates)
    is_new = db.log_attendance(person_id, name)
    if is_new:
        _logged_today.add(key)
        logger.info("Attendance logged: %s (id=%d) on %s", name, person_id, today)
    else:
        # Already in DB — sync our in-memory set
        _logged_today.add(key)

    return is_new


def is_present_today(person_id: int) -> bool:
    """Return True if this person has already been logged today."""
    today = date.today().isoformat()
    return (person_id, today) in _logged_today


def get_today_records() -> list[dict]:
    """Return today's attendance records from the database."""
    return db.get_today_attendance()


def get_records_by_date(target_date: str) -> list[dict]:
    """
    Return attendance records for a specific date.
    *target_date* should be 'YYYY-MM-DD'.
    """
    return db.get_attendance(target_date)


def get_all_records() -> list[dict]:
    """Return all attendance records, most recent first."""
    return db.get_attendance()


def reset_today_cache() -> None:
    """
    Clear the in-memory cache (call at midnight / app restart
    so the next day's attendance starts fresh).
    """
    _logged_today.clear()
    logger.info("Attendance in-memory cache reset.")


def export_attendance_csv(filepath: str) -> bool:
    """Export all attendance records to a CSV file."""
    records = get_all_records()
    return utils.export_to_csv(records, filepath)
