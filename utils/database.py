"""
database.py
SQLite helper for storing / retrieving scan history.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "phishguard.db")


def _get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the scans table if it does not exist."""
    conn = _get_connection()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            url         TEXT    NOT NULL,
            result      TEXT    NOT NULL,
            ml_pred     TEXT    NOT NULL,
            ml_conf     REAL    NOT NULL,
            nn_pred     TEXT    NOT NULL,
            nn_conf     REAL    NOT NULL,
            scanned_at  TEXT    NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_scan(url, result, ml_pred, ml_conf, nn_pred, nn_conf):
    """Insert a new scan record."""
    conn = _get_connection()
    conn.execute(
        """
        INSERT INTO scans (url, result, ml_pred, ml_conf, nn_pred, nn_conf, scanned_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (url, result, ml_pred, round(ml_conf, 4),
         nn_pred, round(nn_conf, 4),
         datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()


def get_all_scans(limit=100):
    """Return the most recent scans as a list of dicts."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT * FROM scans ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_stats():
    """Return aggregate statistics."""
    conn = _get_connection()
    row = conn.execute(
        """
        SELECT
            COUNT(*)                                              AS total,
            COALESCE(SUM(CASE WHEN result='Phishing'   THEN 1 ELSE 0 END), 0) AS phishing,
            COALESCE(SUM(CASE WHEN result='Legitimate' THEN 1 ELSE 0 END), 0) AS legitimate
        FROM scans
        """
    ).fetchone()
    conn.close()
    return dict(row) if row else {"total": 0, "phishing": 0, "legitimate": 0}
