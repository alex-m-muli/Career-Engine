"""
Application tracking database module for the Career Engine.

Provides a lightweight SQLite wrapper to track applied jobs, prevent 
duplicate submissions, and log pipeline statuses.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict

from models import ApplicationRecord

logger = logging.getLogger("ApplicationDB")

DB_NAME = "applications.db"


def _get_connection() -> sqlite3.Connection:
    """
    Creates a new SQLite connection.
    Enables WAL mode for better concurrency and sets a busy timeout.
    """
    conn = sqlite3.connect(DB_NAME, timeout=10.0)
    conn.row_factory = sqlite3.Row
    # WAL mode allows concurrent reads and writes, avoiding "database is locked" errors
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Creates the applications table if it does not already exist."""
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS applications (
                job_url TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                title TEXT NOT NULL,
                fit_score REAL NOT NULL,
                pdf_path TEXT,
                submitted_at TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)
        conn.commit()
    logger.info("Database initialized.")


def is_applied(job_url: str) -> bool:
    """
    Checks if a job URL has already been recorded in the database.
    
    Returns:
        True if the URL exists, False otherwise.
    """
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT 1 FROM applications WHERE job_url = ? LIMIT 1", 
            (job_url,)
        )
        row = cursor.fetchone()
        return row is not None


def record_application(record: ApplicationRecord) -> None:
    """
    Inserts or updates an application record.
    Uses INSERT OR REPLACE to handle updates if the same URL is processed again.
    """
    with _get_connection() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO applications 
            (job_url, company, title, fit_score, pdf_path, submitted_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            record.job_url,
            record.company,
            record.title,
            record.fit_score,
            record.pdf_path,
            record.submitted_at,
            record.status
        ))
        conn.commit()
    logger.info(f"Recorded application: {record.company} - {record.title} [{record.status}]")


def update_status(job_url: str, status: str) -> None:
    """
    Updates the status of an existing application (e.g., from 'pending' to 'applied').
    """
    with _get_connection() as conn:
        conn.execute(
            "UPDATE applications SET status = ? WHERE job_url = ?", 
            (status, job_url)
        )
        conn.commit()
    logger.info(f"Updated status for {job_url} to '{status}'.")


def get_stats() -> Dict[str, int]:
    """
    Returns a summary of application counts grouped by status.
    
    Returns:
        A dictionary mapping status strings to counts (e.g., {"applied": 5, "skipped": 10}).
    """
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT status, COUNT(*) as count FROM applications GROUP BY status"
        )
        rows = cursor.fetchall()
        
    stats = {row["status"]: row["count"] for row in rows}
    return stats