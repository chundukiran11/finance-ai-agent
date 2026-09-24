"""SQLite schema and connection helpers."""

from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS transactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    txn_date TEXT NOT NULL,
    description TEXT NOT NULL,
    amount REAL NOT NULL,
    account TEXT NOT NULL DEFAULT 'checking',
    category TEXT,
    source_file TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(txn_date);
CREATE INDEX IF NOT EXISTS idx_txn_category ON transactions(category);
"""


def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a SQLite connection with Row factory enabled."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str) -> None:
    """Create tables if they do not exist."""
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
