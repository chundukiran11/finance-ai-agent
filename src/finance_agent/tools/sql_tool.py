"""Read-only SQL tool with strict validation guardrails."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from finance_agent.db.schema import get_connection

# Keywords that indicate a mutating / dangerous statement.
_BLOCKED = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|DETACH|"
    r"PRAGMA|VACUUM|REINDEX|TRIGGER|GRANT|REVOKE|TRUNCATE|INTO)\b",
    re.IGNORECASE,
)


def validate_select_sql(sql: str) -> str:
    """Validate that ``sql`` is a single read-only SELECT (or WITH…SELECT).

    Raises:
        ValueError: If the statement is empty, multi-statement, or not SELECT.
    """
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise ValueError("SQL query is empty")
    if ";" in cleaned:
        raise ValueError("Multiple SQL statements are not allowed")
    if _BLOCKED.search(cleaned):
        raise ValueError("Only read-only SELECT queries are allowed")
    # Must start with SELECT or WITH (CTE)
    head = cleaned.split(None, 1)[0].upper()
    if head not in {"SELECT", "WITH"}:
        raise ValueError("Query must start with SELECT or WITH")
    return cleaned


def run_readonly_sql(db_path: str, sql: str, limit: int = 100) -> dict[str, Any]:
    """Execute a validated SELECT and return rows as dictionaries.

    Guardrail: wraps the user SQL as a subquery with a hard ``LIMIT`` so a
    runaway query cannot dump the whole table by accident.
    """
    cleaned = validate_select_sql(sql)
    wrapped = f"SELECT * FROM ({cleaned}) AS q LIMIT {int(limit)}"
    try:
        with get_connection(db_path) as conn:
            cur = conn.execute(wrapped)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
    except sqlite3.Error as exc:
        return {"ok": False, "error": str(exc), "rows": []}
    return {"ok": True, "columns": cols, "rows": rows, "n": len(rows)}
