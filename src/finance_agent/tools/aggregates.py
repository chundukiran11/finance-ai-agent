"""Pre-built aggregate queries for common personal-finance questions."""

from __future__ import annotations

from typing import Any

from finance_agent.db.schema import get_connection


def compute_aggregates(
    db_path: str,
    *,
    group_by: str = "category",
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, Any]:
    """Sum amounts grouped by category (or account / month).

    Args:
        db_path: SQLite path.
        group_by: One of ``category``, ``account``, ``month``.
        start_date / end_date: Optional ISO date filters (inclusive).
    """
    allowed = {
        "category": "COALESCE(category, 'uncategorised')",
        "account": "account",
        "month": "substr(txn_date, 1, 7)",
    }
    if group_by not in allowed:
        raise ValueError(f"group_by must be one of {list(allowed)}")

    clauses = ["1=1"]
    params: list[Any] = []
    if start_date:
        clauses.append("txn_date >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("txn_date <= ?")
        params.append(end_date)

    expr = allowed[group_by]
    sql = f"""
        SELECT {expr} AS key,
               COUNT(*) AS n_txns,
               ROUND(SUM(amount), 2) AS total,
               ROUND(SUM(CASE WHEN amount < 0 THEN amount ELSE 0 END), 2) AS spending,
               ROUND(SUM(CASE WHEN amount > 0 THEN amount ELSE 0 END), 2) AS income
        FROM transactions
        WHERE {' AND '.join(clauses)}
        GROUP BY {expr}
        ORDER BY total ASC
    """
    with get_connection(db_path) as conn:
        cur = conn.execute(sql, params)
        rows = [dict(row) for row in cur.fetchall()]
    return {"group_by": group_by, "rows": rows}
