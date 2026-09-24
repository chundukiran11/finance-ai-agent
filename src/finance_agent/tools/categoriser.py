"""Rule-based transaction categoriser with optional LLM fallback hook."""

from __future__ import annotations

import re
from typing import Callable

from finance_agent.db.schema import get_connection

# Ordered rules: first match wins. Patterns are matched against description.
RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"payroll|salary|direct deposit|interest credit", re.I), "income"),
    (re.compile(r"rent|landlord|mortgage", re.I), "rent"),
    (re.compile(r"wholefoods|trader joe|grocery|kroger|safeway|aldi", re.I), "groceries"),
    (re.compile(r"uber|lyft|shell|chevron|gas station|parking", re.I), "transport"),
    (re.compile(r"netflix|spotify|amc|hulu|disney\+|steam", re.I), "entertainment"),
    (re.compile(r"chipotle|starbucks|restaurant|doordash|grubhub|mcdonald", re.I), "dining"),
    (re.compile(r"pg&e|comcast|utility|electric|water bill|internet", re.I), "utilities"),
    (re.compile(r"cvs|pharmacy|fitness|dental|hospital|clinic", re.I), "health"),
    (re.compile(r"amazon|target|walmart|best buy|ikea", re.I), "shopping"),
    (re.compile(r"airbnb|airline|hotel|marriott|hilton|united air", re.I), "travel"),
    (re.compile(r"atm|cash withdrawal", re.I), "cash"),
    (re.compile(r"venmo|zelle|paypal|transfer", re.I), "transfers"),
]


def categorise_description(
    description: str,
    *,
    llm_fallback: Callable[[str], str] | None = None,
) -> str:
    """Return a category for ``description``.

    Tries rule-based patterns first. If none match and ``llm_fallback`` is
    provided, calls it. Otherwise returns ``uncategorised``.
    """
    for pattern, category in RULES:
        if pattern.search(description):
            return category
    if llm_fallback is not None:
        try:
            return llm_fallback(description)
        except Exception:
            return "uncategorised"
    return "uncategorised"


def categorise_all(
    db_path: str,
    *,
    only_null: bool = True,
    llm_fallback: Callable[[str], str] | None = None,
) -> dict:
    """Categorise transactions in the database in place."""
    with get_connection(db_path) as conn:
        if only_null:
            rows = conn.execute(
                "SELECT id, description FROM transactions WHERE category IS NULL"
            ).fetchall()
        else:
            rows = conn.execute("SELECT id, description FROM transactions").fetchall()

        updated = 0
        for row in rows:
            cat = categorise_description(row["description"], llm_fallback=llm_fallback)
            conn.execute(
                "UPDATE transactions SET category = ? WHERE id = ?",
                (cat, row["id"]),
            )
            updated += 1
        conn.commit()
    return {"updated": updated}
