"""CSV ingestion and synthetic bank-statement generation.

The synthetic generator is clearly labelled: every description is prefixed
with ``[SYNTHETIC]`` and the README documents that the data is fake.
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

from finance_agent.db.schema import get_connection, init_db

# Merchant templates used by the synthetic generator (NOT real transactions).
SYNTHETIC_MERCHANTS = [
    ("WHOLEFOODS MARKET", "groceries", -45, -120),
    ("TRADER JOES", "groceries", -30, -90),
    ("UBER TRIP", "transport", -8, -35),
    ("LYFT RIDE", "transport", -8, -40),
    ("SHELL GAS STATION", "transport", -25, -70),
    ("NETFLIX.COM", "entertainment", -15.99, -15.99),
    ("SPOTIFY USA", "entertainment", -10.99, -10.99),
    ("AMC THEATRES", "entertainment", -12, -40),
    ("CHIPOTLE", "dining", -10, -28),
    ("STARBUCKS", "dining", -4, -12),
    ("PG&E UTILITY", "utilities", -60, -140),
    ("COMCAST CABLE", "utilities", -50, -90),
    ("CVS PHARMACY", "health", -8, -55),
    ("PLANET FITNESS", "health", -22.99, -22.99),
    ("AMAZON.COM", "shopping", -15, -150),
    ("TARGET STORE", "shopping", -20, -100),
    ("AIRBNB", "travel", -80, -350),
    ("UNITED AIRLINES", "travel", -150, -600),
    ("LANDLORD RENT", "rent", -1800, -1800),
    ("ACME CORP PAYROLL", "income", 3200, 4500),
    ("INTEREST CREDIT", "income", 1, 15),
    ("ATM WITHDRAWAL", "cash", -40, -200),
    ("VENMO PAYMENT", "transfers", -20, -200),
    ("ZELLE TRANSFER", "transfers", -25, -300),
]


def generate_synthetic_statement(
    output_path: str | Path,
    n_transactions: int = 300,
    seed: int = 42,
    start: date | None = None,
) -> Path:
    """Write a synthetic bank-statement CSV with ~``n_transactions`` rows.

    Every description is prefixed with ``[SYNTHETIC]`` so the data is
    unambiguously fake.
    """
    rng = random.Random(seed)
    start = start or date(2024, 1, 1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i in range(n_transactions):
        merchant, category, lo, hi = SYNTHETIC_MERCHANTS[i % len(SYNTHETIC_MERCHANTS)]
        # Mix in a few random day offsets for realism
        day = start + timedelta(days=rng.randint(0, 364))
        amount = round(rng.uniform(lo, hi), 2)
        # Income stays positive; expenses stay negative
        if category == "income":
            amount = abs(amount)
        elif amount > 0 and category != "income":
            amount = -abs(amount)
        rows.append(
            {
                "txn_date": day.isoformat(),
                "description": f"[SYNTHETIC] {merchant}",
                "amount": amount,
                "account": rng.choice(["checking", "checking", "savings"]),
                "category": category,  # ground-truth label for evaluation
            }
        )

    rows.sort(key=lambda r: r["txn_date"])
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["txn_date", "description", "amount", "account", "category"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def ingest_csv(db_path: str, csv_path: str | Path, *, use_csv_category: bool = False) -> int:
    """Load a bank-statement CSV into SQLite.

    Args:
        db_path: SQLite file path.
        csv_path: CSV with at least txn_date, description, amount.
        use_csv_category: If True, store the CSV ``category`` column
            (used when loading labelled eval data). Otherwise leave
            category NULL for the categoriser to fill.

    Returns:
        Number of rows inserted.
    """
    init_db(db_path)
    csv_path = Path(csv_path)
    inserted = 0
    with get_connection(db_path) as conn, csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            category = row.get("category") if use_csv_category else None
            conn.execute(
                """
                INSERT INTO transactions
                    (txn_date, description, amount, account, category, source_file)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row["txn_date"],
                    row["description"],
                    float(row["amount"]),
                    row.get("account") or "checking",
                    category,
                    csv_path.name,
                ),
            )
            inserted += 1
        conn.commit()
    return inserted
