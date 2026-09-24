from __future__ import annotations

import csv
from pathlib import Path

import pytest

from finance_agent.db.ingest import generate_synthetic_statement, ingest_csv
from finance_agent.db.schema import init_db
from finance_agent.agent.graph import build_agent
from finance_agent.tools.categoriser import categorise_all

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    csv_path = tmp_path / "stmt.csv"
    generate_synthetic_statement(csv_path, n_transactions=60, seed=0)
    db = str(tmp_path / "finance.db")
    init_db(db)
    ingest_csv(db, csv_path, use_csv_category=False)
    categorise_all(db)
    return db


@pytest.fixture
def agent(db_path: str):
    return build_agent(db_path, mock_llm=True)
