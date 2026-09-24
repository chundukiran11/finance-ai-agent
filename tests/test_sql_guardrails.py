import pytest

from finance_agent.tools.sql_tool import run_readonly_sql, validate_select_sql


def test_allows_select() -> None:
    assert validate_select_sql("SELECT * FROM transactions LIMIT 5").startswith("SELECT")


def test_blocks_drop() -> None:
    with pytest.raises(ValueError):
        validate_select_sql("DROP TABLE transactions")


def test_blocks_insert() -> None:
    with pytest.raises(ValueError):
        validate_select_sql("INSERT INTO transactions VALUES (1)")


def test_blocks_multi_statement() -> None:
    with pytest.raises(ValueError):
        validate_select_sql("SELECT 1; SELECT 2")


def test_run_readonly(db_path: str) -> None:
    result = run_readonly_sql(db_path, "SELECT COUNT(*) AS n FROM transactions")
    assert result["ok"] is True
    assert result["rows"][0]["n"] == 60
