"""LangChain ``@tool`` wrappers bound to a specific SQLite database path."""

from __future__ import annotations

import json
from typing import Any, Callable

from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field

from finance_agent.tools.aggregates import compute_aggregates
from finance_agent.tools.categoriser import categorise_all
from finance_agent.tools.sql_tool import run_readonly_sql


class RunSQLInput(BaseModel):
    sql: str = Field(..., description="A single read-only SELECT (or WITH…SELECT) query.")


class AggregatesInput(BaseModel):
    group_by: str = Field(
        default="category",
        description="One of: category, account, month",
    )
    start_date: str | None = Field(default=None, description="Optional ISO start date")
    end_date: str | None = Field(default=None, description="Optional ISO end date")


class CategoriseInput(BaseModel):
    only_null: bool = Field(
        default=True,
        description="If true, only fill rows where category IS NULL",
    )


def _as_json(payload: Any) -> str:
    return json.dumps(payload, default=str)


def build_tools(db_path: str) -> list[BaseTool]:
    """Build the three finance tools closed over ``db_path``."""

    def _run_sql(sql: str) -> str:
        # Catch validation / execution errors so ToolNode returns a ToolMessage
        # instead of crashing the graph; the validate node then decides to retry.
        try:
            return _as_json(run_readonly_sql(db_path, sql))
        except Exception as exc:  # noqa: BLE001
            return _as_json({"ok": False, "error": str(exc), "rows": []})

    def _aggregates(
        group_by: str = "category",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> str:
        try:
            return _as_json(
                compute_aggregates(
                    db_path, group_by=group_by, start_date=start_date, end_date=end_date
                )
            )
        except Exception as exc:  # noqa: BLE001
            return _as_json({"ok": False, "error": str(exc)})

    def _categorise(only_null: bool = True) -> str:
        return _as_json(categorise_all(db_path, only_null=only_null))

    return [
        StructuredTool.from_function(
            name="run_sql",
            description=(
                "Execute a read-only SELECT against the transactions table. "
                "Mutating statements are rejected by a hard validator."
            ),
            func=_run_sql,
            args_schema=RunSQLInput,
        ),
        StructuredTool.from_function(
            name="compute_aggregates",
            description="Sum income/spending grouped by category, account, or month.",
            func=_aggregates,
            args_schema=AggregatesInput,
        ),
        StructuredTool.from_function(
            name="categorise_transactions",
            description="Fill missing transaction categories using rule-based patterns.",
            func=_categorise,
            args_schema=CategoriseInput,
        ),
    ]


def tool_result_failed(content: str) -> bool:
    """Return True if a tool message payload indicates failure."""
    try:
        data = json.loads(content)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(data, dict):
        return False
    if data.get("ok") is False:
        return True
    if "error" in data and data.get("ok") is not True and "rows" not in data:
        return True
    return False
