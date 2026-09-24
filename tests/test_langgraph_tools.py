"""Exercise the real LangGraph path with a mocked LLM that emits tool calls."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from finance_agent.agent.graph import build_agent
from finance_agent.llm.factory import MockChatModel


def test_graph_tool_call_aggregates(db_path: str) -> None:
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "compute_aggregates",
                    "args": {"group_by": "category"},
                    "id": "call_agg_1",
                }
            ],
        ),
        AIMessage(content="Here is your spending breakdown by category from the tools."),
    ]
    llm = MockChatModel(script=script)
    agent = build_agent(db_path, llm=llm)
    result = agent.chat(
        "Please use tools to give me a custom spending analysis by category",
        use_llm=True,
    )
    assert result["refused"] is False
    assert result["route"] == "llm"
    assert "category" in result["answer"].lower() or "breakdown" in result["answer"].lower()


def test_graph_tool_call_sql(db_path: str) -> None:
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_sql",
                    "args": {
                        "sql": "SELECT ROUND(SUM(amount), 2) AS spending FROM transactions WHERE amount < 0"
                    },
                    "id": "call_sql_1",
                }
            ],
        ),
        AIMessage(content="Total spending was computed from the ledger."),
    ]
    llm = MockChatModel(script=script)
    agent = build_agent(db_path, llm=llm)
    result = agent.chat("Compute my spending via SQL please", use_llm=True)
    assert result["refused"] is False
    assert "spending" in result["answer"].lower()


def test_graph_retries_on_bad_sql(db_path: str) -> None:
    script = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_sql",
                    "args": {"sql": "DELETE FROM transactions"},
                    "id": "call_bad_1",
                }
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "compute_aggregates",
                    "args": {"group_by": "category"},
                    "id": "call_agg_2",
                }
            ],
        ),
        AIMessage(content="Recovered with a category spending summary."),
    ]
    llm = MockChatModel(script=script)
    agent = build_agent(db_path, llm=llm)
    result = agent.chat(
        "Run a bad SQL then summarise my spending by category",
        use_llm=True,
    )
    assert result["refused"] is False
    assert result["answer"]
    assert "spending" in result["answer"].lower() or "category" in result["answer"].lower() or "Recovered" in result["answer"]


def test_mock_bind_tools() -> None:
    llm = MockChatModel()
    bound = llm.bind_tools([])
    assert bound is llm
    assert bound.bound_tools == []
