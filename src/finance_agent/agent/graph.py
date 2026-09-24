"""Personal finance agent built on a real LangGraph ``StateGraph``.

Graph topology::

    START → guardrail → (refuse → END)
                      → rules_router → (hit → END)
                                     → planner ⇄ tools → validate
                                              ↘ final_answer → END

Nodes
-----
- **guardrail** — refuse out-of-scope / unsafe questions before any tool runs.
- **rules_router** — deterministic offline answers for common intents
  (``use_llm=False`` path; keeps demos and CI free of API keys).
- **planner** — LLM with ``bind_tools`` decides the next tool call or final prose.
- **tools** — LangGraph ``ToolNode`` executing ``run_sql`` /
  ``compute_aggregates`` / ``categorise_transactions``.
- **validate** — inspect tool results; on failure, nudge the planner to retry
  (bounded by ``max_tool_retries``).
- **final_answer** — extract / produce the user-facing answer string.

SQL SELECT-only validation lives inside ``run_sql`` (unchanged).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Sequence, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from finance_agent.agent.tools_langchain import build_tools, tool_result_failed
from finance_agent.config import Settings, get_settings
from finance_agent.llm.factory import get_chat_model
from finance_agent.tools.aggregates import compute_aggregates
from finance_agent.tools.categoriser import categorise_all
from finance_agent.tools.sql_tool import run_readonly_sql

OUT_OF_SCOPE_PATTERNS = [
    re.compile(r"\b(hack|exploit|password|ssn|social security)\b", re.I),
    re.compile(r"\b(write code|build a website|tell me a joke)\b", re.I),
    re.compile(r"\b(medical advice|diagnose|prescribe)\b", re.I),
    re.compile(r"\b(invest in stocks for me|place a trade)\b", re.I),
]

FINANCE_HINT = re.compile(
    r"\b(spend|spent|spending|income|balance|transaction|category|categor|"
    r"budget|total|how much|sql|account|grocer|rent|dining|travel|"
    r"aggregate|summary|month)\b",
    re.I,
)

REFUSAL_TEXT = (
    "I can only help with personal-finance questions about your "
    "transactions (spending, income, categories, aggregates). "
    "Please rephrase within that scope."
)

SYSTEM_PROMPT = """You are a careful personal-finance assistant.
You may ONLY answer questions about the user's bank transactions using tools.
Available tools: run_sql (SELECT only), compute_aggregates, categorise_transactions.
Never invent numbers — always use tools for figures.
If a tool fails, try a simpler SELECT or aggregates call."""


class AgentState(TypedDict, total=False):
    """Typed state flowing through the LangGraph."""

    question: str
    use_llm: bool
    messages: Annotated[list[BaseMessage], add_messages]
    answer: str
    refused: bool
    route: str
    retry_count: int
    max_retries: int
    tool_error: str | None


def is_out_of_scope(question: str) -> bool:
    """Return True when the question must be refused."""
    if any(p.search(question) for p in OUT_OF_SCOPE_PATTERNS):
        return True
    if not FINANCE_HINT.search(question) and len(question.split()) > 3:
        if re.search(r"\b(hello|hi|hey|help|what can you)\b", question, re.I):
            return False
        return True
    return False


def rule_route_answer(db_path: str, question: str) -> str | None:
    """Deterministic answers for common intents (no LLM required)."""
    q = question.lower()

    if re.search(r"categoris|categorize|label (my )?transactions", q):
        result = categorise_all(db_path, only_null=True)
        return f"Categorised {result['updated']} transactions using rule-based patterns."

    if re.search(r"by category|spending summary|breakdown|aggregate", q) or (
        "how much" in q and "category" in q
    ):
        result = compute_aggregates(db_path, group_by="category")
        lines = [
            f"- {row['key']}: total={row['total']}, spending={row['spending']}, n={row['n_txns']}"
            for row in result["rows"][:15]
        ]
        return "Spending by category:\n" + "\n".join(lines)

    if re.search(r"by month|monthly", q):
        result = compute_aggregates(db_path, group_by="month")
        lines = [
            f"- {row['key']}: total={row['total']}, spending={row['spending']}"
            for row in result["rows"][:12]
        ]
        return "Monthly totals:\n" + "\n".join(lines)

    if re.search(r"total (income|earned)", q) or "how much did i earn" in q:
        result = run_readonly_sql(
            db_path,
            "SELECT ROUND(SUM(amount), 2) AS income FROM transactions WHERE amount > 0",
        )
        if result.get("ok") and result["rows"]:
            return f"Total income: {result['rows'][0].get('income')}"
        return f"Could not compute income: {result.get('error')}"

    if re.search(r"total (spend|spent|spending|expenses)", q) or "how much did i spend" in q:
        result = run_readonly_sql(
            db_path,
            "SELECT ROUND(SUM(amount), 2) AS spending FROM transactions WHERE amount < 0",
        )
        if result.get("ok") and result["rows"]:
            return f"Total spending: {result['rows'][0].get('spending')}"
        return f"Could not compute spending: {result.get('error')}"

    m = re.search(r"spend(?:ing)? on ([a-z &]+)", q)
    if m:
        cat = m.group(1).strip().rstrip("?")
        result = run_readonly_sql(
            db_path,
            "SELECT ROUND(SUM(amount), 2) AS spending, COUNT(*) AS n "
            f"FROM transactions WHERE lower(category) = '{cat}' AND amount < 0",
        )
        if result.get("ok") and result["rows"]:
            row = result["rows"][0]
            return f"Spending on {cat}: {row.get('spending')} across {row.get('n')} transactions."

    if "top" in q and "transaction" in q:
        result = run_readonly_sql(
            db_path,
            "SELECT txn_date, description, amount, category FROM transactions "
            "ORDER BY amount ASC LIMIT 5",
        )
        if result.get("ok"):
            return "Largest expenses:\n" + json.dumps(result["rows"], indent=2)

    return None


def _last_ai_message(messages: Sequence[BaseMessage]) -> AIMessage | None:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _has_tool_calls(messages: Sequence[BaseMessage]) -> bool:
    ai = _last_ai_message(messages)
    return bool(ai and getattr(ai, "tool_calls", None))


def build_finance_graph(
    db_path: str,
    llm: BaseChatModel,
    *,
    max_tool_retries: int = 1,
) -> Any:
    """Compile the LangGraph StateGraph for this database + LLM."""

    tools = build_tools(db_path)
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def guardrail(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        if is_out_of_scope(question):
            return {
                "refused": True,
                "answer": REFUSAL_TEXT,
                "route": "guardrail",
                "messages": [HumanMessage(content=question)],
            }
        return {
            "refused": False,
            "retry_count": 0,
            "max_retries": state.get("max_retries", max_tool_retries),
            "tool_error": None,
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ],
        }

    def after_guardrail(state: AgentState) -> Literal["end", "rules_router"]:
        return "end" if state.get("refused") else "rules_router"

    def rules_router(state: AgentState) -> dict[str, Any]:
        # Only short-circuit when the caller did not ask for the LLM path.
        if state.get("use_llm"):
            return {"route": "llm"}
        hit = rule_route_answer(db_path, state["question"])
        if hit is not None:
            return {"answer": hit, "route": "rules", "refused": False}
        return {"route": "llm"}

    def after_rules(state: AgentState) -> Literal["end", "planner"]:
        if state.get("route") == "rules" and state.get("answer"):
            return "end"
        return "planner"

    def planner(state: AgentState) -> dict[str, Any]:
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "route": "llm"}

    def after_planner(state: AgentState) -> Literal["tools", "final_answer"]:
        return "tools" if _has_tool_calls(state.get("messages") or []) else "final_answer"

    def validate(state: AgentState) -> dict[str, Any]:
        """Inspect the latest tool messages; schedule a retry on failure."""
        messages = state.get("messages") or []
        errors: list[str] = []
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                # Stop once we leave the most recent tool-message block.
                if isinstance(msg, AIMessage):
                    break
                continue
            if tool_result_failed(str(msg.content)):
                errors.append(str(msg.content))
        if not errors:
            return {"tool_error": None}

        retry_count = int(state.get("retry_count") or 0)
        max_retries = int(state.get("max_retries") or max_tool_retries)
        err_text = errors[0]
        if retry_count < max_retries:
            nudge = HumanMessage(
                content=(
                    f"Tool failed: {err_text}. "
                    "Retry with a simpler valid SELECT-only query or use compute_aggregates."
                )
            )
            return {
                "tool_error": err_text,
                "retry_count": retry_count + 1,
                "messages": [nudge],
            }
        return {
            "tool_error": err_text,
            "answer": (
                f"I could not complete that request after retries. Last error: {err_text}. "
                "Try asking for a spending summary by category."
            ),
            "route": "llm",
        }

    def after_validate(state: AgentState) -> Literal["planner", "final_answer"]:
        # Exhausted retries → validate already wrote a fallback answer.
        if state.get("answer"):
            return "final_answer"
        # Tool failed and a retry nudge was appended → plan again.
        if state.get("tool_error"):
            messages = state.get("messages") or []
            if messages and isinstance(messages[-1], HumanMessage):
                return "planner"
        return "final_answer"

    def final_answer(state: AgentState) -> dict[str, Any]:
        if state.get("answer"):
            return {"refused": bool(state.get("refused")), "route": state.get("route") or "llm"}

        messages = list(state.get("messages") or [])
        ai = _last_ai_message(messages)
        # If the last AI already has prose and no pending tool calls, use it.
        if ai and not getattr(ai, "tool_calls", None) and str(ai.content).strip():
            return {
                "answer": str(ai.content),
                "refused": False,
                "route": state.get("route") or "llm",
            }

        # Otherwise ask the LLM (without forcing tools) for a grounded wrap-up.
        wrap = llm.invoke(
            messages
            + [
                HumanMessage(
                    content=(
                        "Using only the tool results above, answer the user's question "
                        "concisely. If tools failed, say so."
                    )
                )
            ]
        )
        return {
            "answer": str(wrap.content),
            "refused": False,
            "route": state.get("route") or "llm",
        }

    graph = StateGraph(AgentState)
    graph.add_node("guardrail", guardrail)
    graph.add_node("rules_router", rules_router)
    graph.add_node("planner", planner)
    graph.add_node("tools", tool_node)
    graph.add_node("validate", validate)
    graph.add_node("final_answer", final_answer)

    graph.add_edge(START, "guardrail")
    graph.add_conditional_edges(
        "guardrail",
        after_guardrail,
        {"end": END, "rules_router": "rules_router"},
    )
    graph.add_conditional_edges(
        "rules_router",
        after_rules,
        {"end": END, "planner": "planner"},
    )
    graph.add_conditional_edges(
        "planner",
        after_planner,
        {"tools": "tools", "final_answer": "final_answer"},
    )
    graph.add_edge("tools", "validate")
    graph.add_conditional_edges(
        "validate",
        after_validate,
        {"planner": "planner", "final_answer": "final_answer"},
    )
    graph.add_edge("final_answer", END)

    return graph.compile()


@dataclass
class FinanceAgent:
    """Façade that owns a compiled LangGraph and preserves the public chat API."""

    db_path: str
    llm: BaseChatModel | None = None
    settings: Settings = field(default_factory=get_settings)
    max_tool_retries: int = 1
    _graph: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.llm is None:
            self.llm = get_chat_model(self.settings, mock=True)
        self._graph = build_finance_graph(
            self.db_path, self.llm, max_tool_retries=self.max_tool_retries
        )

    def is_out_of_scope(self, question: str) -> bool:
        return is_out_of_scope(question)

    def tool_run_sql(self, sql: str) -> dict[str, Any]:
        return run_readonly_sql(self.db_path, sql)

    def tool_aggregates(
        self,
        group_by: str = "category",
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        return compute_aggregates(
            self.db_path, group_by=group_by, start_date=start_date, end_date=end_date
        )

    def tool_categorise(self) -> dict[str, Any]:
        return categorise_all(self.db_path, only_null=True)

    def chat(self, question: str, *, use_llm: bool = False) -> dict[str, Any]:
        """Run the compiled graph and return ``{answer, refused, route}``."""
        assert self._graph is not None
        result = self._graph.invoke(
            {
                "question": question,
                "use_llm": use_llm,
                "messages": [],
                "retry_count": 0,
                "max_retries": self.max_tool_retries,
            }
        )
        return {
            "answer": result.get("answer")
            or "I could not produce an answer. Please try rephrasing.",
            "refused": bool(result.get("refused")),
            "route": result.get("route"),
        }


def build_agent(
    db_path: str,
    *,
    mock_llm: bool = False,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
) -> FinanceAgent:
    """Factory used by CLI / API / tests."""
    cfg = settings or get_settings()
    model = llm if llm is not None else get_chat_model(cfg, mock=mock_llm)
    return FinanceAgent(db_path=db_path, llm=model, settings=cfg)
