"""LangGraph-based personal finance agent with tool calling and guardrails.

Design:
- Intent gate refuses out-of-scope questions before any tool runs.
- Tools: read-only SQL, aggregates, categorise.
- On tool failure, the agent retries once with a fallback message.
- When no LLM key is available, a deterministic rule-based router answers
  common questions so demos and tests still work.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from finance_agent.config import Settings, get_settings
from finance_agent.llm.factory import get_chat_model
from finance_agent.tools.aggregates import compute_aggregates
from finance_agent.tools.categoriser import categorise_all, categorise_description
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

SYSTEM_PROMPT = """You are a careful personal-finance assistant.
You may ONLY answer questions about the user's bank transactions using tools.
Available tools: run_sql (SELECT only), compute_aggregates, categorise_transactions.
Refuse questions outside personal finance. Never invent numbers — always use tools.
If a tool fails, explain the error and suggest a simpler query."""


class AgentState(TypedDict, total=False):
    question: str
    messages: list
    tool_results: list
    answer: str
    refused: bool


@dataclass
class FinanceAgent:
    """High-level façade over the finance tool loop."""

    db_path: str
    llm: BaseChatModel | None = None
    settings: Settings = field(default_factory=get_settings)
    max_tool_retries: int = 1

    def __post_init__(self) -> None:
        if self.llm is None:
            self.llm = get_chat_model(self.settings, mock=True)

    # ----- Guardrails -----------------------------------------------------

    def is_out_of_scope(self, question: str) -> bool:
        if any(p.search(question) for p in OUT_OF_SCOPE_PATTERNS):
            return True
        # Soft check: if it looks nothing like finance, refuse.
        if not FINANCE_HINT.search(question) and len(question.split()) > 3:
            # Allow short greetings through the soft check
            if re.search(r"\b(hello|hi|hey|help|what can you)\b", question, re.I):
                return False
            return True
        return False

    # ----- Tools exposed to the LLM / router ------------------------------

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

    # ----- Deterministic router (no live LLM required) --------------------

    def _rule_route(self, question: str) -> str | None:
        """Answer common questions without an LLM so tests stay offline."""
        q = question.lower()

        if re.search(r"categoris|categorize|label (my )?transactions", q):
            result = self.tool_categorise()
            return f"Categorised {result['updated']} transactions using rule-based patterns."

        if re.search(r"by category|spending summary|breakdown|aggregate", q) or (
            "how much" in q and "category" in q
        ):
            result = self.tool_aggregates(group_by="category")
            lines = [
                f"- {row['key']}: total={row['total']}, spending={row['spending']}, n={row['n_txns']}"
                for row in result["rows"][:15]
            ]
            return "Spending by category:\n" + "\n".join(lines)

        if re.search(r"by month|monthly", q):
            result = self.tool_aggregates(group_by="month")
            lines = [
                f"- {row['key']}: total={row['total']}, spending={row['spending']}"
                for row in result["rows"][:12]
            ]
            return "Monthly totals:\n" + "\n".join(lines)

        if re.search(r"total (income|earned)", q) or "how much did i earn" in q:
            result = self.tool_run_sql(
                "SELECT ROUND(SUM(amount), 2) AS income FROM transactions WHERE amount > 0"
            )
            if result.get("ok") and result["rows"]:
                return f"Total income: {result['rows'][0].get('income')}"
            return f"Could not compute income: {result.get('error')}"

        if re.search(r"total (spend|spent|spending|expenses)", q) or "how much did i spend" in q:
            result = self.tool_run_sql(
                "SELECT ROUND(SUM(amount), 2) AS spending FROM transactions WHERE amount < 0"
            )
            if result.get("ok") and result["rows"]:
                return f"Total spending: {result['rows'][0].get('spending')}"
            return f"Could not compute spending: {result.get('error')}"

        m = re.search(r"spend(?:ing)? on ([a-z &]+)", q)
        if m:
            cat = m.group(1).strip().rstrip("?")
            result = self.tool_run_sql(
                f"SELECT ROUND(SUM(amount), 2) AS spending, COUNT(*) AS n "
                f"FROM transactions WHERE lower(category) = '{cat}' AND amount < 0"
            )
            if result.get("ok") and result["rows"]:
                row = result["rows"][0]
                return f"Spending on {cat}: {row.get('spending')} across {row.get('n')} transactions."

        if "top" in q and "transaction" in q:
            result = self.tool_run_sql(
                "SELECT txn_date, description, amount, category FROM transactions "
                "ORDER BY amount ASC LIMIT 5"
            )
            if result.get("ok"):
                return "Largest expenses:\n" + json.dumps(result["rows"], indent=2)

        return None

    def _llm_tool_loop(self, question: str) -> str:
        """Best-effort LLM tool-calling loop with one retry on failure."""
        assert self.llm is not None
        tools_desc = (
            "Tools (call by writing JSON on its own line like "
            '{"tool":"run_sql","sql":"SELECT ..."} or '
            '{"tool":"aggregates","group_by":"category"} or '
            '{"tool":"categorise"}):\n'
            "- run_sql: read-only SELECT against transactions\n"
            "- aggregates: group sums by category|account|month\n"
            "- categorise: fill null categories with rules\n"
        )
        messages = [
            SystemMessage(content=SYSTEM_PROMPT + "\n" + tools_desc),
            HumanMessage(content=question),
        ]
        last_error: str | None = None
        for attempt in range(self.max_tool_retries + 1):
            try:
                response = self.llm.invoke(messages)
                text = str(response.content)
                # Try to detect a tool call JSON blob
                tool_match = re.search(r"\{[^{}]+\}", text)
                if tool_match:
                    payload = json.loads(tool_match.group())
                    tool = payload.get("tool")
                    if tool == "run_sql":
                        result = self.tool_run_sql(payload.get("sql", "SELECT 1"))
                    elif tool == "aggregates":
                        result = self.tool_aggregates(
                            group_by=payload.get("group_by", "category"),
                            start_date=payload.get("start_date"),
                            end_date=payload.get("end_date"),
                        )
                    elif tool == "categorise":
                        result = self.tool_categorise()
                    else:
                        result = {"ok": False, "error": f"Unknown tool {tool}"}
                    if isinstance(result, dict) and result.get("ok") is False:
                        last_error = str(result.get("error"))
                        messages.append(AIMessage(content=text))
                        messages.append(
                            HumanMessage(
                                content=f"Tool failed: {last_error}. Try a simpler SELECT."
                            )
                        )
                        continue
                    messages.append(AIMessage(content=text))
                    messages.append(
                        HumanMessage(
                            content=f"Tool result:\n{json.dumps(result, default=str)[:3000]}\n"
                            "Now answer the user concisely using only these numbers."
                        )
                    )
                    final = self.llm.invoke(messages)
                    return str(final.content)
                return text
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                continue
        return (
            f"I could not complete that request after retries. Last error: {last_error}. "
            "Try asking for a spending summary by category."
        )

    def chat(self, question: str, *, use_llm: bool = False) -> dict[str, Any]:
        """Answer a user question with guardrails and tools.

        Args:
            question: Natural-language question.
            use_llm: If True, use the configured LLM tool loop. If False
                (default for tests), use the deterministic rule router and
                only fall back to the (possibly mock) LLM when needed.
        """
        if self.is_out_of_scope(question):
            return {
                "answer": (
                    "I can only help with personal-finance questions about your "
                    "transactions (spending, income, categories, aggregates). "
                    "Please rephrase within that scope."
                ),
                "refused": True,
                "route": "guardrail",
            }

        # Prefer deterministic router for reliability / offline use
        routed = self._rule_route(question)
        if routed is not None and not use_llm:
            return {"answer": routed, "refused": False, "route": "rules"}

        if use_llm:
            answer = self._llm_tool_loop(question)
            return {"answer": answer, "refused": False, "route": "llm"}

        # Fallback: try rules again, then mock/live LLM prose
        if routed is not None:
            return {"answer": routed, "refused": False, "route": "rules"}

        assert self.llm is not None
        response = self.llm.invoke(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=question),
            ]
        )
        return {"answer": str(response.content), "refused": False, "route": "llm-fallback"}


def build_agent(
    db_path: str,
    *,
    mock_llm: bool = False,
    settings: Settings | None = None,
) -> FinanceAgent:
    """Factory used by CLI / API."""
    cfg = settings or get_settings()
    llm = get_chat_model(cfg, mock=mock_llm)
    return FinanceAgent(db_path=db_path, llm=llm, settings=cfg)
