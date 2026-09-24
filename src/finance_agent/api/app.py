"""FastAPI /chat endpoint for the personal finance agent."""

from __future__ import annotations

from fastapi import FastAPI

from finance_agent.agent.graph import FinanceAgent, build_agent
from finance_agent.api.schemas import ChatRequest, ChatResponse
from finance_agent.config import get_settings

app = FastAPI(
    title="Finance AI Agent",
    description="Personal finance agent with LangGraph-style tool calling and SQL guardrails.",
    version="0.1.0",
)

_agent: FinanceAgent | None = None


def get_agent() -> FinanceAgent:
    global _agent
    if _agent is None:
        settings = get_settings()
        # Default to mock LLM so the API boots without keys; pass use_llm in body for live.
        _agent = build_agent(settings.database_path, mock_llm=True, settings=settings)
    return _agent


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    result = get_agent().chat(body.message, use_llm=body.use_llm)
    return ChatResponse(
        answer=result["answer"],
        refused=bool(result.get("refused")),
        route=result.get("route"),
    )
