"""Provider-agnostic LLM factory (gemini | openai | anthropic)."""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field, PrivateAttr

from finance_agent.config import Settings, get_settings


class MockChatModel(BaseChatModel):
    """Deterministic chat model for tests.

    Supports ``bind_tools`` and an optional ``script`` of ``AIMessage``s so
    LangGraph tool-calling loops can be exercised without an API key.
    """

    canned_response: str = Field(
        default="I can help with your finances using the available tools."
    )
    script: list[AIMessage] = Field(default_factory=list)
    bound_tools: list[Any] = Field(default_factory=list)
    _call_index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "mock-chat"

    def bind_tools(
        self,
        tools: Sequence[BaseTool | dict[str, Any] | type],
        **kwargs: Any,
    ) -> "MockChatModel":
        """Attach tools and return ``self`` so scripted call indices stay shared.

        Real chat models return a binding over the same underlying client; we
        mirror that by mutating in place instead of ``model_copy``.
        """
        self.bound_tools = list(tools)
        return self

    def _next_message(self) -> AIMessage:
        if self.script:
            idx = min(self._call_index, len(self.script) - 1)
            self._call_index += 1
            return self.script[idx]
        return AIMessage(content=self.canned_response)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next_message())])


def get_chat_model(
    settings: Settings | None = None,
    *,
    mock: bool = False,
    mock_response: str | None = None,
    mock_script: list[AIMessage] | None = None,
) -> BaseChatModel:
    if mock:
        return MockChatModel(
            canned_response=mock_response
            or "I can help with your finances using the available tools.",
            script=list(mock_script or []),
        )

    cfg = settings or get_settings()
    provider = cfg.llm_provider.lower()

    if provider == "openai":
        if not cfg.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=cfg.llm_model, api_key=cfg.openai_api_key, temperature=0)

    if provider == "anthropic":
        if not cfg.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=cfg.llm_model, api_key=cfg.anthropic_api_key, temperature=0)

    if provider == "gemini":
        if not cfg.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=cfg.llm_model, google_api_key=cfg.gemini_api_key, temperature=0
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
