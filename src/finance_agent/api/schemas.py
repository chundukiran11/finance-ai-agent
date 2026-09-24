from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    use_llm: bool = False


class ChatResponse(BaseModel):
    answer: str
    refused: bool = False
    route: str | None = None
