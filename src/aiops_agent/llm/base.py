"""LLM 抽象层 — 定义所有 LLM Provider 必须实现的接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class ToolCall:
    id: str; name: str; arguments: dict


@dataclass
class LLMResponse:
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class BaseLLM(ABC):
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @abstractmethod
    def invoke(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse: ...

    @abstractmethod
    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Generator[str, None, LLMResponse]: ...

    def count_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            c = m.get("content", "")
            total += len(c) // 4 if isinstance(c, str) else sum(len(str(b)) // 4 for b in c if isinstance(b, dict))
        return total
