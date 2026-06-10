# d:\workspace\aiops-agent\src\aiops_agent\llm\base.py
"""LLM 抽象层 — 定义所有 LLM Provider 必须实现的接口。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator


@dataclass
class ToolCall:
    """LLM 请求调用工具的指令。"""
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    """LLM 调用的统一返回格式。"""
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"


class BaseLLM(ABC):
    """LLM 抽象基类。

    所有 Provider 适配器都必须继承此类并实现 invoke 和 stream 方法。
    """

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @abstractmethod
    def invoke(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """调用 LLM 并返回完整响应。"""
        ...

    @abstractmethod
    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[str, None, LLMResponse]:
        """流式调用 LLM。

        Yields: 逐块文本内容
        Returns: 完整的 LLMResponse（包含可能的 tool_calls）
        """
        ...

    def count_tokens(self, messages: list[dict]) -> int:
        """估算消息的 token 数量（近似值，非精确）。"""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        total += len(block["text"]) // 4
                    else:
                        total += len(str(block)) // 4
            elif isinstance(content, str):
                total += len(content) // 4
        return total
