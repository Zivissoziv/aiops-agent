# d:\workspace\aiops-agent\src\aiops_agent\tools\base.py
"""工具系统基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolResult:
    """工具执行结果。"""
    success: bool
    output: str = ""
    error: str = ""
    execution_time: float = 0.0


class Tool(ABC):
    """工具基类。

    所有运维工具必须继承此类。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（给 LLM 看的标识符）。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述（给 LLM 看的说明，影响 LLM 选择工具的准确性）。"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """工具参数 JSON Schema 定义。"""
        ...

    def to_openai_tool(self) -> dict:
        """转换为 OpenAI 工具定义格式。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具。"""
        ...
