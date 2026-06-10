# d:\workspace\aiops-agent\src\aiops_agent\tools\registry.py
"""工具注册中心 — 管理所有可用工具。"""

from typing import Any

from .base import Tool, ToolResult


class ToolRegistry:
    """工具注册中心。

    负责:
    1. 注册/注销工具
    2. 生成 LLM 可理解的工具定义列表
    3. 根据名称调度工具执行
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册一个工具。"""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """注销一个工具。"""
        self._tools.pop(name, None)

    def get_tool(self, name: str) -> Tool | None:
        """根据名称获取工具。"""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """获取所有已注册的工具列表。"""
        return list(self._tools.values())

    def get_openai_tool_defs(self) -> list[dict]:
        """获取所有工具的 OpenAI 格式定义。"""
        return [tool.to_openai_tool() for tool in self._tools.values()]

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        """根据名称和参数执行工具。

        Args:
            name: 工具名称
            arguments: 工具参数字典

        Returns:
            工具执行结果
        """
        tool = self.get_tool(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"未知工具: '{name}'。可用工具: {', '.join(self._tools.keys())}",
            )
        return tool.execute(**arguments)
