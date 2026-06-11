# d:\workspace\aiops-agent\src\aiops_agent\tools\__init__.py
"""工具注册 — 每个工具模块提供 get_tools()，集中构建 TOOL_MAP。

扩展方式:
  1. 在 tools/ 下新建文件（如 file_tools.py）
  2. 定义函数，用 @tool 装饰器
  3. 实现 get_tools() 返回列表
  4. 在此文件的 get_tools() 中加上调用
"""

from langchain_core.tools import StructuredTool

from . import shell


def get_tools() -> dict[str, StructuredTool]:
    """获取所有工具，自动用 StructuredTool 包装。"""
    tool_map: dict[str, StructuredTool] = {}

    for tool_fn in shell.get_tools():
        # 函数名转工具名：execute_shell → shell
        name = tool_fn.__name__.replace("execute_", "").replace("tool_", "")
        tool_map[name] = StructuredTool.from_function(
            name=name,
            description=tool_fn.__doc__ or "",
            func=tool_fn,
        )

    return tool_map


__all__ = ["get_tools"]
