# d:\workspace\aiops-agent\src\aiops_agent\tools\__init__.py
"""工具注册 — 自动扫描 @tool 装饰器。"""

import inspect

from langchain_core.tools import StructuredTool

from . import file_tools, shell


def _iter_tools():
    for mod_name in ["file_tools", "shell"]:
        mod = globals().get(mod_name)
        if mod is None:
            continue
        for name, obj in inspect.getmembers(mod):
            if isinstance(obj, StructuredTool):
                yield obj


def get_tools() -> dict[str, StructuredTool]:
    return {tool.name: tool for tool in _iter_tools()}


__all__ = ["get_tools"]
