# d:\workspace\aiops-agent\src\aiops_agent\tools\__init__.py
"""工具注册 — 自动扫描 @tool 装饰器。

用法:
  from aiops_agent.tools import get_tools
  TOOL_MAP = get_tools()  # 自动发现所有 @tool 装饰的函数

加工具:
  1. 在 tools/ 下新建文件
  2. 用 @tool 装饰函数
  3. 在此文件 import 你的模块
"""

import inspect

from langchain_core.tools import StructuredTool

# 导入工具模块（确保 @tool 装饰的函数被扫描到）
from . import shell


def _iter_tools():
    """遍历所有已导入的工具模块，yield 被 @tool 装饰的函数。"""
    for mod_name in ["shell"]:
        mod = globals().get(mod_name)
        if mod is None:
            continue
        for name, obj in inspect.getmembers(mod):
            if isinstance(obj, StructuredTool):
                yield obj


def get_tools() -> dict[str, StructuredTool]:
    """自动扫描并返回所有工具。"""
    return {tool.name: tool for tool in _iter_tools()}


__all__ = ["get_tools"]
