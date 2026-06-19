"""工具注册 — 自动扫描 @tool 装饰器。"""
import inspect
from langchain_core.tools import StructuredTool
from . import file_tools, knowledge_tool, shell, todo_tool

def _iter_tools():
    for mod_name in ["file_tools", "knowledge_tool", "shell", "todo_tool"]:
        m = globals().get(mod_name)
        if m:
            for _, obj in inspect.getmembers(m):
                if isinstance(obj, StructuredTool): yield obj

def get_tools() -> dict[str, StructuredTool]:
    return {t.name: t for t in _iter_tools()}

__all__ = ["get_tools"]
