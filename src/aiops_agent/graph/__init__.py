# d:\workspace\aiops-agent\src\aiops_agent\graph\__init__.py
"""LangGraph Workflow 定义 — Entry + Complex 双图架构。

用法:
    from aiops_agent.graph import build_entry_graph, build_complex_graph

    entry_graph = build_entry_graph(llm, memory)
    # ... intent routing ...
    complex_graph = build_complex_graph(config, llm, memory)
"""

from .complex import TOOL_MAP, build_complex_graph
from .entry import build_entry_graph
from .state import AppState, TodoItem

__all__ = [
    "AppState",
    "TodoItem",
    "TOOL_MAP",
    "build_entry_graph",
    "build_complex_graph",
]
