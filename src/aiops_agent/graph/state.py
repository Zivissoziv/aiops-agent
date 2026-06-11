# d:\workspace\aiops-agent\src\aiops_agent\graph\state.py
"""AppState — LangGraph 全局状态定义。"""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AppState(TypedDict):
    """贯穿 entry 和 complex 两个 graph 的全局状态。"""

    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    need_worker: bool
    todos: list[str]

    # ── 路由字段 — entry workflow 写入，cli.py 读取后决定是否跑 complex ──
    intent_route: str
    intent_reason: str
    intent_confidence: float

    # ── Chat 分支 — entry workflow 写入，cli.py 读取后输出 ──
    chat_response: str

    # ── Session 上下文 — cli.py 写入，entry/complex 都读取 ──
    session_context: str
