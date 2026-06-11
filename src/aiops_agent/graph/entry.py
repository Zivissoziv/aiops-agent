# d:\workspace\aiops-agent\src\aiops_agent\graph\entry.py
"""Entry Workflow — 意图路由 + 轻量 Chat 回复。"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from ..core.chat_responder import chat_responder as chat_responder_fn
from ..core.intent_router import classify_intent
from ._utils import _get_writer
from .state import AppState


def _build_intent_router_node(llm):
    """Intent Router 节点工厂 — 判断 chat/task，写入 state。"""
    def intent_router_node(state: AppState) -> dict:
        writer = _get_writer()

        user_input = state.get("task", "")
        session_context = state.get("session_context", "")

        result = classify_intent(llm, user_input, session_context=session_context)

        writer({
            "type": "intent_decision",
            "route": result["route"],
            "reason": result["reason"],
            "confidence": result["confidence"],
        })

        return {
            "intent_route": result["route"],
            "intent_reason": result["reason"],
            "intent_confidence": result["confidence"],
        }
    return intent_router_node


def _build_chat_responder_node(llm, memory):
    """Chat Responder 节点工厂 — 轻量回复，不走工具。"""
    def chat_responder_node(state: AppState) -> dict:
        writer = _get_writer()

        user_input = state.get("task", "")
        session_context = state.get("session_context", "")

        result = chat_responder_fn(
            llm,
            user_input,
            session_context=session_context,
            reason=state.get("intent_reason", ""),
        )
        text = result["response"]

        writer({
            "type": "chat_response",
            "route": "chat",
            "response": text,
        })

        # 同步到三层记忆
        memory.add_message({"role": "user", "content": user_input})
        memory.add_message({"role": "assistant", "content": text})

        return {
            "chat_response": text,
            "messages": [
                HumanMessage(content=user_input),
                AIMessage(content=text),
            ],
        }
    return chat_responder_node


def _intent_route_fn(state: AppState) -> str:
    """条件边路由：chat → chat_responder，其他 → END。"""
    if state.get("intent_route") == "chat":
        return "chat_responder"
    return "route_to_end"


def build_entry_graph(llm, memory) -> StateGraph:
    """构建 Entry Workflow。

    流程：
      START → intent_router
                ├── chat → chat_responder → END
                └── task → END（cli.py 监听 intent_decision 事件后决定是否跑 complex）
    """
    builder = StateGraph(AppState)

    builder.add_node("intent_router", _build_intent_router_node(llm))
    builder.add_node("chat_responder", _build_chat_responder_node(llm, memory))

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        _intent_route_fn,
        {
            "chat_responder": "chat_responder",
            "route_to_end": END,
        },
    )
    builder.add_edge("chat_responder", END)

    return builder.compile()
