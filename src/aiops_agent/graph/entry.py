"""Entry Workflow — 意图路由 + 轻量 Chat 回复。"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from ..core.chat_responder import chat_responder as chat_fn
from ..core.intent_router import classify_intent
from ._utils import _get_writer
from .state import AppState

def _build_intent_router_node(llm):
    def node(state: AppState) -> dict:
        w = _get_writer()
        r = classify_intent(llm, state["task"], session_context=state.get("session_context", ""))
        w({"type": "intent_decision", "route": r["route"], "reason": r["reason"], "confidence": r["confidence"]})
        return {"intent_route": r["route"], "intent_reason": r["reason"], "intent_confidence": r["confidence"]}
    return node

def _build_chat_responder_node(llm, memory):
    def node(state: AppState) -> dict:
        w = _get_writer()
        r = chat_fn(llm, state["task"], session_context=state.get("session_context", ""), reason=state.get("intent_reason", ""))
        w({"type": "chat_response", "route": "chat", "response": r["response"]})
        memory.add_message({"role": "user", "content": state["task"]})
        memory.add_message({"role": "assistant", "content": r["response"]})
        return {"chat_response": r["response"], "messages": [HumanMessage(content=state["task"]), AIMessage(content=r["response"])]}
    return node

def build_entry_graph(llm, memory) -> StateGraph:
    builder = StateGraph(AppState)
    builder.add_node("intent_router", _build_intent_router_node(llm))
    builder.add_node("chat_responder", _build_chat_responder_node(llm, memory))
    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges("intent_router", lambda s: "chat_responder" if s.get("intent_route") == "chat" else END, {"chat_responder": "chat_responder", END: END})
    builder.add_edge("chat_responder", END)
    return builder.compile()
