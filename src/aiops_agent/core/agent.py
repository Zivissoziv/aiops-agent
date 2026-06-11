# d:\workspace\aiops-agent\src\aiops_agent\core\agent.py
"""Agent 核心引擎（LangGraph 实现 — ToolNode 版本）。

通用 Agent，由参数决定角色和行为:
  name: Agent 名称
  system_prompt: 角色提示词
  tools: 该 Agent 可用的工具列表

内部图结构:
  call_model → tools → call_model → ... → END
"""

import json
import operator
from dataclasses import dataclass, field
from typing import Annotated, Generator, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ..config import Config
from ..llm import BaseLLM
from ..memory import Memory
from ..tools import Tool


# ── 事件 ──

@dataclass
class AgentEvent:
    type: str  # "text" | "tool_start" | "tool_result" | "error" | "done"
    content: str = ""
    data: dict = field(default_factory=dict)


# ── LangGraph State ──

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_round: int
    max_rounds: int
    events: Annotated[list, operator.add]


# ── 工具适配 ──

def _make_tool_func(tool: Tool):
    """将项目 Tool 包装为 ToolNode 可用的可调用函数。"""
    def fn(command: str, timeout: int = 30) -> str:
        result = tool.execute(command=command, timeout=timeout)
        return json.dumps({
            "success": result.success, "output": result.output, "error": result.error,
        }, ensure_ascii=False)
    fn.__name__ = tool.name
    fn.__doc__ = tool.description
    return fn


# ── 消息转换 ──

def _lc_to_dict(msg) -> dict:
    """将 LangChain 消息对象转回 dict。"""
    if isinstance(msg, AIMessage):
        d = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"],
                              "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                for tc in msg.tool_calls
            ]
        return d
    if isinstance(msg, ToolMessage):
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    return {"role": "user", "content": msg.content}


# ── Agent 类 ──

class Agent:
    """通用 Agent，由参数决定角色和行为。

    Args:
        name: Agent 名称（如 "planner", "worker"）
        system_prompt: 角色提示词
        llm: LLM 实例
        tools: 该 Agent 可用的工具列表
        config: 应用配置
        memory: 可选的记忆模块
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: BaseLLM,
        tools: list[Tool],
        config: Config,
        memory: Memory | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools
        self.config = config
        self.memory = memory

        self._tool_funcs = [_make_tool_func(t) for t in tools]
        self._tool_defs = [t.to_openai_tool() for t in tools] or None
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 Agent 内部图: call_model → tools → call_model → ... → END"""
        builder = StateGraph(AgentState)

        builder.add_node("call_model", self._call_model)
        if self._tool_funcs:
            builder.add_node("tools", ToolNode(self._tool_funcs))
            builder.set_entry_point("call_model")
            builder.add_conditional_edges(
                "call_model", self._should_continue,
                {"continue": "tools", "end": END},
            )
            builder.add_edge("tools", "call_model")
        else:
            builder.set_entry_point("call_model")
            builder.add_edge("call_model", END)

        return builder.compile()

    # ── 节点函数 ──

    def _call_model(self, state: AgentState) -> dict:
        dict_messages = [
            _lc_to_dict(m) if not isinstance(m, dict) else m
            for m in state["messages"]
        ]

        response = self.llm.invoke(dict_messages, tools=self._tool_defs)

        ai_msg = AIMessage(content=response.content or "")
        if response.tool_calls:
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": tc.name, "args": tc.arguments, "id": tc.id}
                    for tc in response.tool_calls
                ],
            )

        events = []
        if response.content:
            events.append({"type": "text", "content": response.content, "data": {}})
        else:
            for tc in response.tool_calls:
                events.append({
                    "type": "tool_start",
                    "content": f"🔧 正在使用工具: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})",
                    "data": {"tool_name": tc.name, "arguments": tc.arguments},
                })

        ai_dict = _lc_to_dict(ai_msg)
        if self.memory is not None:
            self.memory.add_message(ai_dict)

        return {"messages": [ai_msg], "events": events}

    def _should_continue(self, state: AgentState) -> str:
        last = state["messages"][-1]
        has_tc = isinstance(last, AIMessage) and bool(last.tool_calls)
        under = state["tool_round"] < state["max_rounds"]
        return "continue" if (has_tc and under) else "end"

    def _check_compaction(self) -> None:
        if hasattr(self.memory, 'check_compaction'):
            self.memory.check_compaction()

    # ── 公开接口 ──

    def run(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> Generator[AgentEvent, None, list[dict]]:
        if self.memory is not None:
            self.memory.add_message({"role": "user", "content": user_input})
            base = [
                {"role": "system", "content": self.system_prompt},
                *self.memory.get_messages(),
            ]
        else:
            history = list(history) if history else []
            base = [
                {"role": "system", "content": self.system_prompt},
                *history,
                {"role": "user", "content": user_input},
            ]

        state: AgentState = {
            "messages": base,
            "tool_round": 0,
            "max_rounds": self.config.max_tool_rounds,
            "events": [],
        }

        max_iter = self.config.max_tool_rounds + 3
        for _ in range(max_iter):
            next_state = self._graph.invoke(state)
            state = next_state

            for evt in next_state.get("events", []):
                yield AgentEvent(type=evt["type"], content=evt["content"], data=evt["data"])

            last = next_state["messages"][-1]
            has_tc = isinstance(last, AIMessage) and bool(last.tool_calls)
            if not has_tc:
                break

            for msg in next_state["messages"]:
                if isinstance(msg, ToolMessage):
                    try:
                        content = json.loads(msg.content)
                        yield AgentEvent(
                            type="tool_result",
                            content=content.get("output", content.get("error", "")),
                            data={"tool_name": msg.name or "",
                                  "success": content.get("success", True)},
                        )
                    except (json.JSONDecodeError, TypeError):
                        yield AgentEvent(type="tool_result", content=str(msg.content)[:200], data={})

                    if self.memory is not None:
                        self.memory.add_message(_lc_to_dict(msg))

            if self.memory is not None:
                self._check_compaction()

            state["tool_round"] += 1
            if state["tool_round"] >= self.config.max_tool_rounds:
                yield AgentEvent(type="error", content=f"⚠️ 已达到最大工具调用轮次（{self.config.max_tool_rounds}）。")
                break
        else:
            yield AgentEvent(type="error", content="⚠️ 已达到最大迭代次数。")

        yield AgentEvent(type="done", content="")

        if self.memory is not None:
            return self.memory.get_messages()
        # 将最终消息转回 dict，兼容 dict 和 LC 消息混用
        result = []
        for m in state["messages"][1:]:
            if isinstance(m, dict):
                result.append(m)
            elif not isinstance(m, SystemMessage):
                result.append(_lc_to_dict(m))
        return result
