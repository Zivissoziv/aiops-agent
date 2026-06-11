# d:\workspace\aiops-agent\src\aiops_agent\core\agent.py
"""Agent 核心引擎（LangGraph 实现 — ToolNode 版本）。

使用 LangGraph 状态机构造 Plan-and-Execute 循环:
  plan → call_model → tools → call_model → ... → END

节点:
  plan       : 首次调用 LLM 做任务规划
  call_model : LLM 推理，回复转为 AIMessage 供 ToolNode 消费
  tools      : ToolNode — 自动执行 tool_calls 并追加 ToolMessage

条件边:
  should_continue : 有 tool_calls 且未超限 → tools，否则 → END

消息格式转换:
  _call_model 中 LLMResponse → AIMessage (带 tool_calls)
  ToolNode 输出 ToolMessage → run() 中转为 dict 供 CLI/记忆使用
"""

import json
import operator
from dataclasses import dataclass, field
from typing import Annotated, Generator, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from ..config import Config
from ..llm import BaseLLM
from ..memory import Memory
from ..tools import Tool, ToolRegistry


# ── 事件 ──

@dataclass
class AgentEvent:
    type: str  # "plan" | "text" | "tool_start" | "tool_result" | "error" | "done"
    content: str = ""
    data: dict = field(default_factory=dict)


# ── LangGraph State ──

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_round: int
    max_rounds: int
    events: Annotated[list, operator.add]
    plan_done: bool


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

def _dict_to_lc(msg: dict):
    """将 OpenAI 格式 dict 转为 LangChain 消息对象。"""
    role = msg.get("role")
    content = msg.get("content", "")
    if role == "system":
        return SystemMessage(content=content)
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        tc = msg.get("tool_calls")
        if tc:
            return AIMessage(content=content or "", tool_calls=[
                {"name": t["function"]["name"],
                 "args": json.loads(t["function"]["arguments"]),
                 "id": t["id"]}
                for t in tc
            ])
        return AIMessage(content=content)
    if role == "tool":
        return ToolMessage(content=msg.get("content", ""), tool_call_id=msg.get("tool_call_id", ""))
    return HumanMessage(content=str(msg))


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
    """Agent 核心引擎（LangGraph Plan-and-Execute + ToolNode）。"""

    def __init__(
        self,
        config: Config,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        memory: Memory | None = None,
    ):
        self.config = config
        self.llm = llm
        self.tool_registry = tool_registry
        self.memory = memory
        # 统一工具定义：ToolNode 和 OpenAI API 共用
        self._tool_funcs = [_make_tool_func(t) for t in tool_registry.list_tools()]
        self._tool_defs = tool_registry.get_openai_tool_defs() or None
        self._graph = self._build_graph()

    def _get_tool_descriptions(self) -> str:
        lines = []
        for tool in self.tool_registry.list_tools():
            params = tool.parameters.get("properties", {})
            param_desc = ", ".join(f"{k}({v.get('type', '?')})" for k, v in params.items())
            lines.append(f"  - {tool.name}: {tool.description} 参数: {param_desc}")
        return "\n".join(lines)

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(AgentState)

        builder.add_node("plan", self._plan)
        builder.add_node("call_model", self._call_model)
        builder.add_node("tools", ToolNode(self._tool_funcs))

        builder.set_entry_point("plan")

        builder.add_conditional_edges(
            "plan", self._after_plan,
            {"call_model": "call_model", "end": END},
        )
        builder.add_conditional_edges(
            "call_model", self._should_continue,
            {"continue": "tools", "end": END},
        )
        builder.add_edge("tools", "call_model")

        return builder.compile()

    # ── Plan 节点 ──

    PLAN_PROMPT = """你是一个 AIOps 运维专家。请分析用户的任务，制定一个清晰的执行计划。

可用工具:
{tools}

请输出:
1. 任务分析: 理解用户需要什么
2. 执行计划: 分步骤列出要做的操作（每步一个工具调用）
3. 预期结果: 完成后能给用户什么信息

注意: 不需要执行工具，只需要规划。"""

    def _plan(self, state: AgentState) -> dict:
        tool_desc = self._get_tool_descriptions()
        user_msg = next(
            (m.get("content", "") for m in reversed(state["messages"]) if m.get("role") == "user"), ""
        )

        response = self.llm.invoke([
            {"role": "system", "content": self.PLAN_PROMPT.format(tools=tool_desc)},
            {"role": "user", "content": user_msg},
        ])
        plan_text = response.content or ""
        plan_dict = {"role": "assistant", "content": f"[执行计划]\n{plan_text}"}

        if self.memory is not None:
            self.memory.add_message(plan_dict)

        return {
            "messages": [plan_dict],
            "events": [{"type": "plan", "content": plan_text, "data": {}}],
            "plan_done": True,
        }

    def _after_plan(self, state: AgentState) -> str:
        return "call_model"

    # ── Call Model 节点 ──

    def _call_model(self, state: AgentState) -> dict:
        # ToolNode 输出是 LC 消息，但 LLM API 需要 dict，所以转回去
        dict_messages = [
            _lc_to_dict(m) if not isinstance(m, dict) else m
            for m in state["messages"]
        ]
        tool_defs = self._tool_defs

        response = self.llm.invoke(dict_messages, tools=tool_defs)

        # 构建 AIMessage（含 tool_calls 时 ToolNode 才能识别）
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
            # 有 tool_calls 时，记录 tool_start 事件
            for tc in response.tool_calls:
                events.append({
                    "type": "tool_start",
                    "content": f"🔧 正在使用工具: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})",
                    "data": {"tool_name": tc.name, "arguments": tc.arguments},
                })

        # 同步到 memory（转回 dict）
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
                {"role": "system", "content": self.config.system_prompt},
                *self.memory.get_messages(),
            ]
        else:
            history = list(history) if history else []
            base = [
                {"role": "system", "content": self.config.system_prompt},
                *history,
                {"role": "user", "content": user_input},
            ]

        state: AgentState = {
            "messages": base,
            "tool_round": 0,
            "max_rounds": self.config.max_tool_rounds,
            "events": [],
            "plan_done": False,
        }

        max_iter = self.config.max_tool_rounds + 3
        for _ in range(max_iter):
            next_state = self._graph.invoke(state)
            state = next_state

            for evt in next_state.get("events", []):
                yield AgentEvent(type=evt["type"], content=evt["content"], data=evt["data"])

            # 判断是否还有工具调用
            last = next_state["messages"][-1]
            has_tc = isinstance(last, AIMessage) and bool(last.tool_calls)
            if not has_tc:
                break

            # tool_result 事件从 ToolNode 的输出中提取
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

                    # 同步到 memory
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
        # 将最终 LC 消息转回 dict
        return [_lc_to_dict(m) for m in state["messages"][1:] if not isinstance(m, (SystemMessage,))]
