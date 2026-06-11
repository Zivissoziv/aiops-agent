# d:\workspace\aiops-agent\src\aiops_agent\core\agent.py
"""Agent 核心引擎（LangGraph 实现）。

使用 LangGraph 状态机构造 Plan-and-Execute 循环:
  plan → call_model → execute_tools → call_model → ... → END

节点:
  plan          : 首次调用 LLM 做任务规划，生成执行计划
  call_model    : LLM 推理，决定调用工具
  execute_tools : 执行所有 tool_calls，追加结果到消息列表

条件边:
  should_continue : 有 tool_calls 且未超限 → execute_tools，否则 → END

注: 工具执行使用手写循环而非 ToolNode，避免引入 langchain 消息类型的依赖。
"""

import json
import operator
from dataclasses import dataclass, field
from typing import Annotated, Generator, TypedDict

from langgraph.graph import END, StateGraph

from ..config import Config
from ..llm import BaseLLM
from ..memory import Memory
from ..tools import ToolRegistry


# ── 事件 ──

@dataclass
class AgentEvent:
    """Agent 执行过程中发出的事件，供 CLI/UI 展示。"""
    type: str  # "plan" | "text" | "tool_start" | "tool_result" | "error" | "done"
    content: str = ""
    data: dict = field(default_factory=dict)


# ── LangGraph State ──

class AgentState(TypedDict):
    """LangGraph 状态定义。"""
    messages: Annotated[list, operator.add]
    tool_round: int
    max_rounds: int
    events: Annotated[list, operator.add]
    plan_done: bool


# ── Agent 类 ──

class Agent:
    """Agent 核心引擎（LangGraph Plan-and-Execute）。"""

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
        self._graph = self._build_graph()

    def _get_tool_descriptions(self) -> str:
        """生成工具描述文本，供 plan 节点参考。"""
        lines = []
        for tool in self.tool_registry.list_tools():
            params = tool.parameters.get("properties", {})
            param_desc = ", ".join(
                f"{k}({v.get('type', '?')})" for k, v in params.items()
            )
            lines.append(f"  - {tool.name}: {tool.description} 参数: {param_desc}")
        return "\n".join(lines)

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态机图。

        图结构:
          plan (首次) → call_model → execute_tools → call_model (循环)
                           ↓ (无 tool_calls)
                          END
        """
        builder = StateGraph(AgentState)

        builder.add_node("plan", self._plan)
        builder.add_node("call_model", self._call_model)
        builder.add_node("execute_tools", self._execute_tools)

        builder.set_entry_point("plan")

        builder.add_conditional_edges(
            "plan", self._after_plan,
            {"call_model": "call_model", "end": END},
        )
        builder.add_conditional_edges(
            "call_model", self._should_continue,
            {"continue": "execute_tools", "end": END},
        )
        builder.add_edge("execute_tools", "call_model")

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
        plan_prompt = self.PLAN_PROMPT.format(tools=tool_desc)

        user_msg = next(
            (m.get("content", "") for m in reversed(state["messages"])
             if m.get("role") == "user"), ""
        )

        response = self.llm.invoke([
            {"role": "system", "content": plan_prompt},
            {"role": "user", "content": user_msg},
        ])
        plan_text = response.content or ""

        plan_msg = {"role": "assistant", "content": f"[执行计划]\n{plan_text}"}
        if self.memory is not None:
            self.memory.add_message(plan_msg)

        return {
            "messages": [plan_msg],
            "events": [{"type": "plan", "content": plan_text, "data": {}}],
            "plan_done": True,
        }

    def _after_plan(self, state: AgentState) -> str:
        return "call_model"

    # ── Call Model 节点 ──

    def _call_model(self, state: AgentState) -> dict:
        messages = state["messages"]
        tool_defs = self.tool_registry.get_openai_tool_defs() or None

        response = self.llm.invoke(messages, tools=tool_defs)

        assistant_msg: dict = {"role": "assistant"}
        if response.content:
            assistant_msg["content"] = response.content
        elif response.tool_calls:
            assistant_msg["content"] = ""
        if response.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                    },
                }
                for tc in response.tool_calls
            ]

        events = []
        if response.content:
            events.append({"type": "text", "content": response.content, "data": {}})

        if self.memory is not None:
            self.memory.add_message(assistant_msg)

        return {"messages": [assistant_msg], "events": events}

    # ── Execute Tools 节点 ──

    def _execute_tools(self, state: AgentState) -> dict:
        """执行所有 tool_calls，追加结果到消息列表。"""
        last_msg = state["messages"][-1]
        tool_calls = last_msg.get("tool_calls", [])
        events = []
        tool_messages = []

        for tc_data in tool_calls:
            func_name = tc_data["function"]["name"]
            arguments = json.loads(tc_data["function"]["arguments"])

            events.append({
                "type": "tool_start",
                "content": f"🔧 正在使用工具: {func_name}({json.dumps(arguments, ensure_ascii=False)})",
                "data": {"tool_name": func_name, "arguments": arguments},
            })

            result = self.tool_registry.execute_tool(func_name, arguments)

            tool_msg = {
                "role": "tool",
                "tool_call_id": tc_data["id"],
                "content": json.dumps(
                    {"success": result.success, "output": result.output, "error": result.error},
                    ensure_ascii=False,
                ),
            }
            tool_messages.append(tool_msg)

            display = result.error if not result.success else result.output
            events.append({
                "type": "tool_result",
                "content": display,
                "data": {"tool_name": func_name, "success": result.success,
                         "execution_time": result.execution_time},
            })

            if self.memory is not None:
                self.memory.add_message(tool_msg)

        if self.memory is not None:
            self._check_compaction()

        return {"messages": tool_messages, "events": events}

    def _should_continue(self, state: AgentState) -> str:
        last_msg = state["messages"][-1]
        has_tool_calls = bool(last_msg.get("tool_calls"))
        under_limit = state["tool_round"] < state["max_rounds"]
        return "continue" if (has_tool_calls and under_limit) else "end"

    def _check_compaction(self) -> None:
        if hasattr(self.memory, 'check_compaction'):
            self.memory.check_compaction()

    # ── 公开接口 ──

    def run(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> Generator[AgentEvent, None, list[dict]]:
        """运行 Agent（Plan-and-Execute）。"""
        if self.memory is not None:
            self.memory.add_message({"role": "user", "content": user_input})
            base_messages = [
                {"role": "system", "content": self.config.system_prompt},
                *self.memory.get_messages(),
            ]
        else:
            history = list(history) if history else []
            base_messages = [
                {"role": "system", "content": self.config.system_prompt},
                *history,
                {"role": "user", "content": user_input},
            ]

        current_state: AgentState = {
            "messages": base_messages,
            "tool_round": 0,
            "max_rounds": self.config.max_tool_rounds,
            "events": [],
            "plan_done": False,
        }

        max_iter = self.config.max_tool_rounds + 3
        for _ in range(max_iter):
            next_state = self._graph.invoke(current_state)
            current_state = next_state

            for evt in next_state.get("events", []):
                yield AgentEvent(type=evt["type"], content=evt["content"], data=evt["data"])

            has_tool_calls = bool(next_state["messages"][-1].get("tool_calls"))
            if not has_tool_calls:
                break

            current_state["tool_round"] += 1
            if current_state["tool_round"] >= self.config.max_tool_rounds:
                yield AgentEvent(type="error", content=f"⚠️ 已达到最大工具调用轮次（{self.config.max_tool_rounds}）。")
                break
        else:
            yield AgentEvent(type="error", content="⚠️ 已达到最大迭代次数。")

        yield AgentEvent(type="done", content="")

        if self.memory is not None:
            return self.memory.get_messages()
        return current_state["messages"][1:]
