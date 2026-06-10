# d:\workspace\aiops-agent\src\aiops_agent\core\agent.py
"""Agent 核心引擎（LangGraph 实现）。

使用 LangGraph 状态机构造"思考-行动-观察"循环:
  call_model → execute_tools → call_model → ... → END

节点:
  call_model    : 调用 LLM，追加回复到消息列表
  execute_tools : 执行所有 tool_calls，追加结果到消息列表

条件边:
  should_continue : 有 tool_calls 且未超限 → execute_tools
                    否则 → END
"""

import json
import operator
from dataclasses import dataclass, field, asdict
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
    type: str  # "text" | "tool_start" | "tool_result" | "error" | "done"
    content: str = ""
    data: dict = field(default_factory=dict)


# ── LangGraph State ──

class AgentState(TypedDict):
    """LangGraph 状态定义。

    messages: OpenAI 格式消息列表（用 operator.add 实现 reducer）
    tool_round: 当前工具调用轮次
    max_rounds: 最大轮次限制
    events: 待输出的事件列表
    system_prompt: system prompt 文本
    llm: LLM 实例（序列化用，实际通过闭包访问）
    tool_registry: 工具注册中心（通过闭包访问）
    memory: 可选的记忆模块（通过闭包访问）
    tool_defs: OpenAI 格式工具定义
    """
    messages: Annotated[list, operator.add]
    tool_round: int
    max_rounds: int
    events: Annotated[list, operator.add]


# ── Agent 类 ──

class Agent:
    """Agent 核心引擎（LangGraph 实现）。

    使用 LangGraph 状态机驱动"思考-行动-观察"循环。
    保持与旧版相同的接口: __init__() + run()。

    用法:
      agent = Agent(config=config, llm=llm, tool_registry=registry, memory=memory)
      for event in agent.run("查一下磁盘"):
          print_event(event)
    """

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

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 状态机图。"""
        builder = StateGraph(AgentState)

        builder.add_node("call_model", self._call_model)
        builder.add_node("execute_tools", self._execute_tools)

        builder.set_entry_point("call_model")

        builder.add_conditional_edges(
            "call_model",
            self._should_continue,
            {"continue": "execute_tools", "end": END},
        )
        builder.add_edge("execute_tools", "call_model")

        return builder.compile()

    # ── 节点函数 ──

    def _call_model(self, state: AgentState) -> dict:
        """调用 LLM，将回复追加到消息列表。"""
        messages = state["messages"]
        tool_defs = self.tool_registry.get_openai_tool_defs()
        tool_defs = tool_defs or None

        response = self.llm.invoke(messages, tools=tool_defs)

        # 构建 assistant 消息
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

        # 输出文本事件
        if response.content:
            events.append({
                "type": "text",
                "content": response.content,
                "data": {},
            })

        # 同步到 memory
        if self.memory is not None:
            self.memory.add_message(assistant_msg)

        return {
            "messages": [assistant_msg],
            "events": events,
        }

    def _execute_tools(self, state: AgentState) -> dict:
        """执行所有 tool_calls，追加结果到消息列表。"""
        last_msg = state["messages"][-1]
        tool_calls = last_msg.get("tool_calls", [])
        events = []
        tool_messages = []

        for tc_data in tool_calls:
            tc_id = tc_data["id"]
            func_name = tc_data["function"]["name"]
            arguments = json.loads(tc_data["function"]["arguments"])

            # 输出 tool_start 事件
            events.append({
                "type": "tool_start",
                "content": f"🔧 正在使用工具: {func_name}({json.dumps(arguments, ensure_ascii=False)})",
                "data": {"tool_name": func_name, "arguments": arguments},
            })

            # 执行工具
            result = self.tool_registry.execute_tool(func_name, arguments)

            # 构建 tool 消息
            tool_msg = {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps(
                    {
                        "success": result.success,
                        "output": result.output,
                        "error": result.error,
                    },
                    ensure_ascii=False,
                ),
            }
            tool_messages.append(tool_msg)

            # 输出 tool_result 事件
            display_output = result.error if not result.success else result.output
            events.append({
                "type": "tool_result",
                "content": display_output,
                "data": {
                    "tool_name": func_name,
                    "success": result.success,
                    "execution_time": result.execution_time,
                },
            })

            # 同步到 memory
            if self.memory is not None:
                self.memory.add_message(tool_msg)

        # 压缩检查
        if self.memory is not None:
            self._check_compaction()

        return {
            "messages": tool_messages,
            "events": events,
        }

    def _should_continue(self, state: AgentState) -> str:
        """条件边：决定继续执行工具还是结束。"""
        last_msg = state["messages"][-1]
        has_tool_calls = bool(last_msg.get("tool_calls"))
        under_limit = state["tool_round"] < state["max_rounds"]

        if has_tool_calls and under_limit:
            return "continue"
        return "end"

    def _check_compaction(self) -> None:
        """检查并执行记忆压缩（如果 memory 支持）。"""
        if hasattr(self.memory, 'check_compaction'):
            self.memory.check_compaction()

    # ── 公开接口 ──

    def run(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> Generator[AgentEvent, None, list[dict]]:
        """运行 Agent 主循环。

        Args:
            user_input: 用户输入
            history: 历史消息列表（仅在无 memory 时使用）

        Yields:
            供 UI 展示的 AgentEvent

        Returns:
            更新后的消息历史
        """
        # 构建初始消息列表
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

        # 初始状态
        initial_state: AgentState = {
            "messages": base_messages,
            "tool_round": 0,
            "max_rounds": self.config.max_tool_rounds,
            "events": [],
        }

        # 运行 LangGraph 图
        current_state = initial_state
        max_iterations = self.config.max_tool_rounds + 2  # 安全限制
        for _ in range(max_iterations):
            next_state = self._graph.invoke(current_state)
            current_state = next_state

            # 收集并输出事件
            for evt_data in next_state.get("events", []):
                yield AgentEvent(
                    type=evt_data["type"],
                    content=evt_data["content"],
                    data=evt_data["data"],
                )

            # 检查是否应该结束
            last_msg = next_state["messages"][-1]
            has_tool_calls = bool(last_msg.get("tool_calls"))
            if not has_tool_calls:
                break

            # 增加 tool_round
            current_state["tool_round"] = current_state.get("tool_round", 0) + 1

            # 检查是否超限
            if current_state["tool_round"] >= self.config.max_tool_rounds:
                yield AgentEvent(
                    type="error",
                    content=f"⚠️ 已达到最大工具调用轮次（{self.config.max_tool_rounds}），请尝试拆分任务。",
                )
                break
        else:
            yield AgentEvent(
                type="error",
                content=f"⚠️ 已达到最大迭代次数，请尝试拆分任务。",
            )

        yield AgentEvent(type="done", content="")

        # 返回更新后的历史
        if self.memory is not None:
            return self.memory.get_messages()
        return current_state["messages"][1:]
