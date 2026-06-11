# d:\workspace\aiops-agent\src\aiops_agent\core\agent.py
"""Agent 核心 — LangGraph 状态机 + 按节点绑定工具。

核心设计：
  1. 消息统一使用 LC 对象（AIMessage, ToolMessage, HumanMessage, SystemMessage）
  2. State 中包含 messages、memory_snapshot、agent_handoffs 等字段
  3. 每个节点独立绑定工具（model.bind_tools）
  4. Memory 从 State 中现场构建，不在外部持有单独实例
"""

import json
from dataclasses import dataclass, field
from typing import Generator

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from ..config import Config
from ..llm import BaseLLM


# ── 事件 ──

@dataclass
class AgentEvent:
    type: str  # "text" | "tool_start" | "tool_result" | "handoff" | "memory" | "error" | "done"
    content: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class AgentHandoff:
    """Agent 间的交接记录。"""
    from_agent: str
    to_agent: str
    instruction: str
    result: str = ""


# ── Agent 类 ──

class Agent:
    """通用 Agent。

    不持有 memory，不持有全局工具注册表。
    每次 run() 接收消息列表，执行 think-act-observe 循环。

    用法:
      agent = Agent(name="worker", system_prompt="...", llm=llm, tools=[ShellTool()], config=config)
      messages, events = agent.run([HumanMessage("查磁盘")])
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: BaseLLM,
        tools: list[StructuredTool],
        config: Config,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools
        self.config = config

        # 工具定义
        self._tool_defs = [convert_to_openai_tool(t) for t in tools]
        self._tool_map = {t.name: t for t in tools}

    def _build_messages(self, input_messages: list[BaseMessage]) -> list[BaseMessage]:
        """构建完整消息列表：system prompt + 输入消息。"""
        result: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        result.extend(input_messages)
        return result

    def _execute_tool(self, tool_name: str, tool_args: dict) -> str:
        """执行一个工具并返回 JSON 字符串结果。"""
        tool = self._tool_map.get(tool_name)
        if not tool:
            return json.dumps({"success": False, "error": f"未知工具: {tool_name}"}, ensure_ascii=False)
        # StructuredTool.invoke 返回的就是函数的返回值（已经是 JSON 字符串）
        return tool.invoke(tool_args)

    def run(
        self,
        input_messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], list[AgentEvent]]:
        """运行 Agent 的 think-act-observe 循环。

        Args:
            input_messages: 输入消息列表（不包含 system prompt）

        Returns:
            (produced_messages, events)
            produced_messages: Agent 产生的消息（AIMessage + ToolMessage）
            events: 供 UI 展示的事件
        """
        messages = self._build_messages(input_messages)
        produced: list[BaseMessage] = []
        events: list[AgentEvent] = []

        for _round in range(self.config.max_tool_rounds):
            # ── Think ──
            response = self.llm.invoke(messages, tools=self._tool_defs or None)

            ai_msg = AIMessage(content=response.content or "")
            if response.tool_calls:
                ai_msg = AIMessage(
                    content="",
                    tool_calls=[
                        {"name": tc.name, "args": tc.arguments, "id": tc.id}
                        for tc in response.tool_calls
                    ],
                )

            produced.append(ai_msg)
            messages.append(ai_msg)

            if response.content:
                events.append(AgentEvent(type="text", content=response.content))

            # 没有工具调用 → 结束
            if not response.tool_calls:
                break

            # ── Act ──
            for tc in response.tool_calls:
                events.append(AgentEvent(
                    type="tool_start",
                    content=f"🔧 [{self.name}] 正在使用工具: {tc.name}",
                    data={"tool_name": tc.name, "arguments": tc.arguments},
                ))

                tool_result = self._execute_tool(tc.name, tc.arguments)

                tool_msg = ToolMessage(content=tool_result, tool_call_id=tc.id, name=tc.name)
                produced.append(tool_msg)
                messages.append(tool_msg)

                try:
                    parsed = json.loads(tool_result)
                    display = parsed.get("output", parsed.get("error", ""))
                except json.JSONDecodeError:
                    display = tool_result[:200]

                events.append(AgentEvent(
                    type="tool_result",
                    content=display,
                    data={"tool_name": tc.name, "success": True},
                ))

        else:
            events.append(AgentEvent(
                type="error",
                content=f"⚠️ 已达到最大工具调用轮次（{self.config.max_tool_rounds}）。",
            ))

        events.append(AgentEvent(type="done", content=""))
        return produced, events
