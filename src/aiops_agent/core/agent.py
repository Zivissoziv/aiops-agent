# d:\workspace\aiops-agent\src\aiops_agent\core\agent.py
"""Agent 核心 — LangGraph 状态机 + 按节点绑定工具 + stream writer 实时事件。

支持 get_stream_writer() 在工具执行过程中推送实时事件，
CLI 通过 stream_mode=["updates", "custom"] 消费。
"""

import json
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.config import get_stream_writer

from ..config import Config
from ..llm import BaseLLM


# ── 事件 ──

@dataclass
class AgentEvent:
    type: str
    content: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class AgentHandoff:
    from_agent: str
    to_agent: str
    instruction: str
    result: str = ""


# ── Agent 类 ──

class Agent:
    """通用 Agent。"""

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

        self._tool_defs = [convert_to_openai_tool(t) for t in tools]
        self._tool_map = {t.name: t for t in tools}

    def _execute_tool(self, tool_name: str, tool_args: dict, writer) -> str:
        """执行工具并通过 writer 发送实时事件。"""
        tool = self._tool_map.get(tool_name)
        if not tool:
            return json.dumps({"success": False, "error": f"未知工具: {tool_name}"}, ensure_ascii=False)

        # 发送 tool_start 事件
        try:
            writer({"type": "tool_start", "tool": tool_name, "args": tool_args, "agent": self.name})
        except Exception:
            pass

        # 执行
        result_str = tool.invoke(tool_args)

        # 发送 tool_result 事件
        try:
            parsed = json.loads(result_str)
            writer({
                "type": "tool_result",
                "tool": tool_name,
                "success": parsed.get("success", False),
                "output": parsed.get("output", "")[:500],
                "error": parsed.get("error", "")[:200],
                "agent": self.name,
            })
        except Exception:
            pass

        return result_str

    def run(
        self,
        input_messages: list[BaseMessage],
    ) -> tuple[list[BaseMessage], list[AgentEvent]]:
        """运行 Agent 的 think-act-observe 循环。"""
        writer = self._get_writer()
        messages = [SystemMessage(content=self.system_prompt), *input_messages]
        produced: list[BaseMessage] = []
        events: list[AgentEvent] = []

        # 发送 agent_start 事件
        try:
            writer({"type": "agent_start", "agent": self.name})
        except Exception:
            pass

        for _round in range(self.config.max_tool_rounds):
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

            if not response.tool_calls:
                break

            for tc in response.tool_calls:
                events.append(AgentEvent(
                    type="tool_start",
                    content=f"🔧 [{self.name}] 正在使用工具: {tc.name}",
                    data={"tool_name": tc.name, "arguments": tc.arguments},
                ))

                tool_result = self._execute_tool(tc.name, tc.arguments, writer)

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
            events.append(AgentEvent(type="error", content=f"⚠️ 已达到最大工具调用轮次。"))

        events.append(AgentEvent(type="done", content=""))
        return produced, events

    @staticmethod
    def _get_writer():
        """安全的 get_stream_writer 封装。"""
        try:
            return get_stream_writer()
        except Exception:
            # 不在 graph 流中时返回 no-op
            return lambda _: None
