"""Agent 核心 — think-act-observe 循环 + stream writer 实时事件。"""

import json, logging
from dataclasses import dataclass, field
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.config import get_stream_writer
from ..config import Config
from ..llm import BaseLLM

logger = logging.getLogger(__name__)


@dataclass
class AgentEvent:
    type: str
    content: str = ""
    data: dict = field(default_factory=dict)


@dataclass
class AgentHandoff:
    from_agent: str; to_agent: str; instruction: str; result: str = ""


class Agent:
    def __init__(self, name: str, system_prompt: str, llm: BaseLLM, tools: list[StructuredTool], config: Config):
        self.name, self.system_prompt, self.llm, self.tools, self.config = name, system_prompt, llm, tools, config
        self._tool_defs = [convert_to_openai_tool(t) for t in tools]
        self._tool_map = {t.name: t for t in tools}

    def _execute_tool(self, name: str, args: dict, writer) -> str:
        tool = self._tool_map.get(name)
        if not tool:
            return json.dumps({"success": False, "error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            writer({"type": "tool_start", "tool": name, "args": args, "agent": self.name})
        except Exception:
            pass
        result = tool.invoke(args)
        try:
            p = json.loads(result)
            writer({"type": "tool_result", "tool": name, "success": p.get("success", False), "output": p.get("output", "")[:500], "error": p.get("error", "")[:200], "agent": self.name})
        except Exception:
            pass
        return result

    def run(self, input_messages: list[BaseMessage]) -> tuple[list[BaseMessage], list[AgentEvent]]:
        w = self._get_writer()
        msgs = [SystemMessage(content=self.system_prompt), *input_messages]
        produced, events = [], []
        try:
            w({"type": "agent_start", "agent": self.name})
        except Exception:
            pass

        for _ in range(self.config.max_tool_rounds):
            resp = self.llm.invoke(msgs, tools=self._tool_defs or None)
            ai = AIMessage(content=resp.content or "")
            if resp.tool_calls:
                ai = AIMessage(content="", tool_calls=[{"name": tc.name, "args": tc.arguments, "id": tc.id} for tc in resp.tool_calls])
            produced.append(ai); msgs.append(ai)
            if resp.content:
                events.append(AgentEvent(type="text", content=resp.content))
            if not resp.tool_calls:
                break
            for tc in resp.tool_calls:
                events.append(AgentEvent(type="tool_start", content=f"🔧 [{self.name}] 正在使用工具: {tc.name}", data={"tool_name": tc.name, "arguments": tc.arguments}))
                r = self._execute_tool(tc.name, tc.arguments, w)
                tm = ToolMessage(content=r, tool_call_id=tc.id, name=tc.name)
                produced.append(tm); msgs.append(tm)
                try:
                    display = (json.loads(r).get("output") or json.loads(r).get("error", ""))
                except json.JSONDecodeError:
                    display = r[:200]
                events.append(AgentEvent(type="tool_result", content=display, data={"tool_name": tc.name, "success": True}))
        else:
            events.append(AgentEvent(type="error", content="⚠️ 已达到最大工具调用轮次。"))

        events.append(AgentEvent(type="done", content=""))
        return produced, events

    @staticmethod
    def _get_writer():
        try:
            return get_stream_writer()
        except Exception:
            return lambda _: None
