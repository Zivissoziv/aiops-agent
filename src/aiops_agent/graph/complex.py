# d:\workspace\aiops-agent\src\aiops_agent\graph\complex.py
"""Complex Workflow — planner → worker 执行链。"""

import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.types import StreamWriter

from ..agents import ALL_AGENTS
from ..config import Config
from ..core import Agent
from ..memory.tiered import TieredMemory
from ..tools import get_tools
from ._utils import _get_writer
from .state import AppState


# ── 工具注册 ──

TOOL_MAP: dict[str, Any] = get_tools()


# ── Planner Prompt ──


def _build_planner_prompt() -> str:
    """动态生成 planner 的 system_prompt，注入其他 Agent 的描述。"""
    others = [a for a in ALL_AGENTS if a["name"] != "planner"]
    agent_descs = "\n".join(
        f"  - {a['name']}: {a.get('description', '未描述')}" for a in others
    )
    return (
        "你是一个 AIOps 运维规划专家。你的职责:\n"
        "1. 分析用户的任务\n"
        "2. 将任务拆解为具体的 TODO 步骤，每个 TODO 一步操作\n"
        "3. 用 [TODO] 标记每个步骤\n"
        "4. 每行一个 TODO，格式: - [TODO] 具体操作描述\n"
        "5. 根据任务类型，分配给合适的 Agent 执行\n\n"
        f"可用 Agent:\n{agent_descs}\n\n"
        "6. 如果任务无法由任何 Agent 完成（没有合适的工具），"
        "直接告知用户原因，**不要**输出 [NEED_WORKER]\n"
        "7. 如果任务可以分配给其他 Agent 执行，在**最后一行**单独输出 [NEED_WORKER]\n"
        "8. 如果只是打招呼、问简单问题，直接回复即可\n"
        "注意: [NEED_WORKER] 只能出现在最后一行，不要在前文出现\n\n"
        "不要执行工具，只需要输出规划。"
    )



def _make_node(name: str, agent: Agent, memory: TieredMemory):
    """创建 LangGraph 节点函数。

    支持 session_context 注入到 planner 的输入中。
    自动将 LLM 输出同步到 TieredMemory。
    """
    def node_fn(state: AppState, writer: StreamWriter) -> dict:
        writer({"type": "agent_start", "agent": name})
        input_msgs: list[BaseMessage] = list(state.get("messages", []))
        if not input_msgs:
            input_msgs = [HumanMessage(content=state.get("task", ""))]

        # planner 只取本轮用户输入 + session_context，避免看到执行细节后重复执行
        if name == "planner":
            task = state.get("task", "")
            session_context = state.get("session_context", "")
            parts = [f"Task: {task}"]
            if session_context:
                parts.append("Session context for this multi-turn session:\n" + session_context)
            input_msgs = [HumanMessage(content="\n\n".join(parts))]

        # 从三层记忆注入额外上下文（core + episodic）
        mem_context = memory.get_messages()
        extra_context = [m for m in mem_context if m.get("role") in ("system", "assistant")]
        if extra_context:
            ctx_msgs: list[BaseMessage] = []
            for ctx in extra_context:
                if ctx["role"] == "system":
                    ctx_msgs.append(SystemMessage(content=ctx["content"]))
                elif ctx["role"] == "assistant":
                    ctx_msgs.append(AIMessage(content=ctx["content"]))
            input_msgs = [*ctx_msgs, *input_msgs]

        produced_msgs, events = agent.run(input_msgs)

        reply = ""
        for m in reversed(produced_msgs):
            if hasattr(m, "content") and m.content:
                reply = m.content
                break

        # 同步到三层记忆
        for msg in produced_msgs:
            if hasattr(msg, "type"):
                role_map = {"human": "user", "ai": "assistant", "tool": "tool"}
                role = role_map.get(getattr(msg, "type", ""), "assistant")
                if role == "tool":
                    memory.add_message({
                        "role": "tool",
                        "content": msg.content,
                        "tool_call_id": getattr(msg, "tool_call_id", ""),
                        "name": getattr(msg, "name", ""),
                    })
                else:
                    memory.add_message({"role": role, "content": msg.content or ""})
        memory.check_compaction()

        result: dict[str, Any] = {}

        if name == "planner":
            todos = re.findall(r'- \[TODO\]\s*(.+)', reply)
            result["todos"] = todos
            last_lines = reply.strip().split("\n")[-3:]
            result["need_worker"] = any("[NEED_WORKER]" in line for line in last_lines)
        else:
            result["need_worker"] = state.get("need_worker", True)

        result["messages"] = produced_msgs
        return result

    return node_fn


def build_complex_graph(config: Config, llm, memory: TieredMemory) -> StateGraph:
    """构建 Complex Workflow：
       planner → (need_worker) → worker → END
       planner → (no worker) → END
    """
    builder = StateGraph(AppState)

    for adef in ALL_AGENTS:
        name = adef["name"]
        if name == "planner" and adef.get("system_prompt") is None:
            sp = _build_planner_prompt()
        else:
            sp = adef["system_prompt"]

        tools = [TOOL_MAP[t] for t in adef["tools"]]
        agent = Agent(name=name, system_prompt=sp, llm=llm, tools=tools, config=config)

        builder.add_node(name, _make_node(name, agent, memory))

    names = [a["name"] for a in ALL_AGENTS]
    builder.set_entry_point(names[0])

    if len(names) >= 2:
        def route(state: AppState) -> str:
            return names[1] if state.get("need_worker", True) else END
        builder.add_conditional_edges(names[0], route, {names[1]: names[1], END: END})
        for i in range(1, len(names) - 1):
            builder.add_edge(names[i], names[i + 1])
        builder.add_edge(names[-1], END)

    return builder.compile()
