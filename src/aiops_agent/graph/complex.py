# d:\workspace\aiops-agent\src\aiops_agent\graph\complex.py
"""Complex Workflow — planner → worker 执行链。"""

import json
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
    agent_names = [a["name"] for a in others]
    return (
        "你是一个 AIOps 运维规划专家。你的职责:\n"
        "1. 分析用户的任务，将任务拆解为具体的步骤\n"
        "2. 为每个步骤指定合适的执行 Agent\n"
        "3. 如果任务无法由任何 Agent 完成（没有合适的工具），直接回复原因\n"
        "4. 不要指定具体的命令或参数，只需描述做什么即可\n\n"
        f"可用 Agent:\n{agent_descs}\n\n"
        f"Agent 名称: {agent_names}\n\n"
        "在回复末尾用 ```json 代码块返回规划，格式:\n"
        "```json\n"
        "{\n"
        '  "plan_summary": "一句话描述整体计划",\n'
        '  "todos": [\n'
        '    {"content": "步骤描述（不指定具体命令）", "assignee": "agent名称"},\n'
        '    ...\n'
        '  ],\n'
        '  "need_worker": true\n'
        "}\n"
        "```\n\n"
        "如果不需要其他 Agent 执行（如纯聊天、纯规划），need_worker 设为 false。\n"
        "不要执行工具，只需要输出规划和 JSON。"
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

        # 将结构化 todos 附加到 worker 的上下文
        if name != "planner":
            todos = state.get("todos", [])
            if todos:
                todo_lines = "\n".join(
                    f"  [{todo.get('id', '?')}] {todo.get('content', '')} (assignee: {todo.get('assignee', 'worker')})"
                    for todo in todos
                )
                input_msgs.insert(0, HumanMessage(
                    content=f"下面是规划好的 TODO 列表，请按顺序执行:\n{todo_lines}"
                ))
                input_msgs.insert(0, SystemMessage(
                    content="以下 TODO 列表由 planner 分配給你。请按顺序执行，完成后在对话中报告状态。"
                ))

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
            parsed = _extract_json(reply)
            if parsed:
                raw_todos = parsed.get("todos") or []
                todos = [
                    {
                        "id": f"todo-{i+1}",
                        "content": str(todo.get("content", "")),
                        "assignee": str(todo.get("assignee", "worker")),
                        "status": "pending",
                    }
                    for i, todo in enumerate(raw_todos)
                    if isinstance(todo, dict) and todo.get("content")
                ]
                result["todos"] = todos
                result["need_worker"] = bool(parsed.get("need_worker")) and len(todos) > 0
            else:
                # JSON 解析失败，回退：默认有 worker
                result["todos"] = _fallback_todos(reply)
                result["need_worker"] = len(result["todos"]) > 0
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


# ── JSON 解析 ──


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型回复中提取 JSON 对象（最外层大括号）。"""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _fallback_todos(reply: str) -> list[dict]:
    """当 JSON 解析失败时，从旧格式文本中提取 TODO 作为回退。"""
    import re
    items = re.findall(r'- \[TODO\]\s*(.+)', reply)
    if not items:
        items = re.findall(r'- \*?\*?TODO\*?\*?\s*:?\s*(.+)', reply, re.IGNORECASE)
    return [
        {
            "id": f"todo-{i+1}",
            "content": item.strip(),
            "assignee": "worker",
            "status": "pending",
        }
        for i, item in enumerate(items)
    ]
