# d:\workspace\aiops-agent\src\aiops_agent\graph\complex.py
"""Complex Workflow — planner → worker 执行链。"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
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
        "4. 不要指定具体的命令或参数，只需描述做什么即可\n"
        "5. **不要询问用户是否执行**——直接规划，后续会自动执行\n\n"
        f"可用 Agent:\n{agent_descs}\n\n"
        f"Agent 名称: {agent_names}\n\n"
        "使用 SubmitPlan 工具输出你的规划，格式如:\n"
        '  SubmitPlan(plan_summary="一句话描述计划", '
        'todos=["[worker] 步骤1描述", "[worker] 步骤2描述"], '
        "need_worker=true)\n"
        "如果不需要其他 Agent 执行（如纯规划反馈），need_worker 设为 false。\n"
        "注意：除了 SubmitPlan，不要调用任何其他工具。"
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

        produced_msgs, _ = agent.run(input_msgs)

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
            plan = _extract_plan_from_tool_calls(produced_msgs)
            if plan:
                raw_todos = plan.get("todos") or []
                todos = _parse_todo_items(raw_todos)
                result["todos"] = todos
                result["need_worker"] = bool(plan.get("need_worker")) and len(todos) > 0
            else:
                result["todos"] = []
                result["need_worker"] = False
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
        elif "system_prompt_template" in adef:
            tool_names = adef.get("tools", [])
            sp = adef["system_prompt_template"].format(
                tools_list=", ".join(tool_names)
            )
        else:
            sp = adef["system_prompt"]

        tools = [TOOL_MAP[t] for t in adef["tools"]]
        # planner 额外绑定 SubmitPlan 工具，强制结构化 JSON 输出
        if name == "planner":
            tools = [*tools, _build_submit_plan_tool()]
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


# ── TODO 解析 ──


_TODO_PATTERN = r'\[(\w+)\]\s*(.+)'


def _parse_todo_items(raw_todos: list[str]) -> list[dict]:
    """将 SubmitPlan 返回的字符串 TODO 列表解析为标准结构。

    输入: ["[worker] 查看磁盘使用率", "[worker] 检查日志"]
    输出: [{id, content, assignee, status}, ...]
    """
    result = []
    for i, item in enumerate(raw_todos):
        m = re.match(_TODO_PATTERN, item.strip())
        result.append({
            "id": f"todo-{i+1}",
            "content": m.group(2) if m else item.strip(),
            "assignee": m.group(1) if m else "worker",
            "status": "pending",
        })
    return result


# ── SubmitPlan Tool — 通过 tool calling 强制结构化输出 ──


from pydantic import BaseModel, Field


class _SubmitPlanArgs(BaseModel):
    plan_summary: str = Field(description="一句话描述整体计划")
    todos: list[str] = Field(
        description="任务步骤列表，每项格式为 '[agent名称] 步骤描述'，例如 '[worker] 查看磁盘使用率'"
    )
    need_worker: bool = Field(description="是否有 TODO 需要其他 Agent 执行")


def _build_submit_plan_tool() -> StructuredTool:
    """构建 SubmitPlan 工具，通过 tool calling 强制结构化输出。"""
    def _submit_plan(**kwargs) -> str:
        return json.dumps(kwargs, ensure_ascii=False)
    return StructuredTool.from_function(
        name="SubmitPlan",
        description="提交任务规划，包含步骤列表和是否需要其他 Agent 执行",
        func=_submit_plan,
        args_schema=_SubmitPlanArgs,
    )


def _extract_plan_from_tool_calls(produced_msgs: list[BaseMessage]) -> dict | None:
    """从 planner 的 tool_call 中提取 SubmitPlan 参数。

    AIMessage.tool_calls 中的每个条目始终是 dict（由 agent.run 转换），
    因此直接使用 dict 访问路径。
    """
    for msg in produced_msgs:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and tc.get("name") == "SubmitPlan":
                    return tc.get("args") or tc.get("arguments", {})
    return None
