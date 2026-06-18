# d:\\workspace\\aiops-agent\\src\\aiops_agent\\graph\\complex.py
"""Complex Workflow — planner → worker 执行链，带 TODO 状态验证。"""

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


# ── 常量 ──

MAX_WORKER_ROUNDS = 3  # 每个 Worker 最大连续执行轮次（防无限循环）


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
        "你是一个 AIOps 智能助手。根据用户的输入，决定是直接回复还是规划任务。\n\n"
        "三种场景：\n\n"
        "1. 闲聊/问候（如 你好、谢谢、你是谁）\n"
        "   直接回复，使用 SubmitPlan(need_worker=false, plan_summary=\"闲聊回复\", todos=[]) 结束。\n\n"
        "2. 询问公司内部规范/配置（如端口分配、命名规则、告警流程）\n"
        "   调用 retrieve_knowledge 工具查询知识库，然后基于查询结果回复。\n"
        "   使用 SubmitPlan(need_worker=false) 结束。\n\n"
        "3. 需要执行操作的运维任务\n"
        f"   拆解为步骤，指定执行 Agent：\n"
        f"   可用 Agent:\n{agent_descs}\n"
        f"   Agent 名称: {agent_names}\n\n"
        "使用 SubmitPlan 工具输出结果，格式如：\n"
        '  SubmitPlan(plan_summary="一句话描述", '
        'todos=["[worker] 步骤1", "[worker] 步骤2"], '
        "need_worker=true)\n"
        "只有需要其他 Agent 执行时 need_worker 才设为 true。\n"
        "不要询问用户是否执行——直接分析和输出。"
    )


# ── TODO 状态工具调用解析 ──


def _apply_todo_updates_from_messages(
    todos: list[dict],
    produced_msgs: list[BaseMessage],
) -> list[dict]:
    """扫描 produced_msgs 中的 update_todo 工具调用，更新 todos 状态。

    在 worker 执行完成后调用，将 tool_call 中的结构化数据同步到 AppState.todos。
    """
    todo_map = {t["id"]: t for t in todos}
    for msg in produced_msgs:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and tc.get("name") == "update_todo":
                    args = tc.get("args") or tc.get("arguments", {})
                    todo_id = args.get("todo_id", "")
                    status = args.get("status", "")
                    if todo_id in todo_map and status in (
                        "in_progress", "completed", "blocked",
                    ):
                        todo_map[todo_id]["status"] = status
    return list(todo_map.values())


def _all_todos_completed(todos: list[dict]) -> bool:
    """检查是否所有 TODO 都已完成或阻塞（没有 pending/in_progress）。"""
    if not todos:
        return True
    return all(t.get("status") in ("completed", "blocked") for t in todos)


# ── 路由函数（模块级，可测试） ──

# LangGraph 返回 "__end__" 作为 END 标识
_END_SENTINEL = "__end__"


def _route_from_planner(state: dict, worker_name: str = "worker") -> str:
    """planner 条件边：need_worker=True → 走 worker，否则 END。"""
    return worker_name if state.get("need_worker", True) else _END_SENTINEL


def _route_from_worker(state: dict, worker_name: str = "worker") -> str:
    """worker 条件边：全部完成或超上限 → END，否则继续 worker。"""
    todos = state.get("todos", [])
    worker_round = state.get("worker_round", 0)
    if _all_todos_completed(todos):
        return _END_SENTINEL
    if worker_round >= MAX_WORKER_ROUNDS:
        return _END_SENTINEL
    return worker_name


# ── 节点工厂 ──


def _make_node(name: str, agent: Agent, memory: TieredMemory):
    """创建 LangGraph 节点函数。

    支持 session_context 注入到 planner 的输入中。
    自动将 LLM 输出同步到 TieredMemory。
    Worker 执行后自动扫描 update_todo 工具调用，更新 todos 状态。
    """
    def node_fn(state: AppState, writer: StreamWriter) -> dict:
        writer({"type": "agent_start", "agent": name})
        input_msgs: list[BaseMessage] = list(state.get("messages", []))
        if not input_msgs:
            input_msgs = [HumanMessage(content=state.get("task", ""))]

        # planner 只取本轮用户输入 + session_context
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
                # 只显示未完成的 TODO（已完成的不再提示）
                active = [t for t in todos if t.get("status") not in ("completed", "blocked")]
                if active:
                    todo_lines = "\n".join(
                        f"  [{todo.get('id', '?')}] {todo.get('content', '')} (assignee: {todo.get('assignee', 'worker')})"
                        for todo in active
                    )
                    input_msgs.insert(0, HumanMessage(
                        content=f"下面是规划好的 TODO 列表，请按顺序执行:\n{todo_lines}"
                    ))
                    input_msgs.insert(0, SystemMessage(
                        content="以下 TODO 列表由 planner 分配給你。请按顺序执行，"
                                "每完成一个步骤调用 update_todo 工具标记状态。\n"
                                "全部完成后在对话中报告最终结果。"
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
            # Worker 执行后：扫描 update_todo 工具调用，更新 todos 状态
            current_todos = state.get("todos", [])
            updated_todos = _apply_todo_updates_from_messages(current_todos, produced_msgs)
            result["todos"] = updated_todos
            result["need_worker"] = state.get("need_worker", True)

            # 跟踪 Worker 执行轮次（防止无限循环）
            current_rounds = state.get("worker_round", 0)
            result["worker_round"] = current_rounds + 1

        writer({"type": "agent_end", "agent": name})

        result["messages"] = produced_msgs
        return result

    return node_fn


def build_complex_graph(config: Config, llm, memory: TieredMemory) -> StateGraph:
    """构建 Complex Workflow：
       planner → (need_worker) → worker → (全部完成?) → END
       planner → (no worker) → END
       worker  → (有未完成) → worker（重试，最多 MAX_WORKER_ROUNDS 次）
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
        # planner 额外绑定 SubmitPlan 工具
        if name == "planner":
            tools = [*tools, _build_submit_plan_tool()]
        agent = Agent(name=name, system_prompt=sp, llm=llm, tools=tools, config=config)

        builder.add_node(name, _make_node(name, agent, memory))

    names = [a["name"] for a in ALL_AGENTS]
    builder.set_entry_point(names[0])

    if len(names) >= 2:
        worker_name = names[1]

        def _planner_route(state):
            return _route_from_planner(state, worker_name)
        def _worker_route(state):
            return _route_from_worker(state, worker_name)

        builder.add_conditional_edges(
            names[0], _planner_route, {worker_name: worker_name, END: END}
        )
        builder.add_conditional_edges(
            names[1], _worker_route, {worker_name: worker_name, END: END},
        )

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
    """从 planner 的 tool_call 中提取 SubmitPlan 参数。"""
    for msg in produced_msgs:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and tc.get("name") == "SubmitPlan":
                    return tc.get("args") or tc.get("arguments", {})
    return None
