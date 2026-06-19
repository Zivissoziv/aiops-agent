"""Complex Workflow — planner → worker 执行链，带 TODO 状态验证。"""

import json, re
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.types import StreamWriter
from pydantic import BaseModel, Field
from ..agents import ALL_AGENTS
from ..config import Config
from ..core import Agent
from ..memory.tiered import TieredMemory
from ..tools import get_tools
from ._utils import _get_writer
from .state import AppState

MAX_WORKER_ROUNDS = 3
TOOL_MAP = get_tools()


# ── Planner Prompt ──

def _build_planner_prompt() -> str:
    others = [a for a in ALL_AGENTS if a["name"] != "planner"]
    descs = "\n".join(f"  - {a['name']}: {a.get('description', '')}" for a in others)
    names = [a["name"] for a in others]
    return (
        "你是一个 AIOps 智能助手。根据用户的输入，决定是直接回复还是规划任务。\n\n"
        "三种场景：\n\n"
        "1. 闲聊/问候 — 直接回复，使用 SubmitPlan(need_worker=false) 结束。\n\n"
        "2. 询问公司内部规范/配置 — 调用 retrieve_knowledge 查询，然后回复。使用 SubmitPlan(need_worker=false) 结束。\n\n"
        "3. 需要执行操作的运维任务\n"
        f"   可用 Agent:\n{descs}\n   名称: {names}\n\n"
        "使用 SubmitPlan 输出：SubmitPlan(plan_summary=..., todos=[\"[worker] 步骤\"], need_worker=true)\n"
        "只有需要其他 Agent 执行时 need_worker 才设为 true。直接分析和输出，不要询问用户。"
    )


# ── TODO 工具调用解析 ──

def _apply_todo_updates(todos: list[dict], msgs: list[BaseMessage]) -> list[dict]:
    m = {t["id"]: t for t in todos}
    for msg in msgs:
        for tc in getattr(msg, "tool_calls", []) or []:
            if isinstance(tc, dict) and tc.get("name") == "update_todo":
                a = tc.get("args") or tc.get("arguments", {})
                tid, st = a.get("todo_id", ""), a.get("status", "")
                if tid in m and st in ("in_progress", "completed", "blocked"):
                    m[tid]["status"] = st
    return list(m.values())

def _all_done(todos: list[dict]) -> bool:
    return all(t.get("status") in ("completed", "blocked") for t in todos) if todos else True


# ── 路由 ──

_END = "__end__"

def _route_planner(state: dict, w: str = "worker") -> str:
    return w if state.get("need_worker", True) else _END

def _route_worker(state: dict, w: str = "worker") -> str:
    return _END if _all_done(state.get("todos", [])) or state.get("worker_round", 0) >= MAX_WORKER_ROUNDS else w


# ── 节点工厂 ──

def _make_node(name: str, agent: Agent, memory: TieredMemory):
    def node_fn(state: AppState, writer: StreamWriter) -> dict:
        writer({"type": "agent_start", "agent": name})
        msgs: list[BaseMessage] = list(state.get("messages", [])) or [HumanMessage(content=state.get("task", ""))]

        if name == "planner":
            parts = [f"Task: {state['task']}"]
            if state.get("session_context"):
                parts.append(f"Session context:\n{state['session_context']}")
            msgs = [HumanMessage(content="\n\n".join(parts))]
        elif state.get("todos"):
            active = [t for t in state["todos"] if t.get("status") not in ("completed", "blocked")]
            if active:
                ls = "\n".join(f"  [{t['id']}] {t['content']}" for t in active)
                msgs = [SystemMessage(content="按 TODO 列表顺序执行，每个完成后调用 update_todo 标记。"), HumanMessage(content=f"TODO:\n{ls}")] + msgs

        ctx = [m for m in memory.get_messages() if m.get("role") in ("system", "assistant")]
        if ctx:
            cls = [SystemMessage(content=c["content"]) if c["role"] == "system" else AIMessage(content=c["content"]) for c in ctx]
            msgs = cls + msgs

        produced, _ = agent.run(msgs)

        for msg in produced:
            t = getattr(msg, "type", "")
            if t == "human": memory.add_message({"role": "user", "content": msg.content})
            elif t == "ai": memory.add_message({"role": "assistant", "content": msg.content or ""})
            elif t == "tool": memory.add_message({"role": "tool", "content": msg.content, "tool_call_id": getattr(msg, "tool_call_id", ""), "name": getattr(msg, "name", "")})
        memory.check_compaction()

        result = {}
        if name == "planner":
            plan = _extract_plan(produced)
            if plan:
                todos = _parse_todos(plan.get("todos") or [])
                result.update(todos=todos, need_worker=bool(plan.get("need_worker")) and len(todos) > 0)
            else:
                result.update(todos=[], need_worker=False)
        else:
            result.update(todos=_apply_todo_updates(state.get("todos", []), produced), need_worker=state.get("need_worker", True), worker_round=state.get("worker_round", 0) + 1)

        writer({"type": "agent_end", "agent": name})
        result["messages"] = produced
        return result
    return node_fn


def build_complex_graph(config: Config, llm, memory: TieredMemory) -> StateGraph:
    builder = StateGraph(AppState)
    names = [a["name"] for a in ALL_AGENTS]
    for adef in ALL_AGENTS:
        n = adef["name"]
        if n == "planner" and adef.get("system_prompt") is None:
            sp = _build_planner_prompt()
        elif "system_prompt_template" in adef:
            sp = adef["system_prompt_template"].format(tools_list=", ".join(adef.get("tools", [])))
        else:
            sp = adef["system_prompt"]
        tools = [TOOL_MAP[t] for t in adef["tools"]]
        if n == "planner":
            tools = [*tools, _build_submit_plan_tool()]
        builder.add_node(n, _make_node(n, Agent(name=n, system_prompt=sp, llm=llm, tools=tools, config=config), memory))

    builder.set_entry_point(names[0])
    if len(names) >= 2:
        wn = names[1]
        builder.add_conditional_edges(names[0], lambda s, w=wn: _route_planner(s, w), {wn: wn, END: END})
        builder.add_conditional_edges(names[1], lambda s, w=wn: _route_worker(s, w), {wn: wn, END: END})
    return builder.compile()


# ── TODO 解析 ──

_TODO_RE = r'\[(\w+)\]\s*(.+)'

def _parse_todos(raw: list[str]) -> list[dict]:
    return [{"id": f"todo-{i+1}", "content": (m.group(2) if (m := re.match(_TODO_RE, item.strip())) else item.strip()), "assignee": m.group(1) if m else "worker", "status": "pending"} for i, item in enumerate(raw)]


# ── SubmitPlan Tool ──

class _PlanArgs(BaseModel):
    plan_summary: str = Field(description="一句话描述整体计划")
    todos: list[str] = Field(description="任务步骤列表，如 ['[worker] 步骤']")
    need_worker: bool = Field(description="是否有 TODO 需要其他 Agent 执行")

def _build_submit_plan_tool() -> StructuredTool:
    def fn(**kw) -> str: return json.dumps(kw, ensure_ascii=False)
    return StructuredTool.from_function(name="SubmitPlan", description="提交任务规划", func=fn, args_schema=_PlanArgs)

def _extract_plan(msgs: list[BaseMessage]) -> dict | None:
    for m in msgs:
        for tc in (getattr(m, "tool_calls", []) or []):
            if isinstance(tc, dict) and tc.get("name") == "SubmitPlan":
                return tc.get("args") or tc.get("arguments", {})
    return None
