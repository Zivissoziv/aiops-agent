# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI 交互界面 — LangGraph StateGraph 多 Agent 编排。

设计要点:
  1. AppState 包含 messages (LC) + memory_snapshot + agent_handoffs
  2. 每个 Agent 节点独立绑定工具
  3. Memory 从 State 中现场构建（build_memory_snapshot）
  4. 所有消息格式统一为 langchain_core.messages 对象
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from langchain_core.tools import StructuredTool

from . import __version__
from .config import Config, _find_project_root
from .core import Agent, AgentEvent, AgentHandoff
from .llm import create_llm
from .tools.shell import execute_shell


# ── 工具注册（LangChain StructuredTool）──

TOOL_MAP: dict[str, StructuredTool] = {
    "shell": StructuredTool.from_function(
        name="shell",
        description="执行 Shell 命令并返回输出。适用于查看系统状态、运行脚本、操作文件等。",
        func=execute_shell,
    ),
}


# ── Agent 配置 ──

@dataclass
class AgentDef:
    system_prompt: str
    tools: list[str]


AGENT_DEFS: dict[str, AgentDef] = {
    "planner": AgentDef(
        system_prompt=(
            "你是一个 AIOps 运维规划专家。你的职责:\n"
            "1. 分析用户的任务\n"
            "2. 制定清晰的执行计划\n"
            "3. 交给运维执行专家去执行\n\n"
            "不要执行工具，只需要输出规划。"
        ),
        tools=[],
    ),
    "worker": AgentDef(
        system_prompt=(
            "你是一个 AIOps 运维执行专家。你的职责:\n"
            "1. 按计划执行运维操作\n"
            "2. 使用 shell 工具查看系统状态\n"
            "3. 给出最终报告\n\n"
            "执行完成后输出最终结果。"
        ),
        tools=["shell"],
    ),
}

TOOL_MAP: dict[str, StructuredTool] = {
    "shell": StructuredTool.from_function(
        name="shell",
        description="执行 Shell 命令并返回输出。适用于查看系统状态、运行脚本、操作文件等。",
        func=execute_shell,
    ),
}


# ── 全局状态 ──

class AppState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str                         # 用户原始任务
    memory_snapshot: dict[str, Any]   # 现场构建的记忆快照
    agent_handoffs: list[AgentHandoff]  # Agent 交接记录
    reply: str                        # 最终回复


# ── 数据目录 ──

DATA_DIR = _find_project_root() / ".aiops_data"


# ── Banner / Help ──

BANNER = """
╔══════════════════════════════════════════╗
║           AIOps Agent v{version:<13}║
║   模型: {model:<29}║
║   模式: {mode:<29}║
║                                          ║
║   输入 /help 查看命令, /exit 退出         ║
╚══════════════════════════════════════════╝
"""

HELP_TEXT = """
可用命令:
  /help              显示此帮助
  /exit              退出程序
  /tools             查看可用工具
  /config            查看当前配置
"""


# ── Memory 快照（从 State 现场构建）──

def build_memory_snapshot(state: AppState, node: str) -> dict[str, Any]:
    """从当前 State 构建三层记忆快照。"""
    messages = state.get("messages", [])
    task = state.get("task", "")

    # 从消息中提取工作记忆
    working = []
    for msg in messages[-10:]:  # 最近 10 条
        if isinstance(msg, HumanMessage):
            working.append({"role": "user", "content": msg.content[:200]})
        elif isinstance(msg, AIMessage):
            working.append({"role": "assistant", "content": msg.content[:200] if msg.content else "(工具调用)"})

    handoffs = []
    for h in state.get("agent_handoffs", []):
        handoffs.append(f"{h.from_agent} → {h.to_agent}: {h.instruction}")

    return {
        "node": node,
        "task": task,
        "working_memory": working,
        "handoffs": handoffs,
    }


# ── 节点函数 ──

def make_agent_node(name: str, agent: Agent, system_prompt: str):
    """创建 Agent 节点函数。"""
    def node_fn(state: AppState) -> dict:
        # 构建输入消息（不含 system prompt，Agent 内部会加）
        input_msgs: list[BaseMessage] = []

        # 添加记忆快照作为 context
        memory = build_memory_snapshot(state, node=name)
        if memory.get("working_memory") or memory.get("handoffs"):
            context_parts = []
            if memory["handoffs"]:
                context_parts.append("历史交接:\n" + "\n".join(memory["handoffs"]))
            if memory["working_memory"]:
                context_parts.append("最近对话:\n" + "\n".join(
                    f"[{m['role']}]: {m['content']}" for m in memory["working_memory"]
                ))
            if context_parts:
                input_msgs.append(HumanMessage(content="\n\n".join(context_parts)))

        # 添加用户任务
        task = state.get("task", "")
        if task:
            input_msgs.append(HumanMessage(content=task))

        # 运行 Agent
        produced_msgs, events = agent.run(input_msgs)

        # 记录交接
        handoff = AgentHandoff(
            from_agent=name,
            to_agent="",
            instruction=f"处理任务: {task[:50]}",
            result=next((m.content for m in produced_msgs if isinstance(m, AIMessage) and m.content), ""),
        )

        # 提取最终文本作为 reply
        reply = ""
        for m in reversed(produced_msgs):
            if isinstance(m, AIMessage) and m.content:
                reply = m.content
                break

        return {
            "messages": produced_msgs,
            "memory_snapshot": build_memory_snapshot({**state, "messages": state["messages"] + produced_msgs}, node=name),
            "agent_handoffs": [handoff],
            "reply": reply,
        }
    return node_fn


# ── 事件打印 ──

def print_event(event: AgentEvent) -> None:
    if event.type == "text":
        print(f"\n{event.content}", flush=True)
    elif event.type == "tool_start":
        print(f"\n{event.content}", flush=True)
        print("─── 输出 ──────────────────────────", flush=True)
    elif event.type == "tool_result":
        print(event.content[:2000])
        if len(event.content) > 2000:
            print("...(输出过长已截断)")
        print("─── 结束 ──────────────────────────", flush=True)
    elif event.type == "error":
        print(f"\n⚠️  {event.content}", flush=True)
    elif event.type == "handoff":
        print(f"\n🔄 {event.content}", flush=True)


# ── 构建图 ──

def build_graph(config: Config, llm) -> StateGraph:
    """根据配置创建 Agent 实例并构建 LangGraph 图。"""
    builder = StateGraph(AppState)

    # 创建 Agent 实例并添加节点
    for name, adef in AGENT_DEFS.items():
        tools = [TOOL_MAP[t] for t in adef.tools]
        agent = Agent(
            name=name,
            system_prompt=adef.system_prompt,
            llm=llm,
            tools=tools,
            config=config,
        )
        builder.add_node(name, make_agent_node(name, agent, adef.system_prompt))

    # 定义边
    names = list(AGENT_DEFS.keys())
    builder.set_entry_point(names[0])
    for i in range(len(names) - 1):
        builder.add_edge(names[i], names[i + 1])
    builder.add_edge(names[-1], END)

    return builder.compile()


# ── 主入口 ──

def main() -> None:
    config = Config.from_env()
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"❌ {err}")
        exit(1)

    llm = create_llm(config)
    graph = build_graph(config, llm)
    mode_label = " → ".join(AGENT_DEFS.keys())

    print(BANNER.format(
        version=__version__,
        model=config.model,
        mode=mode_label,
    ))

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/exit", "/quit"):
                print("再见！")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
                continue
            elif cmd == "/tools":
                print("\n  可用工具:")
                for name, tool in TOOL_MAP.items():
                    print(f"    • {name}: {tool.description}")
                continue
            elif cmd == "/config":
                print(f"\n  Provider: {config.llm_provider}")
                print(f"  Model: {config.model}")
                print(f"  Agent 模式: {mode_label}")
                print(f"  最大工具轮次: {config.max_tool_rounds}")
                continue
            else:
                print(f"未知命令: {user_input}，输入 /help 查看可用命令")
                continue

        # ── 运行图 ──
        try:
            state: AppState = {
                "messages": [HumanMessage(content=user_input)],
                "task": user_input,
                "memory_snapshot": {},
                "agent_handoffs": [],
                "reply": "",
            }

            current_node = ""
            for chunk in graph.stream(state, stream_mode="updates"):
                for node_name, updates in chunk.items():
                    if node_name != current_node:
                        current_node = node_name
                        print(f"\n{'='*50}", flush=True)
                        print(f"  🤖 [{node_name}]", flush=True)
                        print(f"{'='*50}", flush=True)

                    # 从 updates 中提取事件并打印
                    # Agent 节点的输出通过 messages 中的内容推断事件
                    msgs = updates.get("messages", [])
                    for msg in msgs:
                        if isinstance(msg, AIMessage) and msg.content:
                            print(f"\n{msg.content}", flush=True)

            # 显示最终回复
            final_reply = ""
            for m in reversed(state.get("messages", [])):
                if isinstance(m, AIMessage) and m.content:
                    final_reply = m.content
                    break

        except Exception as e:
            import traceback
            print(f"\n❌ 执行出错: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
