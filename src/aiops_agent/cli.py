# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI — LangGraph 流式事件消费 + 审批回调。"""

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from . import __version__
from .agents import ALL_AGENTS
from .config import Config, _find_project_root
from .core import Agent
from .llm import create_llm
from .memory.tiered import TieredMemory
from .tools import get_tools


# ── 工具注册 ──

TOOL_MAP: dict[str, StructuredTool] = get_tools()


# ── 全局状态 ──

from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage


class AppState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    need_worker: bool


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
  /memory            查看三层记忆状态
  /remember <事实>   添加核心记忆
  /forget <事实>     删除核心记忆
  /core              查看核心记忆列表
  /clear             清空对话
  /config            查看当前配置
"""


# ── 待审批队列 ──

_pending_approval: dict = {}


# ── 审批回调 ──

def _approval_hook(command: str, level: str, reason: str):
    """Shell 危险操作审批（不阻塞，返回 None 表示等待确认）。"""
    label = "🔴 高危" if level == "danger" else "⚠️ 警告"
    print(f"\n{label} 操作需要确认: {reason}")
    print(f"  命令: {command}")
    print(f"  输入 /approve 确认，/reject 拒绝")
    _pending_approval["cmd"] = command
    _pending_approval["level"] = level
    _pending_approval["reason"] = reason
    return None


# ── 构建图 ──

def build_graph(config: Config, llm) -> StateGraph:
    builder = StateGraph(AppState)

    for adef in ALL_AGENTS:
        name = adef["name"]
        tools = [TOOL_MAP[t] for t in adef["tools"]]
        agent = Agent(name=name, system_prompt=adef["system_prompt"], llm=llm, tools=tools, config=config)

        def make_node(n: str, a: Agent):
            def node_fn(state: AppState, writer: StreamWriter) -> dict:
                writer({"type": "agent_start", "agent": n})
                input_msgs = [HumanMessage(content=state.get("task", ""))]
                produced_msgs, events = a.run(input_msgs)
                reply = ""
                for m in reversed(produced_msgs):
                    if hasattr(m, "content") and m.content:
                        reply = m.content
                        break
                return {
                    "messages": produced_msgs,
                    "need_worker": "[NEED_WORKER]" in reply if n == "planner" else state.get("need_worker", True),
                }
            return node_fn

        builder.add_node(name, make_node(name, agent))

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


# ── 事件渲染 ──

def print_custom_event(event: dict):
    """渲染 stream_mode='custom' 事件（实时工具调用等）。"""
    t = event.get("type")
    if t == "agent_start":
        print(f"\n{'='*50}", flush=True)
        print(f"  🤖 [{event['agent']}]", flush=True)
        print(f"{'='*50}", flush=True)
    elif t == "tool_start":
        print(f"\n🔧 正在使用工具: {event['tool']}", flush=True)
        print("─── 输出 ──────────────────────────", flush=True)
    elif t == "tool_result":
        output = event.get("output", "")
        if output:
            print(output[:2000], flush=True)
            if len(output) > 2000:
                print("...(输出过长已截断)")
        error = event.get("error", "")
        if error:
            print(f"错误: {error}", flush=True)
        print("─── 结束 ──────────────────────────", flush=True)


def print_graph_update(updates: dict):
    """渲染 stream_mode='updates' 事件（节点返回的文本消息）。"""
    for data in updates.values():
        for msg in data.get("messages", []):
            if hasattr(msg, "content") and msg.content:
                print(f"\n{msg.content}", flush=True)


# ── 主入口 ──

def main() -> None:
    config = Config.from_env()
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"❌ {err}")
        exit(1)

    llm = create_llm(config)

    memory = TieredMemory(
        llm=llm,
        compaction_enabled=True,
        core_persist_path=DATA_DIR / "core_memory.json",
        episodic_persist_path=DATA_DIR / "episodic_memory.json",
    )

    # 注册 Shell 审批回调
    from .tools.shell import set_approval_hook
    set_approval_hook(_approval_hook)

    graph = build_graph(config, llm)
    mode_label = " → ".join(a["name"] for a in ALL_AGENTS)

    print(BANNER.format(version=__version__, model=config.model, mode=mode_label))

    # ── 主循环 ──
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
            cmd_parts = cmd.split()

            if cmd in ("/exit", "/quit"):
                print("再见！")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
                continue
            elif cmd == "/tools":
                print(f"\n  可用工具: {', '.join(TOOL_MAP.keys())}")
                continue
            elif cmd == "/memory":
                stats = memory.get_stats()
                print(f"\n  三层记忆状态:")
                print(f"  ┌─ 工作记忆: {stats['working_messages']}/{stats['working_max_messages']} 条")
                print(f"  ├─ 情景记忆: {stats['episodic_count']} 个片段")
                print(f"  ├─ 核心记忆: {stats['core_facts']} 条事实")
                continue
            elif cmd == "/clear":
                memory.reset()
                print("✅ 对话已清空")
                continue
            elif cmd == "/core":
                facts = memory.get_core_facts()
                if facts:
                    print("\n  核心记忆:")
                    for i, f in enumerate(facts, 1):
                        print(f"    {i}. {f}")
                else:
                    print("  核心记忆为空")
                continue
            elif cmd == "/config":
                print(f"\n  Provider: {config.llm_provider}")
                print(f"  Model: {config.model}")
                print(f"  Agent 模式: {mode_label}")
                continue
            elif cmd == "/approve":
                if _pending_approval:
                    print(f"✅ 已确认: {_pending_approval.pop('cmd', '')}")
                    # 简化版：只是确认记录，后续迭代可改进
                else:
                    print("当前没有待审批的操作")
                continue
            elif cmd == "/reject":
                if _pending_approval:
                    print(f"❌ 已拒绝: {_pending_approval.pop('cmd', '')}")
                else:
                    print("当前没有待审批的操作")
                continue
            else:
                if cmd_parts[0] == "/remember" and len(cmd_parts) >= 2:
                    memory.remember(" ".join(cmd_parts[1:]))
                    print("✅ 已记住")
                    continue
                elif cmd_parts[0] == "/forget" and len(cmd_parts) >= 2:
                    if memory.forget(" ".join(cmd_parts[1:])):
                        print("✅ 已忘记")
                    else:
                        print("⚠️ 未找到")
                    continue
                print(f"未知命令: {user_input}")
                continue

        # ── 运行图（双模式流）──
        try:
            state: AppState = {
                "messages": [HumanMessage(content=user_input)],
                "task": user_input,
                "need_worker": False,
            }

            for mode, event in graph.stream(state, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    print_custom_event(event)
                elif mode == "updates":
                    print_graph_update(event)

        except Exception as e:
            import traceback
            print(f"\n❌ 执行出错: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
