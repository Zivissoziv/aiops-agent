# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI — LangGraph 流式事件消费 + 审批回调。

双 Graph 架构：
  1. Entry Graph（意图路由） → intent_router → chat_responder（chat 路由）或 END（task 路由）
  2. Complex Graph（任务执行） → planner → worker
"""

from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from . import __version__
from .agents import ALL_AGENTS
from .config import Config, _find_project_root
from .graph import AppState, build_complex_graph, build_entry_graph
from .llm import create_llm
from .memory.tiered import TieredMemory
from .tools.file_tools import configure_write_approval
from .tools.file_tools import configure_workspace as configure_file_workspace
from .tools.shell import configure_approval as configure_shell_approval
from .tools.shell import configure_workspace as configure_shell_workspace


# ── 数据目录 ──

DATA_DIR = _find_project_root() / ".aiops_data"
WORKSPACE_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
WORKSPACE_DIR = DATA_DIR / "workspaces" / WORKSPACE_ID


# ── Banner / Help ──

BANNER = """
╔══════════════════════════════════════════════╗
║           AIOps Agent v{version:<13}║
║   模型: {model:<29}║
║   模式: {mode:<29}║
║   Workspace: {workspace:<24}║
║                                          ║
║   输入 /help 查看命令, /exit 退出         ║
╚══════════════════════════════════════════════╝
"""

HELP_TEXT = """
可用命令:
  /help              显示此帮助
  /exit              退出程序
  /tools             查看可用工具
  /memory            查看三层记忆状态
  /workspace         查看当前 Workspace
  /remember <事实>   添加核心记忆
  /forget <事实>     删除核心记忆
  /core              查看核心记忆列表
  /clear             清空对话
  /config            查看当前配置
"""


# ── 审批回调 ──
def _approval_handler(command: str, risk_reason: str) -> bool:
    """Shell 高风险操作审批（阻塞等待用户输入）。"""
    print(f"\n⚠️ 高风险操作需要确认: {risk_reason}")
    print(f"  命令: {command}")
    try:
        resp = input(f"  是否执行? (y/N): ").strip().lower()
        return resp in ("y", "yes", "是")
    except (EOFError, KeyboardInterrupt):
        return False


# ── 事件渲染 ──

# 跟踪已渲染过的 agent，避免重复打印
_seen_agents_in_session: set[str] = set()


def print_custom_event(event: dict):
    """渲染 stream_mode='custom' 事件。"""
    t = event.get("type")
    if t == "intent_decision":
        route = event.get("route", "")
        confidence = event.get("confidence", 0.0)
        reason = event.get("reason", "")
        print(f"\n📋 路由: [{route}] 置信度={confidence:.2f} 理由: {reason}")
    elif t == "chat_response":
        pass  # chat 回复由 print_graph_update 渲染
    elif t == "agent_start":
        agent = event["agent"]
        if agent in _seen_agents_in_session:
            return
        _seen_agents_in_session.add(agent)
        print(f"\n{'='*50}", flush=True)
        print(f"  🤖 [{agent}]", flush=True)
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
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                print(f"\n{msg.content}", flush=True)


def _build_session_context(state: AppState) -> str:
    """从 state 中的 messages 构建多轮对话上下文摘要。"""
    session_msgs = []
    for m in state.get("messages", []):
        if hasattr(m, "type") and m.type in ("human", "ai") and m.content:
            session_msgs.append(f"[{m.type}]: {str(m.content)[:200]}")
    return "\n".join(session_msgs[-6:]) if session_msgs else ""


# ── 主入口 ──

def main() -> None:
    config = Config.from_env()
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"❌ {err}")
        exit(1)

    llm = create_llm(config)

    # 确保 workspace 目录存在
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

    memory = TieredMemory(
        llm=llm,
        compaction_enabled=True,
        working_max_messages=2,
        working_max_tokens=500,
        core_persist_path=DATA_DIR / "core_memory.json",
        episodic_persist_path=WORKSPACE_DIR / "episodic_memory.json",
    )

    # 注册 Shell 审批回调
    configure_shell_approval(handler=_approval_handler, mode="inline")

    # 注册文件工具 workspace 沙箱 + 越界审批回调
    configure_file_workspace(WORKSPACE_DIR)
    configure_write_approval(lambda path, preview: _approval_handler(
        f"访问文件({path})", f"workspace 外路径: {preview}"
    ))
    # 注册 shell workspace 默认工作目录
    configure_shell_workspace(WORKSPACE_DIR)

    entry_graph = build_entry_graph(llm, memory)
    complex_graph = build_complex_graph(config, llm, memory)
    mode_label = " → ".join(a["name"] for a in ALL_AGENTS)

    print(BANNER.format(version=__version__, model=config.model, mode=mode_label, workspace=WORKSPACE_ID))

    # ── 主循环 ──

    # state 全程持续存在，每轮只追加当前输入
    # 启动时从 TieredMemory 恢复历史对话
    working_history = memory.working.get_messages()
    restored_messages: list[BaseMessage] = []
    for m in working_history:
        if m.get("role") == "user":
            restored_messages.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") == "assistant":
            restored_messages.append(AIMessage(content=m.get("content", "")))
        elif m.get("role") == "tool":
            tid = m.get("tool_call_id", "")
            tname = m.get("name", "")
            restored_messages.append(ToolMessage(content=m.get("content", ""), tool_call_id=tid, name=tname or None))

    state: AppState = {
        "messages": restored_messages,
        "task": "",
        "need_worker": False,
        "todos": [],
        "intent_route": "",
        "intent_reason": "",
        "intent_confidence": 0.0,
        "chat_response": "",
        "session_context": "",
    }

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # ── 斜杠命令（不走 graph） ──
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
                from .graph.complex import TOOL_MAP
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
                state["messages"] = []
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
            elif cmd == "/workspace":
                print(f"\n  当前 Workspace: {WORKSPACE_ID}")
                print(f"  路径: {WORKSPACE_DIR}")
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

        # ── Step 1: Entry Graph — 意图路由 ──
        state["task"] = user_input
        state["session_context"] = _build_session_context(state)
        state["chat_response"] = ""
        state["intent_route"] = ""
        state["intent_reason"] = ""
        state["intent_confidence"] = 0.0

        route = "task"
        try:
            for mode, event in entry_graph.stream(state, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    print_custom_event(event)
                    if isinstance(event, dict) and event.get("type") == "intent_decision":
                        route = str(event.get("route") or "task")
                elif mode == "updates":
                    print_graph_update(event)
        except Exception as e:
            import traceback
            print(f"\n❌ 入口路由出错: {e}")
            traceback.print_exc()
            route = "task"  # 路由失败时默认走 task

        # ── Step 2: 根据路由结果分流 ──
        if route == "chat":
            # 回复已由 print_graph_update 渲染（chat_responder 返回 AIMessage 在 messages 中）
            # memory 也已在 entry graph 的 chat_responder 节点中同步
            continue

        # ── Step 3: Complex Graph — 任务执行 ──
        # 将用户输入追加到 state["messages"]，让 worker 能感知原始请求
        user_msg = HumanMessage(content=user_input)
        state["messages"] = list(state["messages"]) + [user_msg]

        try:
            _seen_agents_in_session.clear()

            for mode, event in complex_graph.stream(state, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    print_custom_event(event)
                elif mode == "updates":
                    print_graph_update(event)

            # 图中的 make_node 只同步 AI/tool 消息，用户消息需在此同步
            memory.add_message({"role": "user", "content": user_input})

        except Exception as e:
            import traceback
            print(f"\n❌ 执行出错: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
