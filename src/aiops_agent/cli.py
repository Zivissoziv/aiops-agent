# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI — LangGraph 流式事件消费 + 审批回调。"""

import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import StreamWriter

from . import __version__
from .agents import ALL_AGENTS
from .config import Config, _find_project_root
from .core import Agent
from .core.intent_router import classify_intent
from .llm import create_llm
from .memory.tiered import TieredMemory
from .tools import get_tools
from .tools.file_tools import configure_write_approval
from .tools.file_tools import configure_workspace as configure_file_workspace
from .tools.shell import configure_approval as configure_shell_approval
from .tools.shell import configure_workspace as configure_shell_workspace


# ── 工具注册 ──

TOOL_MAP: dict[str, StructuredTool] = get_tools()


# ── 全局状态 ──


class AppState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    need_worker: bool
    todos: list[str]


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


# ── 构建图 ──

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


def build_graph(config: Config, llm, memory: TieredMemory) -> StateGraph:
    builder = StateGraph(AppState)

    for adef in ALL_AGENTS:
        name = adef["name"]
        if name == "planner" and adef.get("system_prompt") is None:
            sp = _build_planner_prompt()
        else:
            sp = adef["system_prompt"]

        tools = [TOOL_MAP[t] for t in adef["tools"]]
        agent = Agent(name=name, system_prompt=sp, llm=llm, tools=tools, config=config)

        def make_node(n: str, a: Agent, mem: TieredMemory):
            def node_fn(state: AppState, writer: StreamWriter) -> dict:
                writer({"type": "agent_start", "agent": n})
                # 从 state 获取本轮对话历史
                input_msgs: list[BaseMessage] = list(state.get("messages", []))
                if not input_msgs:
                    input_msgs = [HumanMessage(content=state.get("task", ""))]

                # planner 只取本轮用户输入，避免看到之前轮次的执行细节后重复执行
                if n == "planner":
                    input_msgs = [HumanMessage(content=state.get("task", ""))]

                # 从三层记忆注入额外上下文（core + episodic）
                # 这些不是 LangGraph BaseMessage 类型，不放在 state["messages"] 里，
                # 而是以 system/assistant dict 格式注入到 Agent.run() 的输入中
                mem_context = mem.get_messages()
                # 过滤出 system（core）和 assistant（episodic）角色的消息
                extra_context = [m for m in mem_context if m.get("role") in ("system", "assistant")]
                if extra_context:
                    # 转为 BaseMessage 注入
                    ctx_msgs: list[BaseMessage] = []
                    for ctx in extra_context:
                        if ctx["role"] == "system":
                            from langchain_core.messages import SystemMessage
                            ctx_msgs.append(SystemMessage(content=ctx["content"]))
                        elif ctx["role"] == "assistant":
                            ctx_msgs.append(AIMessage(content=ctx["content"]))
                    input_msgs = [*ctx_msgs, *input_msgs]

                produced_msgs, events = a.run(input_msgs)
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
                            mem.add_message({"role": "tool", "content": msg.content, "tool_call_id": getattr(msg, "tool_call_id", "")})
                        else:
                            mem.add_message({"role": role, "content": msg.content or ""})
                mem.check_compaction()

                result = {}

                if n == "planner":
                    import re
                    todos = re.findall(r'- \[TODO\]\s*(.+)', reply)
                    result["todos"] = todos
                    last_lines = reply.strip().split("\n")[-3:]
                    result["need_worker"] = any("[NEED_WORKER]" in line for line in last_lines)
                else:
                    result["need_worker"] = state.get("need_worker", True)

                result["messages"] = produced_msgs
                return result
            return node_fn

        builder.add_node(name, make_node(name, agent, memory))

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

# 跟踪已渲染过的 agent，避免重复打印
_seen_agents_in_session: set[str] = set()


def print_custom_event(event: dict):
    """渲染 stream_mode='custom' 事件。"""
    t = event.get("type")
    if t == "agent_start":
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
            # 只显示 AI 回复的文本内容，不显示 system/tool/human
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
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

    graph = build_graph(config, llm, memory)
    mode_label = " → ".join(a["name"] for a in ALL_AGENTS)

    print(BANNER.format(version=__version__, model=config.model, mode=mode_label, workspace=WORKSPACE_ID))

    # ── 主循环 ──

    # state 全程持续存在，每轮只追加当前输入
    # 启动时从 TieredMemory 恢复历史对话（只恢复 user/assistant/tool，不恢复 core/episodic）
    # core 和 episodic 会在 make_node 中通过 mem.get_messages() 注入
    working_history = memory.working.get_messages()
    restored_messages: list[BaseMessage] = []
    for m in working_history:
        if m.get("role") == "user":
            restored_messages.append(HumanMessage(content=m.get("content", "")))
        elif m.get("role") == "assistant":
            restored_messages.append(AIMessage(content=m.get("content", "")))
        elif m.get("role") == "tool":
            tid = m.get("tool_call_id", "")
            restored_messages.append(ToolMessage(content=m.get("content", ""), tool_call_id=tid))

    state: AppState = {
        "messages": restored_messages,
        "task": "",
        "need_worker": False,
    }

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

        # ── 意图路由 ──
        # 简单关键词快速跳过路由（包含工具关键词的不用调 LLM 分类）
        task_keywords = [
            "查", "看", "读", "写", "创建", "删除", "修改", "编辑", "运行", "执行",
            "安装", "下载", "搜索", "分析", "检测", "检查", "排查", "修复",
            "shell", "read", "write", "edit", "ls", "cat", "grep", "ping",
            "磁盘", "内存", "cpu", "进程", "日志", "配置", "文件", "目录",
            "docker", "k8s", "pod", "service", "deploy",
        ]
        if not any(kw in user_input for kw in task_keywords):
            intent = classify_intent(llm, user_input)

            if intent["route"] == "chat":
                chat_system = "你是一个 AIOps 运维助手。直接友好地回复用户的问题。不要提及工具、文件操作或工作区。"
                response = llm.invoke([
                    SystemMessage(content=chat_system),
                    HumanMessage(content=user_input),
                ])
                reply_text = (response.content or "").strip()
                print(f"\n{reply_text}")
                memory.add_message({"role": "user", "content": user_input})
                memory.add_message({"role": "assistant", "content": reply_text})
                continue

        # ── 运行图（双模式流）──
        try:
            # 每轮重置 seen_agents，让 agent_start banner 重新显示
            _seen_agents_in_session.clear()

            # 将当前用户输入追加到持续存在的 state 中
            new_msg = HumanMessage(content=user_input)
            state["messages"].append(new_msg)
            state["task"] = user_input
            state["need_worker"] = False

            for mode, event in graph.stream(state, stream_mode=["updates", "custom"]):
                if mode == "custom":
                    print_custom_event(event)
                elif mode == "updates":
                    print_graph_update(event)

            # graph.stream 结束后，将本轮用户输入同步到 TieredMemory
            # （AI 回复已在 make_node 中同步）
            memory.add_message({"role": "user", "content": user_input})

        except Exception as e:
            import traceback
            print(f"\n❌ 执行出错: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
