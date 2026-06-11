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
    todos: list[str]


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


def build_graph(config: Config, llm) -> StateGraph:
    builder = StateGraph(AppState)

    for adef in ALL_AGENTS:
        name = adef["name"]
        # planner 的 system_prompt 动态生成
        if name == "planner" and adef.get("system_prompt") is None:
            sp = _build_planner_prompt()
        else:
            sp = adef["system_prompt"]

        tools = [TOOL_MAP[t] for t in adef["tools"]]
        agent = Agent(name=name, system_prompt=sp, llm=llm, tools=tools, config=config)

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

                result = {}

                # planner 节点：检查最后一行是否包含 NEED_WORKER
                if n == "planner":
                    import re
                    todos = re.findall(r'- \[TODO\]\s*(.+)', reply)
                    result["todos"] = todos
                    # 只检查最后 3 行（避免前文误匹配）
                    last_lines = reply.strip().split("\n")[-3:]
                    result["need_worker"] = any("[NEED_WORKER]" in line for line in last_lines)
                else:
                    result["need_worker"] = state.get("need_worker", True)

                result["messages"] = produced_msgs
                return result

                return result
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

def print_custom_event(event: dict, _seen_agents: set = set()):
    """渲染 stream_mode='custom' 事件。"""
    t = event.get("type")
    if t == "agent_start":
        agent = event["agent"]
        if agent in _seen_agents:
            return
        _seen_agents.add(agent)
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

    memory = TieredMemory(
        llm=llm,
        compaction_enabled=True,
        core_persist_path=DATA_DIR / "core_memory.json",
        episodic_persist_path=DATA_DIR / "episodic_memory.json",
    )

    # 注册 Shell 审批回调
    from .tools.shell import configure_approval
    configure_approval(handler=_approval_handler, mode="inline")

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
