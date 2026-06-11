# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI 交互界面 — LangGraph StateGraph 多 Agent 编排 + 三层记忆。"""

from pathlib import Path
from typing import Annotated, Any, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import StructuredTool
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from . import __version__
from .config import Config, _find_project_root
from .core import Agent, AgentEvent, AgentHandoff
from .llm import create_llm
from .memory.tiered import TieredMemory
from .tools import get_tools as get_all_tools


# ── 工具注册（自动扫描）──

TOOL_MAP: dict[str, StructuredTool] = get_all_tools()

from .agents import ALL_AGENTS


# ── 全局状态 ──

class AppState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    memory_snapshot: dict[str, Any]
    agent_handoffs: list[AgentHandoff]
    reply: str
    need_worker: bool  # planner 判断：简单对话直接回，复杂任务交给 worker


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


# ── 节点函数工厂 ──

def make_agent_node(name: str, agent: Agent, memory: TieredMemory):
    """创建 Agent 节点函数，自动处理记忆读写。"""
    def node_fn(state: AppState) -> dict:
        # 从 memory 获取上下文
        memory.add_message({"role": "user", "content": state.get("task", "")})
        context = memory.get_messages()

        # 构建输入消息
        input_msgs: list[BaseMessage] = []
        if context:
            input_msgs.append(HumanMessage(
                content="上下文信息:\n" + "\n".join(
                    f"[{m['role']}]: {m['content'][:200]}" for m in context[-5:]
                )
            ))
        input_msgs.append(HumanMessage(content=state.get("task", "")))

        # 运行 Agent
        produced_msgs, events = agent.run(input_msgs)

        # 将 Agent 产生的消息同步到 memory
        for msg in produced_msgs:
            if hasattr(msg, "type") and hasattr(msg, "content"):
                role_map = {"human": "user", "ai": "assistant", "tool": "tool"}
                role = role_map.get(getattr(msg, "type", ""), "assistant")
                if role == "tool":
                    memory.add_message({"role": "tool", "content": msg.content, "tool_call_id": getattr(msg, "tool_call_id", "")})
                else:
                    memory.add_message({"role": role, "content": msg.content or ""})

        # 压缩检查
        memory.check_compaction()

        # 提取回复
        reply = ""
        for m in reversed(produced_msgs):
            if hasattr(m, "content") and m.content:
                reply = m.content
                break

        return {
            "messages": produced_msgs,
            "memory_snapshot": memory.get_stats(),
            "agent_handoffs": [AgentHandoff(from_agent=name, to_agent="", instruction=state.get("task", ""), result=reply)],
            "reply": reply,
            "need_worker": "[NEED_WORKER]" in reply if name == "planner" else state.get("need_worker", True),
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


# ── 构建图 ──

def build_graph(config: Config, llm, memory: TieredMemory) -> StateGraph:
    builder = StateGraph(AppState)

    for adef in ALL_AGENTS:
        name = adef["name"]
        tools = [TOOL_MAP[t] for t in adef["tools"]]
        agent = Agent(name=name, system_prompt=adef["system_prompt"], llm=llm, tools=tools, config=config)
        builder.add_node(name, make_agent_node(name, agent, memory))

    names = [a["name"] for a in ALL_AGENTS]
    builder.set_entry_point(names[0])

    # 第一个 Agent（planner）之后是条件边：需要 worker 则继续，否则结束
    if len(names) >= 2:
        def route_after_planner(state: AppState) -> str:
            return names[1] if state.get("need_worker", True) else END
        builder.add_conditional_edges(names[0], route_after_planner, {
            names[1]: names[1],
            END: END,
        })
        for i in range(1, len(names) - 1):
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

    # 创建三层记忆
    memory = TieredMemory(
        llm=llm,
        compaction_enabled=True,
        core_persist_path=DATA_DIR / "core_memory.json",
        episodic_persist_path=DATA_DIR / "episodic_memory.json",
    )

    graph = build_graph(config, llm, memory)
    mode_label = " → ".join(a["name"] for a in ALL_AGENTS)

    print(BANNER.format(version=__version__, model=config.model, mode=mode_label))

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
                print("\n  可用工具:")
                for name, tool in TOOL_MAP.items():
                    print(f"    • {name}: {tool.description}")
                continue
            elif cmd == "/memory":
                stats = memory.get_stats()
                print(f"\n  三层记忆状态:")
                print(f"  ┌─ 工作记忆: {stats['working_messages']}/{stats['working_max_messages']} 条")
                print(f"  ├─ 情景记忆: {stats['episodic_count']} 个片段")
                print(f"  ├─ 核心记忆: {stats['core_facts']} 条事实")
                print(f"  └─ 自动压缩: {'开启' if memory._compaction_enabled else '关闭'}")
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
                print(f"  最大工具轮次: {config.max_tool_rounds}")
                print(f"  记忆策略: {config.memory_strategy}")
                continue
            else:
                # 尝试 /remember 和 /forget
                if cmd_parts[0] == "/remember" and len(cmd_parts) >= 2:
                    fact = " ".join(cmd_parts[1:])
                    memory.remember(fact)
                    print(f"✅ 已记住: {fact}")
                    continue
                elif cmd_parts[0] == "/forget" and len(cmd_parts) >= 2:
                    fact = " ".join(cmd_parts[1:])
                    if memory.forget(fact):
                        print(f"✅ 已忘记")
                    else:
                        print(f"⚠️ 未找到")
                    continue
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

                    msgs = updates.get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "content") and msg.content:
                            print(f"\n{msg.content}", flush=True)

        except Exception as e:
            import traceback
            print(f"\n❌ 执行出错: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
