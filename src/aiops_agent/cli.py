# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI 交互界面 — 配置驱动多 Agent 图编排。"""

import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, StateGraph

from . import __version__
from .config import Config, _find_project_root
from .core import Agent, AgentEvent
from .llm import create_llm
from .memory import Memory
from .memory.tiered import TieredMemory
from .tools import ShellTool, Tool


# ── Agent 配置 ──
# 在此处定义 Agent 角色、提示词、和工具
# 加 Agent = 加一条配置，改工具 = 改 tools 列表

@dataclass
class AgentDef:
    system_prompt: str
    tools: list[str]


AGENT_DEFS: dict[str, AgentDef] = {
    "planner": AgentDef(
        system_prompt="你是一个 AIOps 运维规划专家。分析用户的任务，制定一个清晰的执行计划，然后交给运维执行专家去执行。输出计划即可，不要执行工具。",
        tools=[],  # 无工具，只做规划
    ),
    "worker": AgentDef(
        system_prompt="你是一个 AIOps 运维执行专家。请按计划逐步执行运维操作，完成后给出最终报告。",
        tools=["shell"],
    ),
}

TOOL_MAP: dict[str, Tool] = {
    "shell": ShellTool(),
}


# ── 全局状态 ──

class AppState(TypedDict):
    messages: Annotated[list, operator.add]
    events: Annotated[list, operator.add]


# ── 数据目录 ──

DATA_DIR = _find_project_root() / ".aiops_data"


# ── Banner / Help ──

BANNER = """
╔══════════════════════════════════════════╗
║           AIOps Agent v{version:<13}║
║   模型: {model:<29}║
║   记忆: {memory:<29}║
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
  /clear             清空所有记忆
  /config            查看当前配置
"""


# ── 工具描述 ──

def _get_available_tools_text() -> str:
    lines = []
    for name, tool in TOOL_MAP.items():
        params = tool.parameters.get("properties", {})
        param_desc = ", ".join(f"{k}({v.get('type', '?')})" for k, v in params.items())
        lines.append(f"  - {name}: {tool.description} 参数: {param_desc}")
    return "\n".join(lines)


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


# ── 记忆创建 ──

def _create_memory(config: Config, llm) -> Memory | None:
    if config.memory_strategy == "tiered":
        return TieredMemory(
            llm=llm,
            working_max_messages=config.memory_max_messages,
            working_max_tokens=config.memory_max_tokens,
            max_episodes=config.memory_max_episodes,
            recent_episodes=config.memory_recent_episodes,
            core_persist_path=DATA_DIR / "core_memory.json",
            episodic_persist_path=DATA_DIR / "episodic_memory.json",
            compaction_enabled=config.memory_compaction_enabled,
        )
    elif config.memory_strategy == "none":
        return None
    else:
        print(f"⚠️ 未知的记忆策略 '{config.memory_strategy}'，使用 tiered")
        return TieredMemory(
            llm=llm,
            core_persist_path=DATA_DIR / "core_memory.json",
            episodic_persist_path=DATA_DIR / "episodic_memory.json",
        )


# ── 记忆命令 ──

def _handle_memory_command(cmd_parts: list[str], memory: Memory | None) -> bool:
    if memory is None:
        return False
    if not isinstance(memory, TieredMemory):
        return False
    if cmd_parts[0] in ("/remember", "/rem"):
        if len(cmd_parts) < 2:
            print("用法: /remember <事实内容>")
            return True
        memory.remember(" ".join(cmd_parts[1:]))
        print(f"✅ 已记住")
        return True
    if cmd_parts[0] in ("/forget", "/for"):
        if len(cmd_parts) < 2:
            print("用法: /forget <事实内容>")
            return True
        if memory.forget(" ".join(cmd_parts[1:])):
            print(f"✅ 已忘记")
        else:
            print(f"⚠️ 未找到")
        return True
    if cmd_parts[0] == "/core":
        facts = memory.get_core_facts()
        if facts:
            print("\n  核心记忆:")
            for i, f in enumerate(facts, 1):
                print(f"    {i}. {f}")
        else:
            print("  核心记忆为空")
        return True
    return False


# ── 图构建 ──

def build_agents_and_graph(config: Config, llm, memory: Memory | None = None) -> tuple[StateGraph, dict[str, Agent]]:
    """根据配置创建 Agent 实例并构建 LangGraph 图。"""
    builder = StateGraph(AppState)

    agents = {}
    for name, adef in AGENT_DEFS.items():
        tools = [TOOL_MAP[t] for t in adef.tools]
        agents[name] = Agent(
            name=name,
            system_prompt=adef.system_prompt,
            llm=llm,
            tools=tools,
            config=config,
            memory=memory,
        )

    # 添加节点：每个 Agent 作为一个节点
    for name in AGENT_DEFS:
        agent = agents[name]

        def make_node_fn(n: str, a: Agent):
            def node_fn(state: AppState) -> dict:
                # 从 state 中提取 user_input（可能是 dict 或 LangChain 消息）
                user_msg = ""
                for m in reversed(state["messages"]):
                    if isinstance(m, dict):
                        if m.get("role") == "user":
                            user_msg = m.get("content", "")
                            break
                    elif hasattr(m, "content"):
                        user_msg = m.content
                        break
                if not user_msg and state["messages"]:
                    last = state["messages"][-1]
                    if isinstance(last, dict):
                        user_msg = last.get("content", "")
                    else:
                        user_msg = getattr(last, "content", str(last))

                collected_events = []
                for event in a.run(user_msg):
                    collected_events.append({
                        "type": event.type,
                        "content": event.content,
                        "data": event.data,
                    })
                return {"events": collected_events}
            return node_fn

        builder.add_node(name, make_node_fn(name, agent))

    # 定义边：planner → worker → END
    names = list(AGENT_DEFS.keys())
    builder.set_entry_point(names[0])
    for i in range(len(names) - 1):
        builder.add_edge(names[i], names[i + 1])
    builder.add_edge(names[-1], END)

    return builder.compile(), agents


# ── 主入口 ──

def main() -> None:
    config = Config.from_env()
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"❌ {err}")
        exit(1)

    llm = create_llm(config)
    memory = _create_memory(config, llm)
    memory_label = config.memory_strategy if memory else "none"

    # 构建多 Agent 图
    graph, agents = build_agents_and_graph(config, llm, memory)
    mode_label = " → ".join(AGENT_DEFS.keys())
    tool_names = list(TOOL_MAP.keys())

    print(BANNER.format(
        version=__version__,
        model=config.model,
        memory=memory_label,
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
            cmd_parts = cmd.split()

            if cmd_parts[0] in ("/remember", "/rem", "/forget", "/for", "/core"):
                _handle_memory_command(cmd_parts, memory)
                continue

            if cmd in ("/exit", "/quit"):
                print("再见！")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
                continue
            elif cmd == "/tools":
                print(f"\n  可用工具: {', '.join(tool_names)}")
                print(_get_available_tools_text())
                continue
            elif cmd == "/memory":
                if isinstance(memory, TieredMemory):
                    stats = memory.get_stats()
                    print(f"\n  记忆系统: 三层记忆 (Three-Tier)")
                    print(f"  ┌─ 工作记忆: {stats['working_messages']}/{stats['working_max_messages']} 条")
                    print(f"  ├─ 情景记忆: {stats['episodic_count']} 个片段")
                    print(f"  ├─ 核心记忆: {stats['core_facts']} 条事实")
                else:
                    print("  记忆未启用")
                continue
            elif cmd == "/clear":
                if memory:
                    memory.reset()
                print("✅ 对话历史已清空")
                continue
            elif cmd == "/config":
                print(f"\n  Provider: {config.llm_provider}")
                print(f"  Model: {config.model}")
                print(f"  最大工具轮次: {config.max_tool_rounds}")
                print(f"  记忆策略: {config.memory_strategy}")
                print(f"  多 Agent 模式: {mode_label}")
                continue
            else:
                print(f"未知命令: {user_input}，输入 /help 查看可用命令")
                continue

        # 运行多 Agent 图
        try:
            state: AppState = {
                "messages": [HumanMessage(content=user_input)],
                "events": [],
            }
            current_agent_label = ""
            for chunk in graph.stream(state):
                for node_name, updates in chunk.items():
                    if node_name != current_agent_label:
                        current_agent_label = node_name
                        print(f"\n{'='*50}", flush=True)
                        print(f"  🤖 [{node_name}]", flush=True)
                        print(f"{'='*50}", flush=True)
                    for evt_data in updates.get("events", []):
                        print_event(AgentEvent(
                            type=evt_data["type"],
                            content=evt_data["content"],
                            data=evt_data.get("data", {}),
                        ))

        except Exception as e:
            print(f"\n❌ Agent 执行出错: {e}")


if __name__ == "__main__":
    main()
