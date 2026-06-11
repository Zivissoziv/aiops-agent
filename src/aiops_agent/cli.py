# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI 交互界面 — 用户通过终端与 Agent 对话。"""

from pathlib import Path

from . import __version__
from .config import Config, _find_project_root
from .core import Agent, AgentEvent
from .llm import create_llm
from .memory import Memory
from .memory.tiered import TieredMemory
from .tools import ShellTool, ToolRegistry

# 数据目录（持久化记忆文件）
DATA_DIR = _find_project_root() / ".aiops_data"


BANNER = """
╔══════════════════════════════════════════╗
║           AIOps Agent v{version:<13}║
║   模型: {model:<29}║
║   记忆: {memory:<29}║
║   工具: {tools:<29}║
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


def print_event(event: AgentEvent) -> None:
    """格式化输出 Agent 事件。"""
    if event.type == "plan":
        print(f"\n📋 执行计划\n{event.content}", flush=True)
    elif event.type == "text":
        print(f"\n助手: {event.content}", flush=True)
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


def _create_memory(config: Config, llm) -> Memory | None:
    """根据配置创建 Memory 实例。"""
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


def _handle_memory_command(cmd_parts: list[str], memory: Memory | None) -> bool:
    """处理 /remember 和 /forget 命令。

    Returns:
        True 表示命令已处理，False 表示需要继续交给主循环处理。
    """
    if memory is None:
        return False

    if not isinstance(memory, TieredMemory):
        return False

    if cmd_parts[0] in ("/remember", "/rem"):
        if len(cmd_parts) < 2:
            print("用法: /remember <事实内容>")
            return True
        fact = " ".join(cmd_parts[1:])
        memory.remember(fact)
        print(f"✅ 已记住: {fact}")
        return True

    if cmd_parts[0] in ("/forget", "/for"):
        if len(cmd_parts) < 2:
            print("用法: /forget <事实内容>")
            return True
        fact = " ".join(cmd_parts[1:])
        if memory.forget(fact):
            print(f"✅ 已忘记: {fact}")
        else:
            print(f"⚠️ 未找到: {fact}")
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


def main() -> None:
    """CLI 主入口。"""
    # 加载配置
    config = Config.from_env()
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"❌ {err}")
        exit(1)

    # 初始化 LLM
    llm = create_llm(config)

    # 注册工具
    registry = ToolRegistry()
    registry.register(ShellTool())
    available_tools = ", ".join(t.name for t in registry.list_tools())

    # 创建记忆实例
    memory = _create_memory(config, llm)
    memory_label = config.memory_strategy if memory else "none"

    # 初始化 Agent
    agent = Agent(config=config, llm=llm, tool_registry=registry, memory=memory)

    # 显示 Banner
    print(BANNER.format(
        version=__version__,
        model=config.model,
        memory=memory_label,
        tools=available_tools,
    ))

    # 向后兼容：无 memory 时使用 raw history
    history: list[dict] = []

    # 主循环
    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 处理内部命令
        if user_input.startswith("/"):
            cmd = user_input.lower()
            cmd_parts = cmd.split()

            # 先尝试 /remember /forget /core（需要 TieredMemory）
            if cmd_parts[0] in ("/remember", "/rem", "/forget", "/for", "/core"):
                if _handle_memory_command(cmd_parts, memory):
                    continue
                else:
                    print("当前记忆策略不支持该命令")
                    continue

            if cmd in ("/exit", "/quit"):
                print("再见！")
                break
            elif cmd == "/help":
                print(HELP_TEXT)
                continue
            elif cmd == "/tools":
                for tool in registry.list_tools():
                    print(f"\n  • {tool.name}: {tool.description}")
                continue
            elif cmd == "/memory":
                if isinstance(memory, TieredMemory):
                    stats = memory.get_stats()
                    print(f"\n  记忆系统: 三层记忆 (Three-Tier)")
                    print(f"  ┌─ 工作记忆: {stats['working_messages']}/{stats['working_max_messages']} 条消息")
                    print(f"  ├─ 情景记忆: {stats['episodic_count']} 个片段")
                    print(f"  ├─ 核心记忆: {stats['core_facts']} 条事实")
                    print(f"  └─ Token 估算: {memory.count(llm.count_tokens)}")
                elif memory:
                    print(f"\n  策略: {config.memory_strategy}")
                    print(f"  消息数: {len(memory)}")
                    print(f"  Token 估算: {memory.count(llm.count_tokens)}")
                else:
                    print("  记忆未启用")
                continue
            elif cmd == "/clear":
                if memory:
                    memory.reset()
                else:
                    history.clear()
                print("✅ 对话历史已清空")
                continue
            elif cmd == "/config":
                print(f"\n  Provider: {config.llm_provider}")
                print(f"  Base URL: {config.base_url}")
                print(f"  Model: {config.model}")
                print(f"  最大工具轮次: {config.max_tool_rounds}")
                print(f"  记忆策略: {config.memory_strategy}")
                print(f"  工作记忆消息数: {config.memory_max_messages}")
                print(f"  工作记忆 Token: {config.memory_max_tokens}")
                print(f"  情景记忆上限: {config.memory_max_episodes}")
                print(f"  上下文包含情景数: {config.memory_recent_episodes}")
                print(f"  自动压缩: {'开启' if config.memory_compaction_enabled else '关闭'}")
                continue
            else:
                print(f"未知命令: {user_input}，输入 /help 查看可用命令")
                continue

        # 运行 Agent
        try:
            collected = ""
            run_history = None if memory else history
            for event in agent.run(user_input, run_history):
                print_event(event)
                if event.type == "text":
                    collected += event.content

            # 向后兼容: 无 memory 时手动记录历史
            if memory is None and collected:
                history.append({"role": "user", "content": user_input})
                history.append({"role": "assistant", "content": collected})

        except Exception as e:
            print(f"\n❌ Agent 执行出错: {e}")
            if "Incorrect API key" in str(e):
                print("  提示: API Key 可能无效，请检查 .env 配置")
            elif "timeout" in str(e).lower():
                print("  提示: 请求超时，请检查网络连接或 Base URL 配置")


if __name__ == "__main__":
    main()
