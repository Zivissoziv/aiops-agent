# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI 交互界面 — 用户通过终端与 Agent 对话。"""

from . import __version__
from .config import Config
from .core import Agent, AgentEvent
from .llm import create_llm
from .memory import SlidingWindowMemory, SummarizingMemory
from .tools import ShellTool, ToolRegistry


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
  /help      显示此帮助
  /exit      退出程序
  /tools     查看可用工具
  /memory    查看记忆状态
  /clear     清空对话历史
  /config    查看当前配置
"""


def print_event(event: AgentEvent) -> None:
    """格式化输出 Agent 事件。"""
    if event.type == "text":
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


def _create_memory(config: Config, llm):
    """根据配置创建 Memory 实例。"""
    if config.memory_strategy == "window":
        return SlidingWindowMemory(max_messages=config.memory_max_messages)
    elif config.memory_strategy == "summarizing":
        return SummarizingMemory(llm=llm, max_tokens=config.memory_max_tokens)
    elif config.memory_strategy == "none":
        return None
    else:
        print(f"⚠️ 未知的记忆策略 '{config.memory_strategy}'，使用默认 window")
        return SlidingWindowMemory(max_messages=config.memory_max_messages)


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
                if memory:
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
                print(f"  最大记忆消息数: {config.memory_max_messages}")
                print(f"  摘要触发 Token: {config.memory_max_tokens}")
                continue
            else:
                print(f"未知命令: {user_input}，输入 /help 查看可用命令")
                continue

        # 运行 Agent
        try:
            collected = ""
            # memory 路径: 不需要传 history，Agent 内部管理
            # 向后兼容路径: 传 history
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
