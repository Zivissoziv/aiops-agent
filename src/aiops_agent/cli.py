# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI 交互界面 — 用户通过终端与 Agent 对话。"""

from . import __version__
from .config import Config
from .core import Agent, AgentEvent
from .llm import create_llm
from .tools import ShellTool, ToolRegistry


BANNER = """
╔══════════════════════════════════════════╗
║           AIOps Agent v{version:<13}║
║   模型: {model:<29}║
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

    # 初始化 Agent
    agent = Agent(config=config, llm=llm, tool_registry=registry)

    # 显示 Banner
    print(BANNER.format(
        version=__version__,
        model=config.model,
        tools=available_tools,
    ))

    # 对话历史
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
            elif cmd == "/clear":
                history.clear()
                print("✅ 对话历史已清空")
                continue
            elif cmd == "/config":
                print(f"\n  Provider: {config.llm_provider}")
                print(f"  Base URL: {config.base_url}")
                print(f"  Model: {config.model}")
                print(f"  最大工具轮次: {config.max_tool_rounds}")
                continue
            else:
                print(f"未知命令: {user_input}，输入 /help 查看可用命令")
                continue

        # 运行 Agent
        try:
            collected = ""
            new_messages = None
            for event in agent.run(user_input, history):
                print_event(event)
                if event.type == "text":
                    collected += event.content

            # 保存对话历史
            if collected:
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
