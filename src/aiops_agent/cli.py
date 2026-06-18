# d:\workspace\aiops-agent\src\aiops_agent\cli.py
"""CLI — TUI 应用入口，启动 Textual TUI。"""

from .config import Config
from .core.tui_app import AiOpsTUI


def main() -> None:
    config = Config.from_env()
    errors = config.validate()
    if errors:
        for err in errors:
            print(f"❌ {err}")
        exit(1)

    app = AiOpsTUI(config=config)
    app.run()
