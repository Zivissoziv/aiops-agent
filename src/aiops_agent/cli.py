"""CLI — TUI 应用入口。"""
from .config import Config
from .core.tui_app import AiOpsTUI

def main() -> None:
    config = Config.from_env()
    for err in config.validate():
        print(f"❌ {err}")
    if config.validate(): exit(1)
    AiOpsTUI(config=config).run()
