"""NOTEPAD 工具 — 持久化的工作区笔记。Agent 可读写 NOTEPAD.md 来记录研究发现和决策。"""

from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool

NOTEPAD_FILE = "NOTEPAD.md"


@tool
def read_notepad() -> str:
    """读取工作区笔记（NOTEPAD.md）。包含研究发现、决策记录和重要上下文。"""
    p = Path(NOTEPAD_FILE)
    if not p.exists():
        return "(笔记为空)"
    return p.read_text(encoding="utf-8")


@tool
def append_notepad(heading: str, content: str) -> str:
    """向工作区笔记（NOTEPAD.md）追加内容。适用于记录研究发现、决策和重要观察。

    Args:
        heading: 节标题（如 "性能分析结果"）
        content: 笔记内容
    """
    p = Path(NOTEPAD_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n## {heading}\n\n_{timestamp}_\n\n{content}\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(entry)
    return f"✅ 已追加笔记「{heading}」"
