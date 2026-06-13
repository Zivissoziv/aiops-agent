"""
examples/_ui.py — 教学示例共用 UI 组件

基于 Rich 库的美观终端输出，替代手工 === 和 print 排版。

使用方式:
  from _ui import console, title, subtitle, note, success, table, panel

依赖:
  pip install rich
"""

import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.text import Text
from rich import box

# 全局 Console 实例（所有示例共用）
console = Console()


def title(text: str) -> None:
    """章节标题，例如 'Part 2: Embedding — 文本的向量化'"""
    console.print()
    console.print(Panel(
        Text(text, style="bold cyan", justify="left"),
        border_style="cyan",
        box=box.HEAVY,
    ))


def subtitle(text: str) -> None:
    """子标题，例如 '步骤 1: 加载文档'"""
    console.print()
    console.print(Text(f"▌{text}", style="bold yellow"))


def note(text: str) -> None:
    """提示信息"""
    console.print(f"  [dim]{text}[/dim]")


def success(text: str) -> None:
    """成功消息"""
    console.print(f"  [bold green]OK  {text}[/bold green]")


def info(text: str) -> None:
    """普通信息输出"""
    console.print(f"  {text}")


def diagram(ascii_art: str) -> None:
    """ASCII 图表（用 Panel 包裹）"""
    console.print(Panel(
        Text(ascii_art, style="dim"),
        border_style="bright_black",
        box=box.SIMPLE,
    ))


def divider(label: str = "") -> None:
    """分隔线"""
    console.print(Text("─" * 55, style="dim") + (f" {label}" if label else ""))


def wait_for_enter(prompt: str = "按 Enter 继续...") -> None:
    """等待用户按 Enter，带样式"""
    console.print()
    console.print(Panel(
        Text(prompt, style="dim white", justify="center"),
        border_style="bright_black",
        box=box.SIMPLE_HEAVY,
    ))
    input()


def make_table(title: str = "", headers: list[str] | None = None) -> Table:
    """创建一个预设样式的表格"""
    t = Table(
        title=title,
        title_style="bold",
        box=box.ROUNDED,
        border_style="blue",
        header_style="bold cyan",
    )
    if headers:
        for h in headers:
            t.add_column(h)
    return t


def progress() -> Progress:
    """创建一个进度条实例"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    )
