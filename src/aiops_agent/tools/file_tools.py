# d:\workspace\aiops-agent\src\aiops_agent\tools\file_tools.py
"""文件读写工具 — 安全的查看和编辑文件。写操作需要审批。"""

from pathlib import Path

from langchain_core.tools import tool

# 写操作审批回调
_write_approval_handler = None


def configure_write_approval(handler=None):
    """设置写操作审批回调。handler(path, content_preview) -> bool"""
    global _write_approval_handler
    _write_approval_handler = handler


# 工作区沙箱路径
_workspace_path: Path | None = None


def configure_workspace(workspace_path: str | Path | None) -> None:
    """设置沙箱路径。路径在 workspace 内的文件操作免审批，越界需审批。"""
    global _workspace_path
    _workspace_path = Path(workspace_path).resolve() if workspace_path else None


def _is_in_workspace(path: str | Path) -> bool:
    """检查路径是否在 workspace 沙箱内。"""
    if _workspace_path is None:
        return True  # 没有设置沙箱时，不限制
    try:
        resolved = Path(path).resolve()
        return _workspace_path in resolved.parents or resolved == _workspace_path
    except (OSError, ValueError):
        return False


def _check_workspace_access(path: str, desc: str) -> str | None:
    """检查路径越界，返回 None 表示允许访问，返回字符串表示拒绝原因。"""
    if _is_in_workspace(path):
        return None
    if _write_approval_handler:
        if not _write_approval_handler(path, desc):
            return f"⚠️ 用户拒绝了 {path}"
        return None
    return f"错误: 路径不在工作区内且未配置审批回调: {path}"


@tool
def read_file(path: str, lines: int = 50) -> str:
    """读取文件内容，返回前 N 行。

    适用于查看日志文件、配置文件、代码文件等。

    Args:
        path: 文件路径
        lines: 读取的行数，默认 50
    """
    denied = _check_workspace_access(path, f"读取文件（前 {lines} 行）")
    if denied:
        return denied

    try:
        p = Path(path)
        if not p.exists():
            return f"错误: 文件不存在: {path}"
        if not p.is_file():
            return f"错误: 不是文件: {path}"

        content_lines = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    content_lines.append(f"\n... (文件共 {i}+ 行，仅显示前 {lines} 行)")
                    break
                content_lines.append(line.rstrip())

        return "\n".join(content_lines) if content_lines else "(文件为空)"

    except PermissionError:
        return f"错误: 无权限读取: {path}"
    except Exception as e:
        return f"错误: {e}"


@tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """写入内容到文件（越界需要用户确认）。

    适用于创建文件、追加日志、修改配置等。
    在工作区内的写操作直接执行，越界会提示用户确认。

    Args:
        path: 文件路径
        content: 要写入的内容
        append: 是否追加到文件末尾，默认 False（覆盖写入）
    """
    denied = _check_workspace_access(path, content[:100] if content else "")
    if denied:
        return denied

    try:
        p = Path(path)
        mode = "a" if append else "w"

        # 创建父目录（如果不存在）
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, mode, encoding="utf-8") as f:
            f.write(content)

        action = "追加到" if append else "写入"
        return f"✅ 已{action} {path}（{len(content)} 字符）"

    except PermissionError:
        return f"错误: 无权限写入: {path}"
    except Exception as e:
        return f"错误: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str, dry_run: bool = False) -> str:
    """编辑文件中的文本（精确替换）。适用于修改配置、修复 bug、重构代码等。
    在工作区内的编辑直接执行，越界需用户确认。

    Args:
        path: 文件路径
        old_text: 要替换的原始文本（必须在文件中唯一匹配）
        new_text: 替换后的新文本
        dry_run: 是否仅预览匹配结果而不实际写入，默认 False
    """
    denied = _check_workspace_access(path, f"编辑文件: {old_text[:50]}...")
    if denied:
        return denied

    try:
        p = Path(path)
        if not p.exists():
            return f"错误: 文件不存在: {path}"
        if not p.is_file():
            return f"错误: 不是文件: {path}"

        content = p.read_text(encoding="utf-8")
        count = content.count(old_text)

        if count == 0:
            return f"错误: 未找到要替换的文本:\n```\n{old_text[:200]}\n```"
        elif count > 1:
            return f"错误: 找到 {count} 处匹配，请提供更多上下文使 old_text 唯一匹配:\n```\n{old_text[:200]}\n```"

        if dry_run:
            return f"✅ 将替换 1 处匹配:\n```\n{old_text[:200]}\n```\n→\n```\n{new_text[:200]}\n```"

        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"✅ 已编辑 {path}（替换 1 处，{len(old_text)} → {len(new_text)} 字符）"

    except PermissionError:
        return f"错误: 无权限编辑: {path}"
    except Exception as e:
        return f"错误: {e}"
