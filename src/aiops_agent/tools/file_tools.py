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


@tool
def read_file(path: str, lines: int = 50) -> str:
    """读取文件内容，返回前 N 行。

    适用于查看日志文件、配置文件、代码文件等。

    Args:
        path: 文件路径
        lines: 读取的行数，默认 50
    """
    # workspace 边界检查：越界路径需要审批
    if not _is_in_workspace(path):
        if _write_approval_handler:
            if not _write_approval_handler(path, f"读取文件（前 {lines} 行）"):
                return f"⚠️ 用户拒绝了读取 {path}"
        else:
            return f"错误: 路径不在工作区内且未配置审批回调: {path}"
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
    # workspace 边界检查：越界路径需要审批
    if not _is_in_workspace(path):
        if _write_approval_handler:
            preview = content[:100]
            if not _write_approval_handler(path, preview):
                return f"⚠️ 用户拒绝了写入 {path}"
        else:
            return f"错误: 路径不在工作区内且未配置审批回调: {path}"

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
