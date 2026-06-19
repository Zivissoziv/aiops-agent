"""文件读写工具 — 安全的查看和编辑文件。写操作需要审批。"""
from pathlib import Path
from langchain_core.tools import tool

_write_handler = None
_ws_path: Path | None = None

def configure_write_approval(handler=None):
    global _write_handler; _write_handler = handler

def configure_workspace(path: str | Path | None):
    global _ws_path; _ws_path = Path(path).resolve() if path else None

def _in_ws(path: str | Path) -> bool:
    if _ws_path is None: return True
    try:
        r = Path(path).resolve()
        return _ws_path in r.parents or r == _ws_path
    except (OSError, ValueError): return False

def _check(path: str, desc: str) -> str | None:
    if _in_ws(path): return None
    if _write_handler: return None if _write_handler(path, desc) else f"⚠️ 用户拒绝了 {path}"
    return f"错误: 路径不在工作区内: {path}"

@tool
def read_file(path: str, lines: int = 50) -> str:
    """读取文件内容，返回前 N 行。适用于查看日志、配置、代码等。"""
    d = _check(path, f"读取（前{lines}行）")
    if d: return d
    try:
        p = Path(path)
        if not p.exists(): return f"错误: 文件不存在: {path}"
        if not p.is_file(): return f"错误: 不是文件: {path}"
        cl = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= lines: cl.append(f"\n...（共{i+1}行，仅显示前{lines}行）"); break
                cl.append(line.rstrip())
        return "\n".join(cl) if cl else "(文件为空)"
    except PermissionError: return f"错误: 无权限: {path}"
    except Exception as e: return f"错误: {e}"

@tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """写入内容到文件。在工作区内直接执行，越界需确认。"""
    d = _check(path, content[:100] if content else "")
    if d: return d
    try:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a" if append else "w", encoding="utf-8") as f: f.write(content)
        return f"✅ 已{'追加' if append else '写入'} {path}（{len(content)} 字符）"
    except PermissionError: return f"错误: 无权限写入: {path}"
    except Exception as e: return f"错误: {e}"

@tool
def edit_file(path: str, old_text: str, new_text: str, dry_run: bool = False) -> str:
    """编辑文件中的文本（精确替换）。"""
    d = _check(path, f"编辑: {old_text[:50]}...")
    if d: return d
    try:
        p = Path(path)
        if not p.exists(): return f"错误: 文件不存在: {path}"
        c = p.read_text(encoding="utf-8"); cnt = c.count(old_text)
        if cnt == 0: return f"错误: 未找到要替换的文本:\n```\n{old_text[:200]}\n```"
        if cnt > 1: return f"错误: 找到{cnt}处匹配，请提供唯一匹配:\n```\n{old_text[:200]}\n```"
        if dry_run: return f"✅ 将替换1处:\n```\n{old_text[:200]}\n```→\n```\n{new_text[:200]}\n```"
        p.write_text(c.replace(old_text, new_text, 1), encoding="utf-8")
        return f"✅ 已编辑{path}（替换1处，{len(old_text)}→{len(new_text)}字符）"
    except PermissionError: return f"错误: 无权限编辑: {path}"
    except Exception as e: return f"错误: {e}"
