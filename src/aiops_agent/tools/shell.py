# d:\workspace\aiops-agent\src\aiops_agent\tools\shell.py
"""Shell 命令执行工具 — 风险分级 + 审批模式。

安全策略:
  1. 危险命令（rm -rf、格式化等）→ 直接拒绝，不执行
  2. 高风险命令（安装包、下载、写磁盘等）→ 需要用户审批
  3. 普通命令（ls、df、ps 等）→ 直接放行

审批模式:
  - inline: 需要用户确认（默认）
  - auto: 自动放行所有高风险命令
  - deny: 自动拒绝所有高风险命令
"""

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Callable

from langchain_core.tools import tool

# ── 风险模式 ──

# 直接拒绝的危险模式（不执行、不审批）
DANGEROUS_PATTERNS = [
    r"\brm\s+-rf\b",
    r"\bdel\s+(?:/[fqs]+\s*)+",  # del /f /s /q 等强制删除
    r"\bRemove-Item\b.*\b-Recurse\b.*\b-Force\b",
    r"\bmkfs\b",
    r"\bformat\b",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bdd\s+if=",
    r":\s*\(\)\s*\{",
]

# 需要审批的高风险模式
RISK_PATTERNS: list[tuple[str, str]] = [
    (r"\bsudo\b", "提权操作"),
    (r"\b(?:pip|npm|apt|yum|brew|choco|scoop)\s+(?:install|remove|update|upgrade)\b", "安装/更新软件"),
    (r"\b(?:curl|wget)\s+.*?(?:\||[`$])", "远程脚本执行"),
    (r"\bchmod\s+777\b", "修改文件权限为 777"),
    (r">\s*(?:[A-Za-z]:\\)", "写入系统路径"),
    (r"\bdel\s+", "删除文件"),
    (r"\brm\s+", "删除文件"),
    (r"\brmdir\b", "删除目录"),
    (r"\bRemove-Item\b", "删除文件"),
    (r"\bmove\b", "移动文件"),
    (r"\bcopy\b", "复制文件"),
    (r"\brename\b", "重命名文件"),
]

# 审批回调
_approval_handler: Callable | None = None
_approval_mode: str = "inline"
_workspace_path: str | None = None


def configure_workspace(workspace_path: str | Path | None) -> None:
    """设置 shell 默认工作目录。"""
    global _workspace_path
    _workspace_path = str(workspace_path) if workspace_path else None


def configure_approval(
    handler: Callable | None = None,
    mode: str = "inline",
):
    """配置审批回调。

    handler(command, risk_reason) -> bool
    mode: "inline" | "auto" | "deny"
    """
    global _approval_handler, _approval_mode
    _approval_handler = handler
    _approval_mode = mode


def _classify_command(command: str) -> tuple[str, str | None]:
    """分类命令: ("danger"|"risk"|"safe", reason_or_None)"""
    cmd = command.strip()

    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd):
            return "danger", f"危险命令被拦截: {pattern}"

    for pattern, reason in RISK_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return "risk", reason

    return "safe", None


def _decode(data: bytes | None) -> str:
    if not data:
        return ""
    for enc in ["utf-8", "gbk", "gb18030"]:
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace")


@tool
def shell(command: str, timeout: int = 60) -> str:
    """执行 Shell 命令（风险分级 + 审批模式）。

    安全命令直接执行，高风险命令需要用户确认，危险命令直接拒绝。

    Args:
        command: 要执行的 Shell 命令
        timeout: 超时时间（秒），默认 60
    """
    # ── 风险分类 ──
    level, reason = _classify_command(command)

    if level == "danger":
        return json.dumps({"success": False, "error": reason, "output": ""}, ensure_ascii=False)

    if level == "risk":
        if _approval_mode == "deny":
            return json.dumps({
                "success": False,
                "error": f"高风险操作被自动拒绝: {reason}",
                "output": "",
            }, ensure_ascii=False)

        if _approval_mode == "inline" and _approval_handler:
            approved = _approval_handler(command, reason)
            if not approved:
                return json.dumps({
                    "success": False,
                    "error": f"用户拒绝了高风险操作: {reason}",
                    "output": "",
                }, ensure_ascii=False)
        # auto 模式直接放行

    # ── 执行 ──
    start = time.time()
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, timeout=timeout,
            cwd=_workspace_path,
        )
        elapsed = time.time() - start
        stdout = _decode(result.stdout)
        stderr = _decode(result.stderr)

        return json.dumps({
            "success": result.returncode == 0,
            "output": stdout.strip() or ("(无输出)" if result.returncode == 0 else ""),
            "error": stderr.strip() or ("" if result.returncode == 0 else f"退出码: {result.returncode}"),
            "execution_time": round(elapsed, 2),
        }, ensure_ascii=False)

    except subprocess.TimeoutExpired:
        return json.dumps({"success": False, "error": f"超时（{timeout}秒）", "output": ""}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"success": False, "error": str(e), "output": ""}, ensure_ascii=False)
