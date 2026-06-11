# d:\workspace\aiops-agent\src\aiops_agent\tools\shell.py
"""Shell 命令执行工具（审批模式）。

危险操作需要用户确认后才执行。
"""

import json
import re
import subprocess
import time

from langchain_core.tools import tool

# ── 危险命令定义 ──

DANGEROUS_PATTERNS: list[dict] = [
    {"pattern": r"rm\s+-rf", "level": "danger", "reason": "递归强制删除"},
    {"pattern": r"mkfs\.|format", "level": "danger", "reason": "格式化磁盘"},
    {"pattern": r"dd\s+if=", "level": "danger", "reason": "磁盘写入"},
    {"pattern": r"shutdown|reboot|halt|poweroff", "level": "danger", "reason": "系统关机/重启"},
    {"pattern": r"chmod\s+777|chown", "level": "warning", "reason": "修改权限"},
    {"pattern": r"sudo|su\s", "level": "warning", "reason": "提权操作"},
    {"pattern": r"useradd|userdel|passwd", "level": "danger", "reason": "用户管理"},
    {"pattern": r"apt\s+(install|remove|purge)|yum\s+(install|remove)|pip\s+install", "level": "warning", "reason": "安装/卸载软件"},
    {"pattern": r"docker\s+(rm|rmi|stop|kill)", "level": "warning", "reason": "Docker 管理"},
    {"pattern": r"\bdanger\b", "level": "danger", "reason": "危险操作"},  # 占位，暂无匹配
    {"pattern": r"wget.*\|\s*(ba|z)?sh|curl.*\|\s*(ba|z)?sh", "level": "danger", "reason": "远程脚本执行"},
]


def _check_dangerous(command: str) -> tuple[bool, str, str]:
    """检查命令是否危险。

    Returns:
        (is_dangerous, level, reason)
    """
    cmd = command.strip()
    for entry in DANGEROUS_PATTERNS:
        if re.search(entry["pattern"], cmd):
            return True, entry["level"], entry["reason"]
    return False, "", ""


# ── 全局审批缓存 ──
# 在 CLI 中设置的审批函数
_approval_hook = None


def set_approval_hook(hook):
    """设置审批回调函数。

    hook(command, level, reason) -> bool
    """
    global _approval_hook
    _approval_hook = hook


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
def shell(command: str, timeout: int = 30) -> str:
    """执行 Shell 命令（审批模式）。

    危险操作（删除、格式化、重启等）需要用户确认后才执行。

    Args:
        command: 要执行的 Shell 命令
        timeout: 超时时间（秒），默认 30
    """
    # 安全检查
    is_dangerous, level, reason = _check_dangerous(command)

    if is_dangerous:
        if _approval_hook:
            approved = _approval_hook(command, level, reason)
            if not approved:
                return json.dumps({
                    "success": False,
                    "error": f"危险操作已被用户拒绝: {reason}",
                    "output": "",
                }, ensure_ascii=False)
        else:
            return json.dumps({
                "success": False,
                "error": f"危险操作需要确认: {reason}（当前未设置审批回调）",
                "output": "",
            }, ensure_ascii=False)

    # 执行
    start = time.time()
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, timeout=timeout,
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
