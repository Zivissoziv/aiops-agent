"""Shell 命令执行工具 — 风险分级 + 审批模式。"""
import json, re, subprocess, time
from pathlib import Path
from langchain_core.tools import tool

DANGER = [r"\brm\s+-rf\b", r"\bmkfs\b", r"\bformat\b", r"\bshutdown\b", r"\breboot\b", r"\bdd\s+if="]
RISK = [(r"\bsudo\b", "提权"), (r"\b(?:pip|npm|apt|yum|brew)\s+install\b", "安装软件"), (r"\b(?:curl|wget)\s+.*?(?:\||[`$])", "远程执行"), (r"\bchmod\s+777\b", "777权限"), (r"\b(?:rm|del|rmdir|Remove-Item)\b", "删除"), (r"\bmove|copy|rename\b", "文件操作")]

_handler, _mode, _ws = None, "inline", None

def configure_workspace(path: str | Path | None):
    global _ws; _ws = str(path) if path else None

def configure_approval(handler=None, mode: str = "inline"):
    global _handler, _mode; _handler, _mode = handler, mode

def _classify(cmd: str):
    for p in DANGER:
        if re.search(p, cmd): return "danger", f"危险命令: {p}"
    for p, r in RISK:
        if re.search(p, cmd, re.IGNORECASE): return "risk", r
    return "safe", None

def _decode(data: bytes | None) -> str:
    if not data: return ""
    for enc in ["utf-8", "gbk", "gb18030"]:
        try: return data.decode(enc)
        except UnicodeError: continue
    return data.decode("utf-8", errors="replace")

@tool
def shell(command: str, timeout: int = 60) -> str:
    """执行 Shell 命令（风险分级+审批）。安全命令直接执行，高风险需确认，危险直接拒绝。"""
    level, reason = _classify(command)
    if level == "danger":
        return json.dumps({"success": False, "error": reason, "output": ""}, ensure_ascii=False)
    if level == "risk":
        if _mode == "deny": return json.dumps({"success": False, "error": f"自动拒绝: {reason}", "output": ""}, ensure_ascii=False)
        if _mode == "inline" and _handler and not _handler(command, reason):
            return json.dumps({"success": False, "error": f"用户拒绝: {reason}", "output": ""}, ensure_ascii=False)
    try:
        t0 = time.time()
        r = subprocess.run(command, shell=True, capture_output=True, timeout=timeout, cwd=_ws)
        return json.dumps({"success": r.returncode == 0, "output": _decode(r.stdout).strip() or ("(无输出)" if r.returncode == 0 else ""), "error": _decode(r.stderr).strip() or ("" if r.returncode == 0 else f"退出码: {r.returncode}"), "execution_time": round(time.time() - t0, 2)}, ensure_ascii=False)
    except subprocess.TimeoutExpired: return json.dumps({"success": False, "error": f"超时({timeout}秒)", "output": ""}, ensure_ascii=False)
    except Exception as e: return json.dumps({"success": False, "error": str(e), "output": ""}, ensure_ascii=False)
