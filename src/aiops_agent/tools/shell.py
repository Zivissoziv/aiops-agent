# d:\workspace\aiops-agent\src\aiops_agent\tools\shell.py
"""Shell 命令执行工具。"""

import json
import subprocess
import time

from langchain_core.tools import tool


def _decode(data: bytes | None) -> str:
    """解码命令输出，自动尝试 utf-8 和 gbk。"""
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
    """执行 Shell 命令并返回 JSON 结果。

    适用于查看系统状态、运行脚本、操作文件等。
    命令输出会自动尝试 utf-8 / gbk 解码。
    """
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
