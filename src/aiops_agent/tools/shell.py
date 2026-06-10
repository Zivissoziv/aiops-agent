# d:\workspace\aiops-agent\src\aiops_agent\tools\shell.py
"""Shell 命令执行工具。"""

import subprocess
import time

from .base import Tool, ToolResult


class ShellTool(Tool):
    """在本地执行 Shell 命令的工具。"""

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "执行 Shell 命令并返回输出。"
            "适用于查看系统状态、运行脚本、操作文件等。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 Shell 命令",
                },
                "timeout": {
                    "type": "integer",
                    "description": "超时时间（秒），默认 30",
                    "default": 30,
                },
            },
            "required": ["command"],
        }

    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        start = time.time()
        try:
            # 先试试 utf-8，失败时用系统编码
            encoding = "utf-8"
            try:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding=encoding,
                )
            except UnicodeDecodeError:
                # 回退：不用 text=True，手动解码
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    timeout=timeout,
                )
                # 尝试多种编码解码
                for enc in ["gbk", "utf-8", "gb18030"]:
                    try:
                        result.stdout = result.stdout.decode(enc, errors="replace")
                        result.stderr = result.stderr.decode(enc, errors="replace")
                        break
                    except (UnicodeDecodeError, AttributeError):
                        continue
                else:
                    result.stdout = result.stdout.decode("utf-8", errors="replace") if result.stdout else ""
                    result.stderr = result.stderr.decode("utf-8", errors="replace") if result.stderr else ""
            elapsed = time.time() - start

            stdout = result.stdout or ""
            stderr = result.stderr or ""

            if result.returncode == 0:
                output = stdout.strip() or "(命令执行成功，无输出)"
                return ToolResult(
                    success=True,
                    output=output,
                    execution_time=elapsed,
                )
            else:
                return ToolResult(
                    success=True,
                    output=stdout.strip(),
                    error=stderr.strip() or f"退出码: {result.returncode}",
                    execution_time=elapsed,
                )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"命令执行超时（{timeout}秒）",
                execution_time=time.time() - start,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                execution_time=time.time() - start,
            )
