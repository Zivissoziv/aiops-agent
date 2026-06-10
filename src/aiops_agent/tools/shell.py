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

    @staticmethod
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

    def execute(self, command: str, timeout: int = 30) -> ToolResult:
        start = time.time()
        try:
            # 统一用二进制模式，手动解码（避免 Windows text=True 的编码问题）
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                timeout=timeout,
            )
            elapsed = time.time() - start

            stdout = self._decode(result.stdout)
            stderr = self._decode(result.stderr)

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
