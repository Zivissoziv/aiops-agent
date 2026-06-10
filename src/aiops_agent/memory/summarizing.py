# d:\workspace\aiops-agent\src\aiops_agent\memory\summarizing.py
"""摘要记忆策略 — 当 token 数超过阈值时自动摘要历史消息。

原理:
  维护全部消息历史，但在 get_messages() 时检查 token 估算值。
  如果超过 max_tokens，调用 LLM 对最早的非 system 消息做摘要，
  用摘要消息替换被摘要的内容。后续 token 再超限时会增量摘要。

使用场景:
  - 对话需长期上下文但 token 预算有限
  - 需要保留核心事实和决策，允许丢失细节
  - 适合多轮排查场景（摘要保留根因分析路径）

注意:
  - 每次摘要是一次额外的 LLM 调用
  - 摘要会丢失原始对话的细节
"""

from collections.abc import Callable

from .base import Memory
from ..llm import BaseLLM


# 默认摘要提示词
DEFAULT_SUMMARY_PROMPT = (
    "以下是运维助手的对话历史。请总结对话的核心内容，"
    "保留关键的事实、已经执行的命令、命令的输出摘要、"
    "发现的根因和做出的决策。"
    "摘要应该简洁但完整，以便后续对话可以参考。"
    "用中文回复。"
)

# 摘要前后缀标记，帮助 LLM 在后续对话中识别摘要内容
SUMMARY_PREFIX = "[对话摘要]\n"


class SummarizingMemory(Memory):
    """摘要记忆 — token 超限时自动摘要旧消息。

    Args:
        llm: LLM 实例，用于生成摘要
        max_tokens: 触发摘要的 token 阈值，默认 4000
        summary_prompt: 摘要提示词
    """

    def __init__(
        self,
        llm: BaseLLM,
        max_tokens: int = 4000,
        summary_prompt: str | None = None,
    ):
        self._messages: list[dict] = []
        self._llm = llm
        self.max_tokens = max_tokens
        self._summary_prompt = summary_prompt or DEFAULT_SUMMARY_PROMPT

        # 已摘要到的消息索引（不含该索引本身）
        self._summarized_until: int = 0

        # 缓存的摘要消息
        self._summary_message: dict | None = None

    def add_message(self, message: dict) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        """获取消息列表，必要时触发摘要。"""
        if not self._messages:
            return []

        # 检查是否需要摘要
        estimated = self._llm.count_tokens(self._messages)
        if estimated > self.max_tokens and self._summarized_until < len(self._messages):
            self._do_summarize()

        # 返回 system + 摘要(如有) + 未摘要部分
        system = [m for m in self._messages if m.get("role") == "system"]
        recent = self._messages[self._summarized_until:]

        result = list(system)
        if self._summary_message:
            result.append(self._summary_message)
        result.extend(recent)
        return result

    def _do_summarize(self) -> None:
        """执行摘要 — 调用 LLM 对最早的非 system 消息做摘要。"""
        # 收集从 0 到 _summarized_until 的非 system 消息
        to_summarize = [
            m for m in self._messages[:self._summarized_until]
            if m.get("role") != "system"
        ]
        if not to_summarize:
            # 没有可摘要的内容，推进指针到当前位置
            self._summarized_until = len(self._messages)
            return

        # 构建摘要请求
        summary_messages = [
            {"role": "system", "content": self._summary_prompt},
            *to_summarize,
        ]

        try:
            response = self._llm.invoke(summary_messages)
            summary_text = response.content or ""
            self._summary_message = {
                "role": "assistant",
                "content": f"{SUMMARY_PREFIX}{summary_text}",
            }
        except Exception as e:
            # 摘要失败时不影响对话
            self._summary_message = {
                "role": "assistant",
                "content": f"{SUMMARY_PREFIX}[摘要生成失败: {e}]",
            }

        # 推进摘要指针
        self._summarized_until = len(self._messages)

    def count(self, estimation_fn: Callable[[list[dict]], int]) -> int:
        return estimation_fn(self._messages)

    def reset(self) -> None:
        self._messages.clear()
        self._summary_message = None
        self._summarized_until = 0
