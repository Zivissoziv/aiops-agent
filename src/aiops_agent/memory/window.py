# d:\workspace\aiops-agent\src\aiops_agent\memory\window.py
"""滑动窗口记忆策略 — 只保留最近 N 条消息。

原理:
  将历史消息限制在固定数量内，超出部分自动丢弃。
  适合 token 预算固定的场景，代价是会丢失超出窗口的上下文。

使用场景:
  - token 预算严格受限
  - 对话轮次较多但上下文关联性不强
  - 作为其他策略的兜底方案
"""

from collections.abc import Callable

from .base import Memory


class SlidingWindowMemory(Memory):
    """滑动窗口记忆 — 只保留最近 N 条消息。

    Args:
        max_messages: 保留的最大消息数（不包括 system prompt）。默认 20。
        keep_system: 是否始终保留 system prompt 在窗口前。默认 True。
    """

    def __init__(self, max_messages: int = 20, keep_system: bool = True):
        self._messages: list[dict] = []
        self.max_messages = max_messages
        self.keep_system = keep_system

    def add_message(self, message: dict) -> None:
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        """获取经过窗口过滤后的消息列表。"""
        if not self._messages:
            return []

        if not self.keep_system:
            # 不保留 system prompt — 直接取最后 N 条
            return self._messages[-self.max_messages:]

        # 分离 system 消息和其他消息
        system = [m for m in self._messages if m.get("role") == "system"]
        others = [m for m in self._messages if m.get("role") != "system"]

        # 保留所有 system 消息 + 最近 N-len(system) 条其他消息
        keep_count = self.max_messages - len(system)
        if keep_count <= 0:
            # system 消息已经超过限制，只返回 system
            return system[:self.max_messages]

        return system + others[-keep_count:]

    def count(self, estimation_fn: Callable[[list[dict]], int]) -> int:
        return estimation_fn(self._messages)

    def reset(self) -> None:
        self._messages.clear()
