# d:\workspace\aiops-agent\src\aiops_agent\memory\working.py
"""工作记忆 — 当前对话，带压缩触发器。

这是三层记忆系统的第一层，管理当前对话的消息列表。
当消息数或 token 数超过阈值时，触发压缩（compaction），
将最旧的消息提取出来供情景记忆做摘要。

工作记忆不是 Memory 子类，而是 TieredMemory 的内部组件。
"""

from collections.abc import Callable


class WorkingMemory:
    """有界的工作记忆，带压缩触发器。

    Args:
        max_messages: 最大消息数（硬限制），默认 30。
        max_tokens: 最大 token 数（软限制），默认 8000。
        keep_system: 是否保留 system 消息在窗口中。默认 True。
    """

    def __init__(
        self,
        max_messages: int = 30,
        max_tokens: int = 8000,
        keep_system: bool = True,
    ):
        self._messages: list[dict] = []
        self._last_compact_indices: list[int] = []
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.keep_system = keep_system

    # ── 基础操作 ────────────────────────────────────

    def add_message(self, message: dict) -> None:
        """添加一条消息到工作记忆。"""
        self._messages.append(message)

    def get_messages(self) -> list[dict]:
        """获取当前工作记忆中的消息（已做窗口裁剪）。"""
        if not self._messages:
            return []
        if not self.keep_system:
            return self._messages[-self.max_messages:]
        system = [m for m in self._messages if m.get("role") == "system"]
        others = [m for m in self._messages if m.get("role") != "system"]
        keep_count = self.max_messages - len(system)
        if keep_count <= 0:
            return system[:self.max_messages]
        return system + others[-keep_count:]

    def get_all_messages(self) -> list[dict]:
        """获取所有消息（包括已超出窗口的，用于 token 估算）。"""
        return list(self._messages)

    def reset(self) -> None:
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    # ── 压缩逻辑 ────────────────────────────────────

    def should_compact(self, estimation_fn: Callable[[list[dict]], int]) -> bool:
        """检查是否需要触发压缩。

        两个触发条件（任一满足即触发）:
        1. 非 system 消息数 >= max_messages
        2. token 数 >= max_tokens（仅在消息数过半时检查，避免频繁估算）

        Returns:
            True 表示需要压缩
        """
        non_system = [m for m in self._messages if m.get("role") != "system"]
        if len(non_system) < self.max_messages // 2:
            return False  # 消息太少，不需要压缩
        if len(non_system) >= self.max_messages:
            return True
        if estimation_fn(self._messages) >= self.max_tokens:
            return True
        return False

    def compact(self) -> list[dict]:
        """提取需要压缩的最旧消息。

        策略: 取最旧的约 75% 非 system 消息供压缩，
        保留最近的 25% 在工作记忆中。

        Returns:
            需要压缩的消息列表（按时间顺序）。
            如果没有足够的消息可供压缩，返回空列表。
        """
        non_system_indices = [
            i for i, m in enumerate(self._messages)
            if m.get("role") != "system"
        ]
        if len(non_system_indices) <= 2:
            return []  # 至少保留 2 条非 system 消息

        # 压缩最旧的 75%
        compact_count = max(2, int(len(non_system_indices) * 0.75))
        self._last_compact_indices = non_system_indices[:compact_count]
        return [self._messages[i] for i in self._last_compact_indices]

    def prune_compacted(self) -> None:
        """从工作记忆中移除已压缩的消息。"""
        if not self._last_compact_indices:
            return
        compact_set = set(self._last_compact_indices)
        self._messages = [
            m for i, m in enumerate(self._messages)
            if i not in compact_set
        ]
        self._last_compact_indices = []
