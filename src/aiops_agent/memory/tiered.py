# d:\workspace\aiops-agent\src\aiops_agent\memory\tiered.py
"""三层记忆编排器 — 协调工作记忆、情景记忆、核心记忆。

此类的职责:
  1. 实现 Memory 接口，供 Agent 和 CLI 使用
  2. 在 get_messages() 中组合三层的消息视图
  3. 管理压缩流程（工作记忆 → 情景记忆）

组合上下文顺序:
  system_prompt（由 Agent 添加）
  core_memory 事实（system 角色）
  episodic 摘要（assistant 角色，最近 K 条）
  working_memory 消息（当前对话）
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..llm import BaseLLM, LLMResponse
from .base import Memory
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .core import CoreMemory


class TieredMemory(Memory):
    """三层记忆编排器。

    Args:
        llm: LLM 实例，用于压缩时生成摘要。
        working_max_messages: 工作记忆最大消息数。默认 30。
        working_max_tokens: 工作记忆触发器 token 阈值。默认 8000。
        max_episodes: 情景记忆最大 episode 数。默认 50。
        recent_episodes: 上下文中包含的最近 episode 数。默认 3。
        core_persist_path: 核心记忆持久化路径。
        episodic_persist_path: 情景记忆持久化路径。
        compaction_enabled: 是否启用自动压缩。默认 True。
    """

    def __init__(
        self,
        llm: BaseLLM,
        working_max_messages: int = 30,
        working_max_tokens: int = 8000,
        max_episodes: int = 50,
        recent_episodes: int = 3,
        core_persist_path: str | Path | None = None,
        episodic_persist_path: str | Path | None = None,
        compaction_enabled: bool = True,
    ):
        self._llm = llm
        self._recent_episodes = recent_episodes
        self._compaction_enabled = compaction_enabled

        self.working = WorkingMemory(
            max_messages=working_max_messages,
            max_tokens=working_max_tokens,
        )
        self.episodic = EpisodicMemory(
            max_episodes=max_episodes,
            persist_path=episodic_persist_path,
        )
        self.core = CoreMemory(
            persist_path=core_persist_path,
        )

        self._compacting = False

    # ── Memory 接口 ──────────────────────────────────

    def add_message(self, message: dict) -> None:
        """添加一条消息到工作记忆。"""
        self.working.add_message(message)

    def get_messages(self) -> list[dict]:
        """构建组合上下文。

        顺序: core → episodic → working
        """
        result: list[dict] = []

        # 1. 核心记忆（system 角色）
        result.extend(self.core.format_for_context())

        # 2. 情景摘要（assistant 角色）
        result.extend(
            self.episodic.format_for_context(k=self._recent_episodes)
        )

        # 3. 工作记忆
        result.extend(self.working.get_messages())

        return result

    def count(self, estimation_fn: Callable[[list[dict]], int]) -> int:
        """估算当前组合上下文的 token 数。"""
        return estimation_fn(self.get_messages())

    def reset(self) -> None:
        """重置所有三层记忆。"""
        self.working.reset()
        self.episodic.reset()
        self.core.clear()

    # ── 压缩逻辑 ─────────────────────────────────────

    def check_compaction(self) -> bool:
        """检查并执行压缩。

        由 Agent 在每轮对话后调用。
        使用 _compacting 标志防止重入。

        Returns:
            True 表示执行了压缩，False 表示无需压缩。
        """
        if not self._compaction_enabled or self._compacting:
            return False
        if not self.working.should_compact(self._llm.count_tokens):
            return False

        self._compacting = True
        try:
            self._do_compaction()
            return True
        finally:
            self._compacting = False

    def _do_compaction(self) -> None:
        """执行压缩: 提取 → 摘要 → 存储 → 修剪。"""
        # 1. 提取待压缩消息
        to_compact = self.working.compact()
        if not to_compact:
            return

        # 2. 构建压缩提示
        compaction_messages = self.episodic.get_compaction_messages(to_compact)

        # 3. 调用 LLM
        try:
            response: LLMResponse = self._llm.invoke(compaction_messages)
            raw = response.content or ""
        except Exception as e:
            raw = ""

        # 4. 解析结果
        parsed = self.episodic.parse_compaction_result(
            raw,
            error_context=f"压缩了 {len(to_compact)} 条消息但摘要生成失败",
        )

        # 5. 存储 episode
        self.episodic.add_episode(
            summary=parsed["summary"],
            key_facts=parsed["key_facts"],
            decisions=parsed["decisions"],
            unresolved=parsed["unresolved"],
            message_count=len(to_compact),
        )

        # 6. 修剪工作记忆
        self.working.prune_compacted()

    # ── 核心记忆管理 ─────────────────────────────────

    def remember(self, fact: str) -> None:
        """添加一条核心记忆。"""
        self.core.add_fact(fact)

    def forget(self, fact: str) -> bool:
        """删除一条核心记忆。"""
        return self.core.remove_fact(fact)

    def get_core_facts(self) -> list[str]:
        """获取所有核心记忆。"""
        return self.core.get_all_facts()

    # ── 统计信息 ─────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """获取三层记忆的统计信息。"""
        return {
            "working_messages": len(self.working),
            "working_max_messages": self.working.max_messages,
            "episodic_count": self.episodic.count(),
            "core_facts": self.core.count(),
            "recent_episodes_included": self._recent_episodes,
            "compaction_enabled": self._compaction_enabled,
        }
