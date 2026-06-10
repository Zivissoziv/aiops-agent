# d:\workspace\aiops-agent\src\aiops_agent\memory\episodic.py
"""情景记忆 — 时间索引的对话摘要存档。

每段"情节"（Episode）由工作记忆压缩而来，包含:
  - summary: LLM 生成的对话摘要
  - timestamp: 创建时间
  - key_facts: 重要事实
  - decisions: 决策
  - unresolved: 未解决问题

情景记忆自动持久化到 JSON 文件，重启后仍然可用。
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class Episode:
    """一个压缩后的对话情节。"""
    summary: str
    timestamp: float = 0.0
    key_facts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    message_count: int = 0

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "timestamp": self.timestamp or time.time(),
            "key_facts": self.key_facts,
            "decisions": self.decisions,
            "unresolved": self.unresolved,
            "message_count": self.message_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        return cls(
            summary=data.get("summary", ""),
            timestamp=data.get("timestamp", 0.0),
            key_facts=data.get("key_facts", []),
            decisions=data.get("decisions", []),
            unresolved=data.get("unresolved", []),
            message_count=data.get("message_count", 0),
        )


# 压缩提示词：要求 LLM 返回结构化 JSON
COMPACTION_PROMPT = (
    "你是一个运维助手，请分析以下对话记录，生成结构化的摘要。\n\n"
    "请提取以下信息并以 JSON 格式返回（不要使用 markdown 代码块标记）：\n"
    "1. summary: 对话的核心内容摘要，保留关键的技术细节\n"
    "2. key_facts: 重要的事实信息列表，如服务器配置、发现的根因、命令输出、系统状态等\n"
    "3. decisions: 做出的决策列表\n"
    "4. unresolved: 未解决的问题或待办事项列表\n\n"
    "JSON 格式示例：\n"
    '{"summary": "...", "key_facts": ["..."], "decisions": ["..."], "unresolved": ["..."]}\n\n'
    "请确保 JSON 是有效的，不要包含对话中未提到的信息。"
)


class EpisodicMemory:
    """情景记忆 — 时间索引的摘要存档。

    Args:
        max_episodes: 最大保留的 episode 数量，超出时 FIFO 淘汰。默认 50。
        persist_path: JSON 持久化文件路径。如果为 None，不持久化。
    """

    def __init__(
        self,
        max_episodes: int = 50,
        persist_path: str | Path | None = None,
    ):
        self._episodes: list[Episode] = []
        self.max_episodes = max_episodes
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    def add_episode(
        self,
        summary: str,
        key_facts: list[str] | None = None,
        decisions: list[str] | None = None,
        unresolved: list[str] | None = None,
        message_count: int = 0,
    ) -> Episode:
        """创建并存储一个新的 episode。"""
        episode = Episode(
            summary=summary,
            timestamp=time.time(),
            key_facts=key_facts or [],
            decisions=decisions or [],
            unresolved=unresolved or [],
            message_count=message_count,
        )
        self._episodes.append(episode)

        # FIFO 淘汰
        if len(self._episodes) > self.max_episodes:
            self._episodes = self._episodes[-self.max_episodes:]

        self._save()
        return episode

    def get_recent_episodes(self, k: int = 3) -> list[Episode]:
        """获取最近 K 个 episode。"""
        return self._episodes[-k:]

    def count(self) -> int:
        return len(self._episodes)

    def reset(self) -> None:
        self._episodes.clear()
        self._save()

    # ── 压缩工具 ────────────────────────────────────

    def get_compaction_messages(self, compacted_messages: list[dict]) -> list[dict]:
        """构建压缩请求消息列表。

        将待压缩的对话历史格式化为一条 system 消息，
        供 LLM 生成结构化摘要。

        Returns:
            直接可发送给 LLM 的消息列表
        """
        lines = []
        for msg in compacted_messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                texts = [b.get("text", "") for b in content if isinstance(b, dict)]
                content = " ".join(texts)
            lines.append(f"[{role}]: {content}")

        conversation_text = "\n".join(lines)
        prompt = COMPACTION_PROMPT + "\n\n对话记录:\n" + conversation_text
        return [{"role": "system", "content": prompt}]

    def parse_compaction_result(self, raw_text: str, error_context: str = "") -> dict:
        """解析 LLM 返回的压缩结果（JSON）。

        Args:
            raw_text: LLM 返回的原始文本
            error_context: 解析失败时的备选文本

        Returns:
            包含 summary / key_facts / decisions / unresolved 的字典
        """
        text = raw_text.strip()
        # 去掉 markdown 代码块包裹
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
            return {
                "summary": parsed.get("summary", str(parsed)),
                "key_facts": parsed.get("key_facts", []),
                "decisions": parsed.get("decisions", []),
                "unresolved": parsed.get("unresolved", []),
            }
        except json.JSONDecodeError:
            return {
                "summary": text or error_context or "[摘要生成失败]",
                "key_facts": [],
                "decisions": [],
                "unresolved": [],
            }

    # ── 上下文格式化 ─────────────────────────────────

    def format_for_context(self, k: int = 3) -> list[dict]:
        """将最近 K 个 episode 格式化为 LLM 上下文消息。

        Returns:
            一条 assistant 角色的消息，包含所有摘要。
            如果没有 episode，返回空列表。
        """
        episodes = self.get_recent_episodes(k)
        if not episodes:
            return []

        parts = ["以下是之前对话中提取的历史摘要信息："]
        for i, ep in enumerate(episodes, 1):
            time_str = time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(ep.timestamp))
            parts.append(f"\n--- 历史片段 {i} ({time_str}) ---")
            parts.append(f"摘要: {ep.summary}")
            if ep.key_facts:
                parts.append(f"关键事实: {'; '.join(ep.key_facts)}")
            if ep.decisions:
                parts.append(f"决策: {'; '.join(ep.decisions)}")
            if ep.unresolved:
                parts.append(f"未解决: {'; '.join(ep.unresolved)}")

        return [{"role": "assistant", "content": "\n".join(parts)}]

    # ── 持久化 ───────────────────────────────────────

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._episodes = [Episode.from_dict(ep) for ep in data]
        except (json.JSONDecodeError, KeyError, IOError):
            self._episodes = []

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(
                    [ep.to_dict() for ep in self._episodes],
                    f, ensure_ascii=False, indent=2,
                )
        except IOError:
            pass
