"""三层记忆编排器 — 协调工作记忆、情景记忆、核心记忆。"""
from pathlib import Path
from ..llm import BaseLLM, LLMResponse
from .base import Memory
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .core import CoreMemory

class TieredMemory(Memory):
    def __init__(self, llm: BaseLLM, working_max_messages: int = 30, working_max_tokens: int = 8000, max_episodes: int = 50, recent_episodes: int = 3, core_persist_path: str | Path | None = None, episodic_persist_path: str | Path | None = None, compaction_enabled: bool = True):
        self._llm, self._recent_episodes, self._compaction_enabled = llm, recent_episodes, compaction_enabled
        self.working = WorkingMemory(max_messages=working_max_messages, max_tokens=working_max_tokens)
        self.episodic = EpisodicMemory(max_episodes=max_episodes, persist_path=episodic_persist_path)
        self.core = CoreMemory(persist_path=core_persist_path)
        self._compacting = False
    def __bool__(self) -> bool: return True

    def add_message(self, message: dict) -> None: self.working.add_message(message)
    def get_messages(self) -> list[dict]:
        r: list[dict] = []
        r.extend(self.core.format_for_context())
        r.extend(self.episodic.format_for_context(k=self._recent_episodes))
        r.extend(self.working.get_messages())
        return r
    def count(self, fn) -> int: return fn(self.get_messages())
    def reset(self) -> None: self.working.reset(); self.episodic.reset(); self.core.clear()

    def check_compaction(self) -> bool:
        if not self._compaction_enabled or self._compacting: return False
        if not self.working.should_compact(self._llm.count_tokens): return False
        self._compacting = True
        try: self._do_compaction(); return True
        finally: self._compacting = False

    def _do_compaction(self) -> None:
        to_c = self.working.compact()
        if not to_c: return
        try:
            r = self._llm.invoke(self.episodic.get_compaction_messages(to_c))
            raw = r.content or ""
        except Exception: raw = ""
        p = self.episodic.parse_compaction_result(raw, f"压缩了 {len(to_c)} 条消息但摘要生成失败")
        self.episodic.add_episode(summary=p["summary"], key_facts=p["key_facts"], decisions=p["decisions"], unresolved=p["unresolved"], message_count=len(to_c))
        self.working.prune_compacted()

    def remember(self, fact: str) -> None: self.core.add_fact(fact)
    def forget(self, fact: str) -> bool: return self.core.remove_fact(fact)
    def get_core_facts(self) -> list[str]: return self.core.get_all_facts()
    def get_stats(self) -> dict:
        return {"working_messages": len(self.working), "working_max_messages": self.working.max_messages, "episodic_count": self.episodic.count(), "core_facts": self.core.count(), "recent_episodes_included": self._recent_episodes, "compaction_enabled": self._compaction_enabled}
