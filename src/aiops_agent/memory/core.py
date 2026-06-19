"""核心记忆 — 持久化的长期知识。"""
import json, logging
from pathlib import Path
logger = logging.getLogger(__name__)

CORE_MEMORY_HEADER = "以下是关于系统和用户的核心知识：\n"

class CoreMemory:
    def __init__(self, persist_path: str | Path | None = None):
        self._facts: list[str] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    def add_fact(self, fact: str) -> None: self._facts.append(fact); self._save()
    def add_facts(self, facts: list[str]) -> None: self._facts.extend(facts); self._save()
    def remove_fact(self, fact: str) -> bool:
        if fact in self._facts: self._facts.remove(fact); self._save(); return True
        return False
    def get_all_facts(self) -> list[str]: return list(self._facts)
    def clear(self) -> None: self._facts.clear(); self._save()
    def count(self) -> int: return len(self._facts)

    def format_for_context(self) -> list[dict]:
        if not self._facts: return []
        return [{"role": "system", "content": CORE_MEMORY_HEADER + "\n".join(f"- {f}" for f in self._facts)}]

    def _load(self):
        if not self._persist_path or not self._persist_path.exists(): return
        try:
            with open(self._persist_path, encoding="utf-8") as f:
                d = json.load(f)
            self._facts = d if isinstance(d, list) else []
        except (json.JSONDecodeError, IOError): self._facts = []

    def _save(self):
        if not self._persist_path: return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._persist_path.write_text(json.dumps(self._facts, ensure_ascii=False, indent=2), encoding="utf-8")
        except IOError as e:
            logger.warning("核心记忆写入失败 (%s): %s", self._persist_path, e)
