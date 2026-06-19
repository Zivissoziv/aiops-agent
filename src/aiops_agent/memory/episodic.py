"""情景记忆 — 时间索引的对话摘要存档。"""
import json, logging, textwrap, time
from dataclasses import dataclass, field
from pathlib import Path
logger = logging.getLogger(__name__)

COMPACT_PROMPT = textwrap.dedent("""\
分析以下运维对话，生成JSON摘要：
{"summary":"...","key_facts":["..."],"decisions":["..."],"unresolved":["..."]}
不要用markdown代码块。""")

@dataclass
class Episode:
    summary: str; timestamp: float = 0.0
    key_facts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    message_count: int = 0

    def to_dict(self) -> dict:
        return {"summary": self.summary, "timestamp": self.timestamp or time.time(), "key_facts": self.key_facts, "decisions": self.decisions, "unresolved": self.unresolved, "message_count": self.message_count}
    @classmethod
    def from_dict(cls, d: dict) -> "Episode":
        return cls(summary=d.get("summary",""), timestamp=d.get("timestamp",0.0), key_facts=d.get("key_facts",[]), decisions=d.get("decisions",[]), unresolved=d.get("unresolved",[]), message_count=d.get("message_count",0))

class EpisodicMemory:
    def __init__(self, max_episodes: int = 50, persist_path: str | Path | None = None):
        self._episodes: list[Episode] = []; self.max_episodes = max_episodes
        self._persist_path = Path(persist_path) if persist_path else None; self._load()

    def add_episode(self, summary: str, key_facts=None, decisions=None, unresolved=None, message_count=0) -> Episode:
        ep = Episode(summary=summary, timestamp=time.time(), key_facts=key_facts or [], decisions=decisions or [], unresolved=unresolved or [], message_count=message_count)
        self._episodes.append(ep)
        if len(self._episodes) > self.max_episodes: self._episodes = self._episodes[-self.max_episodes:]
        self._save(); return ep

    def get_recent_episodes(self, k: int = 3) -> list[Episode]: return self._episodes[-k:]
    def count(self) -> int: return len(self._episodes)
    def reset(self) -> None: self._episodes.clear(); self._save()

    def get_compaction_messages(self, msgs: list[dict]) -> list[dict]:
        lines = [f"[{m['role']}]: {' '.join(b.get('text','') for b in m['content'] if isinstance(b,dict)) if isinstance(m.get('content'),list) else m.get('content','')}" for m in msgs]
        return [{"role": "system", "content": COMPACT_PROMPT + "\n\n对话:\n" + "\n".join(lines)}]

    def parse_compaction_result(self, raw: str, fallback: str = "") -> dict:
        t = raw.strip()
        if t.startswith("```"):
            ls = t.split("\n"); t = "\n".join(ls[1:] if ls[0].startswith("```") else ls)[:-3].strip() if ls[-1].startswith("```") else "\n".join(ls[1:]).strip()
        try:
            p = json.loads(t); return {"summary": p.get("summary", str(p)), "key_facts": p.get("key_facts",[]), "decisions": p.get("decisions",[]), "unresolved": p.get("unresolved",[])}
        except json.JSONDecodeError:
            return {"summary": t or fallback or "[失败]", "key_facts":[], "decisions":[], "unresolved":[]}

    def format_for_context(self, k: int = 1) -> list[dict]:
        eps = self.get_recent_episodes(k)
        if not eps: return []
        parts = ["以下是之前对话的历史摘要："]
        for i, ep in enumerate(eps, 1):
            ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(ep.timestamp))
            parts.append(f"\n--- 历史 {i} ({ts}) ---\n摘要: {ep.summary[:200]}")
            for f in ep.key_facts[:3]: parts.append(f"  • {f[:100]}")
            for d in ep.decisions[:2]: parts.append(f"  决策: {d[:100]}")
            for u in ep.unresolved[:2]: parts.append(f"  未解决: {u[:100]}")
        return [{"role": "assistant", "content": "\n".join(parts)}]

    def _load(self):
        if not self._persist_path or not self._persist_path.exists(): return
        try:
            self._episodes = [Episode.from_dict(ep) for ep in json.loads(self._persist_path.read_text(encoding="utf-8"))]
        except Exception: self._episodes = []

    def _save(self):
        if not self._persist_path: return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        try: self._persist_path.write_text(json.dumps([ep.to_dict() for ep in self._episodes], ensure_ascii=False, indent=2), encoding="utf-8")
        except IOError as e: logger.warning("情景记忆写入失败: %s", e)
