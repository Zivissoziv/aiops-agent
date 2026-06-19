"""工作记忆 — 当前对话，带压缩触发器。"""

class WorkingMemory:
    def __init__(self, max_messages: int = 30, max_tokens: int = 8000, keep_system: bool = True):
        self._messages: list[dict] = []; self._last_compact: list[int] = []
        self.max_messages, self.max_tokens, self.keep_system = max_messages, max_tokens, keep_system

    def add_message(self, m: dict) -> None: self._messages.append(m)

    def get_messages(self) -> list[dict]:
        if not self._messages: return []
        if not self.keep_system: return self._messages[-self.max_messages:]
        sys = [m for m in self._messages if m.get("role") == "system"]
        others = [m for m in self._messages if m.get("role") != "system"]
        keep = self.max_messages - len(sys)
        return sys[:self.max_messages] if keep <= 0 else sys + others[-keep:]

    def get_all_messages(self) -> list[dict]: return list(self._messages)
    def reset(self) -> None: self._messages.clear()
    def __len__(self) -> int: return len(self._messages)

    def should_compact(self, fn) -> bool:
        ns = [m for m in self._messages if m.get("role") != "system"]
        if len(ns) < self.max_messages // 2: return False
        return len(ns) >= self.max_messages or fn(self._messages) >= self.max_tokens

    def compact(self) -> list[dict]:
        idx = [i for i, m in enumerate(self._messages) if m.get("role") != "system"]
        if len(idx) <= 2: return []
        n = max(2, int(len(idx) * 0.75))
        self._last_compact = idx[:n]
        return [self._messages[i] for i in self._last_compact]

    def prune_compacted(self) -> None:
        if not self._last_compact: return
        s = set(self._last_compact)
        self._messages = [m for i, m in enumerate(self._messages) if i not in s]
        self._last_compact = []
