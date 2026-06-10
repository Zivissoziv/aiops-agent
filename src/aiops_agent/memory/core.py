# d:\workspace\aiops-agent\src\aiops_agent\memory\core.py
"""核心记忆 — 持久化的长期知识。

存储关于系统和用户的稳定事实，如:
  - 服务器拓扑
  - 常用运维流程
  - 用户偏好
  - 已做的决策

核心记忆自动持久化到 JSON 文件，重启后仍可用。
"""

import json
from pathlib import Path


# 核心记忆在上下文中显示时的标题
CORE_MEMORY_HEADER = "以下是关于系统和用户的核心知识，请参考这些信息来回答问题：\n"


class CoreMemory:
    """持久化的长期知识存储。

    Args:
        persist_path: JSON 持久化文件路径。如果为 None，不持久化。
    """

    def __init__(self, persist_path: str | Path | None = None):
        self._facts: list[str] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    # ── 事实管理 ─────────────────────────────────────

    def add_fact(self, fact: str) -> None:
        """添加一条核心记忆。"""
        self._facts.append(fact)
        self._save()

    def add_facts(self, facts: list[str]) -> None:
        """批量添加核心记忆。"""
        self._facts.extend(facts)
        self._save()

    def remove_fact(self, fact: str) -> bool:
        """删除一条核心记忆。"""
        if fact in self._facts:
            self._facts.remove(fact)
            self._save()
            return True
        return False

    def get_all_facts(self) -> list[str]:
        """获取所有核心记忆。"""
        return list(self._facts)

    def clear(self) -> None:
        """清空所有核心记忆。"""
        self._facts.clear()
        self._save()

    def count(self) -> int:
        return len(self._facts)

    # ── 上下文格式化 ─────────────────────────────────

    def format_for_context(self) -> list[dict]:
        """将核心记忆格式化为 LLM 上下文消息。

        Returns:
            一条 system 角色的消息，包含所有事实。
            如果没有事实，返回空列表。
        """
        if not self._facts:
            return []
        content = CORE_MEMORY_HEADER + "\n".join(
            f"- {fact}" for fact in self._facts
        )
        return [{"role": "system", "content": content}]

    # ── 持久化 ───────────────────────────────────────

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._facts = data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            self._facts = []

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._persist_path, "w", encoding="utf-8") as f:
                json.dump(self._facts, f, ensure_ascii=False, indent=2)
        except IOError:
            pass
