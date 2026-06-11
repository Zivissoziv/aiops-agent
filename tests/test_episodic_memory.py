"""情景记忆单元测试。"""

import json

import pytest

from aiops_agent.memory.episodic import COMPACTION_PROMPT, EpisodicMemory, Episode


class TestEpisode:
    """Episode 数据类测试。"""

    def test_to_dict(self):
        ep = Episode(summary="test", timestamp=100.0, key_facts=["f1"], decisions=["d1"], unresolved=["u1"], message_count=5)
        d = ep.to_dict()
        assert d["summary"] == "test"
        assert d["timestamp"] == 100.0
        assert d["key_facts"] == ["f1"]

    def test_from_dict(self):
        d = {"summary": "s1", "timestamp": 200.0, "key_facts": ["f1"], "decisions": ["d1"], "unresolved": ["u1"], "message_count": 3}
        ep = Episode.from_dict(d)
        assert ep.summary == "s1"
        assert ep.message_count == 3

    def test_from_dict_empty(self):
        ep = Episode.from_dict({})
        assert ep.summary == ""
        assert ep.timestamp == 0.0
        assert ep.key_facts == []


class TestEpisodicMemory:
    """EpisodicMemory 基础功能测试。"""

    def test_init(self):
        em = EpisodicMemory()
        assert em.count() == 0
        assert em.max_episodes == 50

    def test_add_episode(self):
        em = EpisodicMemory()
        ep = em.add_episode("对话摘要", key_facts=["重要事实"], decisions=["决策1"])
        assert em.count() == 1
        assert ep.summary == "对话摘要"
        assert ep.key_facts == ["重要事实"]

    def test_fifo_eviction(self):
        em = EpisodicMemory(max_episodes=3)
        for i in range(5):
            em.add_episode(f"摘要{i}")
        assert em.count() == 3
        summaries = [ep.summary for ep in em.get_recent_episodes(5)]
        assert summaries == ["摘要2", "摘要3", "摘要4"]

    def test_get_recent_episodes(self):
        em = EpisodicMemory()
        for i in range(10):
            em.add_episode(f"摘要{i}")
        recent = em.get_recent_episodes(3)
        assert len(recent) == 3
        assert recent[0].summary == "摘要7"
        assert recent[-1].summary == "摘要9"

    def test_reset(self):
        em = EpisodicMemory()
        em.add_episode("test")
        em.reset()
        assert em.count() == 0


class TestEpisodicMemoryCompaction:
    """EpisodicMemory 压缩相关功能测试。"""

    def test_get_compaction_messages(self):
        em = EpisodicMemory()
        msgs = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！"},
        ]
        result = em.get_compaction_messages(msgs)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert COMPACTION_PROMPT in result[0]["content"]
        assert "[user]: 你好" in result[0]["content"]

    def test_get_compaction_messages_list_content(self):
        em = EpisodicMemory()
        msgs = [
            {"role": "user", "content": [{"type": "text", "text": "多模态消息"}]},
        ]
        result = em.get_compaction_messages(msgs)
        assert "多模态消息" in result[0]["content"]

    def test_parse_compaction_result_valid(self):
        em = EpisodicMemory()
        raw = '{"summary": "测试摘要", "key_facts": ["f1"], "decisions": ["d1"], "unresolved": ["u1"]}'
        parsed = em.parse_compaction_result(raw)
        assert parsed["summary"] == "测试摘要"
        assert parsed["key_facts"] == ["f1"]

    def test_parse_compaction_result_markdown(self):
        em = EpisodicMemory()
        raw = '```json\n{"summary": "md摘要", "key_facts": []}\n```'
        parsed = em.parse_compaction_result(raw)
        assert parsed["summary"] == "md摘要"

    def test_parse_compaction_result_invalid(self):
        em = EpisodicMemory()
        parsed = em.parse_compaction_result("不是 JSON", error_context="备选文本")
        assert parsed["summary"] == "不是 JSON"
        assert parsed["key_facts"] == []


class TestEpisodicMemoryContext:
    """EpisodicMemory 上下文格式化测试。"""

    def test_format_empty(self):
        em = EpisodicMemory()
        assert em.format_for_context() == []

    def test_format_recent(self):
        em = EpisodicMemory()
        em.add_episode("摘要1", key_facts=["事实1"])
        em.add_episode("摘要2", key_facts=["事实2"])
        result = em.format_for_context(k=1)
        assert len(result) == 1
        assert "摘要2" in result[0]["content"]
        assert "摘要1" not in result[0]["content"]

    def test_format_all_fields(self):
        em = EpisodicMemory()
        em.add_episode("完整摘要", key_facts=["f1", "f2", "f3", "f4"], decisions=["d1", "d2", "d3"], unresolved=["u1"])
        result = em.format_for_context(k=1)
        content = result[0]["content"]
        assert "完整摘要" in content
        # key_facts 最多 3 条
        assert content.count("•") == 3
        # decisions 最多 2 条
        assert content.count("决策:") == 2
        # unresolved 最多 2 条
        assert content.count("未解决:") == 1


class TestEpisodicMemoryPersistence:
    """EpisodicMemory 持久化测试。"""

    def test_persist_and_load(self, temp_json_file):
        em = EpisodicMemory(persist_path=temp_json_file)
        em.add_episode("存档摘要", key_facts=["f1"], decisions=["d1"])

        em2 = EpisodicMemory(persist_path=temp_json_file)
        assert em2.count() == 1
        assert em2.get_recent_episodes(1)[0].summary == "存档摘要"

    def test_load_corrupted(self, temp_json_file):
        temp_json_file.write_text("坏数据", encoding="utf-8")
        em = EpisodicMemory(persist_path=temp_json_file)
        assert em.count() == 0
