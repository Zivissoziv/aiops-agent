"""三层记忆编排器单元测试。"""

from unittest.mock import MagicMock

import pytest

from aiops_agent.memory.tiered import TieredMemory


class TestTieredMemory:

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.count_tokens.return_value = 100
        return llm

    def test_init(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        assert tm._compaction_enabled is True
        assert tm.working.max_messages == 30
        assert tm.episodic.max_episodes == 50

    def test_add_and_get(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hello"})
        msgs = tm.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"

    def test_core_in_context(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.remember("核心事实")
        tm.add_message({"role": "user", "content": "hello"})
        msgs = tm.get_messages()
        assert msgs[0]["role"] == "system"
        assert "核心事实" in msgs[0]["content"]

    def test_count(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hello"})
        assert tm.count(mock_llm.count_tokens) == 100

    def test_reset(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hello"})
        tm.remember("事实")
        tm.reset()
        assert tm.working.get_messages() == []
        assert tm.core.count() == 0

    def test_core_management(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.remember("事实1")
        tm.remember("事实2")
        assert tm.get_core_facts() == ["事实1", "事实2"]
        assert tm.forget("事实1") is True
        assert tm.forget("不存在") is False

    def test_get_stats(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hello"})
        tm.remember("事实")
        stats = tm.get_stats()
        assert stats["working_messages"] == 1
        assert stats["core_facts"] == 1


class TestTieredMemoryCompaction:

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.count_tokens.return_value = 9999
        response = MagicMock()
        response.content = '{"summary": "压缩摘要", "key_facts": ["f1"], "decisions": ["d1"], "unresolved": []}'
        llm.invoke.return_value = response
        return llm

    def test_disabled_when_false(self, mock_llm):
        tm = TieredMemory(llm=mock_llm, compaction_enabled=False)
        assert tm.check_compaction() is False

    def test_check_and_compact(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        # 消息太少 → 不压缩
        tm.add_message({"role": "user", "content": "hi"})
        assert tm.check_compaction() is False

        # 足够消息 → 触发压缩
        for i in range(15):
            tm.add_message({"role": "user", "content": f"msg{i}"})
        assert tm.check_compaction() is True
        assert tm.episodic.count() == 1
        assert tm.episodic.get_recent_episodes(1)[0].summary == "压缩摘要"

        # 压缩锁已释放
        assert tm._compacting is False
