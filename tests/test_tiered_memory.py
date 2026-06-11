"""三层记忆编排器单元测试。"""

from unittest.mock import MagicMock

import pytest

from aiops_agent.memory.tiered import TieredMemory


class TestTieredMemory:
    """TieredMemory 基础功能测试。"""

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
        assert tm.core.count() == 0

    def test_bool_true(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        assert bool(tm) is True

    def test_add_and_get_messages(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hello"})
        msgs = tm.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"

    def test_get_messages_with_core(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.remember("核心事实")
        tm.add_message({"role": "user", "content": "hello"})
        msgs = tm.get_messages()
        # core + working = 2 条
        assert len(msgs) >= 2
        assert msgs[0]["role"] == "system"
        assert "核心事实" in msgs[0]["content"]

    def test_count(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hello"})
        tm.add_message({"role": "assistant", "content": "world"})
        count = tm.count(mock_llm.count_tokens)
        assert count == 100  # mock 返回值

    def test_reset(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hello"})
        tm.remember("事实")
        tm.reset()
        assert tm.working.get_messages() == []
        assert tm.core.count() == 0

    def test_core_memory_management(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.remember("事实1")
        tm.remember("事实2")
        assert tm.get_core_facts() == ["事实1", "事实2"]
        assert tm.forget("事实1") is True
        assert tm.get_core_facts() == ["事实2"]
        assert tm.forget("不存在") is False

    def test_get_stats(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hello"})
        tm.remember("事实")
        stats = tm.get_stats()
        assert stats["working_messages"] == 1
        assert stats["core_facts"] == 1
        assert stats["episodic_count"] == 0
        assert stats["compaction_enabled"] is True


class TestTieredMemoryCompaction:
    """TieredMemory 压缩流程测试。"""

    @pytest.fixture
    def mock_llm(self):
        llm = MagicMock()
        llm.count_tokens.return_value = 9999  # 触发 token 超限
        response = MagicMock()
        response.content = '{"summary": "压缩摘要", "key_facts": ["f1"], "decisions": ["d1"], "unresolved": []}'
        llm.invoke.return_value = response
        return llm

    def test_check_compaction_disabled(self, mock_llm):
        tm = TieredMemory(llm=mock_llm, compaction_enabled=False)
        assert tm.check_compaction() is False

    def test_check_compaction_no_need(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        tm.add_message({"role": "user", "content": "hi"})  # 太少，不需要压缩
        assert tm.check_compaction() is False

    def test_check_compaction_success(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        for i in range(15):
            tm.add_message({"role": "user", "content": f"msg{i}"})
        assert tm.check_compaction() is True
        assert tm.episodic.count() == 1
        assert tm.episodic.get_recent_episodes(1)[0].summary == "压缩摘要"

    def test_check_compaction_prevent_reentry(self, mock_llm):
        tm = TieredMemory(llm=mock_llm)
        assert tm._compacting is False
        # 第一次调用后，_compacting 应恢复为 False
        for i in range(15):
            tm.add_message({"role": "user", "content": f"msg{i}"})
        tm.check_compaction()
        assert tm._compacting is False
