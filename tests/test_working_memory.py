"""工作记忆单元测试。"""

import pytest

from aiops_agent.memory.working import WorkingMemory


class TestWorkingMemory:
    """WorkingMemory 基础操作测试。"""

    def test_init(self):
        wm = WorkingMemory()
        assert wm.max_messages == 30
        assert wm.max_tokens == 8000
        assert wm.keep_system is True
        assert len(wm) == 0
        assert wm._last_compact_indices == []

    def test_add_and_get_messages(self):
        wm = WorkingMemory()
        wm.add_message({"role": "user", "content": "hello"})
        assert len(wm) == 1
        msgs = wm.get_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"

    def test_get_messages_window(self):
        wm = WorkingMemory(max_messages=3)
        for i in range(5):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        msgs = wm.get_messages()
        # 3 条 system 为非，只有 user 消息，取最近 3 条
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg2"

    def test_get_messages_with_system(self):
        wm = WorkingMemory(max_messages=4)
        wm.add_message({"role": "system", "content": "sys1"})
        wm.add_message({"role": "system", "content": "sys2"})
        wm.add_message({"role": "user", "content": "u1"})
        wm.add_message({"role": "user", "content": "u2"})
        wm.add_message({"role": "user", "content": "u3"})
        msgs = wm.get_messages()
        # 保留 2 条 system + 2 条 user
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "system", "user", "user"]

    def test_get_messages_keep_system_false(self):
        wm = WorkingMemory(max_messages=3, keep_system=False)
        wm.add_message({"role": "system", "content": "sys"})
        wm.add_message({"role": "user", "content": "u1"})
        wm.add_message({"role": "user", "content": "u2"})
        wm.add_message({"role": "user", "content": "u3"})
        msgs = wm.get_messages()
        assert len(msgs) == 3
        assert all(m["role"] != "system" for m in msgs)

    def test_get_all_messages(self):
        wm = WorkingMemory(max_messages=3)
        for i in range(10):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        # get_all 返回全部，不受窗口限制
        assert len(wm.get_all_messages()) == 10

    def test_reset(self):
        wm = WorkingMemory()
        wm.add_message({"role": "user", "content": "hello"})
        wm.reset()
        assert len(wm) == 0
        assert wm.get_messages() == []


class TestWorkingMemoryCompaction:
    """WorkingMemory 压缩逻辑测试。"""

    def test_should_compact_few_messages(self):
        wm = WorkingMemory(max_messages=10, max_tokens=100)
        for i in range(4):  # 未过半（< 5）
            wm.add_message({"role": "user", "content": f"msg{i}"})
        assert wm.should_compact(lambda _: 50) is False

    def test_should_compact_by_count(self):
        wm = WorkingMemory(max_messages=10)
        for i in range(10):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        assert wm.should_compact(lambda _: 50) is True

    def test_should_compact_by_tokens(self):
        wm = WorkingMemory(max_messages=20, max_tokens=100)
        for i in range(10):  # 过半
            wm.add_message({"role": "user", "content": f"msg{i}"})
        assert wm.should_compact(lambda _: 200) is True  # token 超限
        assert wm.should_compact(lambda _: 50) is False  # token 未超限

    def test_compact_basic(self):
        wm = WorkingMemory(max_messages=10)
        for i in range(10):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        compacted = wm.compact()
        # 75% of 10 = 7.5 → max(2, 7) = 7
        assert len(compacted) == 7
        assert len(wm) == 10  # 压缩后不自动删除

    def test_compact_too_few(self):
        wm = WorkingMemory(max_messages=5)
        wm.add_message({"role": "user", "content": "u1"})
        wm.add_message({"role": "user", "content": "u2"})
        compacted = wm.compact()
        assert compacted == []  # <= 2 条不压缩

    def test_prune_compacted(self):
        wm = WorkingMemory(max_messages=10)
        for i in range(10):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        wm.compact()
        assert len(wm) == 10
        wm.prune_compacted()
        assert len(wm) == 3  # 保留 3 条

    def test_prune_without_compact(self):
        """未压缩时调用 prune_compacted 不应报错。"""
        wm = WorkingMemory()
        wm.add_message({"role": "user", "content": "hello"})
        wm.prune_compacted()  # 不应抛异常
        assert len(wm) == 1

    def test_compact_with_system_messages(self):
        wm = WorkingMemory(max_messages=10)
        wm.add_message({"role": "system", "content": "sys"})
        for i in range(10):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        compacted = wm.compact()
        # 10 条非 system，压缩 75% = 7 条
        assert len(compacted) == 7
        # system 消息不应被压缩
        assert all(m["role"] != "system" for m in compacted)
