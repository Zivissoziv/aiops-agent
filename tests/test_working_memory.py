"""工作记忆单元测试。"""

import pytest

from aiops_agent.memory.working import WorkingMemory


class TestWorkingMemory:

    def test_init(self):
        wm = WorkingMemory()
        assert wm.max_messages == 30
        assert wm.max_tokens == 8000
        assert len(wm) == 0

    def test_add_and_get(self):
        wm = WorkingMemory()
        wm.add_message({"role": "user", "content": "hello"})
        assert len(wm) == 1
        assert wm.get_messages()[0]["content"] == "hello"

    def test_window(self):
        wm = WorkingMemory(max_messages=3)
        for i in range(5):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        msgs = wm.get_messages()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg2"

    def test_system_preserved(self):
        wm = WorkingMemory(max_messages=4)
        wm.add_message({"role": "system", "content": "sys"})
        for i in range(3):
            wm.add_message({"role": "user", "content": f"u{i}"})
        msgs = wm.get_messages()
        assert msgs[0]["role"] == "system"
        assert len(msgs) == 4

    def test_get_all(self):
        wm = WorkingMemory(max_messages=3)
        for i in range(10):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        assert len(wm.get_all_messages()) == 10

    def test_reset(self):
        wm = WorkingMemory()
        wm.add_message({"role": "user", "content": "hello"})
        wm.reset()
        assert len(wm) == 0


class TestWorkingMemoryCompaction:

    def test_should_compact_thresholds(self):
        wm = WorkingMemory(max_messages=10, max_tokens=100)
        # 消息太少 → 不触发
        for i in range(4):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        assert wm.should_compact(lambda _: 50) is False

        # 数量过半且 token 超限
        for i in range(6):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        assert wm.should_compact(lambda _: 200) is True

    def test_compact_and_prune(self):
        wm = WorkingMemory(max_messages=10)
        for i in range(10):
            wm.add_message({"role": "user", "content": f"msg{i}"})

        compacted = wm.compact()
        assert len(compacted) > 0
        assert len(wm) == 10  # 压缩但不删除

        wm.prune_compacted()
        assert len(wm) < 10  # 裁掉被压缩的部分

    def test_compact_too_few_noop(self):
        wm = WorkingMemory(max_messages=5)
        wm.add_message({"role": "user", "content": "u1"})
        wm.add_message({"role": "user", "content": "u2"})
        assert wm.compact() == []

    def test_prune_without_compact(self):
        """未压缩时调用 prune 不应报错。"""
        wm = WorkingMemory()
        wm.add_message({"role": "user", "content": "hello"})
        wm.prune_compacted()
        assert len(wm) == 1

    def test_compact_skips_system(self):
        """system 消息不被压缩。"""
        wm = WorkingMemory(max_messages=10)
        wm.add_message({"role": "system", "content": "sys"})
        for i in range(10):
            wm.add_message({"role": "user", "content": f"msg{i}"})
        compacted = wm.compact()
        assert all(m["role"] != "system" for m in compacted)
