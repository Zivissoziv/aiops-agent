"""核心记忆单元测试。"""

import json

import pytest

from aiops_agent.memory.core import CoreMemory


class TestCoreMemory:

    def test_init(self):
        cm = CoreMemory()
        assert cm.count() == 0

    def test_add_fact(self):
        cm = CoreMemory()
        cm.add_fact("服务器 IP 是 10.0.0.1")
        assert cm.count() == 1
        assert "10.0.0.1" in cm.get_all_facts()[0]

    def test_add_facts(self):
        cm = CoreMemory()
        cm.add_facts(["fact1", "fact2"])
        assert cm.count() == 2

    def test_remove_fact(self):
        cm = CoreMemory()
        cm.add_fact("fact1")
        cm.add_fact("fact2")
        assert cm.remove_fact("fact1") is True
        assert cm.count() == 1
        assert cm.remove_fact("不存在") is False

    def test_clear(self):
        cm = CoreMemory()
        cm.add_fact("fact1")
        cm.clear()
        assert cm.count() == 0

    def test_format_for_context(self):
        cm = CoreMemory()
        assert cm.format_for_context() == []

        cm.add_fact("服务器 A")
        cm.add_fact("服务器 B")
        result = cm.format_for_context()
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert "服务器 A" in result[0]["content"]


class TestCoreMemoryPersistence:

    def test_persist_and_load(self, temp_json_file):
        cm = CoreMemory(persist_path=temp_json_file)
        cm.add_fact("持久化事实")
        cm.add_fact("另一个事实")

        cm2 = CoreMemory(persist_path=temp_json_file)
        assert cm2.count() == 2
        assert "持久化事实" in cm2.get_all_facts()

    def test_persist_empty(self, temp_json_file):
        cm = CoreMemory(persist_path=temp_json_file)
        cm.add_fact("test")
        cm.clear()
        cm2 = CoreMemory(persist_path=temp_json_file)
        assert cm2.count() == 0

    def test_load_corrupted(self, temp_json_file):
        temp_json_file.write_text("{corrupted json", encoding="utf-8")
        cm = CoreMemory(persist_path=temp_json_file)
        assert cm.count() == 0

    def test_load_non_list(self, temp_json_file):
        temp_json_file.write_text('{"not": "a list"}', encoding="utf-8")
        cm = CoreMemory(persist_path=temp_json_file)
        assert cm.count() == 0
