"""CLI 辅助函数单元测试（不启动交互循环）。"""

import pytest

from aiops_agent.cli import print_custom_event, print_graph_update, _seen_agents_in_session


class FakeWriter:
    """模拟 StreamWriter 用于测试。"""
    def __init__(self):
        self.events = []

    def __call__(self, event):
        self.events.append(event)


class TestPrintCustomEvent:
    """print_custom_event 渲染测试。"""

    def test_agent_start_new(self, capsys):
        _seen_agents_in_session.clear()
        print_custom_event({"type": "agent_start", "agent": "planner"})
        captured = capsys.readouterr()
        assert "[planner]" in captured.out
        assert "planner" in _seen_agents_in_session

    def test_agent_start_duplicate(self, capsys):
        _seen_agents_in_session.clear()
        _seen_agents_in_session.add("planner")
        print_custom_event({"type": "agent_start", "agent": "planner"})
        captured = capsys.readouterr()
        assert captured.out == ""  # 已见过，不输出

    def test_tool_start(self, capsys):
        print_custom_event({"type": "tool_start", "tool": "shell", "args": {}, "agent": "worker"})
        captured = capsys.readouterr()
        assert "shell" in captured.out

    def test_tool_result_with_output(self, capsys):
        print_custom_event({"type": "tool_result", "tool": "shell", "success": True, "output": "done", "error": "", "agent": "worker"})
        captured = capsys.readouterr()
        assert "done" in captured.out

    def test_tool_result_with_error(self, capsys):
        print_custom_event({"type": "tool_result", "tool": "shell", "success": False, "output": "", "error": "失败", "agent": "worker"})
        captured = capsys.readouterr()
        assert "失败" in captured.out
