"""TODO 管理工具单元测试。"""

import json
import pytest

from aiops_agent.tools.todo_tool import update_todo


class TestUpdateTodoTool:

    def test_completed(self):
        result = json.loads(update_todo.invoke({"todo_id": "todo-1", "status": "completed"}))
        assert result["success"] is True
        assert "✅" in result["output"]

    def test_in_progress(self):
        result = json.loads(update_todo.invoke({"todo_id": "todo-1", "status": "in_progress"}))
        assert result["success"] is True

    def test_blocked(self):
        result = json.loads(update_todo.invoke({"todo_id": "todo-1", "status": "blocked"}))
        assert result["success"] is True

    def test_invalid_status(self):
        result = json.loads(update_todo.invoke({"todo_id": "todo-1", "status": "invalid"}))
        assert result["success"] is False

    def test_always_returns_valid_json(self):
        """无论输入如何，返回值始终是合法 JSON。"""
        for args in [{"todo_id": "x", "status": "completed"},
                     {"todo_id": "x", "status": "bad"}]:
            result = json.loads(update_todo.invoke(args))
            assert "success" in result
            assert "output" in result
