"""pytest 共享配置和夹具。"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """提供临时目录用于测试文件持久化。"""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_json_file(temp_dir):
    """创建一个临时 JSON 文件并返回路径。"""
    path = temp_dir / "test_data.json"
    return path


@pytest.fixture
def sample_messages():
    """提供一组示例消息用于测试。"""
    return [
        {"role": "system", "content": "你是运维助手"},
        {"role": "user", "content": "查看服务器状态"},
        {"role": "assistant", "content": "正在执行...", "tool_calls": [{"id": "1", "name": "shell", "args": {"command": "ls"}}]},
        {"role": "tool", "content": '{"success": true, "output": "ok", "error": ""}', "tool_call_id": "1"},
        {"role": "assistant", "content": "执行完毕"},
    ]
