"""LLM 消息转换器单元测试。"""

import json

import pytest

from aiops_agent.llm.openai_compatible import _lc_msg_to_dict


class MockMessage:
    """模拟 LangChain BaseMessage。"""
    def __init__(self, type_, content, tool_calls=None, tool_call_id=None):
        self.type = type_
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


class TestLcMsgToDict:
    """_lc_msg_to_dict 转换测试。"""

    def test_human_message(self):
        msg = MockMessage("human", "hello")
        result = _lc_msg_to_dict(msg)
        assert result == {"role": "user", "content": "hello"}

    def test_ai_message(self):
        msg = MockMessage("ai", "I'm AI")
        result = _lc_msg_to_dict(msg)
        assert result == {"role": "assistant", "content": "I'm AI"}

    def test_system_message(self):
        msg = MockMessage("system", "be helpful")
        result = _lc_msg_to_dict(msg)
        assert result == {"role": "system", "content": "be helpful"}

    def test_tool_message(self):
        msg = MockMessage("tool", '{"ok": true}', tool_call_id="call_123")
        result = _lc_msg_to_dict(msg)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"

    def test_ai_with_tool_calls(self):
        tc = [{"id": "tc1", "name": "shell", "args": {"command": "ls"}}]
        msg = MockMessage("ai", "", tool_calls=tc)
        result = _lc_msg_to_dict(msg)
        assert result["role"] == "assistant"
        assert "tool_calls" in result
        assert result["tool_calls"][0]["function"]["name"] == "shell"

    def test_empty_content(self):
        msg = MockMessage("human", None)
        result = _lc_msg_to_dict(msg)
        assert result["content"] == ""

    def test_list_content(self):
        msg = MockMessage("human", [{"type": "text", "text": "多模态内容"}])
        result = _lc_msg_to_dict(msg)
        assert isinstance(result["content"], str)
        assert "多模态内容" in result["content"]

    def test_list_content_mixed(self):
        msg = MockMessage("human", [
            {"type": "text", "text": "描述"},
            {"type": "image_url", "image_url": {"url": "data:image/..."}},
        ])
        result = _lc_msg_to_dict(msg)
        assert result["content"] == "描述"  # 只提取 text 类型

    def test_unknown_type(self):
        msg = MockMessage("function", "result")
        result = _lc_msg_to_dict(msg)
        assert result["role"] == "user"  # 未知类型兜底为 user
