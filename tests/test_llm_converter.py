"""LLM 消息转换器单元测试。"""

import pytest

from aiops_agent.llm.openai_compatible import _lc_to_dict


class MockMessage:
    """模拟 LangChain BaseMessage。"""
    def __init__(self, type_, content, tool_calls=None, tool_call_id=None):
        self.type = type_
        self.content = content
        self.tool_calls = tool_calls
        self.tool_call_id = tool_call_id


class TestLcMsgToDict:
    """_lc_to_dict 转换测试（合并同类项，保留核心场景）。"""

    def test_role_mapping(self):
        """角色映射: human→user, ai→assistant, system→system, tool→tool, 未知→user。"""
        cases = [
            (MockMessage("human", "hi"), {"role": "user", "content": "hi"}),
            (MockMessage("ai", "ok"), {"role": "assistant", "content": "ok"}),
            (MockMessage("system", "sys"), {"role": "system", "content": "sys"}),
            (MockMessage("tool", "res", tool_call_id="c1"),
             {"role": "tool", "content": "res", "tool_call_id": "c1"}),
            (MockMessage("function", "x"), {"role": "user", "content": "x"}),
        ]
        for msg, expected in cases:
            assert _lc_to_dict(msg) == expected

    def test_ai_with_tool_calls(self):
        """AI 消息包含 tool_calls 时正确转换。"""
        tc = [{"id": "tc1", "name": "shell", "args": {"command": "ls"}}]
        result = _lc_to_dict(MockMessage("ai", "", tool_calls=tc))
        assert result["role"] == "assistant"
        assert result["tool_calls"][0]["function"]["name"] == "shell"
        assert result["tool_calls"][0]["function"]["arguments"] == '{"command": "ls"}'

    def test_content_handling(self):
        """各类 content 格式：None→空字串、列表→拼接文本。"""
        assert _lc_to_dict(MockMessage("human", None))["content"] == ""
        assert _lc_to_dict(MockMessage("human", [{"type": "text", "text": "多模态"}]))["content"] == "多模态"

    def test_list_content_ignores_non_text(self):
        """content 列表中只取 text 类型的字段。"""
        result = _lc_to_dict(MockMessage("human", [
            {"type": "text", "text": "描述"},
            {"type": "image_url", "image_url": {"url": "data:img"}},
        ]))
        assert result["content"] == "描述"
