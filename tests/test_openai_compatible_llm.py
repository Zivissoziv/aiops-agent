"""OpenAICompatibleLLM 单元测试（mock 模式，不真实调用 API）。"""

from unittest.mock import MagicMock, patch

import pytest

from aiops_agent.llm.openai_compatible import OpenAICompatibleLLM


@pytest.fixture
def llm():
    """创建一个 mock 模式的 LLM 实例。"""
    return OpenAICompatibleLLM(
        api_key="sk-test",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )


class TestBuildKwargs:
    """_build_kwargs 测试。"""

    def test_without_tools(self, llm):
        kwargs = llm._build_kwargs(None)
        assert kwargs == {"model": "gpt-4o-mini"}

    def test_with_tools(self, llm):
        tools = [{"type": "function", "function": {"name": "test"}}]
        kwargs = llm._build_kwargs(tools)
        assert "tools" in kwargs
        assert kwargs["tools"] == tools


class TestParseResponse:
    """_parse_response 测试。"""

    def test_text_response(self, llm):
        # 模拟 OpenAI SDK 响应
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello!"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = llm._parse_response(mock_response)
        assert result.content == "Hello!"
        assert result.tool_calls == []
        assert result.finish_reason == "stop"

    def test_tool_call_response(self, llm):
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "shell"
        mock_tc.function.arguments = '{"command": "ls"}'

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = llm._parse_response(mock_response)
        assert result.content is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "shell"
        assert result.tool_calls[0].arguments == {"command": "ls"}

    def test_invalid_tool_arguments(self, llm):
        mock_tc = MagicMock()
        mock_tc.id = "call_1"
        mock_tc.function.name = "test"
        mock_tc.function.arguments = "not-json"

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_choice.message.tool_calls = [mock_tc]
        mock_choice.finish_reason = "tool_calls"

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        result = llm._parse_response(mock_response)
        assert result.tool_calls[0].arguments == {"raw": "not-json"}
