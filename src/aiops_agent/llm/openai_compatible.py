# d:\workspace\aiops-agent\src\aiops_agent\llm\openai_compatible.py
"""OpenAI 兼容接口适配器。

支持: OpenAI / DeepSeek / 通义千问 / 硅基流动 / Ollama 等
所有使用 OpenAI API 格式的 LLM 提供商。
"""

import json
from typing import Generator

from openai import OpenAI

from .base import BaseLLM, LLMResponse, ToolCall


class OpenAICompatibleLLM(BaseLLM):
    """OpenAI 兼容接口的 LLM 适配器。"""

    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__(api_key, base_url, model)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _build_kwargs(self, tools: list[dict] | None) -> dict:
        kwargs = {"model": self.model}
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _parse_response(self, raw) -> LLMResponse:
        """解析 OpenAI SDK 的原始响应为统一格式。"""
        choice = raw.choices[0]
        message = choice.message

        content = message.content

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                    )
                )

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason or "stop",
        )

    def invoke(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        raw = self.client.chat.completions.create(
            messages=messages,
            **self._build_kwargs(tools),
        )
        return self._parse_response(raw)

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Generator[str, None, LLMResponse]:
        stream = self.client.chat.completions.create(
            messages=messages,
            stream=True,
            **self._build_kwargs(tools),
        )

        content_chunks: list[str] = []
        tool_call_chunks: dict[int, dict[str, str]] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                content_chunks.append(delta.content)
                yield delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": tc_delta.id or "",
                            "function_name": tc_delta.function.name or "",
                            "arguments": tc_delta.function.arguments or "",
                        }
                    else:
                        existing = tool_call_chunks[idx]
                        if tc_delta.id:
                            existing["id"] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            existing["function_name"] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            existing["arguments"] += tc_delta.function.arguments

        if not tool_call_chunks:
            return LLMResponse(content="".join(content_chunks))

        tool_calls = []
        for idx in sorted(tool_call_chunks.keys()):
            chunk_data = tool_call_chunks[idx]
            try:
                args = json.loads(chunk_data["arguments"])
            except json.JSONDecodeError:
                args = {"raw": chunk_data["arguments"]}
            tool_calls.append(
                ToolCall(
                    id=chunk_data["id"],
                    name=chunk_data["function_name"],
                    arguments=args,
                )
            )

        return LLMResponse(
            content="".join(content_chunks) if content_chunks else None,
            tool_calls=tool_calls,
            finish_reason="tool_calls",
        )
