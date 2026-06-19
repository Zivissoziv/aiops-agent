"""OpenAI 兼容接口适配器。支持 OpenAI / DeepSeek / 通义千问 / 硅基流动 / Ollama 等。"""

import json
from openai import OpenAI
from .base import BaseLLM, LLMResponse, ToolCall


def _lc_to_dict(msg) -> dict:
    t = getattr(msg, "type", "user")
    role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(t, "user")
    c = getattr(msg, "content", "")
    if isinstance(c, list):
        c = " ".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type") == "text")
    r = {"role": role, "content": c or ""}
    tcs = getattr(msg, "tool_calls", None)
    if tcs:
        r["tool_calls"] = [{"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["args"], ensure_ascii=False)}} for tc in tcs]
    tid = getattr(msg, "tool_call_id", None)
    if tid:
        r["tool_call_id"] = tid
    return r


class OpenAICompatibleLLM(BaseLLM):
    def __init__(self, api_key: str, base_url: str, model: str):
        super().__init__(api_key, base_url, model)
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _parse(self, raw) -> LLMResponse:
        ch = raw.choices[0]; msg = ch.message
        tcs = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    a = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    a = {"raw": tc.function.arguments}
                tcs.append(ToolCall(id=tc.id, name=tc.function.name, arguments=a))
        return LLMResponse(content=msg.content, tool_calls=tcs, finish_reason=ch.finish_reason or "stop")

    def invoke(self, messages, tools=None) -> LLMResponse:
        dm = [_lc_to_dict(m) if hasattr(m, "dict") else m for m in messages]
        kw = {"model": self.model}
        if tools: kw["tools"] = tools
        return self._parse(self.client.chat.completions.create(messages=dm, **kw))

    def stream(self, messages, tools=None):
        kw = {"model": self.model}
        if tools: kw["tools"] = tools
        stream = self.client.chat.completions.create(messages=messages, stream=True, **kw)
        chunks, tcc = [], {}
        for chunk in stream:
            d = chunk.choices[0].delta if chunk.choices else None
            if not d: continue
            if d.content: chunks.append(d.content); yield d.content
            if d.tool_calls:
                for tc in d.tool_calls:
                    if tc.index not in tcc:
                        tcc[tc.index] = {"id": tc.id or "", "name": tc.function.name or "", "args": tc.function.arguments or ""}
                    else:
                        e = tcc[tc.index]
                        if tc.id: e["id"] = tc.id
                        if tc.function and tc.function.name: e["name"] = tc.function.name
                        if tc.function and tc.function.arguments: e["args"] += tc.function.arguments
        if not tcc:
            return LLMResponse(content="".join(chunks))
        tcs = []
        for k in sorted(tcc):
            try:
                a = json.loads(tcc[k]["args"])
            except json.JSONDecodeError:
                a = {"raw": tcc[k]["args"]}
            tcs.append(ToolCall(id=tcc[k]["id"], name=tcc[k]["name"], arguments=a))
        return LLMResponse(content="".join(chunks) if chunks else None, tool_calls=tcs, finish_reason="tool_calls")
