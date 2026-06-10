# d:\workspace\aiops-agent\src\aiops_agent\core\agent.py
"""Agent 核心引擎。

实现"思考-行动-观察"循环:
1. 接收用户消息
2. 调用 LLM（带工具定义）
3. LLM 返回文本 → 输出
4. LLM 返回工具调用 → 执行工具 → 结果反馈给 LLM → 回到 2
5. 达到最大轮次或 LLM 返回纯文本 → 结束

支持可选的 Memory 模块管理对话历史。
"""

import json
from dataclasses import dataclass, field
from typing import Generator

from ..config import Config
from ..llm import BaseLLM
from ..memory import Memory
from ..tools import ToolRegistry


@dataclass
class AgentEvent:
    """Agent 执行过程中发出的事件，供 CLI/UI 展示。"""
    type: str  # "text" | "tool_start" | "tool_result" | "error" | "done"
    content: str = ""
    data: dict = field(default_factory=dict)


class Agent:
    """Agent 核心引擎。"""

    def __init__(
        self,
        config: Config,
        llm: BaseLLM,
        tool_registry: ToolRegistry,
        memory: Memory | None = None,
    ):
        self.config = config
        self.llm = llm
        self.tool_registry = tool_registry
        self.memory = memory

    def _check_compaction(self) -> None:
        """检查并执行记忆压缩（如果 memory 支持）。"""
        if self.memory is not None and hasattr(self.memory, 'check_compaction'):
            self.memory.check_compaction()

    def _build_messages(
        self,
        history: list[dict],
        user_input: str,
    ) -> list[dict]:
        """构建完整的消息列表（无 memory 的向后兼容路径）。"""
        return [
            {"role": "system", "content": self.config.system_prompt},
            *history,
            {"role": "user", "content": user_input},
        ]

    def _build_messages_from_memory(self) -> list[dict]:
        """从 memory 构建消息列表。"""
        return [
            {"role": "system", "content": self.config.system_prompt},
            *self.memory.get_messages(),
        ]

    def run(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> Generator[AgentEvent, None, list[dict]]:
        """运行 Agent 主循环。

        Args:
            user_input: 用户输入
            history: 历史消息列表（仅在无 memory 时使用）

        Yields:
            供 UI 展示的事件

        Returns:
            更新后的消息历史
        """
        # 选择消息构建路径
        if self.memory is not None:
            self.memory.add_message({"role": "user", "content": user_input})
            messages = self._build_messages_from_memory()
        else:
            history = list(history) if history else []
            messages = self._build_messages(history, user_input)

        tool_defs = self.tool_registry.get_openai_tool_defs()
        tool_defs = tool_defs or None

        for _round in range(self.config.max_tool_rounds):
            # 调用 LLM
            response = self.llm.invoke(messages, tools=tool_defs)

            # 将 LLM 回复追加到消息列表
            assistant_msg: dict = {"role": "assistant"}
            if response.content:
                assistant_msg["content"] = response.content
            if response.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments,
                        },
                    }
                    for tc in response.tool_calls
                ]
            messages.append(assistant_msg)

            # 记忆路径: 同步添加消息到 memory
            if self.memory is not None:
                self.memory.add_message(assistant_msg)

            # LLM 回复循环 → 检查是否需要压缩
            if response.content:
                yield AgentEvent(
                    type="text",
                    content=response.content,
                )
            self._check_compaction()

            # 如果没有工具调用，结束本轮
            if not response.tool_calls:
                break

            # 逐个执行工具调用
            for tc in response.tool_calls:
                yield AgentEvent(
                    type="tool_start",
                    content=f"🔧 正在使用工具: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})",
                    data={"tool_name": tc.name, "arguments": tc.arguments},
                )

                result = self.tool_registry.execute_tool(tc.name, tc.arguments)

                # 工具结果追加到消息列表
                tool_result_msg = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(
                        {
                            "success": result.success,
                            "output": result.output,
                            "error": result.error,
                        },
                        ensure_ascii=False,
                    ),
                }
                messages.append(tool_result_msg)

                # 记忆路径: 同步添加工具结果到 memory
                if self.memory is not None:
                    self.memory.add_message(tool_result_msg)

                display_output = result.error if not result.success else result.output
                yield AgentEvent(
                    type="tool_result",
                    content=display_output,
                    data={
                        "tool_name": tc.name,
                        "success": result.success,
                        "execution_time": result.execution_time,
                    },
                )

            # 工具调用循环后 → 检查是否需要压缩
            self._check_compaction()
        else:
            yield AgentEvent(
                type="error",
                content=f"⚠️ 已达到最大工具调用轮次（{self.config.max_tool_rounds}），请尝试拆分任务。",
            )

        yield AgentEvent(type="done", content="")

        # 返回更新后的历史
        if self.memory is not None:
            return self.memory.get_messages()
        return messages[1:]
