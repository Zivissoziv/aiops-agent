# d:\workspace\aiops-agent\src\aiops_agent\memory\base.py
"""记忆管理抽象基类 — 定义所有记忆策略必须实现的接口。"""

from abc import ABC, abstractmethod
from collections.abc import Callable


class Memory(ABC):
    """记忆管理抽象基类。

    所有记忆策略必须继承此类并实现以下方法。

    消息格式遵循 OpenAI 规范:
        {"role": "user"/"assistant"/"tool"/"system", "content": "..."}
    """

    @abstractmethod
    def add_message(self, message: dict) -> None:
        """添加一条消息到记忆。

        Args:
            message: OpenAI 格式的消息字典
        """
        ...

    @abstractmethod
    def get_messages(self) -> list[dict]:
        """获取发送给 LLM 的消息视图。

        实现可以在此过滤、摘要或重排消息。
        注意: 返回的列表不应包含 system prompt ——
        system prompt 由 Agent 管理。
        """
        ...

    @abstractmethod
    def count(self, estimation_fn: Callable[[list[dict]], int]) -> int:
        """估算当前记忆的 token 数量。

        Args:
            estimation_fn: 接受消息列表并返回近似 token 数的可调用对象
                          （例如 llm.count_tokens）

        Returns:
            近似 token 数量
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """清空所有存储的消息。"""
        ...

    def __len__(self) -> int:
        return len(self.get_messages())
