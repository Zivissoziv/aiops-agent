"""记忆管理抽象基类。"""
from abc import ABC, abstractmethod

class Memory(ABC):
    @abstractmethod
    def add_message(self, message: dict) -> None: ...
    @abstractmethod
    def get_messages(self) -> list[dict]: ...
    @abstractmethod
    def count(self, estimation_fn) -> int: ...
    @abstractmethod
    def reset(self) -> None: ...
    def __len__(self) -> int:
        return len(self.get_messages())
