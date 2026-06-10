# d:\workspace\aiops-agent\src\aiops_agent\config.py
"""配置管理 — 从 .env 文件加载配置，提供类型安全的访问。"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv


def _find_project_root() -> Path:
    """从当前文件位置向上查找项目根目录（包含 .env 的目录）。"""
    current = Path(__file__).resolve().parent  # src/aiops_agent/
    for parent in [current, *current.parents]:
        if (parent / ".env").exists():
            return parent
    # 兜底：取当前文件所在目录的父目录
    return current.parent.parent


@dataclass
class Config:
    """应用配置。"""

    # LLM 配置
    llm_provider: str = "openai_compatible"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"

    # Agent 配置
    system_prompt: str = "你是一个 AIOps 运维助手，擅长通过工具执行运维任务。"
    max_tool_rounds: int = 10

    # Memory 配置
    memory_strategy: str = "window"       # "window" | "summarizing" | "none"
    memory_max_messages: int = 20         # 滑窗策略的最大消息数
    memory_max_tokens: int = 4000         # 摘要策略的触发阈值

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Config":
        """从 .env 文件加载配置。"""
        if env_path is None:
            env_path = _find_project_root() / ".env"

        load_dotenv(env_path)

        return cls(
            llm_provider=os.getenv("LLM_PROVIDER", "openai_compatible"),
            api_key=os.getenv("OPENAI_API_KEY", ""),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            system_prompt=os.getenv(
                "SYSTEM_PROMPT",
                "你是一个 AIOps 运维助手，擅长通过工具执行运维任务。",
            ),
            max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "10")),
            memory_strategy=os.getenv("MEMORY_STRATEGY", "window"),
            memory_max_messages=int(os.getenv("MEMORY_MAX_MESSAGES", "20")),
            memory_max_tokens=int(os.getenv("MEMORY_MAX_TOKENS", "4000")),
        )

    def validate(self) -> list[str]:
        """验证配置，返回错误信息列表。"""
        errors = []
        if not self.api_key or len(self.api_key) < 10:
            errors.append("OPENAI_API_KEY 未配置或无效，请在 .env 中设置")
        if not self.base_url.startswith(("http://", "https://")):
            errors.append(f"OPENAI_BASE_URL 格式无效: {self.base_url}")
        return errors
