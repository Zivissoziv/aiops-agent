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
    memory_strategy: str = "tiered"        # "tiered" | "none"
    memory_max_messages: int = 30          # 工作记忆最大消息数
    memory_max_tokens: int = 8000          # 工作记忆 token 阈值
    memory_max_episodes: int = 50          # 情景记忆最大 episode 数
    memory_recent_episodes: int = 3        # 上下文中包含的最近 episode 数
    memory_compaction_enabled: bool = True  # 是否启用自动压缩

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
            memory_strategy=os.getenv("MEMORY_STRATEGY", "tiered"),
            memory_max_messages=int(os.getenv("MEMORY_MAX_MESSAGES", "30")),
            memory_max_tokens=int(os.getenv("MEMORY_MAX_TOKENS", "8000")),
            memory_max_episodes=int(os.getenv("MEMORY_MAX_EPISODES", "50")),
            memory_recent_episodes=int(os.getenv("MEMORY_RECENT_EPISODES", "3")),
            memory_compaction_enabled=os.getenv("MEMORY_COMPACTION_ENABLED", "true").lower() == "true",
        )

    def validate(self) -> list[str]:
        """验证配置，返回错误信息列表。"""
        errors = []
        if not self.api_key or len(self.api_key) < 10:
            errors.append("OPENAI_API_KEY 未配置或无效，请在 .env 中设置")
        if not self.base_url.startswith(("http://", "https://")):
            errors.append(f"OPENAI_BASE_URL 格式无效: {self.base_url}")
        return errors
