"""配置管理 — 从 .env 文件加载配置。"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

def _find_project_root() -> Path:
    c = Path(__file__).resolve().parent
    for p in [c, *c.parents]:
        if (p / ".env").exists(): return p
    return c.parent.parent

@dataclass
class Config:
    llm_provider: str = "openai_compatible"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    system_prompt: str = "你是一个 AIOps 运维助手，擅长通过工具执行运维任务。"
    max_tool_rounds: int = 10
    memory_strategy: str = "tiered"
    memory_max_messages: int = 30
    memory_max_tokens: int = 8000
    memory_max_episodes: int = 50
    memory_recent_episodes: int = 3
    memory_compaction_enabled: bool = True

    @classmethod
    def from_env(cls, env_path: Path | None = None) -> "Config":
        load_dotenv(env_path or _find_project_root() / ".env")
        return cls(llm_provider=os.getenv("LLM_PROVIDER", "openai_compatible"), api_key=os.getenv("OPENAI_API_KEY", ""), base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"), model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), system_prompt=os.getenv("SYSTEM_PROMPT", "你是一个 AIOps 运维助手，擅长通过工具执行运维任务。"), max_tool_rounds=int(os.getenv("MAX_TOOL_ROUNDS", "10")), memory_strategy=os.getenv("MEMORY_STRATEGY", "tiered"), memory_max_messages=int(os.getenv("MEMORY_MAX_MESSAGES", "30")), memory_max_tokens=int(os.getenv("MEMORY_MAX_TOKENS", "8000")), memory_max_episodes=int(os.getenv("MEMORY_MAX_EPISODES", "50")), memory_recent_episodes=int(os.getenv("MEMORY_RECENT_EPISODES", "3")), memory_compaction_enabled=os.getenv("MEMORY_COMPACTION_ENABLED", "true").lower() == "true")

    def validate(self) -> list[str]:
        e = []
        if not self.api_key or len(self.api_key) < 10: e.append("OPENAI_API_KEY 未配置或无效")
        if not self.base_url.startswith(("http://", "https://")): e.append(f"BASE_URL 格式无效: {self.base_url}")
        return e
