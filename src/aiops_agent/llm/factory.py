"""LLM Provider 工厂 — 根据配置创建对应的 LLM 实例。"""

from .base import BaseLLM
from .openai_compatible import OpenAICompatibleLLM
from ..config import Config

_PROVIDERS = {"openai_compatible": OpenAICompatibleLLM}

def create_llm(config: Config) -> BaseLLM:
    cls = _PROVIDERS.get(config.llm_provider)
    if not cls:
        raise ValueError(f"不支持的 LLM provider: '{config.llm_provider}'，支持: {', '.join(_PROVIDERS)}")
    return cls(api_key=config.api_key, base_url=config.base_url, model=config.model)
