# d:\workspace\aiops-agent\src\aiops_agent\llm\factory.py
"""LLM Provider 工厂 — 根据配置创建对应的 LLM 实例。"""

from .base import BaseLLM
from .openai_compatible import OpenAICompatibleLLM
from ..config import Config


_SUPPORTED_PROVIDERS = {
    "openai_compatible": OpenAICompatibleLLM,
}


def create_llm(config: Config) -> BaseLLM:
    """根据配置创建 LLM 实例。

    Args:
        config: 应用配置

    Returns:
        对应 Provider 的 LLM 实例

    Raises:
        ValueError: 不支持的 LLM provider
    """
    provider_class = _SUPPORTED_PROVIDERS.get(config.llm_provider)
    if provider_class is None:
        supported = ", ".join(_SUPPORTED_PROVIDERS.keys())
        raise ValueError(
            f"不支持的 LLM provider: '{config.llm_provider}'。"
            f"当前支持: {supported}"
        )

    return provider_class(
        api_key=config.api_key,
        base_url=config.base_url,
        model=config.model,
    )
