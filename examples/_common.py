"""
examples/_common.py — 教学示例共用模块

提供所有教学示例共用的基础功能:
  - 加载 .env 配置
  - 创建 OpenAI 客户端
  - 估算 token 数
  - 调用 LLM

使用方式:
  from _common import load_config, create_client, call_llm, estimate_tokens

设计意图:
  每个教学示例只需关注自己要讲的新概念，
  不需要重复写配置加载等基础代码。
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


def load_config() -> dict:
    """加载 .env 配置，返回包含 api_key / base_url / model 的字典。

    如果 API Key 未配置，直接退出并提示用户。
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key or len(api_key) < 10:
        print("❌ 请在项目根目录的 .env 文件中配置 OPENAI_API_KEY")
        print("   参考 .env.example 文件")
        exit(1)

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
    }


def create_client(config: dict) -> OpenAI:
    """根据配置创建 OpenAI 客户端。

    支持所有 OpenAI 兼容接口（OpenAI / DeepSeek / 通义千问 等）。
    """
    return OpenAI(api_key=config["api_key"], base_url=config["base_url"])


def call_llm(client: OpenAI, model: str, messages: list[dict]) -> str:
    """调用 LLM 并返回回复内容（非流式，简单教学用）。

    对于需要流式输出的场景（如 01_simple_chat），请直接在示例中实现。
    """
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def estimate_tokens(messages: list[dict]) -> int:
    """近似估算 token 数（字符数 // 4，非精确）。

    Args:
        messages: OpenAI 格式的消息列表

    Returns:
        近似的 token 数量
    """
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += len(block["text"])
    return total // 4
