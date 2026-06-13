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
        print("[x] 请在项目根目录的 .env 文件中配置 OPENAI_API_KEY")
        print("   参考 .env.example 文件")
        exit(1)

    # Embedding 配置（可选，用于 RAG 示例）
    embedding_api_key = os.getenv("EMBEDDING_API_KEY", api_key)
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", base_url)
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    return {
        "api_key": api_key,
        "base_url": base_url,
        "model": model,
        "embedding_api_key": embedding_api_key,
        "embedding_base_url": embedding_base_url,
        "embedding_model": embedding_model,
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


def create_embeddings(
    client: OpenAI,
    texts: list[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 10,
) -> list[list[float]]:
    """批量将文本转为向量（Embedding）。

    用于 RAG 场景：将知识库文档转为向量后存入向量数据库，
    检索时再将问题转为向量，通过相似度匹配找到相关文档。

    Args:
        client: OpenAI 客户端
        texts: 文本列表（每段文本会被转为一个向量）
        model: Embedding 模型名，默认 text-embedding-3-small
        batch_size: 每批处理数量（部分 API 有上限，如 DashScope 限 10）

    Returns:
        向量列表，每个向量是一个 float 列表（维度取决于模型）
    """
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(model=model, input=batch)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings
