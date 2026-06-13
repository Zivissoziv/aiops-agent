# d:\workspace\aiops-agent\src\aiops_agent\tools\knowledge_tool.py
"""知识库检索工具 — 从运维知识库中检索公司内部规范。

自动管理 Chroma 向量数据库的生命周期:
  - 首次启动时自动建库（读取 knowledge_base/*.md）
  - 检测到文件变更时自动重建
  - 无变更时跳过，零开销
"""

import hashlib
import json
import os
from typing import Any

from langchain_core.tools import tool
from openai import OpenAI

from ..config import Config, _find_project_root

# ── 路径常量 ──

PROJECT_ROOT = _find_project_root()  # 复用 config.py 的项目根查找逻辑
KB_DIR = PROJECT_ROOT / "knowledge_base"
AIOP_DATA_DIR = PROJECT_ROOT / ".aiops_data"
CHROMA_DIR = str(AIOP_DATA_DIR / "chroma_db")
TS_FILE = AIOP_DATA_DIR / "kb_timestamp.json"

# ── Embedding 配置（从 .env 读取） ──

_config: Config | None = None
_embedding_client: OpenAI | None = None
_embedding_model: str = "text-embedding-3-small"

# ── Chroma 延迟初始化 ──

_collection = None
_chroma_client = None

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


def _get_config() -> Config:
    global _config
    if _config is None:
        from ..config import Config
        _config = Config.from_env()
    return _config


def _get_embedding_client() -> OpenAI:
    global _embedding_client, _embedding_model
    if _embedding_client is None:
        # _get_config() 会调 Config.from_env() 加载 .env，后续 os.getenv 才能读到
        cfg = _get_config()
        _embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        api_key = os.getenv("EMBEDDING_API_KEY") or cfg.api_key
        base_url = os.getenv("EMBEDDING_BASE_URL") or cfg.base_url
        _embedding_client = OpenAI(api_key=api_key, base_url=base_url)
    return _embedding_client


def _create_embeddings(texts: list[str]) -> list[list[float]]:
    """批量 Embedding，自动分批（部分 API 限 10 条/次）。"""
    client = _get_embedding_client()
    all_embeddings: list[list[float]] = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=_embedding_model, input=batch)
        all_embeddings.extend(item.embedding for item in resp.data)
    return all_embeddings


# ── 时间戳管理 ──


def _get_kb_timestamp() -> str:
    """计算 knowledge_base 目录的时间戳（基于文件数量和最后修改时间）。"""
    if not KB_DIR.is_dir():
        return ""
    files = sorted(KB_DIR.glob("*.md"))
    if not files:
        return ""
    # 用文件名 + mtime 计算 hash
    raw = "\n".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in files)
    return hashlib.md5(raw.encode()).hexdigest()


def _load_saved_timestamp() -> str:
    if TS_FILE.exists():
        try:
            data = json.loads(TS_FILE.read_text(encoding="utf-8"))
            return data.get("timestamp", "")
        except Exception:
            return ""
    return ""


def _save_timestamp(ts: str):
    AIOP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    TS_FILE.write_text(
        json.dumps({"timestamp": ts}, ensure_ascii=False),
        encoding="utf-8",
    )


# ── 建库 ──


def _load_kb_docs() -> list[dict[str, Any]]:
    """加载 knowledge_base/*.md。"""
    docs = []
    for md_file in sorted(KB_DIR.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        title = md_file.name.replace(".md", "")
        for line in text.splitlines():
            if line.startswith("# "):
                title = line.lstrip("# ").strip()
                break
        docs.append({"source": md_file.name, "title": title, "content": text})
    return docs


def _chunk_doc(doc: dict[str, Any], chunk_size: int = 200, overlap: int = 20) -> list[dict[str, Any]]:
    text = doc["content"]
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += ("\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)

    result = []
    prev_tail = ""
    for i, ct in enumerate(chunks):
        if prev_tail and overlap > 0:
            ct = prev_tail + "\n" + ct
        result.append({"id": f"{doc['source']}_chunk_{i:03d}", "source": doc["source"], "title": doc["title"], "content": ct})
        prev_tail = ct[-overlap:] if len(ct) > overlap else ct
    return result


def _get_chroma_client():
    """获取 Chroma 客户端（延迟初始化，单例）。"""
    global _chroma_client
    if _chroma_client is None and CHROMA_AVAILABLE:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _ensure_kb_built():
    """获取 Chroma 客户端（延迟初始化，单例）。"""
    global _chroma_client
    if _chroma_client is None and CHROMA_AVAILABLE:
        import chromadb
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client


def _ensure_kb_built():
    """检查知识库是否需要构建/重建，自动执行。"""
    global _collection
    client = _get_chroma_client()
    if client is None:
        return

    current_ts = _get_kb_timestamp()
    if not current_ts:
        return  # 没有知识库文档

    saved_ts = _load_saved_timestamp()
    if current_ts == saved_ts:
        return  # 未变化，跳过

    # 需要建库，清除旧的集合缓存
    _collection = None

    # 重建集合：删掉旧的，重新获取
    try:
        client.delete_collection("ops_kb")
    except Exception:
        pass
    # delete 成功后 create 可能因异步竞态失败，用 get_collection 兜底
    try:
        collection = client.create_collection(
            "ops_kb", metadata={"description": "AIOps 运维知识库"},
        )
    except chromadb.errors.InternalError:
        collection = client.get_collection("ops_kb")

    docs = _load_kb_docs()
    if not docs:
        return

    all_chunks = []
    for d in docs:
        all_chunks.extend(_chunk_doc(d))

    texts = [c["content"] for c in all_chunks]
    embeddings = _create_embeddings(texts)

    collection.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=embeddings,
        documents=[c["content"] for c in all_chunks],
        metadatas=[{"source": c["source"], "title": c["title"]} for c in all_chunks],
    )

    _save_timestamp(current_ts)


def _get_collection():
    """获取 Chroma 集合（延迟加载）。"""
    global _collection
    if _collection is None and CHROMA_AVAILABLE:
        _ensure_kb_built()
        client = _get_chroma_client()
        if client:
            try:
                _collection = client.get_collection("ops_kb")
            except Exception:
                _collection = None
    return _collection


# ── 工具函数 ──


@tool
def retrieve_knowledge(query: str, top_k: int = 3) -> str:
    """从公司内部运维知识库中检索相关信息。

    当回答问题需要参考公司内部规范时调用此工具。
    知识库包含：服务器命名规范、告警分级与处理流程、端口与中间件配置、日志收集与备份策略。

    Args:
        query: 检索关键词，如 'Nginx 端口'、'P0 告警处理'、'日志目录'
        top_k: 返回的文档数量，默认 3

    Returns:
        检索到的文档内容，如果没有相关知识则返回空字符串
    """
    collection = _get_collection()
    if collection is None:
        return ""

    query_embedding = _create_embeddings([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    if not docs:
        return ""

    parts = []
    for meta, doc in zip(metas, docs):
        source = meta.get("source", "未知")
        parts.append(f"[来源: {source}]\n{doc}")
    return "\n\n---\n\n".join(parts)
