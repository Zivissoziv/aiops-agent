"""知识库检索工具 — 从运维知识库中检索公司内部规范。"""
import hashlib, json, os
from langchain_core.tools import tool
from openai import OpenAI
from ..config import Config, _find_project_root

ROOT = _find_project_root()
KB_DIR = ROOT / "knowledge_base"
DATA_DIR = ROOT / ".aiops_data"
CHROMA_DIR = str(DATA_DIR / "chroma_db")
TS_FILE = DATA_DIR / "kb_timestamp.json"

_config: Config | None = None
_embedding_client: OpenAI | None = None
_embedding_model = "text-embedding-3-small"
_collection, _chroma_client = None, None

try:
    import chromadb
    CHROMA_OK = True
except ImportError:
    CHROMA_OK = False


def _get_config() -> Config:
    global _config
    if _config is None: _config = Config.from_env()
    return _config

def _embed() -> OpenAI:
    global _embedding_client, _embedding_model
    if _embedding_client is None:
        cfg = _get_config()
        _embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        _embedding_client = OpenAI(api_key=os.getenv("EMBEDDING_API_KEY") or cfg.api_key, base_url=os.getenv("EMBEDDING_BASE_URL") or cfg.base_url)
    return _embedding_client

def _create_embeddings(texts: list[str]) -> list[list[float]]:
    c = _embed(); r = []
    for i in range(0, len(texts), 10):
        resp = c.embeddings.create(model=_embedding_model, input=texts[i:i+10])
        r.extend(item.embedding for item in resp.data)
    return r


# ── 时间戳 ──

def _ts() -> str:
    if not KB_DIR.is_dir(): return ""
    fs = sorted(KB_DIR.glob("*.md"))
    if not fs: return ""
    return hashlib.md5("\n".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in fs).encode()).hexdigest()

def _load_ts() -> str:
    if TS_FILE.exists():
        try: return json.loads(TS_FILE.read_text(encoding="utf-8")).get("timestamp", "")
        except Exception: return ""
    return ""

def _save_ts(ts: str):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TS_FILE.write_text(json.dumps({"timestamp": ts}, ensure_ascii=False), encoding="utf-8")


# ── 建库 ──

def _load_docs() -> list[dict]:
    docs = []
    for f in sorted(KB_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8")
        title = f.name.replace(".md", "")
        for line in text.splitlines():
            if line.startswith("# "): title = line.lstrip("# ").strip(); break
        docs.append({"source": f.name, "title": title, "content": text})
    return docs

def _chunk(doc: dict, size: int = 200, overlap: int = 20) -> list[dict]:
    ps = [p.strip() for p in doc["content"].split("\n\n") if p.strip()]
    chunks, cur = [], ""
    for p in ps:
        if len(cur) + len(p) < size:
            cur = (cur + "\n\n" + p) if cur else p
        else:
            if cur: chunks.append(cur); cur = p
    if cur: chunks.append(cur)
    r, tail = [], ""
    for i, c in enumerate(chunks):
        if tail and overlap: c = tail + "\n" + c
        r.append({"id": f"{doc['source']}_chunk_{i:03d}", "source": doc["source"], "title": doc["title"], "content": c})
        tail = c[-overlap:] if len(c) > overlap else c
    return r

def _get_chroma():
    global _chroma_client
    if _chroma_client is None and CHROMA_OK and CHROMA_OK:
        import chromadb; _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client

def _ensure_kb():
    global _collection, _chroma_client
    client = _get_chroma()
    if not client: return
    cur = _ts()
    if not cur: return
    if cur == _load_ts(): return
    _collection = None
    try: client.delete_collection("ops_kb")
    except Exception: pass
    try:
        col = client.create_collection("ops_kb", metadata={"description": "AIOps 运维知识库"})
    except chromadb.errors.InternalError:
        col = client.get_collection("ops_kb")
    docs = _load_docs()
    if not docs: return
    all_c = []
    for d in docs: all_c.extend(_chunk(d))
    texts = [c["content"] for c in all_c]
    col.add(ids=[c["id"] for c in all_c], embeddings=_create_embeddings(texts), documents=texts, metadatas=[{"source": c["source"], "title": c["title"]} for c in all_c])
    _save_ts(cur)

def _get_collection():
    global _collection
    if _collection is None and CHROMA_OK:
        _ensure_kb(); client = _get_chroma()
        if client:
            try: _collection = client.get_collection("ops_kb")
            except Exception: _collection = None
    return _collection


@tool
def retrieve_knowledge(query: str, top_k: int = 3) -> str:
    """从公司内部运维知识库中检索相关信息。知识库：服务器命名规范、告警分级与处理、端口与中间件配置、日志收集与备份。"""
    col = _get_collection()
    if not col: return ""
    qe = _create_embeddings([query])[0]
    r = col.query(query_embeddings=[qe], n_results=top_k)
    docs = r["documents"][0] if r["documents"] else []
    metas = r["metadatas"][0] if r["metadatas"] else []
    if not docs: return ""
    return "\n\n---\n\n".join(f"[来源: {m.get('source','?')}]\n{d}" for m, d in zip(metas, docs))
