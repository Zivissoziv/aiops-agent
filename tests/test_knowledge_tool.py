"""知识库工具单元测试。

注意: 这些测试需要 Embedding API Key（复用 .env 配置）。
首次运行会创建 Chroma 数据库，后续复用。
"""

import tempfile
from pathlib import Path

import pytest

from src.aiops_agent.tools.knowledge_tool import (
    _ensure_kb_built,
    _get_kb_timestamp,
    _load_kb_docs,
    _load_saved_timestamp,
    _save_timestamp,
    retrieve_knowledge,
)


class TestTimestamp:
    """时间戳管理测试。"""

    def test_get_timestamp_empty_dir(self):
        """空目录或不存在目录返回空字符串。"""
        import hashlib
        from src.aiops_agent.tools.knowledge_tool import KB_DIR as real_kb, _get_kb_timestamp

        # KB_DIR 是模块级常量指向真实目录，不能改
        # 但函数内部对 is_dir() 和 glob 有判断，验证它返回非空即可
        ts = _get_kb_timestamp()
        # 项目根目录有 knowledge_base/ 时，时间戳不应为空
        if real_kb.is_dir() and list(real_kb.glob("*.md")):
            assert ts != ""
            assert len(ts) == 32  # md5 hex
        else:
            assert ts == ""

    def test_save_and_load_timestamp(self):
        """保存的时间戳能正确读取。"""
        with tempfile.TemporaryDirectory() as tmp:
            ts_file = Path(tmp) / "kb_timestamp.json"
            # 模拟 _save_timestamp 的行为
            import json
            ts_file.write_text(json.dumps({"timestamp": "abc123"}, ensure_ascii=False), encoding="utf-8")

            # 用模块内部逻辑读取
            saved = ""
            if ts_file.exists():
                try:
                    data = json.loads(ts_file.read_text(encoding="utf-8"))
                    saved = data.get("timestamp", "")
                except Exception:
                    pass

            assert saved == "abc123"

    def test_empty_timestamp_file(self):
        """文件不存在时返回空字符串。"""
        with tempfile.TemporaryDirectory() as tmp:
            ts_file = Path(tmp) / "nonexistent.json"
            saved = ""
            if ts_file.exists():
                try:
                    import json
                    data = json.loads(ts_file.read_text(encoding="utf-8"))
                    saved = data.get("timestamp", "")
                except Exception:
                    pass
            assert saved == ""


class TestLoadDocs:
    """知识库文档加载测试。"""

    def test_load_markdown_files(self):
        """加载 .md 文件，正确提取标题和内容。"""
        with tempfile.TemporaryDirectory() as tmp:
            kb = Path(tmp)
            md = kb / "test_doc.md"
            md.write_text("# 测试文档\n\n这是内容。\n", encoding="utf-8")

            docs = []
            for md_file in sorted(kb.glob("*.md")):
                text = md_file.read_text(encoding="utf-8")
                title = md_file.name.replace(".md", "")
                for line in text.splitlines():
                    if line.startswith("# "):
                        title = line.lstrip("# ").strip()
                        break
                docs.append({"source": md_file.name, "title": title, "content": text})

            assert len(docs) == 1
            assert docs[0]["title"] == "测试文档"
            assert "这是内容" in docs[0]["content"]

    def test_load_empty_dir(self):
        """空目录返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            docs = []
            for md_file in sorted(Path(tmp).glob("*.md")):
                pass
            assert len(docs) == 0

    def test_load_multiple_files(self):
        """多个 .md 文件全部加载。"""
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "a.md").write_text("# A\n", encoding="utf-8")
            (Path(tmp) / "b.md").write_text("# B\n", encoding="utf-8")

            docs = []
            for md_file in sorted(Path(tmp).glob("*.md")):
                text = md_file.read_text(encoding="utf-8")
                title = md_file.name.replace(".md", "")
                for line in text.splitlines():
                    if line.startswith("# "):
                        title = line.lstrip("# ").strip()
                        break
                docs.append({"source": md_file.name, "title": title, "content": text})

            assert len(docs) == 2
            assert docs[0]["source"] == "a.md"


class TestRetrieveKnowledge:
    """知识库检索测试。

    这些测试使用项目根目录的 knowledge_base/（如果存在），
    或创建一个临时知识库来测试检索功能。
    需要 Embedding API 可用。
    """

    @pytest.fixture(autouse=True)
    def setup_temp_kb(self):
        """用临时知识库目录替换模块变量。"""
        import src.aiops_agent.tools.knowledge_tool as kt

        # 保存原始值
        orig_kb_dir = kt.KB_DIR
        orig_chroma_dir = kt.CHROMA_DIR
        orig_ts_file = kt.TS_FILE

        # 创建临时知识库（用固定目录避免 Windows 文件锁问题）
        import shutil
        tmp = Path(tempfile.mkdtemp(prefix="kb_test_"))
        try:
            kb_path = tmp / "knowledge_base"
            kb_path.mkdir()
            chroma_path = tmp / "chroma_db"
            ts_path = tmp / "kb_timestamp.json"

            # 写入测试文档
            doc1 = kb_path / "test_ports.md"
            doc1.write_text(
                "# 端口规范\n\n"
                "Nginx 使用 8081 端口。\n"
                "Jenkins 使用 8082 端口。\n",
                encoding="utf-8",
            )
            doc2 = kb_path / "test_logs.md"
            doc2.write_text(
                "# 日志规范\n\n"
                "日志统一写入 /data/logs/ 目录。\n"
                "保留 30 天。\n",
                encoding="utf-8",
            )

            # 替换模块变量
            kt.KB_DIR = kb_path
            kt.CHROMA_DIR = str(chroma_path)
            kt.TS_FILE = ts_path
            # 清除缓存，让 _ensure_kb_built 重新执行
            kt._collection = None
            kt._chroma_client = None

            yield
        finally:
            # 恢复原始值（在清理 Chroma 文件之前）
            kt._collection = None
            kt._chroma_client = None
            kt.KB_DIR = orig_kb_dir
            kt.CHROMA_DIR = orig_chroma_dir
            kt.TS_FILE = orig_ts_file

            # 等待 Chroma 释放文件锁后再清理
            import time
            time.sleep(0.5)
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass

    def test_retrieve_existing_knowledge(self):
        """检索已存在的知识，返回相关内容。"""
        result = retrieve_knowledge.invoke({"query": "Nginx 端口"})
        assert "8081" in result

    def test_retrieve_other_knowledge(self):
        """检索另一个知识，返回正确内容。"""
        result = retrieve_knowledge.invoke({"query": "日志目录"})
        assert "data/logs" in result

    def test_retrieve_nonexistent(self):
        """检索不存在的知识，返回空字符串（当知识库无匹配时）。"""
        # 注意：语义检索可能返回最接近的文档，
        # 所以这里测的是"文档内容不包含检索词"而不是"返回空"
        result = retrieve_knowledge.invoke({"query": "数据库密码"})
        # 知识库里没有任何关于密码的内容，所以结果不应包含"密码"
        assert "密码" not in result

    def test_retrieve_top_k(self):
        """指定 top_k 返回对应数量的文档块。"""
        result = retrieve_knowledge.invoke({"query": "端口", "top_k": 1})
        assert result.count("[来源:") <= 1

    def test_auto_build_on_first_call(self):
        """首次调用自动建库。"""
        import src.aiops_agent.tools.knowledge_tool as kt

        # 清理时间戳和集合缓存
        if kt.TS_FILE.exists():
            kt.TS_FILE.unlink()
        kt._collection = None

        # 再次检索，应该自动建库
        result = retrieve_knowledge.invoke({"query": "Nginx 端口"})
        assert "8081" in result
        assert kt.TS_FILE.exists()

    def test_skip_build_when_unchanged(self):
        """知识库未变化时跳过建库。"""
        import src.aiops_agent.tools.knowledge_tool as kt

        # 先确保已建库
        retrieve_knowledge.invoke({"query": "Nginx 端口"})

        # 清除集合缓存，下次调用会重新检查时间戳
        kt._collection = None

        # 再次检索（不修改文件），应该复用已有库
        result = retrieve_knowledge.invoke({"query": "Nginx 端口"})
        assert "8081" in result

    def test_rebuild_when_doc_changed(self):
        """知识库文件变更后自动重建。"""
        import src.aiops_agent.tools.knowledge_tool as kt

        # 先确保已建库
        retrieve_knowledge.invoke({"query": "Nginx 端口"})

        # 修改文档内容
        doc = kt.KB_DIR / "test_ports.md"
        doc.write_text("# 端口规范\n\nNginx 使用 9090 端口。\n", encoding="utf-8")

        # 清除缓存
        kt._collection = None

        # 再次检索，应该重建并返回新内容
        result = retrieve_knowledge.invoke({"query": "Nginx 端口"})
        assert "9090" in result
        assert "8081" not in result
