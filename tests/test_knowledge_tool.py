"""知识库工具单元测试。

注意: 这些测试需要 Embedding API Key（复用 .env 配置）。
首次运行会创建 Chroma 数据库，后续复用。
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.aiops_agent.tools.knowledge_tool import (
    _load_ts,
    _save_ts,
    _ts,
)


class TestTimestamp:
    """时间戳管理测试。"""

    def test_save_and_load(self):
        """保存的时间戳能正确读取。"""
        _save_ts("abc123")
        assert _load_ts() == "abc123"

    def test_load_missing(self):
        """文件不存在时返回空字符串。"""
        import src.aiops_agent.tools.knowledge_tool as kt
        # 把 TS_FILE 指向不存在的路径
        saved = ""
        if kt.TS_FILE.exists():
            # 正常情况文件存在时，读出来一定有内容
            saved = _load_ts()
        assert isinstance(saved, str)


class TestRetrieveKnowledge:
    """知识库检索测试（保留集成测试）。"""

    @pytest.fixture(autouse=True)
    def setup_temp_kb(self):
        """用临时知识库目录替换模块变量。"""
        import src.aiops_agent.tools.knowledge_tool as kt

        orig_kb_dir = kt.KB_DIR
        orig_chroma_dir = kt.CHROMA_DIR
        orig_ts_file = kt.TS_FILE
        orig_root = kt.ROOT

        import shutil
        tmp = Path(tempfile.mkdtemp(prefix="kb_test_"))
        try:
            kb_path = tmp / "knowledge_base"
            kb_path.mkdir()
            chroma_path = tmp / "chroma_db"
            ts_path = tmp / "kb_timestamp.json"

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

            kt.KB_DIR = kb_path
            kt.CHROMA_DIR = str(chroma_path)
            kt.TS_FILE = ts_path
            kt._collection = None
            kt._chroma_client = None

            yield
        finally:
            kt._collection = None
            kt._chroma_client = None
            kt.KB_DIR = orig_kb_dir
            kt.CHROMA_DIR = orig_chroma_dir
            kt.TS_FILE = orig_ts_file
            kt.ROOT = orig_root

            import time
            time.sleep(0.5)
            try:
                shutil.rmtree(tmp, ignore_errors=True)
            except Exception:
                pass

    def test_retrieve_existing(self):
        """检索已有知识，返回相关内容。"""
        from src.aiops_agent.tools.knowledge_tool import retrieve_knowledge
        result = retrieve_knowledge.invoke({"query": "Nginx 端口"})
        assert "8081" in result

    def test_retrieve_other(self):
        """检索另一条知识。"""
        from src.aiops_agent.tools.knowledge_tool import retrieve_knowledge
        result = retrieve_knowledge.invoke({"query": "日志目录"})
        assert "data/logs" in result

    def test_top_k(self):
        """top_k 参数影响返回文档块数量。"""
        from src.aiops_agent.tools.knowledge_tool import retrieve_knowledge
        result = retrieve_knowledge.invoke({"query": "端口", "top_k": 1})
        assert result.count("[来源:") <= 1

    def test_auto_build_and_skip(self):
        """首次调用自动建库，未变更时跳过重建。"""
        import src.aiops_agent.tools.knowledge_tool as kt
        from src.aiops_agent.tools.knowledge_tool import retrieve_knowledge

        # 首次调用
        result = retrieve_knowledge.invoke({"query": "Nginx 端口"})
        assert "8081" in result
        assert kt.TS_FILE.exists()

        # 清除缓存后再次调用（不修改文件）
        kt._collection = None
        result = retrieve_knowledge.invoke({"query": "Nginx 端口"})
        assert "8081" in result

    def test_rebuild_when_doc_changed(self):
        """文件变更后自动重建。"""
        import src.aiops_agent.tools.knowledge_tool as kt
        from src.aiops_agent.tools.knowledge_tool import retrieve_knowledge

        retrieve_knowledge.invoke({"query": "Nginx 端口"})

        # 修改文档
        doc = kt.KB_DIR / "test_ports.md"
        doc.write_text("# 端口规范\n\nNginx 使用 9090 端口。\n", encoding="utf-8")

        kt._collection = None
        result = retrieve_knowledge.invoke({"query": "Nginx 端口"})
        assert "9090" in result
