"""文件工具单元测试。

注意: 这些测试使用回调来模拟审批机制，并测试文件读写功能。
"""

import tempfile
from pathlib import Path

import pytest

from src.aiops_agent.tools.file_tools import (
    configure_workspace,
    configure_write_approval,
    read_file,
    write_file,
)


class TestReadFile:
    """读取文件测试。"""

    @pytest.fixture
    def test_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("line1\nline2\nline3\n", encoding="utf-8")
            yield str(path)

    def test_read_existing(self, test_file):
        result = read_file.invoke({"path": test_file})
        assert "line1" in result
        assert "line2" in result

    def test_read_limit_lines(self, test_file):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.txt"
            path.write_text("\n".join(f"line{i}" for i in range(100)), encoding="utf-8")
            result = read_file.invoke({"path": str(path), "lines": 5})
            lines = result.split("\n")
            # 应该只有 5 行 + 省略提示
            assert any("仅显示前 5 行" in line for line in lines)

    def test_read_not_found(self):
        result = read_file.invoke({"path": "/nonexistent/file.txt"})
        assert "不存在" in result

    def test_read_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.txt"
            path.write_text("", encoding="utf-8")
            result = read_file.invoke({"path": str(path)})
            assert "文件为空" in result


class TestWriteFile:
    """写入文件测试。"""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        # 确保每次测试后重置配置
        configure_write_approval(None)
        configure_workspace(None)
        yield
        configure_write_approval(None)
        configure_workspace(None)

    def test_write_without_sandbox(self):
        # 没有沙箱时，行为不变
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.txt"
            result = write_file.invoke({"path": str(path), "content": "hello world"})
            assert "已写入" in result
            assert path.read_text(encoding="utf-8") == "hello world"

    def test_write_within_workspace(self):
        # 在 workspace 内直接写入，无需审批
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            configure_workspace(str(ws))
            path = ws / "within.txt"
            result = write_file.invoke({"path": str(path), "content": "data"})
            assert "已写入" in result
            assert path.read_text(encoding="utf-8") == "data"

    def test_write_outside_workspace_approved(self):
        # 越界且用户批准
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            configure_workspace(str(ws))
            outside = Path(tmp) / "outside.txt"

            approved = []
            def handler(path, preview):
                approved.append((path, preview))
                return True
            configure_write_approval(handler)

            result = write_file.invoke({"path": str(outside), "content": "data"})
            assert "已写入" in result
            assert len(approved) == 1

    def test_write_outside_workspace_denied(self):
        # 越界且用户拒绝
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            configure_workspace(str(ws))
            outside = Path(tmp) / "outside.txt"

            def handler(path, preview):
                return False
            configure_write_approval(handler)

            result = write_file.invoke({"path": str(outside), "content": "data"})
            assert "拒绝" in result
            assert not outside.exists()

    def test_read_within_workspace(self):
        # 在 workspace 内直接读取
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            configure_workspace(str(ws))
            path = ws / "test.txt"
            path.write_text("inside\n", encoding="utf-8")
            result = read_file.invoke({"path": str(path)})
            assert "inside" in result

    def test_read_outside_workspace_approved(self):
        # 越界读取且用户批准
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            configure_workspace(str(ws))
            outside = Path(tmp) / "outside.txt"
            outside.write_text("external\n", encoding="utf-8")

            approved = []
            def handler(path, preview):
                approved.append((path, preview))
                return True
            configure_write_approval(handler)

            result = read_file.invoke({"path": str(outside)})
            assert "external" in result
            assert len(approved) == 1

    def test_read_outside_workspace_denied(self):
        # 越界读取且用户拒绝
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            configure_workspace(str(ws))
            outside = Path(tmp) / "outside.txt"
            outside.write_text("secret\n", encoding="utf-8")

            def handler(path, preview):
                return False
            configure_write_approval(handler)

            result = read_file.invoke({"path": str(outside)})
            assert "拒绝" in result

    def test_append_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "append.txt"
            path.write_text("original\n", encoding="utf-8")
            result = write_file.invoke({"path": str(path), "content": "appended", "append": True})
            assert "追加" in result
            assert path.read_text(encoding="utf-8") == "original\nappended"
