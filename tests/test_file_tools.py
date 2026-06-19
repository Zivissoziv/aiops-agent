# -*- coding: utf-8 -*-
"""文件工具单元测试。"""

import tempfile
from pathlib import Path

import pytest

from src.aiops_agent.tools.file_tools import (
    configure_workspace,
    configure_write_approval,
    edit_file,
    read_file,
    write_file,
)


class TestReadFile:

    def test_read_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.txt"
            path.write_text("line1\nline2\nline3\n", encoding="utf-8")
            result = read_file.invoke({"path": str(path)})
            assert "line1" in result

    def test_read_limit_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "long.txt"
            path.write_text("\n".join(f"line{i}" for i in range(100)), encoding="utf-8")
            result = read_file.invoke({"path": str(path), "lines": 5})
            lines = result.split("\n")
            # 只显示前 5 行 + 省略提示
            assert lines[0] == "line0"
            assert any("..." in line for line in lines)

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

    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        configure_write_approval(None)
        configure_workspace(None)
        yield
        configure_write_approval(None)
        configure_workspace(None)

    def test_write_basic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.txt"
            result = write_file.invoke({"path": str(path), "content": "hello"})
            assert "已写入" in result
            assert path.read_text(encoding="utf-8") == "hello"

    def test_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "append.txt"
            path.write_text("original\n", encoding="utf-8")
            result = write_file.invoke({"path": str(path), "content": "appended", "append": True})
            assert "追加" in result
            assert path.read_text(encoding="utf-8") == "original\nappended"


class TestWorkspaceSandbox:

    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        configure_write_approval(None)
        configure_workspace(None)
        yield
        configure_write_approval(None)
        configure_workspace(None)

    def test_read_write_within_workspace(self):
        """workspace 内直接读写，无需审批。"""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            configure_workspace(str(ws))
            assert "已写入" in write_file.invoke({"path": str(ws / "f.txt"), "content": "data"})
            assert "data" in read_file.invoke({"path": str(ws / "f.txt")})

    def test_read_write_outside_workspace_approved(self):
        """越界操作，经审批通过。"""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            configure_workspace(str(ws))
            outside = Path(tmp) / "outside.txt"

            approved = []
            configure_write_approval(lambda path, preview: (approved.append((path, preview)), True)[1])

            # 越界写入
            result_write = write_file.invoke({"path": str(outside), "content": "data"})
            assert "已写入" in result_write
            assert outside.read_text(encoding="utf-8") == "data"

            # 越界读取
            result_read = read_file.invoke({"path": str(outside)})
            assert "data" in result_read

            assert len(approved) == 2

    def test_write_outside_workspace_denied(self):
        """越界操作被拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            configure_workspace(str(ws))
            outside = Path(tmp) / "outside.txt"

            configure_write_approval(lambda path, preview: False)
            assert "拒绝" in write_file.invoke({"path": str(outside), "content": "data"})
            assert not outside.exists()


class TestEditFile:

    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        configure_write_approval(None)
        configure_workspace(None)
        yield
        configure_write_approval(None)
        configure_workspace(None)

    def test_edit_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.py"
            path.write_text("print('hello')\n", encoding="utf-8")
            result = edit_file.invoke({"path": str(path), "old_text": "hello", "new_text": "hi"})
            assert "已编辑" in result
            assert path.read_text(encoding="utf-8") == "print('hi')\n"

    def test_edit_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.py"
            path.write_text("x\n", encoding="utf-8")
            result = edit_file.invoke({"path": str(path), "old_text": "y", "new_text": "z"})
            assert "未找到" in result

    def test_edit_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.py"
            path.write_text("old\n", encoding="utf-8")
            result = edit_file.invoke({"path": str(path), "old_text": "old", "new_text": "new", "dry_run": True})
            assert "将替换" in result
            assert path.read_text(encoding="utf-8") == "old\n"

    def test_edit_outside_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp) / "workspace"
            ws.mkdir()
            configure_workspace(str(ws))
            path = Path(tmp) / "outside.txt"
            path.write_text("old\n", encoding="utf-8")

            configure_write_approval(lambda path, preview: True)
            assert "已编辑" in edit_file.invoke({"path": str(path), "old_text": "old", "new_text": "new"})

            configure_write_approval(lambda path, preview: False)
            assert "拒绝" in edit_file.invoke({"path": str(path), "old_text": "new", "new_text": "x"})
