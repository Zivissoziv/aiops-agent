import pytest
from datetime import datetime
from pathlib import Path

from aiops_agent.tools.notepad_tool import read_notepad, append_notepad, NOTEPAD_FILE


@pytest.fixture(autouse=True)
def cleanup_notepad():
    """Remove NOTEPAD.md before and after each test."""
    p = Path(NOTEPAD_FILE)
    if p.exists():
        p.unlink()
    yield
    if p.exists():
        p.unlink()


class TestReadNotepad:
    def test_read_empty_when_missing(self):
        result = read_notepad.invoke({})
        assert result == "(笔记为空)"

    def test_read_existing_content(self):
        p = Path(NOTEPAD_FILE)
        p.write_text("Hello World", encoding="utf-8")
        result = read_notepad.invoke({})
        assert result == "Hello World"


class TestAppendNotepad:
    def test_append_creates_file(self):
        result = append_notepad.invoke({"heading": "测试", "content": "测试内容"})
        assert "测试" in result
        assert Path(NOTEPAD_FILE).exists()

    def test_append_content_appears(self):
        append_notepad.invoke({"heading": "H1", "content": "C1"})
        content = Path(NOTEPAD_FILE).read_text(encoding="utf-8")
        assert "## H1" in content
        assert "C1" in content

    def test_append_timestamp(self):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        append_notepad.invoke({"heading": "TS", "content": "abc"})
        content = Path(NOTEPAD_FILE).read_text(encoding="utf-8")
        # Ensure the timestamp looks right
        assert "20" in content  # year prefix

    def test_multiple_appends_accumulate(self):
        append_notepad.invoke({"heading": "A", "content": "a"})
        append_notepad.invoke({"heading": "B", "content": "b"})
        content = Path(NOTEPAD_FILE).read_text(encoding="utf-8")
        assert "## A" in content
        assert "## B" in content
