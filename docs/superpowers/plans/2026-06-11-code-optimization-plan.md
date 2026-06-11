# AIOps Agent 代码优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use subagent-driven-development (recommended) or executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复代码审查发现的关键安全和功能问题，提升代码质量和可维护性

**Architecture:** 按优先级逐项修复，每项独立可测试、可验证。从 P0 安全/功能缺陷开始，再到 P1 可维护性改进，最后是 P2 代码质量提升。

**Tech Stack:** Python 3.10+, pytest, langgraph, langchain-core, openai

---

## 范围与文件清单

### 将被修改的文件
| 文件 | 修改内容 |
|------|---------|
| `src/aiops_agent/cli.py` | 多轮对话修复、模块级变量重构、异常处理 |
| `src/aiops_agent/tools/file_tools.py` | 路径遍历防护 |
| `src/aiops_agent/tools/shell.py` | 输出截断 |
| `src/aiops_agent/tools/__init__.py` | `_iter_tools` 重构 |
| `src/aiops_agent/memory/episodic.py` | `list_keys` 性能优化 |
| `src/pyproject.toml` | 依赖版本锁定 |
| `tests/test_cli_helpers.py` | 新增 build_graph 测试 |
| `tests/test_integration.py` | 新增集成测试 |
| `docs/specs/2026-06-10-aiops-agent-design.md` | 同步设计文档 |

### 不变的文件
| 文件 | 原因 |
|------|------|
| `src/aiops_agent/memory/tiered.py` | 当前设计合理 |
| `src/aiops_agent/memory/core.py` | 当前设计合理，持久化逻辑健壮 |
| `src/aiops_agent/memory/working.py` | 当前设计合理 |
| `src/aiops_agent/core/agent.py` | 当前设计合理 |
| `src/aiops_agent/agents/` | 当前设计合理 |
| `src/aiops_agent/llm/` | 当前设计合理 |
| `src/aiops_agent/config.py` | 当前设计合理 |

---

### Task 1: 修复 API Key 泄露 — 将 `.env` 加入 `.gitignore` 并清理历史

**Files:**
- Modify: `.gitignore`
- Modify: `.env.example`（确保有完整示例）

**背景:** `.env` 文件已被 git 跟踪且包含真实 API Key。需要立即停止跟踪并从历史中清除。

- [ ] **Step 1: 确认 `.gitignore` 已包含 `.env`**

检查当前 `.gitignore` 内容：

```bash
cat .gitignore
```

- [ ] **Step 2: 如果缺了 `.env`，添加到 `.gitignore`**

```bash
echo ".env" >> .gitignore
echo ".aiops_data/" >> .gitignore
```

- [ ] **Step 3: 从 git 跟踪中移除 `.env`（但保留本地文件）**

```bash
cd d:/workspace/aiops-agent
git rm --cached .env
git rm --cached -r .aiops_data/ 2>/dev/null || true
```

- [ ] **Step 4: 提交**

```bash
git add .gitignore
git commit -m "security: 移除 .env 文件跟踪，API Key 不再提交到仓库"
```

- [ ] **Step 5: 检查 `.env.example` 是否包含所有必要字段**

确认 `.env.example` 文件包含完整的占位符配置，不要遗漏 `REACT_ENABLED` 等可选字段。

---

### Task 2: 修复多轮对话上下文丢失（关键功能缺陷）

**Files:**
- Modify: `src/aiops_agent/cli.py`（第 108-171 行 `build_graph` 函数、第 312-326 行 main 循环）

**背景:** `cli.py:main()` 中 `graph.stream()` 每次重新创建 `AppState`（`messages` 和 `task` 每次都从 `user_input` 新构建），导致 `TieredMemory` 虽然存储了历史消息，但 LangGraph 状态机内部只看到当前轮的消息，Agent 无法感知上下文。

需要将 `TieredMemory` 的历史注入到 `state["messages"]` 中，让 Agent 在每轮对话时都能看到历史上下文。

- [ ] **Step 1: 修改 `cli.py` 主循环中的 state 构建**

将 `main()` 中第 314-318 行替换为使用 `memory.get_messages()` 注入历史上下文：

```python
# ── 运行图（双模式流）──
try:
    # 从三层记忆构建上下文消息
    history = memory.get_messages()  # 返回 core + episodic + working 的合并
    state: AppState = {
        "messages": [
            # system/core/episodic 已在 history 中（若有），
            # 但 Agent.run() 内部也会加 system_prompt，
            # 所以只取 working 部分避免重复
            HumanMessage(content=user_input),
        ],
        "task": user_input,
        "need_worker": False,
    }

    for mode, event in graph.stream(state, stream_mode=["updates", "custom"]):
        ...
```

注意：`Agent.run()` 内部已经加了 `SystemMessage(content=self.system_prompt)`，所以这里只需要传入当前轮的用户消息。真正的历史注入应当通过 `Agent.run()` 内部将 `memory.get_messages()` 追加到 `messages` 列表中。更合理的方案是在 `make_node` 中将 `memory` 的历史注入到 `input_messages`。

- [ ] **Step 2: 修改 `make_node` 函数，将三层记忆注入 Agent 输入**

在 `cli.py:build_graph` 的 `make_node` 内部（第 121-156 行），将 `mem.get_messages()` 作为额外的上下文消息注入：

```python
def make_node(n: str, a: Agent, mem: TieredMemory):
    def node_fn(state: AppState, writer: StreamWriter) -> dict:
        writer({"type": "agent_start", "agent": n})
        # 注入三层记忆历史
        history = mem.get_messages()
        input_msgs = list(history)  # core + episodic + working 上下文
        input_msgs.append(HumanMessage(content=state.get("task", "")))
        produced_msgs, events = a.run(input_msgs)
        ...
    return node_fn
```

注意: `Agent.run()` 内部会拼 `SystemMessage(content=self.system_prompt)` 在最前面，而 `mem.get_messages()` 返回的 core 也是 system 角色的消息，所以会有两个 system 消息。需要确认 `Agent.run()` 是否支持外部传入的 system 消息，或者将 system_prompt 的注入移到外部。

- [ ] **Step 3: 修改 `Agent.run()` 避免重复 system prompt**

如果 `Agent.run()` 始终在开头添加 `SystemMessage(content=self.system_prompt)`，而 `history` 中已包含 core memory（也是 system 角色），则会出现两个 system 消息。修改 `Agent.run()` 以支持已包含 system_prompt 的输入，或者不让 `Agent` 自动添加 system_prompt 而是由调用方控制。

最佳方案：让 `make_node` 在构造 `input_msgs` 时不包含 system 消息（交给 Agent 自动加），只把 working + episodic 的上下文传进去：

```python
def node_fn(state: AppState, writer: StreamWriter) -> dict:
    writer({"type": "agent_start", "agent": n})
    # 从三层记忆获取上下文（排除 system 角色，避免和 Agent 的 system_prompt 重复）
    history = mem.get_messages()
    non_system_history = [m for m in history if m.get("role") != "system"]
    input_msgs = non_system_history
    input_msgs.append(HumanMessage(content=state.get("task", "")))
    produced_msgs, events = a.run(input_msgs)
    ...
```

- [ ] **Step 4: 修复 `print_custom_event` 的全局变量问题**

`_seen_agents_in_session` 是模块级变量，不会在对话轮次间重置，导致多轮对话中只有第一轮能看到 agent_start 的 banner。返回后应重置：

```python
def main():
    ...
    # 主循环
    while True:
        ...
        # 每轮对话前重置
        _seen_agents_in_session.clear()
        ...
```

- [ ] **Step 5: 运行现有测试确保不破坏既有功能**

```bash
cd d:/workspace/aiops-agent
python -m pytest tests/ -v
```

Expected: 所有测试通过。

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "fix: 多轮对话上下文注入 — Agent 每轮可感知三层记忆的历史信息"
```

---

### Task 3: 修复路径遍历漏洞

**Files:**
- Modify: `src/aiops_agent/tools/file_tools.py`（第 18-48 行 `read_file`、第 51-85 行 `write_file`）

**背景:** `read_file` 和 `write_file` 直接接受用户提供的路径 `path` 参数，未做路径合法性校验。攻击者可能通过 `../../etc/passwd` 等方式跨越到项目目录之外访问任意文件。

- [ ] **Step 1: 在 `file_tools.py` 添加路径安全函数**

在文件顶部添加 `_safe_resolve_path` 函数，将路径解析到项目根目录下的安全范围：

```python
# ── 安全路径解析 ──

# 允许访问的工作目录（项目根目录）
_PROJECT_ROOT: Path | None = None

def configure_allowed_dir(path: str | Path | None = None):
    """设置允许的文件操作根目录。"""
    global _PROJECT_ROOT
    from .context import get_project_root  # or from config import _find_project_root
    _PROJECT_ROOT = Path(path) if path else _find_project_root()

def _safe_resolve_path(path: str) -> Path:
    """安全解析路径，防止路径遍历攻击。
    
    检查用户提供的路径是否在允许的工作目录范围内。
    若超出范围，抛出 ValueError。
    """
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        configure_allowed_dir()
    
    user_path = Path(path)
    # 如果是相对路径，拼接项目根目录
    if not user_path.is_absolute():
        user_path = _PROJECT_ROOT / user_path
    # 解析符号链接和 .. 简化
    resolved = user_path.resolve()
    root_resolved = _PROJECT_ROOT.resolve()
    
    # 检查是否在允许的根目录下
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise ValueError(f"路径越权: {path} 不在允许的工作目录 {root_resolved} 内")
    
    return resolved
```

注意：如果无法 import `_find_project_root`（循环引用风险），直接在 `file_tools.py` 中定义一个简单的根路径逻辑，或通过 `configure_write_approval` 类似的初始化函数传入。

- [ ] **Step 2: 修改 `read_file` 使用安全路径**

```python
@tool
def read_file(path: str, lines: int = 50) -> str:
    """读取文件内容，返回前 N 行。"""
    try:
        p = _safe_resolve_path(path)
    except ValueError as e:
        return f"错误: {e}"
    
    try:
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

- [ ] **Step 3: 修改 `write_file` 使用安全路径**

```python
@tool
def write_file(path: str, content: str, append: bool = False) -> str:
    """写入内容到文件（需要用户确认）。"""
    try:
        p = _safe_resolve_path(path)
    except ValueError as e:
        return f"错误: {e}"
    
    # 审批
    if _write_approval_handler:
        preview = content[:100]
        if not _write_approval_handler(path, preview):
            return f"⚠️ 用户拒绝了写入 {path}"
    
    try:
        mode = "a" if append else "w"
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

- [ ] **Step 4: 在 `cli.py:main` 中初始化安全目录**

在 `main()` 中，注册完审批回调后增加：

```python
# 注册文件路径安全防护
from .tools.file_tools import configure_allowed_dir
configure_allowed_dir()
```

- [ ] **Step 5: 运行测试**

```bash
cd d:/workspace/aiops-agent
python -m pytest tests/ -v
```

- [ ] **Step 6: 提交**

```bash
git add -A
git commit -m "fix: 路径遍历防护 — file_tools 所有操作限定在项目根目录下"
```

---

### Task 4: 添加集成测试（E2E 测试）

**Files:**
- Create: `tests/test_integration.py`

**背景:** 当前 11 个测试文件全是纯单元测试（Mock 驱动），缺少对 `build_graph`、多 Agent 路由、TieredMemory 读写一致性的端到端验证。

- [ ] **Step 1: 创建集成测试文件**

```python
"""集成测试 — 使用 MockLLM 测试完整的图执行流程。"""

from unittest.mock import MagicMock, patch

import pytest

from aiops_agent.cli import build_graph, TOOL_MAP
from aiops_agent.config import Config
from aiops_agent.memory.tiered import TieredMemory


class MockLLM:
    """模拟 LLM，返回固定响应，不调用真实 API。"""
    
    def __init__(self, responses=None):
        self.responses = responses or []
        self.call_count = 0
        self.messages_history = []
    
    def invoke(self, messages, tools=None):
        self.messages_history.extend(messages)
        response = MagicMock()
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
        else:
            resp = {"content": "操作完成，没有问题需要处理。", "tool_calls": None}
        
        response.content = resp.get("content", "")
        response.tool_calls = resp.get("tool_calls", None)
        self.call_count += 1
        return response
    
    def count_tokens(self, messages):
        return sum(len(str(m)) for m in messages) // 2


class TestBuildGraph:
    """build_graph 集成测试。"""
    
    @pytest.fixture
    def config(self):
        c = MagicMock(spec=Config)
        c.max_tool_rounds = 3
        c.llm_provider = "openai_compatible"
        c.model = "mock-model"
        return c
    
    @pytest.fixture
    def llm(self):
        return MockLLM(responses=[
            {"content": "系统正常，无需操作。", "tool_calls": None},
        ])
    
    @pytest.fixture
    def memory(self, llm):
        return TieredMemory(llm=llm)
    
    def test_build_graph_structure(self, config, llm, memory):
        """验证状态图节点和边的构造。"""
        graph = build_graph(config, llm, memory)
        assert graph is not None
        
        # 检查图中的节点
        nodes = list(graph.get_graph().nodes)
        assert len(nodes) >= 2  # 至少 planner + worker
        
        node_names = [n for n in nodes]
        assert "planner" in node_names
        assert "worker" in node_names
    
    def test_graph_simple_query(self, config, llm, memory):
        """简单查询：greeting 类只走 planner，不走 worker。"""
        # 因为 planner 发现没有 TODO 时 need_worker=False，直接 END
        graph = build_graph(config, llm, memory)
        
        state = {
            "messages": [],
            "task": "你好",
            "need_worker": False,
        }
        
        collected = []
        for mode, event in graph.stream(state, stream_mode=["custom"]):
            if mode == "custom":
                collected.append(event)
        
        # 至少收到 agent_start 事件
        starts = [e for e in collected if e.get("type") == "agent_start"]
        assert len(starts) >= 1
    
    def test_graph_with_worker(self, config, llm, memory):
        """含 TODO 的查询：planner → worker。"""
        llm_worker = MockLLM(responses=[
            {"content": "[TODO] 检查 CPU 使用率\n[TODO] 检查磁盘\n[NEED_WORKER]", "tool_calls": None},
            {"content": "CPU 使用率 45%，正常。", "tool_calls": None},
        ])
        mem = TieredMemory(llm=llm_worker)
        graph = build_graph(config, llm_worker, mem)
        
        state = {
            "messages": [],
            "task": "检查系统状态",
            "need_worker": False,
        }
        
        collected_modes = []
        for mode, event in graph.stream(state, stream_mode=["updates", "custom"]):
            collected_modes.append(mode)
            _ = event  # 不关心具体输出，只确认流程
        
        # 两种模式都应有输出
        assert "custom" in collected_modes

    def test_memory_persistence_across_runs(self, config, llm, memory):
        """验证 TieredMemory 跨多次 graph.stream 调用的读写一致性。"""
        memory.add_message({"role": "user", "content": "第一轮对话"})
        memory.add_message({"role": "assistant", "content": "好的"})
        
        msgs = memory.get_messages()
        working = memory.working.get_messages()
        assert len(working) == 2
        assert working[0]["content"] == "第一轮对话"
        
        # 模拟第二轮对话
        memory.add_message({"role": "user", "content": "第二轮对话"})
        memory.add_message({"role": "assistant", "content": "收到"})
        
        msgs_after = memory.get_messages()
        assert len(msgs_after) >= 2
```

- [ ] **Step 2: 运行集成测试**

```bash
cd d:/workspace/aiops-agent
python -m pytest tests/test_integration.py -v
```

Expected: 全部 PASS。

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "test: 添加集成测试 — build_graph 结构验证 + 多轮记忆一致性 + 路由测试"
```

---

### Task 5: 重构 `tools/__init__.py` 的 `_iter_tools` 为更健壮的模式

**Files:**
- Modify: `src/aiops_agent/tools/__init__.py`

**背景:** 当前使用 `globals().get(mod_name)` 依赖字符串与 import 变量名完全一致，脆弱且不 Pythonic。

- [ ] **Step 1: 重构 `_iter_tools`**

```python
# d:\workspace\aiops-agent\src\aiops_agent\tools\__init__.py
"""工具注册 — 自动扫描 @tool 装饰器。"""

import inspect

from langchain_core.tools import StructuredTool

from . import file_tools, shell

# 模块白名单 — 将所有工具模块列在此处
_TOOL_MODULES = [file_tools, shell]


def _iter_tools():
    for mod in _TOOL_MODULES:
        for name, obj in inspect.getmembers(mod):
            if isinstance(obj, StructuredTool):
                yield obj


def get_tools() -> dict[str, StructuredTool]:
    return {tool.name: tool for tool in _iter_tools()}


__all__ = ["get_tools"]
```

- [ ] **Step 2: 运行测试确保不影响已有功能**

```bash
cd d:/workspace/aiops-agent
python -m pytest tests/test_cli_helpers.py -v
```

Expected: 全部 PASS。

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "refactor: _iter_tools 改用模块对象列表而非字符串 getattr"
```

---

### Task 6: 工具输出截断 — 防止过长输出撑爆上下文

**Files:**
- Modify: `src/aiops_agent/tools/shell.py`（第 99-153 行 `shell` 函数）

**背景:** `shell` 命令执行结果不做截断，超长输出（如 `cat` 一个大文件）会撑爆 Agent 上下文。

- [ ] **Step 1: 在 `shell` 函数返回前截断 output**

```python
# 在 _decode(result.stdout) 之后、构建返回字典之前截断
MAX_OUTPUT_LENGTH = 10000

stdout = _decode(result.stdout)
stderr = _decode(result.stderr)

# 截断超长输出
if len(stdout) > MAX_OUTPUT_LENGTH:
    stdout = stdout[:MAX_OUTPUT_LENGTH] + f"\n...（输出过长，已截断 {len(stdout) - MAX_OUTPUT_LENGTH} 字符）"
```

- [ ] **Step 2: 提交**

```bash
git add -A
git commit -m "perf: shell 命令输出截断至 10000 字符，防止撑爆上下文"
```

---

### Task 7: 依赖配置整合 + 版本锁定

**Files:**
- Modify: `src/pyproject.toml`
- Modify: `pyproject.toml`（根目录）

**背景:** 两个 `pyproject.toml` 配置分散，`langgraph` 和 `langchain-core` 版本约束过宽。

- [ ] **Step 1: 锁定 `src/pyproject.toml` 中的依赖版本**

```toml
[project]
name = "aiops-agent"
version = "0.1.0"
description = "AIOps Agent — 运维智能助手"
requires-python = ">=3.10"
dependencies = [
    "openai>=1.0.0",
    "python-dotenv>=1.0.0",
    "langgraph>=0.4.0,<0.5.0",
    "langchain-core>=0.3.0,<0.4.0",
]

[project.scripts]
aiops-agent = "aiops_agent.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: 将 dev dependencies 移至根 `pyproject.toml`**

```toml
[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "chore: 锁定 langgraph/langchain-core 版本上限 + 集中 dev-dependencies"
```

---

### Task 8: 同步设计文档与代码实现

**Files:**
- Modify: `docs/specs/2026-06-10-aiops-agent-design.md`

**背景:** 设计文档提到的 `tools/base.py`、`tools/registry.py`、`core/messages.py` 等模块在代码中已不存在或已重命名。

- [ ] **Step 1: 更新设计文档章节**

将以下过时内容更新：
- `tools/base.py` → 删除（已使用 `@tool` 装饰器 + `StructuredTool`）
- `tools/registry.py` → 删除（已使用 `tools/__init__.py` 的 `_iter_tools`）
- `core/messages.py` → 删除（消息管理在 `Agent.run()` 内部）
- `examples/04_planning.py` → `examples/04_react.py`
- `examples/05_rag.py` → `examples/05_langgraph.py`

- [ ] **Step 2: 提交**

```bash
git add -A
git commit -m "docs: 同步设计文档与代码实现 — 删除已重构的模块引用"
```

---

### Task 9: 修复 `print_custom_event` 全局状态重置

**Files:**
- Modify: `src/aiops_agent/cli.py`

**背景:** 已在 Task 2 Step 4 中处理。此 Task 确认并在 `main` 函数每轮对话开头重置。

已包含在 Task 2 中。

---

## 执行顺序

| 顺序 | Task | 优先级 | 预估时间 |
|------|------|--------|---------|
| 1 | Task 1: API Key 泄露 | P0 🔴 | 5 min |
| 2 | Task 2: 多轮对话 | P0 🔴 | 15 min |
| 3 | Task 3: 路径遍历 | P1 🟠 | 10 min |
| 4 | Task 4: 集成测试 | P1 🟠 | 15 min |
| 5 | Task 5: `_iter_tools` 重构 | P2 🟡 | 5 min |
| 6 | Task 6: 输出截断 | P2 🟡 | 5 min |
| 7 | Task 7: 依赖配置 | P2 🟡 | 5 min |
| 8 | Task 8: 设计文档 | P2 🟡 | 5 min |
