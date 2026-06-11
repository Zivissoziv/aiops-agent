# AIOps Agent 优化 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 4 optimizations inspired by MokioAgent: FileEditTool, role-based tool sets, intent routing, TODO/NOTEPAD persistence

**Architecture:** All 4 are independent — file_tools.py gets a new tool, cli.py gets tool filtering and intent routing, new todo.py/notepad.py tools with persistence. Each task is self-contained.

**Tech Stack:** Python, langgraph, pathlib, rich (for CLI)

---

### Task 1: FileEditTool — 精确文本替换编辑工具

**Files:**
- Modify: `src/aiops_agent/tools/file_tools.py`
- Test: `tests/test_file_tools.py`

Add an `edit_file` tool that finds a unique `old_text` snippet in a file and replaces it with `new_text`. Follows the same workspace sandbox rules as `read_file`/`write_file`.

**Design:**
- `edit_file(path, old_text, new_text)` — 在文件中查找第一个（且唯一）匹配的 old_text，替换为 new_text
- 如果 old_text 匹配 0 次 → 返回错误
- 如果 old_text 匹配 >1 次 → 返回错误，提示用更多上下文
- 遵循 workspace 沙箱规则（同 read_file/write_file）
- 支持 `dry_run=True` 参数预览匹配而不实际写入

```python
@tool
def edit_file(path: str, old_text: str, new_text: str, dry_run: bool = False) -> str:
    """编辑文件中的文本（精确替换）。适用于修改配置、修复 bug、重构代码等。
    在 workspace 内的编辑直接执行，越界需用户确认。

    Args:
        path: 文件路径
        old_text: 要替换的原始文本（必须在文件中唯一匹配）
        new_text: 替换后的新文本
        dry_run: 是否仅预览匹配结果而不实际写入，默认 False
    """
    # workspace 边界检查
    if not _is_in_workspace(path):
        if _write_approval_handler:
            if not _write_approval_handler(path, f"编辑文件: {old_text[:50]}..."):
                return f"⚠️ 用户拒绝了编辑 {path}"
        else:
            return f"错误: 路径不在工作区内且未配置审批回调: {path}"

    try:
        p = Path(path)
        if not p.exists():
            return f"错误: 文件不存在: {path}"
        if not p.is_file():
            return f"错误: 不是文件: {path}"

        content = p.read_text(encoding="utf-8")
        count = content.count(old_text)

        if count == 0:
            return f"错误: 未找到要替换的文本:\n```\n{old_text[:200]}\n```"
        elif count > 1:
            return f"错误: 找到 {count} 处匹配，请提供更多上下文使 old_text 唯一匹配:\n```\n{old_text[:200]}\n```"

        if dry_run:
            return f"✅ 将替换 1 处匹配:\n```\n{old_text[:200]}\n```\n→\n```\n{new_text[:200]}\n```"

        new_content = content.replace(old_text, new_text, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"✅ 已编辑 {path}（替换 1 处，{len(old_text)} → {len(new_text)} 字符）"

    except PermissionError:
        return f"错误: 无权限编辑: {path}"
    except Exception as e:
        return f"错误: {e}"
```

Tests to add:
- `test_edit_existing` — 替换单处匹配
- `test_edit_not_found` — 文本不存在
- `test_edit_multiple_matches` — 多处匹配报错
- `test_edit_dry_run` — 预览不写入
- `test_edit_within_workspace` — workspace 内直接编辑
- `test_edit_outside_workspace_approved/denied` — 越界审批

---

### Task 2: 按 Agent 角色分配工具集

**Files:**
- Modify: `src/aiops_agent/cli.py`

Currently: `TOOL_MAP` 是所有工具的平铺字典，planner 和 worker 共享同一份工具集。planner 也能看到并调用 `shell`/`read_file`/`write_file`。

Change: 在 `build_graph` 中，为每个 agent 过滤其配置的工具列表，只绑定属于它的工具。

Current structure in `cli.py`:
```python
tools = [TOOL_MAP[t] for t in adef["tools"]]
```

This already reads `adef["tools"]` — so the filtering mechanism exists. Check what each agent's `tools` list contains in `agents/__init__.py`:

```python
# Current assumption — need to verify actual content
ALL_AGENTS = [
    {"name": "planner", "tools": ["shell", "read_file", "write_file", ...]},
    {"name": "worker", "tools": ["shell", "read_file", "write_file", ...]},
]
```

The fix: Remove execution tools from planner's tool list, keep only planning-relevant tools (or leave empty). Worker keeps all tools.

Steps:
1. Read `src/aiops_agent/agents/__init__.py` to see current tool assignments
2. Remove execution tools (`shell`, `read_file`, `write_file`, `edit_file`) from planner's tool list
3. Verify planner's system prompt already says "不要执行工具，只需要输出规划" — keep that as defense-in-depth

---

### Task 3: 意图路由 — 闲聊不进工作流

**Files:**
- Modify: `src/aiops_agent/cli.py`
- New: `src/aiops_agent/core/intent_router.py`

Add an intent routing step before entering the LangGraph workflow. If the query is "chat" (greetings, thanks, simple Q&A), reply directly with a lightweight LLM call. Only enter the planner→workflow pipeline for actual tasks.

**New file: `src/aiops_agent/core/intent_router.py`**

```python
"""意图路由 — 判断用户输入是闲聊还是工作任务。"""

INTENT_ROUTER_PROMPT = """你是一个意图分类器。判断用户最新输入属于哪一类：

- chat: 打招呼、感谢、询问身份、简单问答等不需要操作工作区的纯对话
- task: 需要读写文件、执行命令、查询系统信息、安装软件等任何需要工具操作的任务

只返回以下 JSON 格式（不要加其他内容）：
{"route": "chat" | "task", "reason": "简短原因"}

如果不确定，选 task。"""


def classify_intent(llm, user_input: str, history_summary: str = "") -> dict:
    """判断用户意图，返回 {"route": ..., "reason": ...}"""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=INTENT_ROUTER_PROMPT),
    ]
    if history_summary:
        messages.append(SystemMessage(content=f"对话历史摘要: {history_summary}"))
    messages.append(HumanMessage(content=user_input))

    import json
    try:
        response = llm.invoke(messages)
        parsed = json.loads(response.content.strip())
        route = parsed.get("route", "task")
        reason = parsed.get("reason", "")
        if route not in ("chat", "task"):
            route = "task"
        return {"route": route, "reason": reason}
    except Exception:
        return {"route": "task", "reason": "路由解析失败，默认走工作流"}
```

In `cli.py main()`, wrap the user input handling:

```python
# ── 意图路由 ──
intent = classify_intent(llm, user_input, memory.working.get_summary_context())
if intent["route"] == "chat":
    # 轻量回复，不走 graph
    response = llm.invoke([
        SystemMessage(content="你是一个 AIOps 助手。直接回复用户的问题，简洁友好。不需要提及工作区、工具或文件操作。"),
        HumanMessage(content=user_input),
    ])
    print(f"\n{response.content}")
    memory.add_message({"role": "user", "content": user_input})
    memory.add_message({"role": "assistant", "content": response.content or ""})
    continue
# 否则走正常的工作流...
```

---

### Task 4: TODO.md / NOTEPAD.md 持久化

**Files:**
- Create: `src/aiops_agent/tools/notepad_tool.py`
- Create: `src/aiops_agent/tools/todo_tool.py`
- Modify: `src/aiops_agent/tools/__init__.py`

Agent 可以在 workspace 内维护 TODO.md 和 NOTEPAD.md 文件，这些文件对 agent 可见，也可以在工作流使用。

**TODO Tool:** Agent 读取 TODO.md 了解当前待办，更新进度。
**NOTEPAD Tool:** Agent 读取 NOTEPAD.md 查看研究笔记，追加新笔记。

Both tools follow workspace sandbox rules (files are inside workspace, so no boundary approval needed).

```python
# src/aiops_agent/tools/notepad_tool.py
from pathlib import Path
from langchain_core.tools import tool

NOTEPAD_FILE = "NOTEPAD.md"

@tool
def read_notepad() -> str:
    """读取工作区笔记（NOTEPAD.md），包含研究发现和决策记录。"""
    p = Path(NOTEPAD_FILE)
    if not p.exists():
        return "(笔记为空)"
    return p.read_text(encoding="utf-8")

@tool
def append_notepad(heading: str, content: str) -> str:
    """向工作区笔记追加内容。heading 是标题，content 是笔记内容。"""
    from datetime import datetime
    p = Path(NOTEPAD_FILE)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n## {heading}\n\n_{timestamp}_\n\n{content}\n"
    with open(p, "a", encoding="utf-8") as f:
        f.write(entry)
    return f"✅ 已追加笔记: {heading}"


# src/aiops_agent/tools/todo_tool.py (optional — planner already handles todos via [TODO] tags)
```

Register these tools in `tools/__init__.py` auto-discovery.

---

### Execution Order

1. **Task 1** — FileEditTool（独立，不依赖其他改动）
2. **Task 2** — 按角色分配工具集（只改 agents/__init__.py）
3. **Task 3** — 意图路由（新增文件 + 改 cli.py）
4. **Task 4** — TODO/NOTEPAD 持久化（新增文件 + 注册）

Run full test suite after each task.
