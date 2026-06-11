# 多 Agent 架构重构 — 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Agent 与工具解耦，Agent 直接接受 `tools: list[Tool]` 参数；去掉全局 `ToolRegistry`；去掉 Agent 内部的 plan 节点；改为配置驱动的多 Agent 图编排。

**Architecture:** Agent 通用化（name + system_prompt + tools），不再区分 plan/work 模式。LangGraph StateGraph 作为编排器，不同的 Agent 实例作为不同节点。Agent 角色和流转关系由配置定义。

**Tech Stack:** Python 3.10+, LangGraph, ToolNode

---

## 文件变更

```
src/aiops_agent/
├── core/
│   └── agent.py              # MODIFIED — 去掉 plan 节点 + ToolRegistry → tools 参数
├── tools/
│   ├── __init__.py            # MODIFIED — 去掉 ToolRegistry 导出
│   ├── registry.py            # DELETED
│   └── ...                    # 其他工具文件不变
├── config.py                  # MODIFIED — 新增 agent 配置字段（可选）
├── cli.py                     # MODIFIED — 构建图，创建多 Agent
├── __init__.py                 # MODIFIED — 版本号 v0.4.0
```

---

### Task 1: Agent 重构 — 去掉 plan 节点，tools 参数替换 ToolRegistry

**Files:**
- Modify: `d:\workspace\aiops-agent\src\aiops_agent\core\agent.py`

改动：
- `Agent.__init__` 新增 `name`, `system_prompt`, `tools: list[Tool]` 参数，去掉 `tool_registry` 参数
- `_build_graph` 去掉 plan 节点，只保留 `call_model` → `ToolNode` → `call_model` 循环
- `AgentState` 去掉 `plan_done` 字段
- 工具定义从传入的 `tools` 列表生成，不再从 `tool_registry` 读取
- `_get_tool_descriptions` 改为从 `self._tool_funcs` 生成

**Agent 类签名:**

```python
class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: BaseLLM,
        tools: list[Tool],
        config: Config,
        memory: Memory | None = None,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = llm
        self.tools = tools
        self.config = config
        self.memory = memory
        self._tool_funcs = [_make_tool_func(t) for t in tools]
        self._tool_defs = [t.to_openai_tool() for t in tools] or None
        self._graph = self._build_graph()
```

**内部图结构:**

```python
def _build_graph(self):
    builder = StateGraph(AgentState)
    builder.add_node("call_model", self._call_model)
    builder.add_node("tools", ToolNode(self._tool_funcs))
    builder.set_entry_point("call_model")
    builder.add_conditional_edges(
        "call_model", self._should_continue,
        {"continue": "tools", "end": END},
    )
    builder.add_edge("tools", "call_model")
    return builder.compile()
```

**StateAgent 去掉 plan_done:**

```python
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tool_round: int
    max_rounds: int
    events: Annotated[list, operator.add]
```

**Agent.run() 去掉 plan 相关逻辑:**

```python
def run(self, user_input: str, history: list[dict] | None = None):
    ...
    state: AgentState = {
        "messages": base,
        "tool_round": 0,
        "max_rounds": self.config.max_tool_rounds,
        "events": [],
    }
    ...
```

1. 重写 `agent.py`
2. 测试简单对话和工具调用
3. 提交: `git commit -m "refactor: Agent 去掉 plan 节点 + ToolRegistry，改为 name/system_prompt/tools 参数"`

---

### Task 2: 删除 ToolRegistry

**Files:**
- Delete: `d:\workspace\aiops-agent\src\aiops_agent\tools\registry.py`
- Modify: `d:\workspace\aiops-agent\src\aiops_agent\tools\__init__.py`

**tools/__init__.py:**

```python
from .base import Tool, ToolResult
from .shell import ShellTool

__all__ = ["Tool", "ToolResult", "ShellTool"]
```

1. 删除 `registry.py`，更新 `__init__.py`
2. 提交: `git commit -m "refactor: 删除 ToolRegistry（Agent 直接接受 tools 列表）"`

---

### Task 3: 重构 CLI — 配置驱动多 Agent 图

**Files:**
- Modify: `d:\workspace\aiops-agent\src\aiops_agent\cli.py`

CLI 中不再创建单个 Agent 然后循环调用 `agent.run()`。改为：

1. 定义 Agent 配置
2. 创建多个 Agent 实例
3. 构建 LangGraph 图
4. 运行图

```python
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage

# 全局状态
class AppState(TypedDict):
    messages: Annotated[list, operator.add]
    current_agent: str

# Agent 配置
AGENT_DEFS = {
    "planner": AgentDef(
        system_prompt="你是一个 AIOps 运维规划专家。分析用户任务，制定执行计划，然后交给执行专家。",
        tools=[],
    ),
    "worker": AgentDef(
        system_prompt="你是一个 AIOps 运维执行专家。按计划执行运维操作，完成后给出最终报告。",
        tools=["shell"],
    ),
}

TOOL_MAP = {
    "shell": ShellTool(),
}

def build_agent_graph(config, llm):
    builder = StateGraph(AppState)

    # 创建 Agent 实例
    agents = {}
    for name, adef in AGENT_DEFS.items():
        tools = [TOOL_MAP[t] for t in adef.tools]
        agents[name] = Agent(
            name=name,
            system_prompt=adef.system_prompt,
            llm=llm,
            tools=tools,
            config=config,
        )

    # 添加节点：每个 Agent 作为一个节点
    for name, agent in agents.items():
        def make_node(n=name):
            def node_fn(state: AppState) -> dict:
                # 提取用户输入
                user_msg = next(
                    (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
                )
                events = []
                for event in agent.run(user_msg):
                    events.append({"type": event.type, "content": event.content, "data": event.data})
                return {"messages": state["messages"], "events": events, "current_agent": n}
            return node_fn
        builder.add_node(name, make_node())

    builder.set_entry_point("planner")
    builder.add_edge("planner", "worker")
    builder.add_edge("worker", END)

    return builder.compile(), agents
```

CLI 主循环改为：

```python
# 构建图
graph, agents = build_agent_graph(config, llm)

# 运行
state = {
    "messages": [SystemMessage(content=config.system_prompt), HumanMessage(content=user_input)],
    "current_agent": "",
}
for chunk in graph.stream(state):
    for node_name, updates in chunk.items():
        for evt in updates.get("events", []):
            print_event(AgentEvent(type=evt["type"], content=evt["content"], data=evt["data"]))
```

1. 重写 CLI
2. 测试 `/help` 等命令正常 + 多 Agent 图运行正常
3. 提交: `git commit -m "feat: CLI 改为配置驱动多 Agent 图编排"`

---

### Task 4: 更新版本号 + 清理

**Files:**
- Modify: `d:\workspace\aiops-agent\src\aiops_agent\__init__.py`

```python
__version__ = "0.4.0"
```

1. 更新版本号
2. 提交: `git commit -m "chore: v0.4.0 — 多 Agent 架构"`

---

### Task 5: 端到端测试

1. `cd src && python -m aiops_agent` → CLI 正常启动，Banner 显示
2. 输入"查下磁盘" → 走 planner → worker 流程，正常执行 `df -h` 并输出报告
3. 输入 `/help` → 命令列表正常
4. 本例示例 01-05 仍然正常（他们使用独立的 LLM 调用，不依赖 Agent 类）

---

## 向后兼容

- `Agent.__init__` 签名变了（`tool_registry` → `tools`），但 CLI 是唯一调用者
- 教学示例不依赖 Agent 类，完全不受影响
- `ToolRegistry` 被删除，但项目中未被外部依赖

## 自检

### Spec 覆盖

| Spec 需求 | Task |
|-----------|------|
| Agent 通用化（name + system_prompt + tools） | Task 1 |
| 去掉 plan 节点 | Task 1 |
| 去掉 ToolRegistry | Task 2 |
| 配置驱动多 Agent 图 | Task 3 |
| 版本号更新 | Task 4 |

### 类型一致性

- `Agent.__init__` 的 `tools: list[Tool]` 在 Task 1 定义，Task 3 使用
- `AgentState` 去掉 `plan_done` 后，`run()` 中不再引用
- `AGENT_DEFS` 中 `tools` 列表引用 `TOOL_MAP` 中的 key
