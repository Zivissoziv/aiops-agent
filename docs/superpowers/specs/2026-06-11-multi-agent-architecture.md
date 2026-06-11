# 多 Agent 架构设计文档

> 日期: 2026-06-11
> 状态: 初稿

## 1. 动机

### 1.1 当前问题

当前 `Agent` 类是一个"全能单体":
- 内部包含 `plan` + `call_model` + `tools` 三个节点
- 所有工具挂在一个全局 `ToolRegistry` 中
- 所有 Agent 实例共享同一套工具

这导致:
- **权限不清晰**: PlanAgent 不需要工具但也看得到工具
- **扩展困难**: 加一个新角色（如日志分析 Agent）要改 Agent 内部逻辑
- **配置耦合**: Agent 的职责和工具绑定在代码里，不是配置里

### 1.2 目标

- 工具与 Agent 解耦: 每个 Agent 有独立的工具列表
- 配置驱动: Agent 的角色、工具、system prompt 由配置定义
- 图即编排: LangGraph StateGraph 本身就是编排器，不需要额外的 Orchestrator 类

---

## 2. 架构

```
                    ┌──────────────────────────────────────┐
                    │           LangGraph StateGraph        │
                    │         （即编排器本身）                │
                    │                                      │
                    │   ┌────────────┐                     │
                    │   │ plan_agent │  Agent(tools=[])     │
                    │   └──────┬─────┘                     │
                    │          │ 计划完成                    │
                    │          ▼                            │
                    │   ┌────────────┐                     │
                    │   │work_agent  │  Agent(tools=[...])  │
                    │   └──┬──────┬──┘                     │
                    │      │ 有   │ 无                      │
                    │   ┌──▼────┐  │                       │
                    │   │ Tool  │  │                       │
                    │   │ Node  │  │                       │
                    │   └──┬────┘  │                       │
                    │      │       │                        │
                    │      └───┬───┘                        │
                    │          ▼                            │
                    │       ┌─────┐                         │
                    │       │ END │                         │
                    │       └─────┘                         │
                    └──────────────────────────────────────┘
```

### 2.1 核心原则

1. **Agent 是通用的**: Agent 不再区分"plan 模式"或"work 模式"。区别只在于传入的 `system_prompt` 和 `tools` 不同
2. **图就是编排**: Agent 之间的流转顺序由 StateGraph 的边定义。加 Agent = 加节点和边
3. **配置驱动**: Agent 的角色、工具、system prompt 写在配置里，修改配置即可调整行为

### 2.2 对比当前架构

| 维度 | 当前 | 新架构 |
|------|------|--------|
| Agent 职责 | 内置 plan + execute | 通用，由配置决定角色 |
| 工具挂载 | 全局 ToolRegistry | 每个 Agent 独立的 tools 列表 |
| 扩展 Agent | 改 agent.py 代码 | 加一条配置 + 注册工具 |
| 编排 | Agent 内部 plan 节点 | LangGraph 图边定义 |
| 权限 | 所有工具有效 | Agent 只看得到自己的工具 |

---

## 3. 组件设计

### 3.1 Agent 类（重构）

```python
class Agent:
    """通用 Agent，由配置决定角色和行为。"""

    def __init__(
        self,
        name: str,
        system_prompt: str,
        llm: BaseLLM,
        tools: list[Tool],          # ← 从 ToolRegistry 改为直接传工具列表
        config: Config,
        memory: Memory | None = None,
    ):
        ...
```

**改动点:**
- 新增 `name` 参数: Agent 名称（如 "planner", "worker"）
- 新增 `system_prompt` 参数: 角色提示词（不再从 config 读取）
- `tools` 替代 `tool_registry`: 直接传入工具实例列表
- 移除内部的 `plan` 节点: 规划能力由配置了空工具列表的 Agent 实例提供

**内部图结构（不变的部分）:**
```
call_model → ToolNode → call_model → ... → END
```
这个 think-act-observe 循环是通用逻辑，所有 Agent 共用。

### 3.2 Agent 注册表（新增）

```python
# 定义有哪些 Agent 角色
AGENTS = {
    "planner": AgentDef(
        system_prompt="你是一个 AIOps 运维规划专家。分析任务，制定执行计划。",
        tools=[],  # 无工具，只做规划
    ),
    "worker": AgentDef(
        system_prompt="你是一个 AIOps 运维执行专家。按计划执行运维操作。",
        tools=["shell"],  # 可执行命令
    ),
    "log_analyzer": AgentDef(
        system_prompt="你是一个日志分析专家。分析日志文件找出问题。",
        tools=["read_file", "grep_logs"],
    ),
}
```

### 3.3 工具工厂

```python
# 工具名 → 工具实例的映射
TOOL_MAP = {
    "shell": ShellTool(),
    "read_file": ReadFileTool(),
    "grep_logs": GrepLogsTool(),
}
```

### 3.4 图构建

```python
def build_graph(agents: dict[str, Agent], flow: list[tuple[str, str]]):
    """根据 Agent 定义和流转关系构建 StateGraph。

    Args:
        agents: {name: Agent 实例}
        flow: [("planner", "worker"), ...]  边定义
    """
    builder = StateGraph(GlobalState)
    for name, agent in agents.items():
        builder.add_node(name, agent)  # Agent 本身作为节点函数
    for src, dst in flow:
        builder.add_edge(src, dst)
    builder.set_entry_point(flow[0][0])
    return builder.compile()
```

这样加一个 Agent 就是加配置 + 注册工具，不改代码。

---

## 4. CLI 集成

CLI 启动时根据配置创建多个 Agent 实例，构建图，然后运行。

```python
def main():
    config = Config.from_env()
    llm = create_llm(config)

    # 注册所有工具
    tool_map = {"shell": ShellTool(), "read_file": ReadFileTool()}

    # 创建 Agent 实例
    agents = {}
    for name, adef in AGENTS.items():
        tools = [tool_map[t] for t in adef.tools]
        agents[name] = Agent(name=name, system_prompt=adef.system_prompt,
                             llm=llm, tools=tools, config=config)

    # 构建图
    graph = build_graph(agents, flow=[("planner", "worker")])

    # 运行
    state = graph.invoke(initial_state)
```

---

## 5. 扩展性示例

**场景: 新增一个日志分析 Agent**

```python
# 1. 注册工具
tool_map["grep_logs"] = GrepLogsTool()

# 2. 加一条配置
AGENTS["log_analyzer"] = AgentDef(
    system_prompt="你是一个日志分析专家...",
    tools=["read_file", "grep_logs"],
)

# 3. 改图的边
flow = [("planner", "worker"), ("worker", "log_analyzer")]
```

不需要改 Agent 类，不需要改工具基类。

---

## 6. 向后兼容

- CLI 默认行为不变（单个全能 Agent，tools 直接传入）
- `/multi` 命令启用多 Agent 模式
- 教学示例 01-05 不受影响
- `ToolRegistry` 类删除，工具直接通过列表传入 Agent

---

## 7. 后续可能的增强

- 条件边: 根据 plan_agent 的输出决定调用哪个 worker（如日志问题→log_analyzer）
- 并行执行: 多个 worker 并发运行
- 动态注入: 运行时根据任务类型选择 Agent
