# AIOps Agent 项目设计文档

> 日期: 2026-06-10
> 状态: 初稿

## 1. 项目概述

### 1.1 目标

构建一个**可学习、可实战**的 AIOps Agent 项目。通过 `examples/` 目录循序渐进地教学 Agent 核心概念，通过 `src/` 目录交付一个可部署的运维智能助手。

### 1.2 核心理念

- **教学先行** — 每个概念都有独立可运行的示例文件，注释详细，代码精简
- **实战驱动** — 教学示例的知识点直接映射到实战项目的模块中
- **渐进演进** — 从 CLI 对话开始，逐步叠加工具调用、记忆、规划、RAG 等能力
- **多 Provider** — 通过统一接口支持多种 LLM（DeepSeek、OpenAI 等）

---

## 2. 项目结构

```
aiops-agent/
├── examples/                          # 📚 教学示例
│   ├── 01_simple_chat.py              #   基础对话 — LLM 调用入门
│   ├── 02_tool_calling.py             #   工具调用 — Function Calling 机制
│   ├── 03_memory.py                   #   记忆管理 — 对话历史与上下文
│   ├── 04_planning.py                 #   多步骤规划 — ReAct / Plan-and-Execute
│   └── 05_rag.py                      #   知识库问答 — RAG 检索增强生成
├── src/                               # 🚀 实战项目
│   ├── aiops_agent/
│   │   ├── __init__.py
│   │   ├── __main__.py                #   python -m aiops_agent 入口
│   │   ├── cli.py                     #   CLI 交互界面
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── agent.py               #   Agent 核心引擎
│   │   │   └── messages.py            #   消息管理
│   │   ├── llm/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                #   LLM 抽象基类
│   │   │   ├── openai_compatible.py   #   OpenAI 兼容接口适配
│   │   │   └── factory.py             #   Provider 工厂
│   │   ├── tools/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                #   工具基类
│   │   │   ├── shell.py               #   Shell 命令执行
│   │   │   └── registry.py            #   工具注册中心
│   │   ├── memory/
│   │   │   ├── __init__.py
│   │   │   └── conversation.py        #   对话记忆
│   │   └── config.py                  #   配置管理
│   └── pyproject.toml                 #   uv 项目配置
├── .env                               # 环境变量（不提交）
├── .env.example                       # 环境变量模板
├── .gitignore
├── README.md                          # 项目总览
├── LICENSE
└── docs/
    └── specs/
        └── 2026-06-10-aiops-agent-design.md    # 本文档
```

---

## 3. 教学示例设计 (examples/)

### 3.1 设计原则

- **可独立运行** — 每个文件 `python examples/XX_xxx.py` 即可运行
- **最小依赖** — 仅依赖核心库，不引入复杂框架
- **注释驱动** — 代码即教程，关键行都有中文注释解释"为什么"
- **渐进难度** — 从最简单的 LLM 调用开始，逐步叠加概念

### 3.2 示例规划

| # | 文件 | 核心概念 | 对应实战模块 | 前置知识 |
|---|------|---------|-------------|---------|
| 01 | `01_simple_chat.py` | 基础 LLM 调用、System Prompt、多轮对话 | cli.py | 无 |
| 02 | `02_tool_calling.py` | Function Calling、工具定义、工具执行循环 | tools/ | 01 |
| 03 | `03_memory.py` | 对话历史管理、窗口/摘要记忆策略 | memory/ | 01 |
| 04 | `04_planning.py` | ReAct 模式、任务分解、子步骤执行 | core/agent.py | 02, 03 |
| 05 | `05_rag.py` | 文档加载、向量化、检索、增强生成 | （后续） | 01 |

### 3.3 `01_simple_chat.py` 详细设计

```python
"""
01_simple_chat.py — 基础 LLM 对话

学习目标:
  1. 理解 LLM 的基本调用方式
  2. 理解 System Prompt 的作用
  3. 理解多轮对话的消息结构

运行方式:
  python examples/01_simple_chat.py

前置条件:
  在项目根目录配置 .env 文件
"""

# 1. 加载环境变量
# 2. 读取模型配置（provider、model name、api key、base url）
# 3. 根据配置文件选择 LLM 客户端
# 4. 创建 System Message
# 5. 进入对话循环：
#    a. 接收用户输入
#    b. 构造消息列表（system + 历史 + 最新用户输入）
#    c. 调用 LLM
#    d. 流式输出回复
#    e. 保存到历史
```

---

## 4. 实战项目设计 (src/)

### 4.1 架构概览

```
用户输入 (CLI)
    │
    ▼
┌─────────────────────────────────────────────┐
│              cli.py (交互界面)               │
│    Prompts 用户输入，显示 Agent 输出           │
└─────────────┬───────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────┐
│        core/agent.py (Agent 核心引擎)        │
│    接收消息 → 调用 LLM → 处理工具调用 → 循环  │
└──┬──────────────┬──────────────┬────────────┘
   │              │              │
   ▼              ▼              ▼
┌───────┐  ┌───────────┐  ┌──────────────┐
│ llm/   │  │ tools/    │  │ memory/      │
│ 多     │  │ Shell 等  │  │ 对话记忆管理  │
│Provider│  │ 工具注册  │  │              │
└───────┘  └───────────┘  └──────────────┘
```

### 4.2 Agent 核心循环 (core/agent.py)

Agent 的核心是一个 **"思考-行动-观察"** 循环：

```
1. 接收用户消息
2. 构建完整的消息列表（system + 记忆历史 + 当前输入）
3. 调用 LLM（带工具定义）
4. LLM 返回:
   a. 文本回复 → 直接输出给用户
   b. 工具调用请求 → 执行工具 → 将结果追加到消息列表 → 回到步骤 3
5. 保存对话到记忆
```

关键设计决策：
- **最大工具调用轮次**：默认 10 次，防止无限循环
- **错误处理**：工具执行失败时，将错误信息返回给 LLM 并让它决定下一步
- **流式输出**：LLM 文本回复流式显示，工具调用过程透明展示给用户

### 4.3 LLM 多 Provider 适配 (llm/)

```python
# llm/base.py — 抽象基类
class BaseLLM(ABC):
    @abstractmethod
    def invoke(self, messages: list, tools: list = None) -> dict: ...
    @abstractmethod
    def stream(self, messages: list, tools: list = None) -> Generator: ...

# llm/openai_compatible.py — OpenAI 兼容接口适配
class OpenAICompatibleLLM(BaseLLM):
    """支持 OpenAI、DeepSeek、通义千问等所有兼容 OpenAI 接口的提供商"""
    ...

# llm/factory.py — 工厂
def create_llm() -> BaseLLM:
    """根据 .env 配置创建对应的 LLM 实例"""
```

环境变量设计：
```
# .env
LLM_PROVIDER=openai_compatible    # 当前仅此一种，后续可扩展
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.deepseek.com/v1
OPENAI_MODEL=deepseek-chat
```

### 4.4 工具系统 (tools/)

```python
# tools/base.py
class Tool(ABC):
    name: str              # 工具名称
    description: str       # 工具描述（给 LLM 看）
    parameters: dict       # JSON Schema 参数定义

    @abstractmethod
    def execute(self, **kwargs) -> str: ...

# tools/shell.py
class ShellTool(Tool):
    name = "shell"
    description = "在服务器上执行 Shell 命令"
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的命令"}
        },
        "required": ["command"]
    }

# tools/registry.py
class ToolRegistry:
    """管理所有可用工具，生成 LLM 可理解的工具定义"""
    def list_tool_defs(self) -> list: ...    # 转成 OpenAI tool format
    def execute_tool(self, name: str, args: dict) -> str: ...
```

### 4.5 配置管理 (config.py)

从 `.env` 读取配置，提供类型安全和默认值：

```python
@dataclass
class Config:
    llm_provider: str = "openai_compatible"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    system_prompt: str = "你是一个 AIOps 运维助手..."
    max_tool_rounds: int = 10
```

---

## 5. 第一阶段 (MVP) 实现范围

### 5.1 包含内容

| 模块 | 文件 | 说明 |
|------|------|------|
| 示例 01 | `examples/01_simple_chat.py` | 基础 LLM 对话 |
| 示例 02 | `examples/02_tool_calling.py` | 工具调用演示 |
| 实战 CLI | `src/aiops_agent/cli.py` | CLI 交互界面 |
| Agent 核心 | `src/aiops_agent/core/agent.py` | 工具调用循环 |
| LLM 适配 | `src/aiops_agent/llm/` | 多 Provider 支持 |
| 工具系统 | `src/aiops_agent/tools/` | Shell 工具 |
| 配置 | `src/aiops_agent/config.py` | 配置管理 |
| 打包 | `src/pyproject.toml` | uv 配置 |

### 5.2 不包含内容（后续阶段）

- 记忆管理（第二阶段）
- 多步骤规划（第三阶段）
- RAG 知识库（第四阶段）
- Web UI（后续）
- 监控集成（后续）

### 5.3 技术选型

| 项 | 选择 | 理由 |
|----|------|------|
| 语言 | Python 3.10+ | AI/LLM 生态最成熟 |
| 包管理 | uv | 现代、快速 |
| LLM SDK | openai Python SDK | 兼容性最好，DeepSeek 等均支持 |
| 依赖 | 最小化 | 仅 openai、python-dotenv |

---

## 6. 后续阶段展望

| 阶段 | 教学示例 | 实战新增 | 核心学习点 |
|------|---------|---------|-----------|
| 二 | `03_memory.py` | memory/ 模块 | Token 管理、窗口策略、摘要策略 |
| 三 | `04_planning.py` | core/agent.py 增强 | ReAct、Task Decomposition |
| 四 | `05_rag.py` | 知识库工具 | Embedding、向量检索、RAG |
| 五 | - | Web UI | FastAPI + 前端 |
| 六 | - | 监控集成 | Prometheus/Grafana API |

---

## 7. CLI 交互设计

第一阶段 CLI 交互效果：

```
$ uv run aiops-agent

╔══════════════════════════════════════════╗
║        AIOps Agent v0.1.0               ║
║  模型: deepseek-chat                     ║
║  工具: shell                             ║
║  输入 /help 查看命令, /exit 退出          ║
╚══════════════════════════════════════════╝

你: 查一下当前磁盘使用情况
助手: 🔧 正在使用工具: shell(command="df -h")
─── 输出 ──────────────────────────
Filesystem      Size  Used Avail Use% Mounted on
/dev/sda1       100G   45G   55G  45% /
─── 结束 ──────────────────────────

服务器总的磁盘空间是 100G，已用 45G（45%），剩余 55G，使用率正常。

你: 看看最近的系统日志有没有错误
...
```

---

## 8. 项目规范

### 8.1 代码规范

- 使用 `ruff` 作为 linter 和 formatter（在后续阶段引入）
- 类型注解：所有函数参数和返回值都需要类型注解
- 注释：中文注释，重点解释"为什么"而不是"是什么"

### 8.2 Git 规范

- main 分支始终可运行
- 每完成一个阶段打 tag: `v0.1.0` (第一阶段), `v0.2.0` (第二阶段) ...
- Commit message 使用中文，清晰描述改动

### 8.3 README 结构

```markdown
# AIOps Agent

## 项目介绍

## 快速开始

## 教学路线

| 示例 | 学习内容 |
|------|---------|
| 01_simple_chat | 基础 LLM 调用 |
| ... | ... |

## 实战项目

## 项目结构

## 许可证
```
