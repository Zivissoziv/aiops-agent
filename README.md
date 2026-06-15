# AIOps Agent 🤖

一个**由浅入深**的 AIOps Agent 学习与实战项目。

从基础 LLM 对话开始，逐步实现工具调用、记忆管理、多步骤规划、RAG 知识库等核心能力，最终构建一个可部署的运维智能助手。

## 项目结构

```
├── examples/          # 📚 教学示例 — 每文件一个概念，可独立运行
│   ├── 01_simple_chat.py     # 基础 LLM 对话
│   ├── 02_tool_calling.py    # 工具调用机制
│   ├── 03_memory.py          # 三层记忆系统
│   ├── 04_react.py           # ReAct 多步骤规划
│   ├── 05_langgraph.py       # LangGraph 状态机
│   ├── 06_rag.py             # RAG 知识库问答
│   ├── 07_skills.py          # 技能系统（Skill System）
│   └── skills/               # 📦 技能目录 — SKILL.md + 工具脚本
├── src/               # 🚀 实战项目 — 可部署的 AIOps Agent
│   ├── aiops_agent/
│   │   ├── cli.py            # CLI 交互界面
│   │   ├── config.py         # 配置管理
│   │   ├── core/             # Agent 核心引擎（LangGraph 状态机）
│   │   ├── core/agent_old.py # 旧版 Agent 实现（参考）
│   │   ├── llm/              # 多 Provider 适配
│   │   ├── memory/           # 三层记忆（工作/情景/核心）
│   │   └── tools/            # 运维工具集
│   └── pyproject.toml        # uv 打包配置
├── docs/
│   └── specs/         # 📋 设计文档
└── .env.example
```

## 快速开始

### 1. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key 和模型配置
```

支持任何 OpenAI 兼容接口（OpenAI / DeepSeek / 通义千问 / 硅基流动 等）。

### 2. 运行教学示例

```bash
# 安装依赖
pip install openai python-dotenv

# 示例 01: 基础 LLM 对话
python examples/01_simple_chat.py

# 示例 02: 工具调用
python examples/02_tool_calling.py

# 示例 03: 三层记忆系统
python examples/03_memory.py

# 示例 04: ReAct 多步骤规划
python examples/04_react.py

# 示例 05: LangGraph 状态机
python examples/05_langgraph.py

# 示例 06: RAG 知识库问答
python examples/06_rag.py

# 示例 07: 技能系统
python examples/07_skills.py
```

### 3. 运行实战项目

```bash
cd src
pip install -e .
aiops-agent
```

或直接通过 Python 模块运行：

```bash
cd src
python -m aiops_agent
```

### CLI 命令

进入 CLI 后支持以下命令：

| 命令 | 说明 |
|------|------|
| `/help` | 显示帮助信息 |
| `/exit` | 退出程序 |
| `/tools` | 查看可用工具 |
| `/memory` | 查看三层记忆状态 |
| `/workspace` | 查看当前 Workspace ID 和路径 |
| `/remember <事实>` | 添加核心记忆（跨会话持久） |
| `/forget <事实>` | 删除核心记忆 |
| `/core` | 查看核心记忆列表 |
| `/clear` | 清空当前会话记忆 |
| `/config` | 查看当前配置 |

### Workspace 沙箱

每次启动自动创建独立的 Workspace 目录，隔离会话数据：

```
.aiops_data/
├── core_memory.json              # 核心记忆 — 全局共享
└── workspaces/
    └── 20260611_143025/          # 每次启动自动创建
        ├── episodic_memory.json  # 情景记忆 — 按 Workspace 隔离
        └── ...                   # Agent 操作生成的文件
```

**沙箱规则：**

| 操作 | Workspace 内 | Workspace 外 |
|------|-------------|-------------|
| `read_file` | ✅ 直接读取 | ⚠️ 需用户审批 |
| `write_file` | ✅ 直接写入 | ⚠️ 需用户审批 |
| `shell` | ✅ 默认 cwd 在 Workspace | 现有风险分级管控 |

### ReAct 模式

在 `.env` 中设置 `REACT_ENABLED=true` 启用 ReAct 推理模式。
启用后 Agent 会输出显式的 `Thought:` 推理过程，适合需要多步推理的复杂运维任务。

## 教学路线

| 示例 | 学习内容 | 核心概念 | 状态 |
|------|---------|---------|------|
| 01_simple_chat | 基础 LLM 对话 | System Prompt、消息结构、流式输出 | ✅ |
| 02_tool_calling | 工具调用 | Function Calling、JSON Schema、Tool Loop | ✅ |
| 03_memory | 三层记忆系统 | 工作记忆、情景记忆、核心记忆、压缩机制 | ✅ |
| 04_react | ReAct 多步骤规划 | Thought、Action、Observation 循环 | ✅ |
| 05_langgraph | LangGraph 状态机 | StateGraph、Node、Conditional Edge | ✅ |
| 06_rag | 知识库问答 | Embedding、向量检索、RAG | ✅ |
| 07_skills | 技能系统 | SKILL.md、技能加载器、match_keywords 自动匹配 | ✅ |

## 技术栈

- **语言:** Python 3.10+
- **LLM SDK:** OpenAI Python SDK（兼容 DeepSeek 等）
- **包管理:** uv / pip
- **打包:** hatchling

## 许可证

MIT
