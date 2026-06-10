# AIOps Agent 🤖

一个**由浅入深**的 AIOps Agent 学习与实战项目。

## 项目结构

```
├── examples/    # 📚 教学示例 — 每文件一个概念，可独立运行
└── src/         # 🚀 实战项目 — 可部署的 AIOps Agent
```

## 快速开始

```bash
# 1. 安装 uv（如果未安装）
# pip install uv

# 2. 复制并配置环境变量
cp .env.example .env
# 编辑 .env 填入你的 API Key

# 3. 运行教学示例
uv run python examples/01_simple_chat.py

# 4. 安装并运行实战项目
cd src && uv sync
uv run aiops-agent
```

## 教学路线

| 示例 | 学习内容 | 状态 |
|------|---------|------|
| 01_simple_chat | 基础 LLM 对话 | ✅ |
| 02_tool_calling | 工具调用机制 | ✅ |
| 03_memory | 对话记忆管理 | 📝 规划中 |
| 04_planning | 多步骤规划 | 📝 规划中 |
| 05_rag | 知识库问答 | 📝 规划中 |

## 许可证

MIT
