# 知识库检索嵌入 aiops_agent

## 背景

教学示例 06_rag 实现了 RAG 知识库问答，但生产环境的 `aiops_agent` 项目（`src/aiops_agent/`）尚不具备知识库检索能力。当前 Agent 只能凭 LLM 训练数据回答运维问题，无法引用公司内部规范。

## 设计目标

将知识库检索以混合模式嵌入 aiops_agent：
1. **`retrieve_knowledge` 作为标准工具** — 注入到 worker 的工具列表中，Agent 可自主调用
2. **planner 启动时自动注入相关上下文** — 根据任务自动检索知识文档，帮助规划更精准
3. **复用现有基础设施** — Chroma 向量库、Embedding API 配置、知识库文档

## 方案：混合模式（Tool + 自动注入）

### 架构变更

```
之前:                        之后:
planner → worker             planner → worker (+ retrieve_knowledge tool)
  tools: SubmitPlan            tools: SubmitPlan
  worker tools: shell/read     启动时自动注入相关知识
  (无知识库能力)               worker tools: shell/read/retrieve_knowledge
```

### 文件变更

| 文件 | 变更 | 说明 |
|------|------|------|
| `src/aiops_agent/tools/knowledge_tool.py` | 新增 | 封装 retrieve_knowledge 为 StructuredTool |
| `src/aiops_agent/tools/__init__.py` | 修改 | 注册新工具 |
| `src/aiops_agent/agents/worker.py` | 修改 | tools 列表加入 retrieve_knowledge |
| `src/aiops_agent/graph/complex.py` | 修改 | planner 节点启动时自动注入知识 |
| `knowledge_base/` | 迁移 | 从 examples/ 移到项目根目录 |
| `.env` | 已有 | 复用 EMBEDDING_* 配置 |
| `.gitignore` | 修改 | chroma_db 路径改为 .aiops_data/chroma_db/ |

### 实现细节

#### 1. knowledge_tool.py

- 使用 Chroma 持久化客户端，路径 `.aiops_data/chroma_db/`
- Embedding 通过 API 调用（复用 .env 的 EMBEDDING_API_KEY/BASE_URL/MODEL）
- 如知识库不存在（首次运行），返回空字符串并提示
- 支持批量建库：首次检测到 `knowledge_base/` 有 .md 文件但 chroma_db 不存在时，自动建库

#### 2. tools/__init__.py

```python
def get_tools() -> dict:
    return {
        "shell": shell_tool,
        "read_file": read_file_tool,
        "write_file": write_file_tool,
        "retrieve_knowledge": retrieve_knowledge_tool,
    }
```

#### 3. worker 注册

worker 的 tools 列表增加 `"retrieve_knowledge"`，使其可在执行过程中主动查知识库。

#### 4. planner 自动注入

在 `complex.py` 的 `_make_node` 中，planner 节点启动时自动检索任务相关文档，作为 SystemMessage 注入。

#### 5. 知识库文档

将 `examples/knowledge_base/` 迁移到项目根目录 `knowledge_base/`，包含已有的 4 份公司内部规范文档。

## 边界说明

- 首次运行：如果 knowledge_base 目录存在但 chroma_db 未初始化，自动建库
- 知识库不存在：retrieve_knowledge 返回空，不影响其他功能
- 依赖：chromadb（已有）、openai（已有，Embedding 复用 .env 配置）
