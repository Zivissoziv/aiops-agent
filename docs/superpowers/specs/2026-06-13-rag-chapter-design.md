# 第六章 RAG 教学设计

## 概述

- **示例文件:** `examples/06_rag.py`
- **知识库目录:** `examples/knowledge_base/`
- **核心概念:** Embedding、向量检索、RAG（Retrieval-Augmented Generation）
- **前置知识:** 已完成 01-05，理解基础 LLM 调用和 Tool Calling
- **新增依赖:** chromadb

## 设计原则

1. **代码即教程** — 与 01-05 风格一致，单文件多 Part 结构，中文注释解释"为什么"
2. **零配置可运行** — 内置运维知识库文档，`python 06_rag.py` 直接运行
3. **渐进式教学** — 从 Embedding → 建库 → 检索 → 结合 Agent，每个 Part 聚焦一个新概念
4. **最小依赖** — 仅新增 chromadb，Embedding 通过 OpenAI API 完成

## 文件结构

```
examples/
├── _common.py                  # [修改] 新增 create_embeddings()
├── 06_rag.py                   # [新增] RAG 教学示例
└── knowledge_base/             # [新增] 预置运维文档
    ├── 01_磁盘管理.md
    ├── 02_网络诊断.md
    ├── 03_日志分析.md
    └── 04_进程管理.md
```

## Part 详细设计

### Part 1: 为什么需要 RAG？（~30 行）

- 演示纯 LLM 回答运维问题的局限性 — 回答泛泛、缺乏针对性
- 引出 RAG 核心思想：检索 + 生成，让 LLM 先查资料再回答
- 对比图：无知识库 vs 有知识库的回答流程

### Part 2: Embedding — 文本向量化（~60 行）

- 概念：把文本变成向量（数字序列）
- 调用 OpenAI `text-embedding-3-small` API
- 余弦相似度计算，演示语义相近的文本向量距离更近
- 新增 `_common.py` 中的 `create_embeddings()` 函数

### Part 3: 构建知识库（~100 行）

- 递归加载 `knowledge_base/*.md`，解析 Markdown 标题
- 分块策略：按段落分块，每块 ~200 tokens，重叠 20 tokens
- 使用 Chroma 创建持久化向量数据库
- 打印建库统计：文档数、块数、向量维度

### Part 4: 检索问答（~80 行）

- 问题 → Embedding → Chroma Top-K 检索 → 拼接上下文 Prompt → LLM 生成
- 对比展示：有 RAG vs 无 RAG 的回答质量差异
- 让学生直观理解"检索增强生成"的工作原理

### Part 5: RAG + Agent（~80 行）

- 将检索封装为 `retrieve_knowledge` 工具
- Agent 在回答运维问题时自主调用知识库
- 展示 Agent 的思考过程和工具调用决策
- 为后续多工具协同做铺垫

## 预置知识库文档

| 文档 | 内容 | 约字数 |
|------|------|--------|
| 01_服务器命名规范.md | 命名规则、环境划分、安全规则 | 300 |
| 02_告警分级与处理流程.md | P0-P3 分级、应急流程、磁盘阈值 | 350 |
| 03_端口与中间件配置.md | 端口规范、版本标准、K8s 规范 | 300 |
| 04_日志收集与备份策略.md | 日志路径、保留策略、备份恢复 | 300 |

## _common.py 修改

新增 `create_embeddings(client, texts, model)` 函数，封装 OpenAI Embedding API。

## 边界说明

本示例聚焦最朴素的 RAG 流水线，**不涉及**：
- Reranking 重排序
- Hybrid Search（混合检索）
- Graph RAG / Agentic RAG 等进阶模式
- 生产级向量数据库（Milvus、Weaviate 等）
