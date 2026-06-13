"""
examples/06_rag.py — RAG 知识库问答

学习目标:
  1. 理解 RAG（Retrieval-Augmented Generation）的核心概念
  2. 理解 Embedding（文本向量化）和语义相似度
  3. 学会使用 Chroma 搭建向量知识库
  4. 掌握检索 → 增强 → 生成的完整 RAG 流水线
  5. 学会将 RAG 封装为 Agent 工具

运行方式:
  pip install chromadb
  python examples/06_rag.py

前置条件:
  已完成 02_tool_calling.py，理解工具调用的概念

核心概念:
  - Embedding: 将文本转为向量（数字序列），语义相近的文本向量距离更近
  - 余弦相似度: 衡量两个向量方向的接近程度（-1 到 1）
  - Chroma: 轻量级向量数据库，存储向量并提供相似度检索
  - RAG: 检索（Retrieve）→ 增强（Augment）→ 生成（Generate）的流水线
"""

import json
import math
from pathlib import Path
from typing import Any

from openai import OpenAI

from _common import load_config, create_embeddings
from _ui import console, title, subtitle, note, success, info, diagram, divider, wait_for_enter, make_table


# ============================================================
# 第一部分: 依赖检查
# ============================================================

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False


# ============================================================
# 配置加载
# ============================================================
config = load_config()
model = config["model"]

EMBEDDING_MODEL = config["embedding_model"]
embedding_client = OpenAI(
    api_key=config["embedding_api_key"],
    base_url=config["embedding_base_url"],
)
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量的余弦相似度。

    公式: cos(θ) = (A·B) / (||A|| × ||B||)

    返回值范围 [-1, 1]:
      - 1 → 方向完全相同（语义最相似）
      - 0 → 正交（不相关）
      - -1 → 方向相反（语义相反）
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_knowledge_docs(kb_dir: str | Path) -> list[dict[str, Any]]:
    """加载 knowledge_base 目录下的所有 .md 文件。"""
    docs = []
    kb_path = Path(kb_dir)

    for md_file in sorted(kb_path.glob("*.md")):
        source = md_file.name
        text = md_file.read_text(encoding="utf-8")

        title_name = source.replace(".md", "")
        for line in text.splitlines():
            if line.startswith("# "):
                title_name = line.lstrip("# ").strip()
                break

        docs.append({
            "source": source,
            "title": title_name,
            "content": text,
        })

    return docs


def chunk_document(doc: dict[str, Any], chunk_size: int = 200, overlap: int = 20) -> list[dict[str, Any]]:
    """将文档分块。

    分块是 RAG 的关键步骤:
      - 块太大 → 包含太多不相关信息，降低检索精度
      - 块太小 → 可能丢失上下文，LLM 无法理解
      - 块重叠 → 避免恰好切断重要信息
    """
    text = doc["content"]
    chunks = []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += ("\n\n" + para) if current else para
        else:
            if current:
                chunks.append(current)
            current = para

    if current:
        chunks.append(current)

    result = []
    prev_tail = ""
    for i, chunk_text in enumerate(chunks):
        if prev_tail and overlap > 0:
            chunk_text = prev_tail + "\n" + chunk_text

        result.append({
            "id": f"{doc['source']}_chunk_{i:03d}",
            "source": doc["source"],
            "title": doc["title"],
            "content": chunk_text,
        })

        if overlap > 0:
            prev_tail = chunk_text[-overlap:] if len(chunk_text) > overlap else chunk_text

    return result


# ============================================================
# Part 1: 为什么需要 RAG？
# ============================================================

def run_part1():
    title("Part 1: 为什么需要 RAG？")

    console.print("""
  先看一个[bold yellow]对比实验[/bold yellow] — 让 LLM 回答一个公司内部问题:

    [italic]"生产环境的 Nginx 用哪个端口？"[/italic]

  [bold]纯 LLM[/bold] 会回答:
    [dim]"Nginx 默认端口是 80（HTTP）和 443（HTTPS）"[/dim]

  但[bold]公司内部规范[/bold]是:
    "Nginx 使用 8081 端口（因为 80/443 被 LB 占用）"

  [bold yellow]LLM 训练数据里不可能知道你们公司内部的端口分配规范！[/bold yellow]
  纯 LLM 只能凭通用知识回答，[bold yellow]根本不知道你的公司具体配置[/bold yellow]。\n""")

    diagram("""
  无知识库:
    用户提问 → LLM（凭训练数据记忆）→ 答 80 端口 ❌

  有知识库（RAG）:
    用户提问 → 检索公司内部文档 → LLM（基于实时资料）→ 答 8081 端口 ✅
    """)

    console.print("""
  RAG = [bold cyan]R[/bold cyan]etrieval-[bold cyan]A[/bold cyan]ugmented [bold cyan]G[/bold cyan]eneration
       = 检索增强生成

  三步流程:
    1. [bold]Retrieve[/bold]（检索）: 从知识库中找到相关文档
    2. [bold]Augment[/bold]（增强）: 把文档拼到 Prompt 中作为上下文
    3. [bold]Generate[/bold]（生成）: LLM 基于上下文回答问题
    """)

    subtitle("演示: 纯 LLM 回答（无知识库）")

    demo_question = "生产环境的 Nginx 用哪个端口？"
    messages = [
        {"role": "system", "content": "你是一个运维助手，请简洁回答用户的问题。"},
        {"role": "user", "content": demo_question},
    ]
    response = client.chat.completions.create(
        model=model, messages=messages, extra_body={"thinking": {"type": "disabled"}},
    )
    plain_answer = response.choices[0].message.content or ""

    console.print(f"  [bold]问题:[/bold] {demo_question}")
    console.print(f"  [bold]LLM 回答:[/bold]")
    console.print(f"  [dim]{plain_answer}[/dim]\n")
    note("LLM 说的是通用知识（80/443），但我们公司内部规范是 8081。")
    note("没有知识库，LLM 不可能知道公司内部的端口分配。")

    wait_for_enter("按 Enter 进入 Part 2...")


# ============================================================
# Part 2: Embedding — 文本的"向量化"
# ============================================================

def run_part2():
    title("Part 2: Embedding — 文本的向量化")

    console.print("""
  [bold]什么是 Embedding？[/bold]

    把一段文字变成一串数字（向量）:
      [dim]"磁盘满了怎么办"[/dim] → [green][0.023, -0.015, 0.089, 0.047, ...][/green]

    [bold]关键特性:[/bold]
      - 语义相近的文字 → 向量距离近
      - 语义无关的文字 → 向量距离远

    就像把文字"映射"到高维空间中的某个位置。
    """)

    subtitle("实验: 语义相似度")

    texts = [
        "磁盘满了怎么排查",         # 0: 磁盘相关
        "如何查看磁盘使用率",        # 1: 磁盘相关
        "网络端口不通怎么办",        # 2: 网络相关
        "如何检查端口是否在监听",     # 3: 网络相关
        "今天天气真好",              # 4: 无关
    ]
    labels = ["磁盘排查", "磁盘使用率", "网络排查", "端口检查", "天气"]

    info(f"Embedding 模型: [cyan]{EMBEDDING_MODEL}[/cyan]")
    info(f"文本数量: {len(texts)}")

    embeddings = create_embeddings(embedding_client, texts, EMBEDDING_MODEL)
    info(f"向量维度: [bold]{len(embeddings[0])}[/bold] 维")
    info(f"前 5 个维度示例: {[f'{v:.4f}' for v in embeddings[0][:5]]}")

    # 用 Rich Table 展示相似度矩阵
    console.print()
    table = make_table(title="语义相似度矩阵（余弦相似度）", headers=[""] + labels)
    for i in range(len(texts)):
        row = [labels[i]]
        for j in range(len(texts)):
            sim = cosine_similarity(embeddings[i], embeddings[j])
            # 高亮对角线和自己
            if i == j:
                row.append(f"[bold green]{sim:.4f}[/bold green]")
            elif sim > 0.5:
                row.append(f"[green]{sim:.4f}[/green]")
            elif sim < 0.2:
                row.append(f"[dim]{sim:.4f}[/dim]")
            else:
                row.append(f"{sim:.4f}")
        table.add_row(*row)
    console.print(table)

    console.print("""
  [bold]观察:[/bold]
    [green]v[/green] '磁盘满了怎么排查' 与 '如何查看磁盘使用率' [green]相似度高[/green]
    [green]v[/green] '网络端口不通怎么办' 与 '端口检查' [green]相似度高[/green]
    [green]v[/green] '磁盘' 与 '网络' 文本 [dim]相似度低[/dim]
    [green]v[/green] '天气' 与所有运维文本 [dim]相似度低[/dim]

  这就是向量数据库做检索的基础:
  将用户问题转为向量 → 找知识库中最相似的文本 → 返回。
    """)

    wait_for_enter("按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 构建知识库
# ============================================================

def run_part3():
    title("Part 3: 构建知识库")

    if not CHROMA_AVAILABLE:
        console.print("\n  [bold red][red]x[/red] 请安装依赖: pip install chromadb[/bold red]")
        return

    console.print("""
  现在把运维文档建成可检索的知识库:

  [bold]步骤:[/bold]
    1. 加载 [cyan]knowledge_base/[/cyan] 下的 Markdown 文档
    2. 文档分块（[bold]Chunking[/bold]）
    3. 每块转为向量（[bold]Embedding[/bold]）
    4. 存入 Chroma 向量数据库
    """)

    # ── 步骤 1: 加载文档 ──
    subtitle("步骤 1: 加载文档")

    kb_dir = Path(__file__).resolve().parent / "knowledge_base"
    docs = load_knowledge_docs(kb_dir)

    info(f"知识库目录: [cyan]{kb_dir}[/cyan]")
    info(f"加载文档: [bold]{len(docs)}[/bold] 个")
    for d in docs:
        size = len(d["content"])
        console.print(f"    [yellow]{d['source']}[/yellow] ({size} 字)")

    # ── 步骤 2: 文档分块 ──
    subtitle("步骤 2: 文档分块")

    all_chunks: list[dict[str, Any]] = []
    for d in docs:
        chunks = chunk_document(d, chunk_size=200, overlap=20)
        all_chunks.extend(chunks)
        info(f"{d['source']}: {len(chunks)} 块")

    info(f"共 [bold]{len(all_chunks)}[/bold] 个文本块")
    info("分块策略: 按段落分块，每块 ~200 字符，重叠 20 字符")
    console.print()
    info(f"分块示例（第一块前 80 字）:")
    info(f"  [dim]{all_chunks[0]['content'][:80]}...[/dim]")

    # ── 步骤 3: 向量化 ──
    subtitle("步骤 3: 向量化所有文本块")

    chunk_texts = [c["content"] for c in all_chunks]
    info(f"正在调用 Embedding API ([cyan]{EMBEDDING_MODEL}[/cyan])...")
    chunk_embeddings = create_embeddings(embedding_client, chunk_texts, EMBEDDING_MODEL)
    success(f"已完成 {len(chunk_embeddings)} 个向量，维度 {len(chunk_embeddings[0])}")

    # ── 步骤 4: 存入 Chroma ──
    subtitle("步骤 4: 存入 Chroma")

    CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
    info(f"持久化目录: [cyan]{CHROMA_DIR}[/cyan]")

    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        chroma_client.delete_collection("ops_kb")
    except Exception:
        pass

    collection = chroma_client.create_collection(
        name="ops_kb",
        metadata={"description": "AIOps 运维知识库"},
    )

    collection.add(
        ids=[c["id"] for c in all_chunks],
        embeddings=chunk_embeddings,
        documents=[c["content"] for c in all_chunks],
        metadatas=[{"source": c["source"], "title": c["title"]} for c in all_chunks],
    )

    success("知识库构建完成!")
    info(f"集合名: ops_kb")
    info(f"文档数: {collection.count()}")
    info(f"向量维度: {len(chunk_embeddings[0])}")
    note("Chroma 已将数据持久化到磁盘，下次运行可重复使用。")

    wait_for_enter("按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 检索问答（RAG 核心流程）
# ============================================================

def run_part4():
    title("Part 4: 检索问答 — RAG 核心流程")

    if not CHROMA_AVAILABLE:
        console.print("\n  [bold red][red]x[/red] 请安装依赖: pip install chromadb[/bold red]")
        return

    CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = chroma_client.get_collection("ops_kb")
    except Exception:
        console.print("\n  [bold red][red]x[/red] 知识库不存在！请先运行 Part 3 构建知识库。[/bold red]")
        return

    diagram("""
    用户问题
       │
       ▼
  ① Embedding 向量化
       │
       ▼
  ② Chroma 检索（取 Top-3）
       │
       ▼
  ③ 拼接上下文 Prompt
       │
       ▼
  ④ LLM 基于上下文生成回答
       │
       ▼
    最终答案
    """)

    test_questions = [
        "生产环境的 Nginx 用哪个端口？",
        "应用日志应该写到哪个目录？保留多久？",
        "生产服务器的命名规则是什么？",
    ]

    for q_idx, question in enumerate(test_questions):
        divider(f"问题 {q_idx + 1}")
        console.print(f"  [bold]{question}[/bold]")

        # ── ① Embedding 向量化 ──
        console.print("\n  [cyan]① Embedding 向量化...[/cyan]")
        query_embedding = create_embeddings(embedding_client, [question], EMBEDDING_MODEL)[0]

        # ── ② Chroma 检索 ──
        console.print("  [cyan]② Chroma 检索 Top-3...[/cyan]")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3,
        )

        retrieved_docs = results["documents"][0] if results["documents"] else []
        retrieved_metas = results["metadatas"][0] if results["metadatas"] else []
        retrieved_distances = results["distances"][0] if results["distances"] else []

        info(f"找到 [bold]{len(retrieved_docs)}[/bold] 个相关文档块:")
        for i, (meta, dist) in enumerate(zip(retrieved_metas, retrieved_distances)):
            sim = 1 - dist
            source_name = meta.get("source", "?")
            console.print(f"       [{i+1}] {source_name} (相关度: [green]{sim:.3f}[/green])")

        # ── ③ 拼接上下文 Prompt ──
        context = "\n\n---\n\n".join(retrieved_docs)
        rag_prompt = f"""你是一个运维助手。请基于以下运维文档回答问题。

如果文档信息不足以回答，请说明"知识库中没有相关信息"。

运维文档:
{context}

问题: {question}

请基于文档内容给出具体、准确的回答。"""

        # ── 无 RAG 对照 ──
        if q_idx == 0:
            console.print("\n  [bold yellow]─ 对照: 无 RAG 的回答 ─[/bold yellow]")
            plain_messages = [
                {"role": "system", "content": "你是一个运维助手，请简洁回答用户的问题。"},
                {"role": "user", "content": question},
            ]
            plain_response = client.chat.completions.create(
                model=model, messages=plain_messages, extra_body={"thinking": {"type": "disabled"}},
            )
            console.print(f"  [dim]{plain_response.choices[0].message.content}[/dim]")

        # ── ④ LLM 生成 ──
        console.print(f"\n  [bold green]─ 有 RAG 的回答 ─[/bold green]")
        rag_messages = [
            {"role": "system", "content": "你是一个运维助手，严格基于文档内容回答问题。"},
            {"role": "user", "content": rag_prompt},
        ]
        rag_response = client.chat.completions.create(
            model=model, messages=rag_messages, extra_body={"thinking": {"type": "disabled"}},
        )
        console.print(f"  {rag_response.choices[0].message.content}")

        if q_idx == 0:
            note("对比: 无 RAG 的回答是通用的，有 RAG 的回答引用了具体的命令和步骤。")

    wait_for_enter("按 Enter 进入 Part 5...")


# ============================================================
# Part 5: RAG + Agent — 将检索封装为工具
# ============================================================

def run_part5():
    title("Part 5: RAG + Agent — 知识库工具化")

    if not CHROMA_AVAILABLE:
        console.print("\n  [bold red][red]x[/red] 请安装依赖: pip install chromadb[/bold red]")
        return

    CHROMA_DIR = str(Path(__file__).resolve().parent / "chroma_db")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        collection = chroma_client.get_collection("ops_kb")
    except Exception:
        console.print("\n  [bold red][red]x[/red] 知识库不存在！请先运行 Part 3 构建知识库。[/bold red]")
        return

    console.print("""
  前面的 RAG 是手动调用的: 用户提问 → 我们检索 → LLM 回答。

  现在更进一步: 将检索封装为 Agent 工具，
  让 Agent [bold]自己决定[/bold] 什么时候查知识库。

  这更接近实际应用场景:
    - Agent 有多个工具可用（shell、知识库等）
    - Agent [bold yellow]自主判断[/bold yellow]是否需要查知识库
    - Agent 基于查到的信息进一步推理
    """)

    # ── 定义工具 ──

    def retrieve_knowledge(query: str, top_k: int = 3) -> str:
        """从运维知识库中检索相关信息，返回检索到的文档内容。"""
        query_embedding = create_embeddings(embedding_client, [query], EMBEDDING_MODEL)[0]
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []

        if not docs:
            return "知识库中没有找到相关信息。"

        output_parts = []
        for meta, doc in zip(metas, docs):
            source = meta.get("source", "未知")
            output_parts.append(f"[来源: {source}]\n{doc}")

        return "\n\n---\n\n".join(output_parts)

    TOOL_DEFS = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_knowledge",
                "description": "从运维知识库中检索相关信息。当回答运维问题需要参考文档时调用此工具。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "检索关键词，如'磁盘满排查'、'日志查看'等",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "返回的文档数量，默认 3",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ]

    # ── Agent 对话演示 ──
    subtitle("Agent 对话演示")

    agent_messages = [
        {
            "role": "system",
            "content": (
                "你是一个 AIOps 运维助手。你可以使用以下工具:\n"
                "- retrieve_knowledge: 从运维知识库检索信息\n\n"
                "当遇到运维问题时，先查知识库获取准确信息，再回答。\n"
                "如果知识库中有相关信息，请引用来源。"
            ),
        },
        {
            "role": "user",
            "content": "生产环境的 Nginx 用哪个端口？按公司规范来",
        },
    ]

    MAX_TURNS = 5
    for turn in range(MAX_TURNS):
        divider(f"第 {turn + 1} 轮")

        response = client.chat.completions.create(
            model=model,
            messages=agent_messages,
            tools=TOOL_DEFS,
            extra_body={"thinking": {"type": "disabled"}},
        )

        choice = response.choices[0]
        msg = choice.message

        if msg.content:
            console.print(f"  [bold cyan]Assistant:[/bold cyan] {msg.content[:200]}")

        if msg.tool_calls:
            for tc in msg.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)

                console.print(f"  [bold yellow]> 工具:[/bold yellow] {func_name}(query=\"{func_args.get('query', '')}\")")

                if func_name == "retrieve_knowledge":
                    result = retrieve_knowledge(
                        func_args.get("query", ""),
                        func_args.get("top_k", 3),
                    )
                else:
                    result = f"未知工具: {func_name}"

                result_preview = result[:150].replace("\n", " ")
                info(f"检索结果: [dim]{result_preview}...[/dim]")

                agent_messages.append({
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    ],
                })
                agent_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            success("回答完毕")
            break
    else:
        console.print(f"\n  [bold yellow]! 已达到最大轮次 {MAX_TURNS}[/bold yellow]")

    console.print("""
  [bold]RAG + Agent 的核心价值:[/bold]
  1. Agent [bold yellow]自主判断[/bold yellow]是否需要查知识库
  2. 知识库内容被实时检索，不依赖 LLM 训练数据
  3. 可以扩展到多个知识源（运维手册、故障记录、API 文档等）
    """)

    wait_for_enter("按 Enter 进入 Part 6...")


# ============================================================
# Part 6: 总结
# ============================================================

def run_part6():
    title("Part 6: 总结")

    table = make_table(
        title="本章学到的核心概念",
        headers=["概念", "说明", "关键要点"],
    )
    table.add_row(
        "[bold]Embedding[/bold]",
        "文本 → 向量，语义相近 → 向量距离近",
        "余弦相似度衡量语义相似度"
    )
    table.add_row(
        "[bold]Chroma[/bold]",
        "轻量级向量数据库",
        "语义检索 + 持久化"
    )
    table.add_row(
        "[bold]RAG 流水线[/bold]",
        "Retrieve → Augment → Generate",
        "检索增强生成"
    )
    table.add_row(
        "[bold]RAG + Agent[/bold]",
        "将检索封装为工具",
        "Agent 自主决定何时查知识库"
    )
    console.print(table)

    console.print("""
  [bold]为什么 RAG 很重要？[/bold]
    [green]v[/green] 解决 LLM 知识截止日期问题
    [green]v[/green] 让回答可追溯、可验证
    [green]v[/green] 可以接入企业私有知识库
    [green]v[/green] 是 Agent 获取外部信息的关键方式

  [bold]进阶方向（后续章节）:[/bold]
    [dim]-[/dim] [italic]Reranking:[/italic] 对检索结果重排序提高精度
    [dim]-[/dim] [italic]Hybrid Search:[/italic] 向量 + 关键词混合检索
    [dim]-[/dim] [italic]Graph RAG:[/italic] 利用知识图谱增强检索
    [dim]-[/dim] [italic]Agentic RAG:[/italic] Agent 自主决定检索策略
    """)

    console.print("[bold cyan]再见！[/bold cyan]")


# ============================================================
# 运行
# ============================================================
if __name__ == "__main__":
    console.rule("[bold cyan]RAG 知识库问答示例[/bold cyan]")
    console.print(f"  Model: {model}  |  Embedding: {EMBEDDING_MODEL}")
    console.rule()

    run_part1()
    run_part2()
    run_part3()
    run_part4()
    run_part5()
    run_part6()
