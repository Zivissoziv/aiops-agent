"""
examples/05_langgraph.py — LangGraph Agent 状态机

学习目标:
  1. 理解状态机（StateGraph）的概念
  2. 理解 Node（节点）、Edge（边）、Conditional Edge（条件边）
  3. 学会用 LangGraph 构建 Agent 工具调用循环
  4. 观察状态如何在不同节点间流转

运行方式:
  python examples/05_langgraph.py

前置条件:
  已完成 02_tool_calling.py，理解工具调用的概念

核心概念:
  - State: 状态（消息列表、轮次等）
  - Node: 节点（处理函数，读写 State）
  - Edge: 边（节点间的连接）
  - Conditional Edge: 条件边（根据 State 决定下一步去向）
  - LangGraph 让 Agent 的控制流变得清晰可见
"""

import json
import subprocess
from typing import Annotated, TypedDict
import operator

from openai import OpenAI

from _common import load_config


# ============================================================
# 第一部分: 安装检查
# ============================================================

try:
    from langgraph.graph import END, StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ============================================================
# 配置加载
# ============================================================
config = load_config()
model = config["model"]
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

SYSTEM_PROMPT = "你是一个 AIOps 运维助手，可以通过工具执行运维任务。"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "执行 Shell 命令，返回命令输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "要执行的 Shell 命令",
                    },
                },
                "required": ["command"],
            },
        },
    },
]


def execute_shell(command: str) -> str:
    """执行 Shell 命令并返回输出。"""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        return result.stdout.strip() or (
            f"退出码: {result.returncode}\n{result.stderr.strip()}" if result.stderr else "(无输出)"
        )
    except Exception as e:
        return f"错误: {e}"


# ============================================================
# Part 1: 为什么需要状态机？
# ============================================================

def run_part1():
    print("\n" + "=" * 60)
    print("  Part 1: 为什么需要状态机？")
    print("=" * 60)
    print("""
  手动 Tool Calling 循环（如 04_react.py）的问题:

  1. 控制流分散在 for 循环 + if break 中
  2. 状态管理隐式（messages 列表全靠手动 append）
  3. 要增加新功能（如重试、记忆、分支）需要改循环逻辑
  4. 代码量和复杂度随功能线性增长

  状态机（StateGraph）的解法:

  ┌──────────────────────────────────────────┐
  │                                          │
  │   ┌───────────┐                          │
  │   │ call_model │ (节点: 调 LLM)           │
  │   └─────┬─────┘                          │
  │         │                                │
  │   ┌─────▼──────┐                         │
  │   │ 有条件吗？   │ (条件边)                │
  │   │ 有 tool?   │                         │
  │   └──┬──────┬──┘                         │
  │      │ 有   │ 无                          │
  │   ┌──▼────┐  │                           │
  │   │exec   │  │                           │
  │   │tools  │  │  (节点: 执行工具)          │
  │   └──┬────┘  │                           │
  │      │       │                            │
  │      └───┬───┘                            │
  │          ▼                                │
  │       ┌─────┐                             │
  │       │ END  │                            │
  │       └─────┘                             │
  │                                          │
  └──────────────────────────────────────────┘

  每次状态更新都通过 State（一个 TypedDict）显式传递，
  每个节点只读/写自己关心的 State 字段。
""")
    input("\n  按 Enter 进入 Part 2...")


# ============================================================
# Part 2: LangGraph 核心概念
# ============================================================

def run_part2():
    print("\n" + "=" * 60)
    print("  Part 2: LangGraph 核心概念")
    print("=" * 60)
    print("""
  LangGraph 的三个核心概念:

  1. State（状态）
     - 一个 TypedDict，描述"整个系统的状态"
     - 字段可以定义 reducer（如 operator.add 实现列表追加）
     - 每个节点都可以读/写 State

  2. Node（节点）
     - 一个函数，接收 State，返回 State 的更新
     - 每个节点只做一件事

  3. Edge（边）/ Conditional Edge（条件边）
     - Edge: 从一个节点到另一个节点的固定连接
     - Conditional Edge: 根据 State 中的值决定去向

  类比:
  ┌──────────┬──────────────┬────────────────────┐
  │ 概念      │ 传统代码      │ LangGraph          │
  ├──────────┼──────────────┼────────────────────┤
  │ 状态      │ messages 列表 │ AgentState         │
  │ 循环      │ for _ in ... │ call → exec → call │
  │ 分支      │ if/else      │ conditional edge   │
  │ 控制流    │ 函数内部逻辑   │ 图结构（声明式）    │
  └──────────┴──────────────┴────────────────────┘

  LangGraph 的优势:
  - 控制流声明式声明（add_node, add_edge），不是写在代码里
  - 状态管理自动（reducer 处理追加/合并）
  - 每个节点可独立测试
  - 可视化（可以打印图结构）
""")
    input("\n  按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 用 LangGraph 构建 Agent
# ============================================================

def run_part3():
    print("\n" + "=" * 60)
    print("  Part 3: 用 LangGraph 构建 Agent")
    print("=" * 60)

    if not LANGGRAPH_AVAILABLE:
        print("\n  ❌ 未安装 langgraph，请运行: pip install langgraph")
        return

    # ── Step 1: 定义 State ──
    print("\n  步骤 1: 定义 State")

    class AgentState(TypedDict):
        """Agent 的状态。"""
        # messages 使用 operator.add 作为 reducer
        # 这样每次 return {"messages": [new_msg]} 时会自动追加到列表中
        messages: Annotated[list, operator.add]
        tool_round: int
        max_rounds: int

    print("    ✅ class AgentState(TypedDict):")
    print("    ✅   messages: Annotated[list, operator.add]")
    print("    ✅   tool_round: int")
    print("    ✅   max_rounds: int")

    # ── Step 2: 定义节点函数 ──
    print("\n  步骤 2: 定义节点")

    def call_model(state: AgentState) -> dict:
        """节点: 调用 LLM。"""
        response = client.chat.completions.create(
            model=model,
            messages=state["messages"],
            tools=TOOLS,
        )
        choice = response.choices[0]
        msg = choice.message

        assistant_msg = {"role": "assistant"}
        if msg.content:
            assistant_msg["content"] = msg.content
        if msg.tool_calls:
            assistant_msg["content"] = ""
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]

        print(f"    🤖 LLM 回复: {msg.content[:60] if msg.content else '(工具调用)'}...")

        return {"messages": [assistant_msg]}

    print("    ✅ def call_model(state) -> dict:")
    print("    ✅   调用 LLM，返回 assistant 消息")

    def execute_tools(state: AgentState) -> dict:
        """节点: 执行所有工具调用。"""
        last_msg = state["messages"][-1]
        tool_calls = last_msg.get("tool_calls", [])
        tool_msgs = []

        for tc in tool_calls:
            func_name = tc["function"]["name"]
            args = json.loads(tc["function"]["arguments"])
            print(f"    🔧 执行: {func_name}({json.dumps(args)})")

            result = execute_shell(args.get("command", ""))

            tool_msgs.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": result,
            })
            print(f"    👁 结果: {result[:60]}...")

        return {"messages": tool_msgs}

    print("    ✅ def execute_tools(state) -> dict:")
    print("    ✅   执行所有 tool_calls，返回 tool 消息")

    def should_continue(state: AgentState) -> str:
        """条件边: 决定继续还是结束。"""
        last_msg = state["messages"][-1]
        has_tools = bool(last_msg.get("tool_calls"))
        under_limit = state["tool_round"] < state["max_rounds"]

        if has_tools and under_limit:
            print(f"    🔄 有工具调用(轮次 {state['tool_round']+1}/{state['max_rounds']})，继续")
            return "continue"
        elif has_tools:
            print(f"    ⛔ 达到最大轮次 {state['max_rounds']}，结束")
        else:
            print("    ✅ 无工具调用，结束")
        return "end"

    # ── Step 3: 构建图 ──
    print("\n  步骤 3: 构建图")

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("execute_tools", execute_tools)
    builder.set_entry_point("call_model")

    builder.add_conditional_edges(
        "call_model",
        should_continue,
        {"continue": "execute_tools", "end": END},
    )
    builder.add_edge("execute_tools", "call_model")

    graph = builder.compile()
    print("    ✅ StateGraph 构建完成")

    # ── Step 4: 运行图 ──
    print("\n  步骤 4: 运行图\n")

    task = "查下当前磁盘使用情况"
    initial_state: AgentState = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ],
        "tool_round": 0,
        "max_rounds": 5,
    }

    print(f"  任务: {task}")
    print(f"  {'─' * 50}")

    current = initial_state
    for i in range(10):
        next_state = graph.invoke(current)
        current = next_state
        current["tool_round"] = current.get("tool_round", 0) + 1

        last_msg = current["messages"][-1]
        has_tools = bool(last_msg.get("tool_calls"))
        if not has_tools:
            break

    # 展示最终结果
    print(f"\n  {'─' * 50}")
    print(f"  最终状态: {len(current['messages'])} 条消息")
    for msg in current["messages"]:
        role = msg["role"]
        if role == "user":
            print(f"    👤 {msg['content'][:40]}...")
        elif role == "assistant":
            content = msg.get("content", "")
            tc = msg.get("tool_calls")
            if tc:
                print(f"    🤖 工具调用: {tc[0]['function']['name']}")
            elif content:
                print(f"    🤖 {content[:60]}...")
        elif role == "tool":
            print(f"    🔧 工具结果: {msg['content'][:40]}...")

    input("\n  按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 对比与讨论
# ============================================================

def run_part4():
    print("\n" + "=" * 60)
    print("  Part 4: 对比与讨论")
    print("=" * 60)
    print("""
  手动循环 vs LangGraph 状态机:

  ┌─────────────┬────────────────────┬──────────────────────┐
  │              │ 手动循环 (04_react) │ LangGraph (05)       │
  ├─────────────┼────────────────────┼──────────────────────┤
  │ 控制流       │ for + if break     │ 声明式的图结构        │
  │ 状态管理     │ 手动 list.append   │ reducer 自动追加      │
  │ 增加功能     │ 改循环逻辑          │ 加节点/改边           │
  │ 可测试性     │ 需要 mock 整个循环  │ 每个节点单独测        │
  │ 可视化       │ 无                 │ 可以打印图结构        │
  │ 学习曲线     │ 低                 │ 中等                 │
  └─────────────┴────────────────────┴──────────────────────┘

  实战项目中的 LangGraph Agent:
    src/aiops_agent/core/agent.py 已用 LangGraph 重写，
    保留相同的 Agent/AgentEvent 接口，
    但内部使用 StateGraph 驱动。

  下一阶段:
    RAG 知识库 — 让 Agent 能检索运维文档和 Runbook 来回答问题
""")
    print("再见！\n")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  LangGraph Agent 状态机示例 (Model: {model})")
    print(f"{'='*60}")

    run_part1()
    run_part2()
    run_part3()
    run_part4()
