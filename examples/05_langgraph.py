"""
examples/05_langgraph.py — LangGraph Agent 状态机（ToolNode 版）

学习目标:
  1. 理解状态机（StateGraph）的概念
  2. 理解 Node（节点）、Edge（边）、Conditional Edge（条件边）
  3. 学会用 ToolNode 自动执行工具调用
  4. 理解 AIMessage / ToolMessage 的消息流转

运行方式:
  python examples/05_langgraph.py

前置条件:
  已完成 02_tool_calling.py，理解工具调用的概念

核心概念:
  - State: 状态（消息列表、轮次等）
  - Node: 节点（处理函数，读写 State）
  - ToolNode: LangGraph 内置的工具执行节点
  - AIMessage: 带 tool_calls 的消息 → ToolNode 自动识别
  - ToolMessage: 工具执行结果
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
    from langgraph.prebuilt import ToolNode
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
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
  2. 消息管理全靠手动 append
  3. 增加新功能（重试、记忆、分支）需要改循环逻辑

  状态机（StateGraph）的解法:

  ┌──────────────────────────────────┐
  │                                  │
  │   ┌───────────┐                  │
  │   │ call_model │ (LLM 推理)       │
  │   └─────┬─────┘                  │
  │         │                        │
  │   ┌─────▼──────┐                 │
  │   │  有 tool?   │ (条件边)        │
  │   └──┬──────┬──┘                 │
  │      │ 有   │ 无                  │
  │   ┌──▼────┐  │                   │
  │   │ ToolNode│ │ (自动执行工具)     │
  │   └──┬────┘  │                   │
  │      │       │                    │
  │      └───┬───┘                    │
  │          ▼                        │
  │       ┌─────┐                     │
  │       │ END  │                    │
  │       └─────┘                     │
  │                                  │
  └──────────────────────────────────┘

  ToolNode 自动从 AIMessage.tool_calls 中提取工具调用，
  执行后自动追加 ToolMessage 到消息列表。
""")
    input("\n  按 Enter 进入 Part 2...")


# ============================================================
# Part 2: LangGraph + ToolNode 核心概念
# ============================================================

def run_part2():
    print("\n" + "=" * 60)
    print("  Part 2: LangGraph + ToolNode 核心概念")
    print("=" * 60)
    print("""
  LangGraph 的三个核心概念:

  1. State（状态）
     - TypedDict，描述系统状态
     - 字段可定义 reducer（如 operator.add 实现追加）

  2. Node（节点）
     - 函数，接收 State，返回 State 更新
     - ToolNode 是内置节点，自动执行工具

  3. Conditional Edge（条件边）
     - 根据 State 决定下一步去向

  消息流转:
  ┌────────────────────────────────────────────┐
  │                                            │
  │   call_model 节点                           │
  │     ↓ 返回 AIMessage(tool_calls=[...])      │
  │                                            │
  │   ToolNode 节点                             │
  │     ↓ 自动执行工具，返回 ToolMessage          │
  │                                            │
  │   call_model 节点（循环）                    │
  │     ↓ 返回 AIMessage(content="最终答案")     │
  │                                            │
  │   END                                       │
  │                                            │
  └────────────────────────────────────────────┘
""")
    input("\n  按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 用 LangGraph + ToolNode 构建 Agent
# ============================================================

def run_part3():
    print("\n" + "=" * 60)
    print("  Part 3: 用 LangGraph + ToolNode 构建 Agent")
    print("=" * 60)

    if not LANGGRAPH_AVAILABLE:
        print("\n  ❌ 请安装依赖: pip install langgraph langchain-core")
        return

    # ── Step 1: 定义工具函数（供 ToolNode + OpenAI 共用）──
    print("\n  步骤 1: 定义工具函数")

    def shell_tool(command: str, timeout: int = 30) -> str:
        """执行 Shell 命令，返回命令输出。"""
        result = execute_shell(command)
        return json.dumps({"output": result}, ensure_ascii=False)

    shell_tool.__name__ = "shell"
    shell_tool.__doc__ = "执行 Shell 命令，查看系统状态"

    # 工具定义（用于传给 OpenAI API 做 function calling）
    TOOL_DEFS = [{
        "type": "function",
        "function": {
            "name": "shell",
            "description": "执行 Shell 命令，返回命令输出",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "要执行的命令"},
                    "timeout": {"type": "integer", "description": "超时秒数", "default": 30},
                },
                "required": ["command"],
            },
        },
    }]

    print("    ✅ shell_tool 函数 + TOOL_DEFS 定义完成")
    print("    ✅ ToolNode([shell_tool]) + OpenAI API 共用")

    # ── Step 2: 定义 State ──
    print("\n  步骤 2: 定义 State")

    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]
        tool_round: int
        max_rounds: int

    print("    ✅ class AgentState(TypedDict):")
    print("    ✅   messages: Annotated[list, operator.add]")
    print("    ✅   tool_round: int")

    # ── Step 3: 定义节点 ──
    print("\n  步骤 3: 定义节点")

    def call_model(state: AgentState) -> dict:
        """节点: 调用 LLM，返回 AIMessage。"""
        # 将 state 中的消息转为 OpenAI API 格式
        dict_messages = []
        for m in state["messages"]:
            if isinstance(m, AIMessage):
                d = {"role": "assistant", "content": m.content or ""}
                if m.tool_calls:
                    d["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"],
                                      "arguments": json.dumps(tc["args"], ensure_ascii=False)}}
                        for tc in m.tool_calls
                    ]
                dict_messages.append(d)
            elif isinstance(m, ToolMessage):
                dict_messages.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif isinstance(m, SystemMessage):
                dict_messages.append({"role": "system", "content": m.content})
            else:
                dict_messages.append({"role": "user", "content": m.content})

        # OpenAI 工具定义（与 ToolNode 共用同一份）
        tools = TOOL_DEFS

        response = client.chat.completions.create(
            model=model,
            messages=dict_messages,
            tools=tools,
        )
        choice = response.choices[0]
        msg = choice.message

        # 返回 AIMessage（含 tool_calls 时 ToolNode 会自动识别）
        if msg.tool_calls:
            ai_msg = AIMessage(
                content="",
                tool_calls=[
                    {"name": tc.function.name,
                     "args": json.loads(tc.function.arguments),
                     "id": tc.id}
                    for tc in msg.tool_calls
                ],
            )
            print(f"    🤖 工具调用: {msg.tool_calls[0].function.name}")
        else:
            ai_msg = AIMessage(content=msg.content or "")
            print(f"    🤖 回复: {msg.content[:40]}...")

        return {"messages": [ai_msg]}

    print("    ✅ def call_model(state)  → 返回 AIMessage")
    print("    ✅   AIMessage(tool_calls=[...]) → ToolNode 自动识别")

    # ── Step 4: 用 ToolNode ──
    print("\n  步骤 4: 使用 ToolNode")

    tool_node = ToolNode([shell_tool])
    print("    ✅ tool_node = ToolNode([shell_tool])")

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        has_tc = isinstance(last, AIMessage) and bool(last.tool_calls)
        if has_tc and state["tool_round"] < state["max_rounds"]:
            print(f"    🔄 有工具调用(轮次 {state['tool_round']+1})，继续")
            return "continue"
        print("    ✅ 无工具调用，结束")
        return "end"

    # ── Step 5: 构建图 ──
    print("\n  步骤 5: 构建图")

    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("call_model")

    builder.add_conditional_edges(
        "call_model", should_continue,
        {"continue": "tools", "end": END},
    )
    builder.add_edge("tools", "call_model")

    graph = builder.compile()
    print("    ✅ StateGraph + ToolNode 构建完成")

    # ── Step 6: 运行图 ──
    print("\n  步骤 6: 运行图\n")

    task = "查下当前磁盘使用情况"
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=task),
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

        last = current["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            break

    # 展示最终结果
    print(f"\n  {'─' * 50}")
    print(f"  最终状态: {len(current['messages'])} 条消息")
    for msg in current["messages"]:
        if isinstance(msg, SystemMessage):
            print(f"    📋 System: {msg.content[:30]}...")
        elif isinstance(msg, HumanMessage):
            print(f"    👤 {msg.content[:40]}...")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"    🤖 工具调用: {msg.tool_calls[0]['name']}")
            elif msg.content:
                print(f"    🤖 {msg.content[:60]}...")
        elif isinstance(msg, ToolMessage):
            print(f"    🔧 工具结果: {msg.content[:40]}...")

    input("\n  按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 对比与讨论
# ============================================================

def run_part4():
    print("\n" + "=" * 60)
    print("  Part 4: 对比与讨论")
    print("=" * 60)
    print("""
  手动执行 vs ToolNode:

  ┌──────────────┬────────────────────┬──────────────────────┐
  │               │ 手写 execute_tools │ ToolNode              │
  ├──────────────┼────────────────────┼──────────────────────┤
  │ 工具识别      │ 手动解析 tool_calls │ 自动从 AIMessage 读取 │
  │ 消息追加      │ 手动 append        │ 自动追加 ToolMessage  │
  │ 错误处理      │ 自己写 try/except   │ 内置错误处理          │
  │ 并发执行      │ 需要手动实现        │ 可配置并行            │
  │ 代码量        │ ~40 行             │ 1 行                  │
  └──────────────┴────────────────────┴──────────────────────┘

  AIMessage ↔ ToolMessage 消息流转:
    call_model 返回 AIMessage(tool_calls=[...])
      → ToolNode 自动执行
      → 追加 ToolMessage 到 state.messages
      → call_model 看到工具结果继续推理
      → 直到返回无 tool_calls 的 AIMessage → END

  实战项目中的 ToolNode:
    src/aiops_agent/core/agent.py 同样使用 ToolNode,
    加上 plan 节点实现 Plan-and-Execute 模式。
""")
    print("再见！\n")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  LangGraph + ToolNode 示例 (Model: {model})")
    print(f"{'='*60}")

    run_part1()
    run_part2()
    run_part3()
    run_part4()
