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
from _ui import console, title, subtitle, note, success, info, diagram, divider, wait_for_enter, make_table


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
    title("Part 1: 为什么需要状态机？")

    diagram("""
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
    """)

    console.print("""
  ToolNode 自动从 [bold]AIMessage.tool_calls[/bold] 中提取工具调用，
  执行后自动追加 [bold]ToolMessage[/bold] 到消息列表。
    """)

    wait_for_enter("按 Enter 进入 Part 2...")


# ============================================================
# Part 2: LangGraph + ToolNode 核心概念
# ============================================================

def run_part2():
    title("Part 2: LangGraph + ToolNode 核心概念")

    diagram("""
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

    wait_for_enter("按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 用 LangGraph + ToolNode 构建 Agent
# ============================================================

def run_part3():
    title("Part 3: 用 LangGraph + ToolNode 构建 Agent")

    if not LANGGRAPH_AVAILABLE:
        console.print("\n  [bold red]x 请安装依赖: pip install langgraph langchain-core[/bold red]")
        return

    # ── Step 1 ──
    subtitle("步骤 1: 定义工具函数")

    def shell_tool(command: str, timeout: int = 30) -> str:
        result = execute_shell(command)
        return json.dumps({"output": result}, ensure_ascii=False)

    shell_tool.__name__ = "shell"
    shell_tool.__doc__ = "执行 Shell 命令，查看系统状态"

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

    success("shell_tool 函数 + TOOL_DEFS 定义完成")
    success("ToolNode([shell_tool]) + OpenAI API 共用")

    # ── Step 2 ──
    subtitle("步骤 2: 定义 State")

    class AgentState(TypedDict):
        messages: Annotated[list, operator.add]
        tool_round: int
        max_rounds: int

    success("class AgentState(TypedDict):")
    success("  messages: Annotated[list, operator.add]")
    success("  tool_round: int")

    # ── Step 3 ──
    subtitle("步骤 3: 定义节点")

    def call_model(state: AgentState) -> dict:
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

        response = client.chat.completions.create(
            model=model,
            messages=dict_messages,
            tools=TOOL_DEFS,
            extra_body={"thinking": {"type": "disabled"}},
        )
        choice = response.choices[0]
        msg = choice.message

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
            console.print(f"    [cyan]> 工具调用:[/cyan] {msg.tool_calls[0].function.name}")
        else:
            ai_msg = AIMessage(content=msg.content or "")
            console.print(f"    > 回复: {msg.content[:40]}...")

        return {"messages": [ai_msg]}

    success("def call_model(state) → 返回 AIMessage")
    success("  AIMessage(tool_calls=[...]) → ToolNode 自动识别")

    # ── Step 4 ──
    subtitle("步骤 4: 使用 ToolNode")

    tool_node = ToolNode([shell_tool])
    success("tool_node = ToolNode([shell_tool])")

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        has_tc = isinstance(last, AIMessage) and bool(last.tool_calls)
        if has_tc and state["tool_round"] < state["max_rounds"]:
            console.print(f"    [yellow]> 有工具调用(轮次 {state['tool_round']+1})，继续[/yellow]")
            return "continue"
        success("无工具调用，结束")
        return "end"

    # ── Step 5 ──
    subtitle("步骤 5: 构建图")

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
    success("StateGraph + ToolNode 构建完成")

    # ── Step 6 ──
    subtitle("步骤 6: 运行图")

    task = "查下当前磁盘使用情况"
    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=task),
        ],
        "tool_round": 0,
        "max_rounds": 5,
    }

    console.print(f"  [bold]任务:[/bold] {task}")
    divider()

    current = initial_state
    for i in range(10):
        next_state = graph.invoke(current)
        current = next_state
        current["tool_round"] = current.get("tool_round", 0) + 1
        last = current["messages"][-1]
        if isinstance(last, AIMessage) and not last.tool_calls:
            break

    divider("最终状态")
    info(f"共 {len(current['messages'])} 条消息")
    for msg in current["messages"]:
        if isinstance(msg, SystemMessage):
            console.print(f"    > System: {msg.content[:30]}...")
        elif isinstance(msg, HumanMessage):
            console.print(f"    > User: {msg.content[:40]}...")
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                console.print(f"    [cyan]> 工具调用:[/cyan] {msg.tool_calls[0]['name']}")
            elif msg.content:
                console.print(f"    > {msg.content[:60]}...")
        elif isinstance(msg, ToolMessage):
            console.print(f"    > 工具结果: {msg.content[:40]}...")

    wait_for_enter("按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 对比与讨论
# ============================================================

def run_part4():
    title("Part 4: 对比与讨论")

    table = make_table(headers=["", "手写 execute_tools", "ToolNode"])
    table.add_row("工具识别", "手动解析 tool_calls", "[green]自动从 AIMessage 读取[/green]")
    table.add_row("消息追加", "手动 append", "[green]自动追加 ToolMessage[/green]")
    table.add_row("错误处理", "自己写 try/except", "[green]内置错误处理[/green]")
    table.add_row("并发执行", "需要手动实现", "[green]可配置并行[/green]")
    table.add_row("代码量", "~40 行", "[green]1 行[/green]")
    console.print(table)

    console.print("""
  [bold]AIMessage <-> ToolMessage 消息流转:[/bold]
    call_model 返回 [cyan]AIMessage(tool_calls=[...])[/cyan]
      → ToolNode 自动执行
      → 追加 [cyan]ToolMessage[/cyan] 到 state.messages
      → call_model 看到工具结果继续推理
      → 直到返回无 tool_calls 的 AIMessage → END

  实战项目中的 ToolNode:
    [cyan]src/aiops_agent/core/agent.py[/cyan] 同样使用 ToolNode,
    加上 plan 节点实现 Plan-and-Execute 模式。
    """)

    console.print("[bold cyan]再见！[/bold cyan]")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    console.rule("[bold cyan]LangGraph + ToolNode 示例[/bold cyan]")
    console.print(f"  Model: {model}")
    console.rule()

    run_part1()
    run_part2()
    run_part3()
    run_part4()
