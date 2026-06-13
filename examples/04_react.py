"""
examples/04_react.py — ReAct 多步骤规划

学习目标:
  1. 理解 ReAct（Reasoning + Acting）的核心概念
  2. 理解 Thought → Action → Observation 循环
  3. 学会手动实现 ReAct 循环，解析 LLM 的推理过程
  4. 比较普通 Tool Calling 和 ReAct 的差异

运行方式:
  python examples/04_react.py

前置条件:
  已完成 02_tool_calling.py，理解基础工具调用

核心概念:
  - Thought（思考）: LLM 分析当前状态，决定下一步做什么
  - Action（行动）: 调用工具执行操作
  - Observation（观察）: 工具返回结果
  - Final Answer（最终答案）: 完成任务后给出最终回复
  - ReAct 让 LLM 的"推理过程"可见、可控
"""

import json
import os
import subprocess
import re
from openai import OpenAI

from _common import load_config
from _ui import console, title, note, info, diagram, success, divider, wait_for_enter, make_table


# ============================================================
# 第一步: 加载配置
# ============================================================
config = load_config()
model = config["model"]
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

# ReAct 模式的 System Prompt
REACT_SYSTEM_PROMPT = """你是一个 AIOps 运维助手。请按照 ReAct 模式思考和执行任务。

你的工作流程:
1. Thought: 分析当前情况，推理下一步该做什么
2. Action: 调用工具执行操作（如果需要）
3. 观察工具返回的结果
4. 重复 Thought → Action → Observation 直到任务完成
5. Final Answer: 给出最终的完整回答

可用工具:
- shell: 执行 Shell 命令，参数: command (字符串)

注意:
- 如果你认为不需要调用工具，可以直接给出 Final Answer
- 每次只调用一个工具
- 基于 Observation 来调整下一步的行动
- 请用中文输出 Thought 和 Final Answer"""

SAMPLE_TASK = "查一下系统的基本信息，包括磁盘、内存，帮我做一个简单的系统健康报告"

# Windows 兼容的命令列表（如果检测到 Windows，使用 dir/wmic 等）
_IS_WINDOWS = hasattr(os, "name") and os.name == "nt"

if _IS_WINDOWS:
    SAMPLE_TASK = "检查系统的磁盘和内存状态，做一个简单的健康报告"
    REACT_SYSTEM_PROMPT = REACT_SYSTEM_PROMPT.replace(
        "- shell: 执行 Shell 命令，参数: command (字符串)",
        "- shell: 执行命令，参数: command (字符串)",
    )


# 工具定义（同 02_tool_calling.py）
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
        if result.returncode == 0:
            return result.stdout.strip() or "(命令执行成功，无输出)"
        else:
            return f"退出码: {result.returncode}\n{result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时（30秒）"
    except Exception as e:
        return f"错误: {e}"


def extract_text_thought(text: str) -> str | None:
    """从文本中提取 Thought 部分。"""
    m = re.search(r"Thought:\s*(.+?)(?=Action:|Final Answer:)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def extract_final_answer(text: str) -> str | None:
    """从文本中提取 Final Answer 部分。"""
    m = re.search(r"Final Answer:\s*(.+)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


# ============================================================
# Part 1: 问题引出 — 为什么需要 ReAct
# ============================================================

def run_part1():
    title("Part 1: 普通 Tool Calling 的局限")

    console.print("""
  当前 Agent 的工具调用是这样的:
    用户: "查一下系统状态"
    LLM: → [yellow]shell("df -h")[/yellow]
    工具: → 返回结果
    LLM: → 直接汇总回答

  [bold]问题:[/bold]
  1. LLM 的"推理过程"不可见 — 不知道为什么调这个工具
  2. 无法处理需要多步推理的复杂场景
  3. 如果工具返回异常，LLM 可能不知道下一步该做什么

  示例复杂场景: [dim]"查磁盘 → 发现满了 → 查大文件 → 清缓存 → 确认"[/dim]
  这需要多步推理，每一步都依赖上一步的观察结果。
    """)

    wait_for_enter("按 Enter 进入 Part 2...")


# ============================================================
# Part 2: ReAct 概念
# ============================================================

def run_part2():
    title("Part 2: ReAct 概念 — 思考 + 行动")

    diagram("""
  ReAct = Reasoning (推理) + Acting (行动)

  核心循环:
  ┌──────────────────────────────────────────┐
  │                                          │
  │   Thought（思考）                         │
  │   ↓  "磁盘可能满了，先检查使用率"          │
  │                                          │
  │   Action（行动）                          │
  │   ↓  shell("df -h")                      │
  │                                          │
  │   Observation（观察）                     │
  │   ↓  "磁盘使用率 85%，需要清理"            │
  │                                          │
  │   Thought（思考）                         │
  │   ↓  "找到大文件目录，清理日志"            │
  │   ...                                    │
  │                                          │
  │   Final Answer（最终答案）                 │
  │   "系统健康报告: ..."                     │
  │                                          │
  └──────────────────────────────────────────┘
    """)

    console.print("  [bold]对比:[/bold]")
    table = make_table(headers=["", "普通 Tool Calling", "ReAct"])
    table.add_row("推理过程", "隐式（在 LLM 内部）", "[green]显式（输出可见）[/green]")
    table.add_row("可控性", "低", "[green]高[/green]")
    table.add_row("多步推理", "靠 LLM 内部状态", "[green]靠显式的 Thought[/green]")
    table.add_row("调试", "黑盒", "[green]白盒[/green]")
    console.print(table)

    wait_for_enter("按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 实现 ReAct 循环
# ============================================================

def run_part3():
    title("Part 3: 手写 ReAct 循环")

    console.print(f"\n  [bold]任务:[/bold] {SAMPLE_TASK}")
    divider()

    MAX_STEPS = 8
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_TASK},
    ]

    step = 0
    while step < MAX_STEPS:
        step += 1
        divider(f"第 {step} 步")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            extra_body={"thinking": {"type": "disabled"}},
        )

        choice = response.choices[0]
        message = choice.message

        if message.content:
            thought = extract_text_thought(message.content)
            if thought:
                console.print(f"  [cyan]> Thought:[/cyan] {thought}")
            else:
                console.print(f"  > {message.content[:200]}")

        if message.content and "Final Answer:" in message.content:
            final = extract_final_answer(message.content)
            if final:
                success(f"Final Answer:\n{final}")
                break

        if message.tool_calls:
            # Collect all tool call results first
            tool_results = []
            for tc in message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                command = func_args.get("command", "")

                console.print(f"  [yellow]> Action:[/yellow] {func_name}(\"{command}\")")

                result = execute_shell(command)
                console.print(f"  [dim]> Observation:[/dim] {result[:120]}...")

                tool_results.append((tc, result))

            # One assistant message with all tool_calls
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc, _ in tool_results
                ],
            })

            # One tool result message per call
            for tc, result in tool_results:
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            if message.content and "Final Answer:" not in message.content:
                info("(没有工具调用，等待下一步...)")
                messages.append({"role": "assistant", "content": message.content})
            elif not message.tool_calls and not message.content:
                info("没有更多输出，结束")
                break
    else:
        note(f"已达到最大步数 {MAX_STEPS}，循环结束")

    info(f"共执行 {step} 步，消息历史共 {len(messages)} 条")

    wait_for_enter("按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 总结
# ============================================================

def run_part4():
    title("Part 4: 总结")

    console.print("""
  ReAct 的核心价值:

  1. [bold]透明[/bold]
     - 每一步的推理过程都可见
     - 用户可以理解 Agent 为什么做某个操作

  2. [bold]可控[/bold]
     - 可以在 Thought 阶段介入纠正
     - 容易调试和修改

  3. [bold]多步推理[/bold]
     - 每一步都基于 Observation 重新思考
     - 不会"遗忘"之前的步骤

  ReAct 在实战 Agent 中的应用:
    实战的 Agent 已内置 Tool Calling 循环，
    通过启用 ReAct 模式（[yellow]REACT_ENABLED=true[/yellow]），
    可以让 Agent 输出显式的推理过程。
    见 [cyan]src/aiops_agent/core/agent.py[/cyan]
    """)

    console.print("[bold cyan]再见！[/bold cyan]")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    console.rule("[bold cyan]ReAct 多步骤规划示例[/bold cyan]")
    console.print(f"  Model: {model}")
    console.rule()

    run_part1()
    run_part2()
    run_part3()
    run_part4()
