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
import subprocess
import re
from openai import OpenAI

from _common import load_config


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

SAMPLE_TASK = "查一下系统的基本信息，包括磁盘、内存和运行时间，帮我做一个简单的系统健康报告"

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
    print("\n" + "=" * 60)
    print("  Part 1: 普通 Tool Calling 的局限")
    print("=" * 60)
    print("""
  当前 Agent 的工具调用是这样的:
    用户: "查一下系统状态"
    LLM: → shell("df -h")
    工具: → 返回结果
    LLM: → 直接汇总回答

  问题:
  1. LLM 的"推理过程"不可见 — 不知道为什么调这个工具
  2. 无法处理需要多步推理的复杂场景
  3. 如果工具返回异常，LLM 可能不知道下一步该做什么

  示例复杂场景: "查磁盘 → 发现满了 → 查大文件 → 清缓存 → 确认"
  这需要多步推理，每一步都依赖上一步的观察结果。
""")
    input("\n  按 Enter 进入 Part 2...")


# ============================================================
# Part 2: ReAct 概念
# ============================================================

def run_part2():
    print("\n" + "=" * 60)
    print("  Part 2: ReAct 概念 — 思考 + 行动")
    print("=" * 60)
    print("""
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

  对比:
  ┌─────────┬───────────────────┬───────────────────┐
  │          │ 普通 Tool Calling  │ ReAct              │
  ├─────────┼───────────────────┼───────────────────┤
  │ 推理过程 │ 隐式（在 LLM 内部） │ 显式（输出可见）    │
  │ 可控性   │ 低                │ 高                │
  │ 多步推理 │ 靠 LLM 内部状态    │ 靠显式的 Thought   │
  │ 调试    │ 黑盒              │ 白盒              │
  └─────────┴───────────────────┴───────────────────┘
""")
    input("\n  按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 实现 ReAct 循环
# ============================================================

def run_part3():
    print("\n" + "=" * 60)
    print("  Part 3: 手写 ReAct 循环")
    print("=" * 60)
    print(f"\n  任务: {SAMPLE_TASK}")
    print(f"  {'.' * 50}\n")

    MAX_STEPS = 8  # 最大循环步数
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_TASK},
    ]

    step = 0
    while step < MAX_STEPS:
        step += 1
        print(f"\n  ═══ 第 {step} 步 ═══")

        # 调用 LLM（传入工具定义）
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        choice = response.choices[0]
        message = choice.message

        # 输出 Thought（文本内容）
        if message.content:
            # 提取并显示 Thought
            thought = extract_text_thought(message.content)
            if thought:
                print(f"  💭 Thought: {thought}")
            else:
                # 如果没有显式的 Thought 标记，直接显示内容
                print(f"  💬 {message.content[:200]}")

        # 检查是否有 Final Answer
        if message.content and "Final Answer:" in message.content:
            final = extract_final_answer(message.content)
            if final:
                print(f"\n  ✅ Final Answer:\n{final}")
                break

        # 执行工具调用
        if message.tool_calls:
            for tc in message.tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                command = func_args.get("command", "")

                print(f"  🔧 Action: {func_name}(\"{command}\")")

                # 执行工具
                result = execute_shell(command)
                print(f"  👁 Observation: {result[:120]}...")

                # 将 assistant 消息（含 tool_calls）加入历史
                messages.append({
                    "role": "assistant",
                    "content": message.content,
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

                # 将工具结果作为 observation 加入历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
        else:
            # 没有工具调用也没有 Final Answer
            if message.content and "Final Answer:" not in message.content:
                print(f"  (没有工具调用，等待下一步...)")
                messages.append({"role": "assistant", "content": message.content})
            elif not message.tool_calls and not message.content:
                print("  没有更多输出，结束")
                break
    else:
        print(f"\n  ⚠️ 已达到最大步数 {MAX_STEPS}，循环结束")

    print(f"\n  共执行 {step} 步，消息历史共 {len(messages)} 条")

    input("\n  按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 总结
# ============================================================

def run_part4():
    print("\n" + "=" * 60)
    print("  Part 4: 总结")
    print("=" * 60)
    print("""
  ReAct 的核心价值:

  1. 透明
     - 每一步的推理过程都可见
     - 用户可以理解 Agent 为什么做某个操作

  2. 可控
     - 可以在 Thought 阶段介入纠正
     - 容易调试和修改

  3. 多步推理
     - 每一步都基于 Observation 重新思考
     - 不会"遗忘"之前的步骤

  ReAct 在实战 Agent 中的应用:
    实战的 Agent 已内置 Tool Calling 循环，
    通过启用 ReAct 模式（REACT_ENABLED=true），
    可以让 Agent 输出显式的推理过程。
    见 src/aiops_agent/core/agent.py
""")
    print("再见！\n")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  ReAct 多步骤规划示例 (Model: {model})")
    print(f"{'='*60}")

    run_part1()
    run_part2()
    run_part3()
    run_part4()
