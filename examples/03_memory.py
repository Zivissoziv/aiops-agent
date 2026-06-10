"""
examples/03_memory.py — 记忆管理

学习目标:
  1. 理解"无记忆"模式下历史无限制增长的问题
  2. 实现滑动窗口（Sliding Window）策略
  3. 实现摘要（Summarization）策略
  4. 比较不同策略在 token 消耗和上下文保留上的权衡

运行方式:
  python examples/03_memory.py

前置条件:
  已完成 01_simple_chat.py，理解基础对话流程

核心概念:
  - Token 预算: LLM 上下文窗口有限，需要管理历史消息的 token 开销
  - 滑动窗口: 只保留最近 N 条消息，超出则丢弃
  - 摘要: 用 LLM 对旧消息做摘要，用摘要替换原文
  - 权衡: 滑窗简单但丢失上下文，摘要保留上下文但增加 LLM 调用
"""

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# 第一步: 加载配置（同前两个示例）
# ============================================================
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY", "")
base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not api_key or len(api_key) < 10:
    print("❌ 请在项目根目录的 .env 文件中配置 OPENAI_API_KEY")
    exit(1)

client = OpenAI(api_key=api_key, base_url=base_url)

SYSTEM_PROMPT = "你是一个 AIOps 运维助手。请简洁地回答运维相关问题。"


def estimate_tokens(messages: list[dict]) -> int:
    """近似估算 token 数（字符数 // 4）。"""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += len(block["text"])
    return total // 4


def call_llm(messages: list[dict]) -> str:
    """调用 LLM 并返回回复内容。"""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content or ""


# ============================================================
# 模拟对话: 生成一系列运维问答，用于测试记忆策略
# ============================================================

SAMPLE_QUESTIONS = [
    "查看一下当前磁盘使用情况",
    "系统负载高怎么排查",
    "怎么看内存使用",
    "nginx 日志在哪里",
    "怎么重启服务",
    "防火墙端口怎么开放",
    "怎么看进程列表",
    "磁盘 IO 怎么看",
    "网络延迟怎么测",
    "怎么配 crontab",
    "怎么看系统版本",
    "SSH 配置在哪",
    "docker 怎么用",
    "怎么看内核日志",
]


def simulate_conversation(
    messages: list[dict],
    num_rounds: int = 8,
    label: str = "",
) -> list[int]:
    """模拟多轮对话，每轮打印 token 数变化。

    Args:
        messages: 消息列表（会被修改）
        num_rounds: 模拟的对话轮数
        label: 策略名称，用于显示

    Returns:
        每轮后的 token 数列表
    """
    token_history = []
    for i in range(min(num_rounds, len(SAMPLE_QUESTIONS))):
        question = SAMPLE_QUESTIONS[i]
        messages.append({"role": "user", "content": question})
        token_history.append(estimate_tokens(messages))

        # 调用 LLM 获取回复
        reply = call_llm(messages)
        messages.append({"role": "assistant", "content": reply})
        token_history.append(estimate_tokens(messages))

        print(f"    轮 {i+1}: 问题={question[:20]}... "
              f"回复={reply[:30]}... "
              f"token≈{estimate_tokens(messages)}")

    return token_history


# ============================================================
# Part 1: 无记忆 — 历史无限制增长
# ============================================================

def run_part1():
    """Part 1: 无记忆基线 — 展示 token 数无限制增长。"""
    print("\n" + "=" * 60)
    print("  Part 1: 无记忆 — 历史无限制增长")
    print("  " + "=" * 50)
    print("  每次对话都会累加所有历史，token 数持续增长。")
    print("  这是最简单的实现方式，但长对话会超出模型上下文限制。")
    print()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    token_counts = simulate_conversation(messages, num_rounds=8, label="无记忆")

    print(f"\n  📊 结果: 共 {len(messages)} 条消息，"
          f"最终 token≈{token_counts[-1]}")
    print(f"  ⚠️  问题: 对话越长，token 消耗越大。")
    print(f"     8 轮对话后 token 从 0 增长到 {token_counts[-1]}。")
    print(f"     如果继续对话 50 轮，会轻松超过模型的上下文限制。")

    input("\n  按 Enter 进入 Part 2...")


# ============================================================
# Part 2: 滑动窗口 — 只保留最近 N 条消息
# ============================================================

def run_part2():
    """Part 2: 滑动窗口 — 只保留最后 N 条消息。"""
    print("\n" + "=" * 60)
    print("  Part 2: 滑动窗口 — 只保留最近的消息")
    print("  " + "=" * 50)
    print("  策略: 只保留最近 6 条消息（3 轮问答），超出丢弃。")
    print("  好处: token 数被严格限制。")
    print("  代价: 超出窗口的上下文会丢失。")
    print()

    WINDOW_SIZE = 6  # 保留最近 3 轮问答（user + assistant 各 1 条 = 6 条）

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for i in range(min(8, len(SAMPLE_QUESTIONS))):
        question = SAMPLE_QUESTIONS[i]
        messages.append({"role": "user", "content": question})

        reply = call_llm(messages)
        messages.append({"role": "assistant", "content": reply})

        # 滑动窗口: 只保留 system + 最近 WINDOW_SIZE 条
        if len(messages) > WINDOW_SIZE + 1:  # +1 是 system
            # 保留 system 和最近 WINDOW_SIZE 条
            messages = (
                [messages[0]]  # system
                + messages[-WINDOW_SIZE:]
            )

        print(f"    轮 {i+1}: token≈{estimate_tokens(messages):>4}, "
              f"消息数={len(messages)}, "
              f"保留={[m['role'][:3] for m in messages[1:]]}")

    print(f"\n  📊 结果: 最终消息数被控制在 {WINDOW_SIZE + 1} 条，"
          f"token 数稳定在 ~{estimate_tokens(messages)}")
    print(f"  ✅ 好处: token 消耗可预测，不会超限。")
    print(f"  ⚠️  代价: 如果第 1 轮说过 \"磁盘是 SSD\"，")
    print(f"     第 5 轮再问 \"磁盘型号\" 时窗口已不包含这个信息。")

    input("\n  按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 摘要 — 超限时自动摘要旧消息
# ============================================================

def run_part3():
    """Part 3: 摘要 — token 超限时摘要旧消息。"""
    print("\n" + "=" * 60)
    print("  Part 3: 摘要 — token 超限时自动摘要旧消息")
    print("  " + "=" * 50)
    print("  策略: 当 token > 2000 时，调用 LLM 对旧消息做摘要。")
    print("  好处: 保留核心上下文，同时控制 token 消耗。")
    print("  代价: 额外 LLM 调用，摘要会丢失细节。")
    print()

    MAX_TOKENS = 2000  # 触发摘要的阈值
    SUMMARY_PROMPT = (
        "以下是运维助手的对话历史。请总结核心内容，"
        "保留关键的事实、已经执行的命令、发现的根因和做出的决策。"
        "摘要应该简洁但完整。用中文回复。"
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    summarized = False

    for i in range(min(8, len(SAMPLE_QUESTIONS))):
        question = SAMPLE_QUESTIONS[i]
        messages.append({"role": "user", "content": question})

        reply = call_llm(messages)
        messages.append({"role": "assistant", "content": reply})

        current_tokens = estimate_tokens(messages)

        # 检查是否需要摘要
        if current_tokens > MAX_TOKENS and not summarized:
            print(f"\n  ⚡ Token={current_tokens} 超过阈值 {MAX_TOKENS}，触发摘要...")

            # 收集要摘要的非 system 消息
            to_summarize = messages[1:]  # 去掉 system
            summary_messages = [
                {"role": "system", "content": SUMMARY_PROMPT},
                *to_summarize,
            ]

            summary = call_llm(summary_messages)
            summarized = True

            # 用摘要消息替换所有历史
            messages = [
                messages[0],  # system
                {"role": "assistant", "content": f"[对话摘要]\n{summary}"},
            ]

            print(f"  📝 摘要完成，消息数压缩为 {len(messages)} 条")
            print(f"  📝 摘要内容: {summary[:80]}...")

        print(f"    轮 {i+1}: token≈{estimate_tokens(messages):>4}, "
              f"消息数={len(messages)}")

    print(f"\n  📊 结果: 触发摘要后 token 从超过 {MAX_TOKENS} 降低到 "
          f"~{estimate_tokens(messages)}")
    print(f"  ✅ 好处: 核心上下文保留（摘要包含了之前的信息）。")
    print(f"  ⚠️  代价: 摘要是一次额外的 LLM 调用，增加了延迟和成本。")
    print(f"        且摘要会丢失原始对话的精确细节。")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  记忆管理示例 (Model: {model})")
    print(f"  输入 exit 或 quit 退出")
    print(f"{'='*60}")

    run_part1()
    run_part2()
    run_part3()

    print(f"\n{'='*60}")
    print(f"  总结")
    print(f"{'='*60}")
    print(f"""
  策略对比:
  ┌────────────┬────────────┬──────────────┬──────────────┐
  │ 策略       │ Token 控制 │ 上下文保留    │ 额外开销     │
  ├────────────┼────────────┼──────────────┼──────────────┤
  │ 无记忆     │ ❌ 无限制  │ ✅ 全部保留   │ 无           │
  │ 滑动窗口   │ ✅ 可控    │ ⚠️ 仅窗口内   │ 无           │
  │ 摘要       │ ✅ 可控    │ ✅ 核心保留   │ LLM 调用     │
  └────────────┴────────────┴──────────────┴──────────────┘

  实际生产环境中，通常组合使用:
    1. 默认用滑动窗口，保证 token 不超限
    2. 对关键对话用摘要保存核心信息
    3. 设置合理的窗口大小（如 20-50 条消息）
""")
    print("再见！\n")
