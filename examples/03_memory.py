"""
examples/03_memory.py — 三层记忆系统

学习目标:
  1. 理解为什么简单的滑动窗口不够用
  2. 理解三层记忆架构: 工作记忆 → 情景记忆 → 核心记忆
  3. 理解压缩（compaction）机制: 工作记忆超限时自动摘要
  4. 看到三层协同工作的效果

运行方式:
  python examples/03_memory.py

前置条件:
  已完成 01_simple_chat.py，理解基础对话流程

核心概念:
  - 工作记忆 (Working Memory): 当前对话，有界窗口
  - 情景记忆 (Episodic Memory): 压缩后的历史摘要，带时间索引
  - 核心记忆 (Core Memory): 持久化的稳定事实
  - 压缩 (Compaction): 工作记忆超限时 → LLM 摘要 → 存入情景记忆
"""

from openai import OpenAI

from _common import load_config, estimate_tokens, call_llm


# ============================================================
# 配置加载
# ============================================================
config = load_config()
model = config["model"]
client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])

SYSTEM_PROMPT = "你是一个 AIOps 运维助手。请简洁地回答运维相关问题。"

# 用于示例的模拟问题
SAMPLE_QUESTIONS = [
    "查看磁盘使用情况",
    "系统负载高怎么排查",
    "nginx 日志在哪里",
    "防火墙端口怎么开放",
    "怎么看进程列表",
    "磁盘 IO 怎么看",
    "网络延迟怎么测",
    "怎么配 crontab",
    "怎么看系统版本",
    "SSH 配置在哪",
]


# ============================================================
# Part 1: 问题引出 — 无记忆 vs 简单滑窗
# ============================================================

def run_part1():
    print("\n" + "=" * 60)
    print("  Part 1: 为什么需要更好的记忆系统？")
    print("=" * 60)

    # 场景 A: 无记忆
    print("\n  场景 A: 无记忆 — token 无限制增长")
    msgs_a = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(6):
        msgs_a.append({"role": "user", "content": SAMPLE_QUESTIONS[i]})
        reply = call_llm(client, model, msgs_a)
        msgs_a.append({"role": "assistant", "content": reply})
    print(f"    6 轮后: {len(msgs_a)} 条消息, ~{estimate_tokens(msgs_a)} tokens")
    print(f"    10 轮后: 预计 ~{estimate_tokens(msgs_a) * 10 // 6} tokens")
    print(f"    问题: 长对话会超出 LLM 上下文窗口")

    # 场景 B: 简单滑动窗口
    print(f"\n  场景 B: 滑动窗口 — 保留最近 4 条消息")
    msgs_b = [{"role": "system", "content": SYSTEM_PROMPT}]
    for i in range(6):
        msgs_b.append({"role": "user", "content": SAMPLE_QUESTIONS[i]})
        reply = call_llm(client, model, msgs_b)
        msgs_b.append({"role": "assistant", "content": reply})
        if len(msgs_b) > 5:  # 保留 system + 最近 4 条
            msgs_b = [msgs_b[0]] + msgs_b[-4:]
    print(f"    始终 ~5 条消息, ~{estimate_tokens(msgs_b)} tokens")
    print(f"    问题: 第 1 轮的信息被丢弃了")

    print(f"\n  💡 我们需要一种既能控制 token、又能保留核心上下文的方案")
    print(f"     → 三层记忆系统")

    input("\n  按 Enter 进入 Part 2...")


# ============================================================
# Part 2: 三层记忆概念
# ============================================================

def run_part2():
    print("\n" + "=" * 60)
    print("  Part 2: 三层记忆架构")
    print("=" * 60)

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │              发送给 LLM 的组合上下文                   │
  ├─────────────────────────────────────────────────────┤
  │  system_prompt (Agent 设定)                          │
  │  ┌─────────────────────────────────────────────┐    │
  │  │ 核心记忆 (Core Memory)                       │    │
  │  │ 持久化事实: 服务器配置、用户偏好、固定决策       │    │
  │  └─────────────────────────────────────────────┘    │
  │  ┌─────────────────────────────────────────────┐    │
  │  │ 情景记忆 (Episodic Memory)                   │    │
  │  │ 最近 K 个压缩摘要: 历史对话的核心信息           │    │
  │  └─────────────────────────────────────────────┘    │
  │  ┌─────────────────────────────────────────────┐    │
  │  │ 工作记忆 (Working Memory)                    │    │
  │  │ 当前对话的最近 N 条消息                        │    │
  │  └─────────────────────────────────────────────┘    │
  │  user_input (用户当前输入)                          │
  └─────────────────────────────────────────────────────┘

  三层职责:
  ┌──────────────┬──────────────────┬──────────────────┐
  │ 层级          │ 生命周期          │ 管理方式          │
  ├──────────────┼──────────────────┼──────────────────┤
  │ 工作记忆      │ 本次对话           │ 自动滑窗          │
  │ 情景记忆      │ 跨轮次             │ 自动压缩 + 持久化 │
  │ 核心记忆      │ 跨会话             │ 手动/自动管理     │
  └──────────────┴──────────────────┴──────────────────┘

  压缩流程 (Compaction):
  工作记忆超限 → 提取最旧 75% 消息 → LLM 生成摘要
  → 存入情景记忆 → 从工作记忆中移除
""")

    input("\n  按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 模拟三层记忆工作
# ============================================================

def run_part3():
    print("\n" + "=" * 60)
    print("  Part 3: 模拟三层记忆工作")
    print("=" * 60)

    # 用简化的手写实现演示三层概念
    MAX_WORKING_MSGS = 6  # 工作记忆最大消息数

    # 三个"层级"的数据
    working = []           # 工作记忆
    episodes = []          # 情景记忆（episode 列表）
    core_facts = [         # 核心记忆（预设事实）
        "本机是一台 Linux 服务器",
    ]

    COMPACTION_PROMPT = (
        "以下是运维助手的对话记录。请简要总结核心内容，保留关键的技术细节和发现。"
    )

    print(f"\n  初始状态:")
    print(f"    核心记忆: {core_facts}")
    print(f"    情景记忆: 空")
    print(f"    工作记忆: 空")
    print(f"    工作记忆上限: {MAX_WORKING_MSGS} 条消息")

    for i, question in enumerate(SAMPLE_QUESTIONS):
        # 用户提问
        working.append({"role": "user", "content": question})

        # LLM 回复
        full_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 构建组合上下文: system + core + episodes + working
        if core_facts:
            full_msgs.append({
                "role": "system",
                "content": "核心知识:\n" + "\n".join(f"- {f}" for f in core_facts),
            })
        for ep in episodes[-2:]:  # 最近 2 个 episode
            full_msgs.append({"role": "assistant", "content": f"[历史摘要] {ep['summary']}"})
        full_msgs.extend(working)

        reply = call_llm(client, model, full_msgs)
        working.append({"role": "assistant", "content": reply})

        # 显示当前轮次的状态
        combined_tokens = estimate_tokens(full_msgs)
        print(f"\n    第 {i+1} 轮: {question}")
        print(f"      回复: {reply[:40]}...")
        print(f"      组合上下文: {len(full_msgs)} 条消息, ~{combined_tokens} tokens")
        print(f"      工作记忆: {len(working)} 条")

        # 检查是否需要压缩
        non_system = [m for m in working if m.get("role") != "system"]
        if len(non_system) >= MAX_WORKING_MSGS:
            print(f"      ⚡ 工作记忆超限, 触发压缩...")

            # 提取最旧的 70% 供压缩
            compact_count = max(2, int(len(non_system) * 0.7))
            to_compact = non_system[:compact_count]

            # 构建压缩提示
            lines = [f"[{m['role']}]: {m['content']}" for m in to_compact]
            compact_prompt = [{
                "role": "system",
                "content": COMPACTION_PROMPT + "\n\n" + "\n".join(lines)
            }]

            # 调用 LLM 生成摘要
            summary = call_llm(client, model, compact_prompt)

            # 存入情景记忆
            episodes.append({
                "summary": summary[:100],
                "timestamp": f"第 {i+1} 轮",
                "message_count": len(to_compact),
            })
            print(f"      📝 压缩了 {len(to_compact)} 条消息")

            # 从工作记忆中移除
            compact_indices = {id(m) for m in to_compact}
            working = [m for m in working if id(m) not in compact_indices]
            print(f"      工作记忆剩余: {len(working)} 条")

        # 在第 4 轮后添加一条核心记忆
        if i == 3:
            core_facts.append("磁盘使用率超过 80% 时需要告警")
            print(f"      💾 添加核心记忆: 磁盘使用率超过 80% 时需要告警")

    # 最终展示
    print(f"\n  {'='*50}")
    print(f"  最终状态:")
    print(f"    核心记忆: {len(core_facts)} 条事实")
    for f in core_facts:
        print(f"      • {f}")
    print(f"    情景记忆: {len(episodes)} 个片段")
    for ep in episodes:
        print(f"      • [{ep['timestamp']}] {ep['summary'][:50]}...")
    print(f"    工作记忆: {len(working)} 条消息（当前对话）")

    # 展示组合上下文
    print(f"\n  最终组合上下文 ({estimate_tokens(full_msgs)} tokens):")
    print(f"    {len(core_facts)} 条核心记忆 → 作为 system 消息")
    print(f"    {min(2, len(episodes))} 个情景摘要 → 作为 assistant 消息")
    print(f"    {len(working)} 条工作记忆 → 当前对话")

    input("\n  按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 策略对比
# ============================================================

def run_part4():
    print("\n" + "=" * 60)
    print("  Part 4: 策略对比总结")
    print("=" * 60)

    print(f"""
  记忆策略对比:
  ┌─────────────┬───────────┬───────────────┬────────────────┐
  │ 策略         │ Token 控制 │ 上下文保留     │ 关键限制        │
  ├─────────────┼───────────┼───────────────┼────────────────┤
  │ 无记忆       │ ❌ 无限制 │ ✅ 全部        │ 会超限          │
  │ 简单滑窗     │ ✅ 可控   │ ⚠️ 仅窗口内    │ 丢失旧上下文     │
  │ 简单摘要     │ ✅ 可控   │ ⚠️ 一次摘要     │ 多次摘要会丢失   │
  │ 三层记忆     │ ✅ 可控   │ ✅ 分层保留     │ 需要 LLM 摘要   │
  └─────────────┴───────────┴───────────────┴────────────────┘

  三层记忆的优势:
  1. 工作记忆保证当前对话的流畅性
  2. 情景记忆通过多次压缩保留历史全貌
  3. 核心记忆跨会话持久化重要知识
  4. 压缩是增量的——每次只压缩最旧的部分
""")

    print("再见！\n")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"  三层记忆系统示例 (Model: {model})")
    print(f"{'='*60}")

    run_part1()
    run_part2()
    run_part3()
    run_part4()
