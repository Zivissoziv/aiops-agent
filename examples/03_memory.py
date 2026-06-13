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

from _ui import console, title, note, success, info, diagram, divider, wait_for_enter, make_table


# ============================================================
# Part 1: 问题引出 — 无记忆 vs 简单滑窗
# ============================================================

def run_part1():
    title("Part 1: 为什么需要更好的记忆系统？")

    console.print("  [bold]场景 A:[/bold] 无记忆 — token 无限制增长")
    console.print("    对话越长，消息越多，最终[red]超出 LLM 上下文窗口[/red]")

    console.print("\n  [bold]场景 B:[/bold] 滑动窗口 — 只保留最近 N 条")
    console.print("    控制了 token 数量，但[red]旧信息被直接丢弃[/red]")

    diagram("""
  无记忆:  消息1 消息2 消息3 ... 消息N   → 超限 x
  滑窗:    消息1 消息2 消息3 消息4     → 消息5 进来，消息1 丢掉
            丢掉的消息再也找不回来了
    """)

    note("我们需要一种方案: 控制 token，又能保留核心上下文")
    console.print("    → [bold]三层记忆系统[/bold]\n")

    wait_for_enter("按 Enter 进入 Part 2...")


# ============================================================
# Part 2: 三层记忆概念
# ============================================================

def run_part2():
    title("Part 2: 三层记忆架构")

    diagram("""
  ┌─────────────────────────────────────────────┐
  │           发给 LLM 的组合上下文                │
  ├─────────────────────────────────────────────┤
  │  system_prompt (Agent 设定)                  │
  │  ┌──────────────────────────────────────┐   │
  │  │ 核心记忆: 持久化事实                   │   │
  │  │ 跨会话保留 (服务器配置、用户偏好)       │   │
  │  └──────────────────────────────────────┘   │
  │  ┌──────────────────────────────────────┐   │
  │  │ 情景记忆: 最近 K 个压缩摘要            │   │
  │  │ 跨轮次保留 (历史对话的核心信息)         │   │
  │  └──────────────────────────────────────┘   │
  │  ┌──────────────────────────────────────┐   │
  │  │ 工作记忆: 当前对话的最近 N 条消息       │   │
  │  └──────────────────────────────────────┘   │
  │  user_input (用户当前输入)                  │
  └─────────────────────────────────────────────┘
    """)

    console.print("  [bold]三层职责:[/bold]")
    table = make_table(headers=["层级", "生命周期", "管理方式"])
    table.add_row("工作记忆", "本次对话", "自动滑窗")
    table.add_row("情景记忆", "跨轮次", "自动压缩 + 持久化")
    table.add_row("核心记忆", "跨会话", "手动/自动管理")
    console.print(table)

    console.print("""
  [bold]压缩流程 (Compaction):[/bold]
  工作记忆超限 → 提取最旧消息 → LLM 生成摘要
  → 存入情景记忆 → 从工作记忆中移除
    """)

    wait_for_enter("按 Enter 进入 Part 3...")


# ============================================================
# Part 3: 三层记忆运作演示（模拟数据，不调 LLM）
# ============================================================

def run_part3():
    title("Part 3: 三层记忆运作演示")

    # 模拟的对话轮次和关键事件
    # 每轮: (轮次, 用户问题, 助手回复摘要, 事件标记)
    SIMULATED_ROUNDS = [
        (1, "查看磁盘使用情况", "建议用 df -h 查看"),
        (2, "系统负载高怎么排查", "用 top 看 CPU，iotop 看 IO"),
        (3, "nginx 日志在哪里", "默认在 /var/log/nginx/"),
        (4, "防火墙端口怎么开放", "用 firewall-cmd 或 iptables"),
        (5, "怎么看进程列表", "ps aux 查看所有进程"),
        (6, "磁盘 IO 怎么看", "iostat -x 1 监控"),
        (7, "网络延迟怎么测", "ping 和 traceroute"),
        (8, "怎么配 crontab", "crontab -e 编辑"),
        (9, "怎么看系统版本", "cat /etc/os-release"),
        (10, "SSH 配置在哪", "/etc/ssh/sshd_config"),
    ]

    MAX_WORKING = 6     # 工作记忆上限（6 条）
    CORE_FACTS = [      # 核心记忆（预设）
        "本机是一台 Linux 服务器",
    ]
    ADD_CORE_AT = 4     # 第 4 轮后添加一条核心记忆

    working: list[str] = []  # 只存消息描述，不调 LLM
    episodes: list[dict] = []
    cores = list(CORE_FACTS)

    info(f"工作记忆上限: [bold]{MAX_WORKING}[/bold] 条消息")
    info(f"核心记忆: {cores[0]}")
    info("情景记忆: 空")
    info("工作记忆: 空\n")

    for round_num, question, reply in SIMULATED_ROUNDS:
        msg = f"User: {question} -> {reply}"
        working.append(msg)

        # 判断是否需要压缩
        need_compact = len(working) >= MAX_WORKING

        # 显示当前轮次
        if need_compact:
            # 提取最旧的 4 条压缩
            old_msgs = working[:4]
            summary = f"讨论了磁盘、负载、nginx、防火墙等运维问题" if round_num <= 6 else f"讨论了进程、IO、网络、cron 等运维问题"
            episodes.append({"summary": summary, "round": f"第 {round_num} 轮"})
            working = working[4:]

            console.print(f"  [bold]第 {round_num} 轮:[/bold] {question}")
            info(f"助手: {reply}")
            console.print(f"  [yellow]> 工作记忆超限![/yellow] 压缩 [dim]{len(old_msgs)}[/dim] 条 → 情景记忆")
            console.print(f"    摘要: [dim]{summary}[/dim]")
        else:
            console.print(f"  [bold]第 {round_num} 轮:[/bold] {question}")
            info(f"助手: {reply}")

        # 添加核心记忆
        if round_num == ADD_CORE_AT:
            cores.append("磁盘使用率超过 80% 时需要告警")
            console.print(f"  [yellow]+ 添加核心记忆:[/yellow] 磁盘使用率超过 80% 时需要告警")

        # 显示当前三层状态
        console.print(f"  [dim]三层状态 →[/dim] [bold]核心:[/bold]{len(cores)}条  [bold]情景:[/bold]{len(episodes)}个  [bold]工作:[/bold]{len(working)}条\n")

    # 最后展示组合上下文的构造
    divider("组合上下文构造")
    console.print("""
  发给 LLM 的不是全部 10 轮消息，而是:
    """)
    console.print(f"  [bold]核心记忆[/bold] ({len(cores)} 条):")
    for c in cores:
        console.print(f"    - {c}")
    console.print(f"  [bold]情景记忆[/bold] (最近 {min(2, len(episodes))} 个摘要):")
    for ep in episodes[-2:]:
        console.print(f"    - [{ep['round']}] [dim]{ep['summary']}[/dim]")
    console.print(f"  [bold]工作记忆[/bold] (最近 {len(working)} 条消息):")
    for w in working:
        console.print(f"    - {w}")

    console.print(f"""
  [green]v 总共 ~{len(cores) * 20 + len(episodes[-2:]) * 30 + len(working) * 25} tokens[/green]
  [green]v 10 轮对话完整保留（核心+情景摘要+最新工作记忆）[/green]
  [green]v 没有丢失重要信息，也没有超出上下文窗口[/green]
    """)

    wait_for_enter("按 Enter 进入 Part 4...")


# ============================================================
# Part 4: 策略对比
# ============================================================

def run_part4():
    title("Part 4: 记忆策略对比")

    table = make_table(headers=["策略", "Token 控制", "旧信息", "代表场景"])
    table.add_row("无记忆", "[red]x 不控制[/red]", "[green]v 全部保留[/green]", "简单问答")
    table.add_row("滑动窗口", "[green]v 可控[/green]", "[red]x 直接丢弃[/red]", "一次性查询")
    table.add_row("简单摘要", "[green]v 可控[/green]", "[yellow]! 一次摘要[/yellow]", "客服对话")
    table.add_row("[bold]三层记忆[/bold]", "[green]v 可控[/green]", "[green]v 分层保留[/green]", "[bold]运维 Agent[/bold]")
    console.print(table)

    console.print("""
  [bold]三层记忆的关键优势:[/bold]
  1. 工作记忆保证当前对话流畅
  2. 情景记忆通过增量压缩保留历史全貌
  3. 核心记忆跨会话持久化重要知识
  4. 压缩是增量的——每次只压缩最旧部分
    """)

    console.print("[bold cyan]再见！[/bold cyan]")


# ============================================================
# 运行
# ============================================================

if __name__ == "__main__":
    console.rule("[bold cyan]三层记忆系统示例[/bold cyan]")
    console.rule()

    run_part1()
    run_part2()
    run_part3()
    run_part4()
