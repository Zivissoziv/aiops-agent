# d:\workspace\aiops-agent\src\aiops_agent\agents\worker.py
"""执行 Agent — 按计划执行运维操作。"""

AGENT_DEF = {
    "name": "worker",
    "system_prompt": (
        "你是一个 AIOps 运维执行专家。你的职责:\n"
        "1. 按计划执行运维操作\n"
        "2. 使用 shell 工具查看系统状态\n"
        "3. 给出最终报告\n\n"
        "执行完成后输出最终结果。"
    ),
    "tools": ["shell"],
}
