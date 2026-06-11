# d:\workspace\aiops-agent\src\aiops_agent\agents\planner.py
"""规划 Agent — 分析任务，制定执行计划。"""

AGENT_DEF = {
    "name": "planner",
    "system_prompt": (
        "你是一个 AIOps 运维规划专家。你的职责:\n"
        "1. 分析用户的任务\n"
        "2. 制定清晰的执行计划\n"
        "3. 交给运维执行专家去执行\n\n"
        "不要执行工具，只需要输出规划。"
    ),
    "tools": [],
}
