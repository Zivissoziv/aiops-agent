# d:\workspace\aiops-agent\src\aiops_agent\agents\planner.py
"""规划 Agent — 分析任务，制定执行计划。

提示词由 cli.py 动态注入，包含当前所有可用 Agent 的描述。
"""

AGENT_DEF = {
    "name": "planner",
    "system_prompt": None,  # 由 cli.py 动态生成
    "tools": [],
}
