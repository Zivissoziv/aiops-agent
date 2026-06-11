# d:\workspace\aiops-agent\src\aiops_agent\agents\__init__.py
"""Agent 定义注册 — 所有 Agent 的配置和角色声明。

扩展方式:
  1. 在 agents/ 下新建文件（如 log_analyzer.py）
  2. 定义 AGENT_DEF 和 run() 函数
  3. 在此文件中注册
"""

from .planner import AGENT_DEF as PLANNER_DEF
from .worker import AGENT_DEF as WORKER_DEF

# 所有 Agent 定义（顺序 = 图中的执行顺序）
ALL_AGENTS: list[dict] = [
    PLANNER_DEF,
    WORKER_DEF,
]

__all__ = ["ALL_AGENTS"]
