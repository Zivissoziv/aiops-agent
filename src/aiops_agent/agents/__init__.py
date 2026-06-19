from .planner import AGENT_DEF as PLANNER_DEF
from .worker import AGENT_DEF as WORKER_DEF
ALL_AGENTS: list[dict] = [PLANNER_DEF, WORKER_DEF]
__all__ = ["ALL_AGENTS"]
