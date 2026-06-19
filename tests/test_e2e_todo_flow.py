"""
端到端测试：路由函数 + Worker 部分完成续做。
"""

import os
import pytest

from aiops_agent.config import Config
from aiops_agent.graph import AppState, build_complex_graph
from aiops_agent.llm import create_llm
from aiops_agent.memory.tiered import TieredMemory

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ──────────────────────────────────────────────
# 路由函数单元测试
# ──────────────────────────────────────────────

class TestRouteFunctions:

    def test_route_planner(self):
        from aiops_agent.graph.complex import _route_planner
        assert _route_planner({"need_worker": True}, "worker") == "worker"
        assert _route_planner({"need_worker": False}, "worker") == "__end__"
        assert _route_planner({}, "worker") == "worker"  # 默认 True

    def test_route_worker_all_completed(self):
        from aiops_agent.graph.complex import _route_worker
        state = {
            "todos": [{"id": "t1", "status": "completed"}, {"id": "t2", "status": "completed"}],
            "worker_round": 1,
        }
        assert _route_worker(state, "worker") == "__end__"

    def test_route_worker_partial(self):
        from aiops_agent.graph.complex import _route_worker
        state = {
            "todos": [{"id": "t1", "status": "completed"}, {"id": "t2", "status": "pending"}],
            "worker_round": 1,
        }
        assert _route_worker(state, "worker") == "worker"

    def test_route_worker_all_blocked(self):
        from aiops_agent.graph.complex import _route_worker
        state = {
            "todos": [{"id": "t1", "status": "blocked"}, {"id": "t2", "status": "blocked"}],
            "worker_round": 1,
        }
        assert _route_worker(state, "worker") == "__end__"

    def test_route_worker_exceeds_max(self):
        from aiops_agent.graph.complex import _route_worker
        assert _route_worker({"todos": [{"id": "t1", "status": "pending"}], "worker_round": 3}, "worker") == "__end__"

    def test_route_worker_empty(self):
        from aiops_agent.graph.complex import _route_worker
        assert _route_worker({"todos": [], "worker_round": 0}, "worker") == "__end__"
        assert _route_worker({"worker_round": 0}, "worker") == "__end__"


# ──────────────────────────────────────────────
# 解析函数单元测试
# ──────────────────────────────────────────────

class TestParseTodoItems:

    def test_basic(self):
        from aiops_agent.graph.complex import _parse_todos
        result = _parse_todos(["[worker] 查看磁盘", "[worker] 查看内存"])
        assert len(result) == 2
        assert result[0]["id"] == "todo-1"
        assert result[0]["content"] == "查看磁盘"
        assert result[0]["assignee"] == "worker"

    def test_empty(self):
        from aiops_agent.graph.complex import _parse_todos
        assert _parse_todos([]) == []

    def test_missing_brackets_defaults_worker(self):
        from aiops_agent.graph.complex import _parse_todos
        assert _parse_todos(["看磁盘"])[0]["assignee"] == "worker"


# ──────────────────────────────────────────────
# e2e 测试（需要真实 API）
# ──────────────────────────────────────────────

requires_llm = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="需要 LLM API 配置",
)


def _make_memory(llm, data_dir):
    return TieredMemory(
        llm=llm,
        compaction_enabled=False,
        working_max_messages=10,
        working_max_tokens=500,
        core_persist_path=str(data_dir / "core_memory.json"),
        episodic_persist_path=str(data_dir / "episodic_memory.json"),
    )


@requires_llm
def test_e2e_partial_completion_then_continue(tmp_path):
    """
    场景：Worker 第一次只完成 1/2 TODO，第二次续做完成剩余的。
    """
    config = Config.from_env()
    llm = create_llm(config)
    memory = _make_memory(llm, tmp_path)
    graph = build_complex_graph(config, llm, memory)

    initial: AppState = {
        "messages": [],
        "task": "查看磁盘使用率和主机名",
        "need_worker": True,
        "todos": [
            {"id": "todo-1", "content": "使用 df 命令查看磁盘分区使用率", "assignee": "worker", "status": "pending"},
            {"id": "todo-2", "content": "使用 hostname 命令查看主机名", "assignee": "worker", "status": "pending"},
        ],
        "worker_round": 0,
        "intent_route": "",
        "intent_reason": "",
        "intent_confidence": 0.0,
        "chat_response": "",
        "session_context": "",
    }

    max_rounds = 0
    final_todos = []

    for mode, event in graph.stream(initial, stream_mode=["updates", "custom"]):
        if mode == "updates":
            for node_name, data in event.items():
                if isinstance(data, dict):
                    wr = data.get("worker_round", 0)
                    if wr > max_rounds:
                        max_rounds = wr
                    todos = data.get("todos", [])
                    if todos:
                        final_todos = todos
                        statuses = {t["id"]: t["status"] for t in todos}
                        print(f"  [{node_name}] round={wr} {statuses}")

    print(f"\n最终: {max_rounds} 轮完成, todos: {[t['status'] for t in final_todos]}")
    assert max_rounds >= 1, "至少执行了 1 轮"
    assert final_todos, "应有 TODO 项"
    assert all(t["status"] in ("completed", "blocked") for t in final_todos), \
        "所有 TODO 最终应 completed 或 blocked"
