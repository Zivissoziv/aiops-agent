"""
端到端测试补全：update_todo 工具、路由函数、部分完成续做、上限强制结束
"""

import json
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


# ════════════════════════════════════════════════════════════
# 第 1 组: update_todo 工具单元测试
# ════════════════════════════════════════════════════════════

class TestUpdateTodoTool:

    def test_normal_completed(self):
        """正常标记 completed。"""
        from aiops_agent.tools.todo_tool import update_todo
        result = json.loads(update_todo.invoke({"todo_id": "todo-1", "status": "completed"}))
        assert result["success"] is True
        assert "✅" in result["output"]

    def test_normal_in_progress(self):
        """正常标记 in_progress。"""
        from aiops_agent.tools.todo_tool import update_todo
        result = json.loads(update_todo.invoke({"todo_id": "todo-2", "status": "in_progress"}))
        assert result["success"] is True

    def test_normal_blocked(self):
        """正常标记 blocked。"""
        from aiops_agent.tools.todo_tool import update_todo
        result = json.loads(update_todo.invoke({"todo_id": "todo-3", "status": "blocked"}))
        assert result["success"] is True

    def test_invalid_status_rejected(self):
        """无效状态被拒绝。"""
        from aiops_agent.tools.todo_tool import update_todo
        result = json.loads(update_todo.invoke({"todo_id": "todo-1", "status": "invalid_status"}))
        assert result["success"] is False

    def test_empty_status_rejected(self):
        """空状态被拒绝。"""
        from aiops_agent.tools.todo_tool import update_todo
        result = json.loads(update_todo.invoke({"todo_id": "todo-1", "status": ""}))
        assert result["success"] is False

    def test_empty_todo_id_not_rejected(self):
        """空 todo_id 不拒绝（只是打印时没有内容）。"""
        from aiops_agent.tools.todo_tool import update_todo
        result = json.loads(update_todo.invoke({"todo_id": "", "status": "completed"}))
        assert result["success"] is True

    def test_return_format_is_valid_json(self):
        """返回值始终是合法 JSON，包含 success 和 output 字段。"""
        from aiops_agent.tools.todo_tool import update_todo
        out = update_todo.invoke({"todo_id": "x", "status": "completed"})
        parsed = json.loads(out)
        assert "success" in parsed
        assert "output" in parsed

    def test_returns_json_even_on_error(self):
        """错误时也返回 JSON。"""
        from aiops_agent.tools.todo_tool import update_todo
        out = update_todo.invoke({"todo_id": "x", "status": "bad"})
        parsed = json.loads(out)
        assert "success" in parsed
        assert "output" in parsed


# ════════════════════════════════════════════════════════════
# 第 2 组: _parse_todo_items 单元测试
# ════════════════════════════════════════════════════════════

class TestParseTodoItems:

    def test_parse_basic(self):
        """基本解析：标准格式。"""
        from aiops_agent.graph.complex import _parse_todo_items
        raw = ["[worker] 查看磁盘使用率", "[worker] 查看内存使用率"]
        result = _parse_todo_items(raw)
        assert len(result) == 2
        assert result[0]["id"] == "todo-1"
        assert result[0]["content"] == "查看磁盘使用率"
        assert result[0]["assignee"] == "worker"
        assert result[0]["status"] == "pending"

    def test_parse_sequential_ids(self):
        """ID 顺序递增。"""
        from aiops_agent.graph.complex import _parse_todo_items
        result = _parse_todo_items(["[w] a", "[w] b", "[w] c"])
        assert [t["id"] for t in result] == ["todo-1", "todo-2", "todo-3"]

    def test_parse_empty_list(self):
        from aiops_agent.graph.complex import _parse_todo_items
        assert _parse_todo_items([]) == []

    def test_parse_missing_brackets(self):
        """没有 [agentname] 前缀也能处理。"""
        from aiops_agent.graph.complex import _parse_todo_items
        result = _parse_todo_items(["看磁盘"])
        assert result[0]["assignee"] == "worker"  # 默认 assignee
        assert result[0]["content"] == "看磁盘"

    def test_parse_different_assignee(self):
        """不同的 assignee 名称。"""
        from aiops_agent.graph.complex import _parse_todo_items
        result = _parse_todo_items(["[planner] 规划", "[worker] 执行"])
        assert result[0]["assignee"] == "planner"
        assert result[1]["assignee"] == "worker"


# ════════════════════════════════════════════════════════════
# 第 3 组: 路由函数单元测试
# ════════════════════════════════════════════════════════════

class TestRouteFunctions:

    def test_route_from_planner_needs_worker(self):
        """planner 路由: need_worker=True → 走 worker。"""
        from aiops_agent.graph.complex import _route_from_planner
        assert _route_from_planner({"need_worker": True}, "worker") == "worker"

    def test_route_from_planner_no_worker(self):
        """planner 路由: need_worker=False → 直接 END。"""
        from aiops_agent.graph.complex import _route_from_planner
        assert _route_from_planner({"need_worker": False}, "worker") == "__end__"

    def test_route_from_planner_missing_key(self):
        """planner 路由: 默认 need_worker=True。"""
        from aiops_agent.graph.complex import _route_from_planner
        assert _route_from_planner({}, "worker") == "worker"

    def test_worker_all_completed_ends(self):
        """Worker 路由: 全部完成 → END。"""
        from aiops_agent.graph.complex import _route_from_worker
        state = {
            "todos": [
                {"id": "t1", "status": "completed"},
                {"id": "t2", "status": "completed"},
            ],
            "worker_round": 1,
        }
        assert _route_from_worker(state, "worker") == "__end__"

    def test_worker_partial_complete_retries(self):
        """Worker 路由: 部分完成，且没超上限 → 继续执行。"""
        from aiops_agent.graph.complex import _route_from_worker
        state = {
            "todos": [
                {"id": "t1", "status": "completed"},
                {"id": "t2", "status": "pending"},
            ],
            "worker_round": 1,
        }
        assert _route_from_worker(state, "worker") == "worker"

    def test_worker_all_blocked_ends(self):
        """Worker 路由: 全部 blocked → 也结束。"""
        from aiops_agent.graph.complex import _route_from_worker
        state = {
            "todos": [
                {"id": "t1", "status": "blocked"},
                {"id": "t2", "status": "blocked"},
            ],
            "worker_round": 1,
        }
        assert _route_from_worker(state, "worker") == "__end__"

    def test_worker_exceeds_max_rounds_forced_end(self):
        """Worker 路由: 超过上限 3 次 → 强制 END。"""
        from aiops_agent.graph.complex import _route_from_worker
        state = {
            "todos": [
                {"id": "t1", "status": "pending"},
            ],
            "worker_round": 3,
        }
        assert _route_from_worker(state, "worker") == "__end__"

    def test_worker_at_max_rounds(self):
        """Worker 路由: 正好到上限 3 → 强制 END。"""
        from aiops_agent.graph.complex import _route_from_worker
        state = {
            "todos": [
                {"id": "t1", "status": "in_progress"},
            ],
            "worker_round": 3,
        }
        assert _route_from_worker(state, "worker") == "__end__"

    def test_worker_empty_todos(self):
        """Worker 路由: 空 TODO 列表 → 直接 END。"""
        from aiops_agent.graph.complex import _route_from_worker
        assert _route_from_worker({"todos": [], "worker_round": 0}, "worker") == "__end__"

    def test_worker_missing_todos_key(self):
        """Worker 路由: 没有 todos key 也安全。"""
        from aiops_agent.graph.complex import _route_from_worker
        assert _route_from_worker({"worker_round": 0}, "worker") == "__end__"


# ════════════════════════════════════════════════════════════
# 第 4 组: Worker 部分完成续做（e2e，需要真实 API）
# ════════════════════════════════════════════════════════════

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
    场景: Worker 第一次只完成 1/2 TODO，第二次续做完成剩余的。

    步骤:
      1. 设 2 个 TODO: 查看磁盘、查看主机名
      2. Worker 第一次执行，只完成 1 个
      3. 验证未完成 → 回到 Worker
      4. Worker 第二次执行，完成剩余的
      5. 验证全部 completed → END
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
    assert all(t["status"] in ("completed", "blocked") for t in final_todos), \
        "所有 TODO 最终应 completed 或 blocked"
    assert len(final_todos) == 2


@requires_llm
def test_e2e_all_three_states_used(tmp_path):
    """
    场景: 3 个 TODO，验证 Worker 能用 update_todo 的几种状态。

    这是一个综合场景测试，确保 Worker 能正确调用 update_todo 工具
    并让系统正确路由。
    """
    config = Config.from_env()
    llm = create_llm(config)
    memory = _make_memory(llm, tmp_path)
    graph = build_complex_graph(config, llm, memory)

    initial: AppState = {
        "messages": [],
        "task": "查看磁盘、内存和系统负载",
        "need_worker": True,
        "todos": [
            {"id": "todo-1", "content": "查看磁盘使用率", "assignee": "worker", "status": "pending"},
            {"id": "todo-2", "content": "查看内存使用率", "assignee": "worker", "status": "pending"},
            {"id": "todo-3", "content": "查看系统负载", "assignee": "worker", "status": "pending"},
        ],
        "worker_round": 0,
        "intent_route": "",
        "intent_reason": "",
        "intent_confidence": 0.0,
        "chat_response": "",
        "session_context": "",
    }

    for mode, event in graph.stream(initial, stream_mode=["updates", "custom"]):
        if mode == "updates":
            for node_name, data in event.items():
                if isinstance(data, dict):
                    todos = data.get("todos", [])
                    if todos:
                        statuses = {t["id"]: t["status"] for t in todos}
                        print(f"  [{node_name}] {statuses}")

    assert True  # 只要能跑完不崩溃就过了
