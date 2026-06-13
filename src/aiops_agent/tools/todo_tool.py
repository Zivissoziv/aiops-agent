# d:\workspace\aiops-agent\src\aiops_agent\tools\todo_tool.py
"""TODO 管理工具 — Worker 在完成任务时更新状态。

工具本身只做输入校验和格式化返回。
真正的状态更新由 complex.py 的 node_fn 扫描 tool_call 后写入 LangGraph state。
"""

import json

from langchain_core.tools import tool

VALID_STATUSES = ("in_progress", "completed", "blocked")


@tool
def update_todo(todo_id: str, status: str) -> str:
    """更新 TODO 状态。

    每完成一个 TODO 步骤，调用此工具标记完成状态。
    全部 TODO 标记为 completed 后，系统会自动结束流程。

    Args:
        todo_id: TODO 的 ID（如 todo-1, todo-2）
        status: 新状态，可选值: in_progress, completed, blocked

    Returns:
        操作结果 JSON，结构: {"success": bool, "output": str}
    """
    if status not in VALID_STATUSES:
        return json.dumps({
            "success": False,
            "output": f"❌ 无效状态 '{status}'，可选: {', '.join(VALID_STATUSES)}",
        }, ensure_ascii=False)

    return json.dumps({
        "success": True,
        "output": f"✅ {todo_id} 已标记为 {status}",
    }, ensure_ascii=False)
