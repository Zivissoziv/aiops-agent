"""轻量 Chat 回复节点 — 用于闲聊场景，不涉及工具调用。"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

CHAT_RESPONDER_PROMPT = """You are AIOps Agent's lightweight chat node.

Answer the user directly and concisely. Do not claim that you read files,
searched the web, ran commands, edited files, or inspected the workspace.
If the user asks for work requiring tools or project context, say that it
should be handled by the task route (try describing what you need done).

If session context is provided, you may use the recent conversation summary to
answer conversational follow-ups, but do not invent workspace facts.
"""


def chat_responder(
    llm,
    user_input: str,
    session_context: str = "",
    reason: str = "",
) -> dict[str, Any]:
    """轻量聊天回复，不走 graph / 不触发工具。

    返回:
        {"response": str, "reason": str}
    """
    try:
        parts = [f"User input:\n{user_input}"]
        if session_context:
            parts.append(f"Session context:\n{session_context}")
        prompt_input = "\n\n".join(parts)

        response = llm.invoke([
            SystemMessage(content=CHAT_RESPONDER_PROMPT),
            HumanMessage(content=prompt_input),
        ])
        text = str(getattr(response, "content", "") or "").strip()
    except Exception as exc:
        text = f"轻量聊天分支暂不可用: {type(exc).__name__}: {exc}"

    if not text:
        text = "我在。你可以继续提问，或者直接描述一个需要我完成的任务。"

    return {
        "response": text,
        "reason": reason,
    }
