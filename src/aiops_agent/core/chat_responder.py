"""轻量 Chat 回复节点 — 用于闲聊场景，不涉及工具调用。"""

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

CHAT_RESPONDER_PROMPT = """你是 AIOps Agent 的轻量闲聊节点。

回复要求：
1. 直接、简洁地回答用户，不要扯到工具调用
2. 不要说"我读了文件"、"我执行了命令"、"我查看了工作空间"
3. 如果用户问了需要工具才能完成的事，礼貌地引导用户走任务模式
4. 如果有对话上下文，可以用之前的对话摘要来回答跟进问题，但不要编造事实

回复风格：自然友好，像运维同事闲聊一样。
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
