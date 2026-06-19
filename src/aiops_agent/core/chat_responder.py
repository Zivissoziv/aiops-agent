"""轻量 Chat 回复节点 — 用于闲聊场景，不涉及工具调用。"""

from langchain_core.messages import HumanMessage, SystemMessage

PROMPT = "你是 AIOps Agent 的轻量闲聊节点。直接、简洁地回答用户，不要说工具调用相关的内容。"

def chat_responder(llm, user_input: str, session_context: str = "", reason: str = "") -> dict:
    try:
        parts = [f"User input:\n{user_input}"]
        if session_context:
            parts.append(f"Session context:\n{session_context}")
        resp = llm.invoke([SystemMessage(content=PROMPT), HumanMessage(content="\n\n".join(parts))])
        text = str(getattr(resp, "content", "") or "").strip()
    except Exception as e:
        text = f"暂不可用: {e}"
    return {"response": text or "我在。你可以继续提问。", "reason": reason}
