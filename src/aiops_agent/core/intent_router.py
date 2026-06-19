"""意图路由 — 判断用户输入是闲聊还是工作任务。"""

import json
from langchain_core.messages import HumanMessage, SystemMessage

PROMPT = """你是 AIOps Agent 的意图路由。将用户输入分类为 chat（闲聊）或 task（需要工具的运维任务）。
注意：查询公司内部规范/配置必须归为 task。
返回JSON: {"route":"chat"|"task","reason":"理由","confidence":0.0-1.0}
如果不确定，选择 task。confidence >= 0.55 才信任。"""

def classify_intent(llm, user_input: str, session_context: str = "") -> dict:
    route, reason, confidence = "task", "router fallback: default to task", 0.0
    try:
        parts = [f"User input:\n{user_input}"]
        if session_context:
            parts.append(f"Session context:\n{session_context}")
        resp = llm.invoke([SystemMessage(content=PROMPT), HumanMessage(content="\n\n".join(parts))])
        parsed = _extract_json(str(resp.content)) or {}
        r = str(parsed.get("route", "")).strip().lower()
        c = max(0.0, min(1.0, float(parsed.get("confidence", 0) or 0)))
        if r in {"chat", "task"} and c >= 0.55:
            route, confidence, reason = r, c, str(parsed.get("reason") or "")
        else:
            reason = str(parsed.get("reason") or "low confidence")
            confidence = c
    except Exception as e:
        reason = f"router error: {e}"
    return {"route": route, "reason": reason, "confidence": confidence}

def _extract_json(text: str) -> dict | None:
    s, e = text.find("{"), text.rfind("}")
    if s == -1 or e == -1 or e < s: return None
    try:
        p = json.loads(text[s:e+1])
        return p if isinstance(p, dict) else None
    except json.JSONDecodeError:
        return None
