"""意图路由 — 判断用户输入是闲聊还是工作任务。"""

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

INTENT_ROUTER_PROMPT = """你是 AIOps Agent 的意图路由。

将用户最新输入分类为以下两种路由之一：

- chat: 问候、感谢、身份/帮助类问题、普通概念问答、不需要执行工具或访问工作空间的对话
- task: 需要读写文件、执行命令、安装包、搜索网络、查询系统状态、分析日志、产出具体成果的请求

如果有对话上下文，仅用于判断当前输入是否为之前工作任务的延续。
简短跟进（如"继续"、"修一下"、"运行测试"、"continue"）应归类为 task。

返回 JSON，格式如下：
{"route":"chat"|"task","reason":"简要理由","confidence":0.0-1.0}

如果不确定，选择 task。confidence >= 0.55 才信任路由结果。
"""


def classify_intent(
    llm,
    user_input: str,
    session_context: str = "",
) -> dict[str, Any]:
    """判断用户意图，返回 {"route": ..., "reason": ..., "confidence": ...}。

    支持 MokioAgent 风格的结构化 JSON 输出 + 置信度门控：
    - confidence >= 0.55 时信任模型分类
    - 低于阈值 → 默认走 task（安全回退）
    - 异常 → 默认走 task
    """
    route = "task"
    reason = "router fallback: default to task"
    confidence = 0.0

    try:
        parts = [f"User input:\n{user_input}"]
        if session_context:
            parts.append(f"Session context:\n{session_context}")
        prompt_input = "\n\n".join(parts)

        response = llm.invoke([
            SystemMessage(content=INTENT_ROUTER_PROMPT),
            HumanMessage(content=prompt_input),
        ])
        parsed = _extract_json(str(response.content)) or {}
        candidate = str(parsed.get("route", "")).strip().lower()
        parsed_confidence = _coerce_confidence(parsed.get("confidence"))

        if candidate in {"chat", "task"} and parsed_confidence >= 0.55:
            route = candidate
            confidence = parsed_confidence
            reason = str(parsed.get("reason") or "")
        else:
            reason = str(parsed.get("reason") or "router returned low-confidence or invalid route")
            confidence = parsed_confidence
    except Exception as exc:
        reason = f"router error: {type(exc).__name__}: {exc}"

    return {
        "route": route,
        "reason": reason,
        "confidence": confidence,
    }


def _extract_json(text: str) -> dict[str, Any] | None:
    """从模型回复中提取 JSON 对象。

    使用 rfind 找最外层大括号，对 LLM 输出的简单 JSON 更鲁棒，
    不会因 reason 字段中包含 {} 字符而截断。
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_confidence(value: Any) -> float:
    """将任意值转换为 [0.0, 1.0] 范围的置信度分数。"""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))
