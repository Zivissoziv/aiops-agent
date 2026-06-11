"""意图路由 — 判断用户输入是闲聊还是工作任务。"""

INTENT_ROUTER_PROMPT = """你是一个意图分类器。判断用户最新输入属于哪一类：

- chat: 打招呼、感谢、询问身份、简单问答等不需要操作工作区的纯对话
- task: 需要读写文件、执行命令、查询系统信息、安装软件等需要工具操作的任务

只返回以下 JSON 格式（不要加其他内容）：
{"route": "chat" | "task", "reason": "简短原因"}

如果不确定，选 task。"""


def classify_intent(llm, user_input: str, history_hint: str = "") -> dict:
    """判断用户意图，返回 {"route": ..., "reason": ...}"""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=INTENT_ROUTER_PROMPT),
    ]
    if history_hint:
        messages.append(SystemMessage(content=f"对话历史摘要: {history_hint}"))
    messages.append(HumanMessage(content=user_input))

    import json
    try:
        response = llm.invoke(messages)
        parsed = json.loads(response.content.strip())
        route = parsed.get("route", "task")
        reason = parsed.get("reason", "")
        if route not in ("chat", "task"):
            route = "task"
        return {"route": route, "reason": reason, "raw": response.content.strip()[:200]}
    except Exception as e:
        return {"route": "task", "reason": f"路由解析失败: {e}"}
