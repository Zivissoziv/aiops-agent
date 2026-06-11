# d:\workspace\aiops-agent\src\aiops_agent\graph\_utils.py
"""Graph 内部工具 — 不对外暴露。"""


def _get_writer():
    """获取 LangGraph stream writer，回退到 no-op。

    在 graph.stream() 上下文内返回真实的 StreamWriter；
    外部调用时返回 no-op 函数。
    """
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None
