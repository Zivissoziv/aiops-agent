"""Graph 内部工具。"""
def _get_writer():
    try:
        from langgraph.config import get_stream_writer
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None
